"""Training and evaluation CLI for CURV-TAIL.

Usage
-----
Train::

    curv-train --config configs/datacon_website.yaml --seed 42

or equivalently ``python -m curv_tail.train --config <cfg> [--seed S]``.

Evaluate a saved checkpoint once::

    curv-train --config <cfg> --mode test --checkpoint <run>/checkpoints/best_macro_f1.pt

The trainer is deliberately small: beyond the standard library it uses torch,
numpy, PyYAML, and (via :mod:`curv_tail.metrics`) pandas and scikit-learn.
Loops, schedule and precision come from the YAML
config, while the optimizer is always AdamW and the supervised loss always
cross-entropy (see :mod:`curv_tail.config`):

* deterministic per-epoch shuffling via an :class:`EpochRandomSampler`
  (seeded by ``seed + epoch``);
* AdamW + cosine learning-rate schedule with linear warm-up;
* bf16 / fp16 autocast (``training.amp``), optional grad clipping;
* checkpoints saved every epoch under ``checkpoints/``; ``best_macro_f1.pt``
  and ``best_tail_macro_f1.pt`` track the validation-selection metrics;
* a masked length-token reconstruction auxiliary loss (``rec_weight``) is added
  to the cross-entropy loss automatically when the model enables ``rec_head``.

Run layout
----------
By default a run creates ``<experiment.output_root>/<name>_seed<S>_<utc>/``,
holding ``resolved_config.yaml``, ``config.sha256``, ``model_info.json`` and
``artifacts/*``; a training run additionally writes ``logs/epochs.jsonl`` and
``checkpoints/*``, and a test run writes ``test_once/*``.  The run directory is
instead taken from ``--run-dir`` when it
is given, and is otherwise inferred from the checkpoint passed to ``--resume``
or ``--mode test`` so that an existing run is continued / evaluated in place.  A
configuration content hash (:func:`curv_tail.config.config_hash`) is stored with
each checkpoint so a checkpoint can only be tested/resumed against an identical
configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from .config import config_hash, load_config, public_config
from .data import (
    MultiViewTrafficDataset,
    compute_train_normalization,
    cumulative_frequency_groups,
    load_manifest,
    load_multiview_cache,
    load_split,
    records_by_split,
    training_class_counts,
)
from .metrics import compute_metrics, json_ready, per_class_table
from .models import build_model
from .utils import (
    append_jsonl,
    atomic_torch_save,
    capture_rng_state,
    count_parameters,
    restore_rng_state,
    rotate_epoch_snapshots,
    seed_everything,
    torch_load_checkpoint,
    utc_stamp,
)

__all__ = ["main", "run", "run_dir_for_checkpoint"]


class EpochRandomSampler(Sampler[int]):
    """Sampler whose permutation depends deterministically on the epoch.

    The permutation for epoch ``e`` is generated from ``seed + e``.  This makes
    every training run reproducible for a fixed seed while still re-shuffling
    across epochs.

    Args:
        size: Dataset size.
        seed: Base random seed.
    """

    def __init__(self, size: int, seed: int) -> None:
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch (used to derive the shuffle key)."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def make_loader(dataset, cfg: dict[str, Any], *, train: bool, sampler=None) -> DataLoader:
    """Build a DataLoader honoring the config's batch sizes and worker count.

    Windows ``spawn``-based workers can intermittently fail with OSError 22 /
    pickle-truncated errors; because cache reads are mmap-backed and cheap the
    default worker count is 0 (synchronous main-process loading).  This does not
    change results: ordering is fixed by the explicit sampler / ``shuffle=False``.

    Args:
        dataset: A torch Dataset.
        cfg: Full config mapping.
        train: Whether this is the training loader (uses ``batch_size`` and
            ``sampler``); otherwise it is an eval loader (``eval_batch_size``).
        sampler: Optional deterministic sampler (train only).

    Returns:
        A configured ``DataLoader``.
    """
    training = cfg["training"]
    workers = int(training.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(training["batch_size"] if train else training["eval_batch_size"]),
        "sampler": sampler,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(training.get("pin_memory", True)),
        "persistent_workers": bool(training.get("persistent_workers", True)) and workers > 0,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def amp_context(device: torch.device, mode: str):
    """Return an autocast context manager for the configured precision mode.

    Args:
        device: The compute device.
        mode: ``"bf16"``, ``"fp16"`` or ``"none"``.

    Returns:
        A ``torch.autocast`` context (or a null context on CPU / ``none``).
    """
    if device.type != "cuda" or mode == "none":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16)


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Plain cross-entropy (CURV-TAIL's supervised loss).

    Args:
        logits: ``[B, C]`` model logits.
        labels: ``[B]`` integer class ids.

    Returns:
        Scalar cross-entropy loss.
    """
    return F.cross_entropy(logits.float(), labels)


def collect_batch(labels_store, pred_store, topk_store, labels: torch.Tensor, logits: torch.Tensor) -> None:
    """Accumulate labels / argmax predictions / top-k predictions for an epoch."""
    labels_store.append(labels.detach().cpu().numpy())
    topk = torch.topk(logits.detach().float(), k=min(10, logits.shape[1]), dim=1).indices.cpu().numpy()
    topk_store.append(topk)
    pred_store.append(topk[:, 0])


def run_epoch(
    *,
    model,
    loader,
    device,
    groups,
    optimizer,
    scheduler,
    accumulation: int,
    grad_clip: float,
    amp_mode: str,
    global_step: int,
):
    """Run one training or evaluation pass over ``loader``.

    Args:
        model: The CURV-TAIL model.
        loader: DataLoader to iterate.
        device: Compute device.
        groups: Head/body/tail group per class.
        optimizer: Optimizer when training, else ``None``.
        scheduler: LR scheduler (stepped per optimizer step) or ``None``.
        accumulation: Gradient accumulation steps.
        grad_clip: Gradient norm clip (ignored when 0).
        amp_mode: Precision mode for :func:`amp_context`.
        global_step: Step counter to continue from (train only).

    Returns:
        ``(metrics_dict, global_step, (labels, predictions, topk))``.
    """
    training = optimizer is not None
    model.train(training)
    labels_store, pred_store, topk_store = [], [], []
    total_loss = total_samples = 0
    aux_sums: dict[str, float] = {}
    aux_counts: dict[str, int] = {}
    started = time.perf_counter()
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with amp_context(device, amp_mode):
                fwd_bytes = batch["fwd_bytes"].to(device, non_blocking=True)
                bwd_bytes = batch["bwd_bytes"].to(device, non_blocking=True)
                logits, aux = model(x, mask, fwd_bytes, bwd_bytes, return_aux=True)
                loss = classification_loss(logits, labels)
                if "rec_loss" in aux:
                    loss = loss + aux["rec_loss"]
            if training:
                (loss / accumulation).backward()
                should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
                if should_step:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    global_step += 1
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        collect_batch(labels_store, pred_store, topk_store, labels, logits)
        for key in ("manifold_error_mean", "manifold_error_max", "curvature", "packet_lift_scale", "byte_lift_scale"):
            if key in aux:
                aux_sums[key] = aux_sums.get(key, 0.0) + float(aux[key])
                aux_counts[key] = aux_counts.get(key, 0) + 1
    labels_np = np.concatenate(labels_store)
    predictions_np = np.concatenate(pred_store)
    topk_np = np.concatenate(topk_store)
    metrics = compute_metrics(labels_np, predictions_np, topk_np, groups)
    elapsed = time.perf_counter() - started
    metrics.update(
        loss=total_loss / max(total_samples, 1),
        num_samples=total_samples,
        elapsed_seconds=elapsed,
        samples_per_second=total_samples / max(elapsed, 1e-9),
        **{key: value / aux_counts[key] for key, value in aux_sums.items()},
    )
    return metrics, global_step, (labels_np, predictions_np, topk_np)


def scheduler_for(optimizer, loader_steps: int, cfg: dict[str, Any]):
    """Cosine LR schedule with linear warm-up, mirroring the released recipe.

    Args:
        optimizer: Optimizer to schedule.
        loader_steps: Number of batches per epoch.
        cfg: Full config mapping.

    Returns:
        A ``torch.optim.lr_scheduler.LambdaLR`` mapping global step to LR.
    """
    training = cfg["training"]
    updates_per_epoch = math.ceil(loader_steps / int(training["gradient_accumulation"]))
    total = updates_per_epoch * int(training["epochs"])
    warmup = updates_per_epoch * int(training.get("warmup_epochs", 0))
    min_ratio = float(training.get("min_learning_rate", 0.0)) / float(training["learning_rate"])

    def scale(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max((step + 1) / warmup, 1e-3)
        progress = (step - warmup) / max(total - warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def prepare(cfg: dict[str, Any], run_dir: Path):
    """Load the cache + splits, build datasets and long-tail groups.

    Args:
        cfg: Full config mapping.
        run_dir: Run directory (for artifact writing).

    Returns:
        ``(datasets, counts, groups)`` where ``datasets`` maps split names to
        :class:`MultiViewTrafficDataset` objects.
    """
    records = load_manifest(cfg["data"]["manifest"])
    split_records = records_by_split(records, load_split(cfg["data"]["split"]))
    # CURV-TAIL releases only the double-branch multiview view: the cache must
    # hold both the packet sequence and the bidirectional byte payloads.
    if str(cfg["data"]["input_view"]) != "packet_bytes_multiview":
        raise ValueError("CURV-TAIL requires data.input_view = packet_bytes_multiview")
    cache = load_multiview_cache(cfg["data"]["cache_dir"])
    if len(records) != int(cfg["data"]["expected_num_samples"]):
        raise RuntimeError("manifest does not contain the expected sample count")
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    normalization_path = artifact_dir / "train_normalization.json"
    if normalization_path.exists():
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    else:
        normalization = compute_train_normalization(split_records["train"], cache)
        normalization_path.write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    counts = training_class_counts(split_records["train"], int(cfg["data"]["expected_num_classes"]))
    groups = cumulative_frequency_groups(counts)
    (artifact_dir / "class_counts.json").write_text(json.dumps(counts.tolist()), encoding="utf-8")
    (artifact_dir / "class_groups.json").write_text(json.dumps(groups.tolist()), encoding="utf-8")
    # CURV-TAIL recovers the exact length token from the raw signed_log1p
    # channel, so datasets must NOT standardize the sequence.
    datasets = {
        name: MultiViewTrafficDataset(rows, cache, normalization, standardize=False)
        for name, rows in split_records.items()
    }
    return datasets, counts, groups


def checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_macro_f1: float,
    best_tail_macro_f1: float,
    best_macro_epoch: int,
    best_tail_epoch: int,
    config: dict[str, Any],
    config_hash_str: str,
    run_id: str,
) -> dict[str, Any]:
    """Assemble the full checkpoint payload (state + config hash + RNG state)."""
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_macro_f1": float(best_macro_f1),
        "best_tail_macro_f1": float(best_tail_macro_f1),
        "best_macro_epoch": int(best_macro_epoch),
        "best_tail_epoch": int(best_tail_epoch),
        "config": config,
        "config_hash": str(config_hash_str),
        "run_id": str(run_id),
        "rng_state": capture_rng_state(),
    }


def save_payloads(run_dir: Path, payload, epoch: int, keep: int, save_best_macro: bool, save_best_tail: bool) -> None:
    """Write latest / history / best checkpoints for one epoch."""
    checkpoint_dir = run_dir / "checkpoints"
    atomic_torch_save(payload, checkpoint_dir / "latest.pt")
    snapshot_dir = checkpoint_dir / "latest_history"
    atomic_torch_save(payload, snapshot_dir / f"latest_epoch_{epoch:04d}.pt")
    rotate_epoch_snapshots(snapshot_dir, keep)
    if save_best_macro:
        atomic_torch_save(payload, checkpoint_dir / "best_macro_f1.pt")
    if save_best_tail:
        atomic_torch_save(payload, checkpoint_dir / "best_tail_macro_f1.pt")


def run_dir_for_checkpoint(checkpoint: str | Path) -> Path:
    """Infer the run directory that owns a checkpoint file.

    Checkpoints are written to ``<run>/checkpoints/*.pt``, with rolling
    snapshots one level deeper in ``<run>/checkpoints/latest_history/``.

    Args:
        checkpoint: Path to a checkpoint file.

    Returns:
        The run directory the checkpoint belongs to.
    """
    parent = Path(checkpoint).resolve().parent
    if parent.parent.name == "checkpoints":
        parent = parent.parent
    return parent.parent


def run(argv: list[str] | None = None) -> None:
    """CLI entry point (see module docstring for usage)."""
    parser = argparse.ArgumentParser(
        description="CURV-TAIL: train or evaluate the curvature-adaptive Lorentz traffic CNN."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML configuration.")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--run-dir", help="Explicit output run directory (overrides auto-naming).")
    parser.add_argument("--resume", help="Checkpoint to resume training from.")
    parser.add_argument("--checkpoint", help="Checkpoint to evaluate in test mode.")
    parser.add_argument("--seed", type=int, help="Seed override; stored in the resolved config.")
    parser.add_argument("--tag", help="Optional run-name suffix (letters/digits/_/-).")
    args = parser.parse_args(argv)
    # Validate the invocation before any run directory is created or any cache is
    # loaded, so a mistyped checkpoint path fails fast instead of creating a run
    # directory next to it and filling it with stray artifacts.
    if args.mode == "test":
        if not args.checkpoint:
            raise FileNotFoundError("--checkpoint is required for test mode")
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"--checkpoint not found: {args.checkpoint}")
    if args.resume and not Path(args.resume).exists():
        raise FileNotFoundError(f"--resume not found: {args.resume}")

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = int(args.seed)
    if args.tag:
        if not all(character.isalnum() or character in "-_" for character in args.tag):
            raise ValueError("--tag may contain only letters, digits, '-' and '_'")
        cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_{args.tag}"
    cfg_hash = config_hash(cfg)
    seed = int(cfg["experiment"]["seed"])
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.mode == "test" and args.checkpoint:
        # Evaluate in place: reuse the run directory that owns the checkpoint so
        # its artifacts (train normalization, class groupings) are reused.
        run_dir = run_dir_for_checkpoint(args.checkpoint)
    elif args.resume:
        run_dir = run_dir_for_checkpoint(args.resume)
    else:
        output_root = cfg["experiment"].get("output_root", "outputs")
        run_dir = Path(output_root) / f"{cfg['experiment']['name']}_seed{seed}_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    resolved_path = run_dir / "resolved_config.yaml"
    if not resolved_path.exists():
        resolved_path.write_text(
            yaml.safe_dump(public_config(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    (run_dir / "config.sha256").write_text(cfg_hash + "\n", encoding="utf-8")

    datasets, counts, groups = prepare(cfg, run_dir)
    # Sequence width: the model's positional bias must at least cover the cache
    # width.  Prefer the explicit data.max_packets, else read it off the cache.
    max_packets = int(cfg["data"].get("max_packets") or datasets["train"].cache.sequences.shape[1])
    model = build_model(cfg["model"], int(cfg["data"]["expected_num_classes"]), max_packets)
    model.to(device)
    model_info = {
        "run_id": run_id,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "parameters": count_parameters(model),
        "config_hash": cfg_hash,
    }
    (run_dir / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    print(json.dumps(model_info, ensure_ascii=False), flush=True)

    train_cfg = cfg["training"]
    sampler = EpochRandomSampler(len(datasets["train"]), seed)
    train_loader = make_loader(datasets["train"], cfg, train=True, sampler=sampler)
    val_loader = make_loader(datasets["val"], cfg, train=False)
    amp_mode = str(train_cfg.get("amp", "bf16"))

    # ------------------------------------------------------------------ test --
    if args.mode == "test":
        payload = torch_load_checkpoint(args.checkpoint)
        if payload["config_hash"] != cfg_hash:
            raise RuntimeError("checkpoint/config hash mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        test_loader = make_loader(datasets["test"], cfg, train=False)
        with torch.inference_mode():
            metrics, _, arrays = run_epoch(
                model=model, loader=test_loader, device=device, groups=groups, optimizer=None,
                scheduler=None, accumulation=1, grad_clip=0.0, amp_mode=amp_mode,
                global_step=int(payload["global_step"]),
            )
        # Re-running test mode on the same run directory replaces the previous
        # test_once/ artifacts instead of raising FileExistsError.
        result_dir = run_dir / "test_once"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "metrics.json").write_text(json.dumps(json_ready(metrics), indent=2), encoding="utf-8")
        labels, predictions, topk = arrays
        np.savez_compressed(result_dir / "predictions.npz", labels=labels, predictions=predictions, topk=topk)
        per_class_table(labels, predictions, groups, counts).to_csv(result_dir / "per_class.csv", index=False)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        return

    # ---------------------------------------------------------------- train --
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"])
    )
    scheduler = scheduler_for(optimizer, len(train_loader), cfg)
    start_epoch, global_step = 1, 0
    best_macro = best_tail = -1.0
    best_macro_epoch = best_tail_epoch = 0
    if args.resume:
        payload = torch_load_checkpoint(args.resume)
        if payload["config_hash"] != cfg_hash:
            raise RuntimeError("resume checkpoint/config hash mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best_macro, best_tail = float(payload["best_macro_f1"]), float(payload["best_tail_macro_f1"])
        best_macro_epoch, best_tail_epoch = int(payload["best_macro_epoch"]), int(payload["best_tail_epoch"])

    no_improvement = 0
    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        sampler.set_epoch(epoch)
        train_metrics, global_step, _ = run_epoch(
            model=model, loader=train_loader, device=device, groups=groups, optimizer=optimizer,
            scheduler=scheduler, accumulation=int(train_cfg["gradient_accumulation"]),
            grad_clip=float(train_cfg["grad_clip_norm"]), amp_mode=amp_mode, global_step=global_step,
        )
        with torch.inference_mode():
            val_metrics, global_step, val_arrays = run_epoch(
                model=model, loader=val_loader, device=device, groups=groups, optimizer=None,
                scheduler=None, accumulation=1, grad_clip=0.0, amp_mode=amp_mode, global_step=global_step,
            )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_jsonl(run_dir / "logs" / "epochs.jsonl", json_ready(row))
        print(json.dumps(json_ready(row), ensure_ascii=False), flush=True)
        improve_macro = float(val_metrics["macro_f1"]) > best_macro
        improve_tail = float(val_metrics["tail_macro_f1"]) > best_tail
        if improve_macro:
            best_macro, best_macro_epoch = float(val_metrics["macro_f1"]), epoch
        if improve_tail:
            best_tail, best_tail_epoch = float(val_metrics["tail_macro_f1"]), epoch
        no_improvement = 0 if improve_macro else no_improvement + 1
        payload = checkpoint_payload(
            model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, global_step=global_step,
            best_macro_f1=best_macro, best_tail_macro_f1=best_tail, best_macro_epoch=best_macro_epoch,
            best_tail_epoch=best_tail_epoch, config=public_config(cfg), config_hash_str=cfg_hash,
            run_id=run_id,
        )
        save_payloads(run_dir, payload, epoch, int(train_cfg.get("keep_latest", 3)), improve_macro, improve_tail)
        if improve_macro:
            labels, predictions, _ = val_arrays
            per_class_table(labels, predictions, groups, counts).to_csv(
                run_dir / "artifacts" / "best_macro_val_per_class.csv", index=False
            )
        if no_improvement >= int(train_cfg.get("early_stop_patience", 10)):
            print(f"early_stop epoch={epoch} best_macro_epoch={best_macro_epoch}", flush=True)
            break


def main() -> None:
    """Console-script entry point."""
    run()


if __name__ == "__main__":
    main()
