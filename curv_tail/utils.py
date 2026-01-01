"""Small shared helpers: RNG seeding, JSONL logging, checkpoint I/O.

CURV-TAIL requires deterministic, reproducible runs.  :func:`seed_everything`
fixes Python / NumPy / torch / CUDA RNGs, and every checkpoint stores the full
RNG state so that a resumed run continues from the same random stream.  The
data order, the initialisation and the optimisation trajectory are therefore
reproducible, but see :func:`seed_everything` for the two caveats that keep a
GPU run from being bit-exact.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "seed_everything",
    "append_jsonl",
    "utc_stamp",
    "count_parameters",
    "capture_rng_state",
    "restore_rng_state",
    "atomic_torch_save",
    "torch_load_checkpoint",
    "rotate_epoch_snapshots",
]


def seed_everything(seed: int) -> None:
    """Seed all random sources for a reproducible run.

    Args:
        seed: Integer seed.

    Note:
        ``PYTHONHASHSEED`` is exported for child processes; it has no effect on
        the interpreter that is already running.  Bit-exact reproducibility also
        requires deterministic cuDNN kernels, which the training CLI does not
        force (it enables ``cudnn.benchmark`` instead), so GPU runs are
        reproducible only up to non-deterministic kernel selection.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON object as a newline to a JSONL file, flushed immediately.

    Args:
        path: Destination file path.
        row: Dict to serialize.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def utc_stamp() -> str:
    """Return a compact UTC timestamp for run-directory naming.

    Returns:
        String ``YYYYMMDD_HHMMSS``.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters of a module.

    Args:
        model: The module.

    Returns:
        ``{"total": ..., "trainable": ...}``.
    """
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def capture_rng_state() -> dict[str, Any]:
    """Snapshot all RNG states for resumption.

    Returns:
        Dict holding Python, NumPy, torch and (if present) CUDA RNG states.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by :func:`capture_rng_state`.

    Args:
        state: The dict returned by :func:`capture_rng_state`.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Write a torch checkpoint atomically (tmp file + ``os.replace``).

    Args:
        payload: Dict to ``torch.save``.
        path: Destination path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def torch_load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint onto CPU.

    Args:
        path: Checkpoint file path.

    Returns:
        The loaded payload dict.
    """
    return torch.load(path, map_location="cpu")


def rotate_epoch_snapshots(directory: Path, keep: int) -> None:
    """Keep only the ``keep`` most recent ``latest_epoch_*.pt`` snapshots.

    Args:
        directory: Snapshot directory.
        keep: Number of newest snapshots to retain.
    """
    files = sorted(directory.glob("latest_epoch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[int(keep) :]:
        path.unlink(missing_ok=True)
