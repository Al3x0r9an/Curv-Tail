#!/usr/bin/env python
"""Build the small synthetic CURV-TAIL demo cache used by the smoke test.

Writes a data cache that follows the *exact* on-disk schema described in
``README.md -> Data cache format``:

    demo_cache/
        cache_manifest.json   cache metadata (widths, feature names)
        manifest.jsonl        one flow per line: sample_id / label / label_name /
                              split / sequence_path
        split.json            {"train": [...], "val": [...], "test": [...]}
        sequences.npy         float32 [N, T, 2] packet features
        lengths.npy           int64  [N] valid packet count per flow
        fwd_bytes.npy         uint16 [N, M] forward payload bytes (b -> b+1)
        bwd_bytes.npy         uint16 [N, M] backward payload bytes (b -> b+1)
        labels.npy            int64  [N] integer class per flow
        sample_ids.txt        one sample id per line

The packet sizes mix the exact-length regime (<= 1503 bytes) with occasional
large "coalesced" packets that exercise the R4 coarse buckets, so both halves
of :func:`curv_tail.models.discretize_length` are covered by the smoke test.

Usage:
    python scripts/build_demo_cache.py [--out demo_cache] [--flows 600]
                                       [--classes 12]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

_FINE_CEILING = 1503  # must match curv_tail.models._FINE_SIZE_CEILING


def _class_counts(num_flows: int, num_classes: int, rng: np.random.Generator) -> np.ndarray:
    """Long-tail-shaped per-class flow counts, randomly assigned to class ids.

    The *multiset* of counts decays geometrically, so a few classes are frequent
    and most are rare; the decayed values are then permuted over the class ids so
    that class 0 is not systematically the head class.  With many classes and few
    flows the rarest classes can end up with no val / test flow, so the demo's
    test metrics need not cover every class.
    """
    weights = np.exp(-0.35 * np.arange(num_classes))
    raw = np.floor(num_flows * weights / weights.sum()).astype(np.int64)
    raw[0] += num_flows - int(raw.sum())
    perm = rng.permutation(num_classes)  # shuffle which id is "head" to be fair
    counts = np.zeros(num_classes, dtype=np.int64)
    counts[perm] = raw
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="demo_cache", help="Output cache directory.")
    parser.add_argument("--flows", type=int, default=600)
    parser.add_argument("--classes", type=int, default=12)
    args = parser.parse_args()

    num_flows, num_classes = args.flows, args.classes
    max_packets = 16
    max_bytes = 64  # divisible by byte_patch_size (8) in configs/demo.yaml
    rng = np.random.default_rng(0)
    random.seed(0)

    counts = _class_counts(num_flows, num_classes, rng)
    label_of = np.concatenate([np.full(int(count), c) for c, count in enumerate(counts)])
    order = rng.permutation(num_flows)
    labels = label_of[order]

    sequences = np.zeros((num_flows, max_packets, 2), dtype=np.float32)
    lengths = np.zeros(num_flows, dtype=np.int64)
    fwd_bytes = np.zeros((num_flows, max_bytes), dtype=np.uint16)
    bwd_bytes = np.zeros((num_flows, max_bytes), dtype=np.uint16)
    sample_ids = [f"flow_{i:05d}" for i in range(num_flows)]

    for i in range(num_flows):
        n_packets = int(rng.integers(5, max_packets + 1))
        lengths[i] = n_packets
        for t in range(n_packets):
            if rng.random() < 0.08:  # occasional coalesced packet -> coarse bucket
                size = int(rng.integers(_FINE_CEILING + 1, 46_000))
            else:
                size = int(rng.integers(20, _FINE_CEILING + 1))
            direction = 1.0 if rng.random() < 0.5 else -1.0
            iat = float(np.exp(rng.uniform(np.log(1e-4), np.log(2.0))))
            sequences[i, t, 0] = direction * float(np.log1p(size))
            sequences[i, t, 1] = float(np.log1p(iat))
        for stream in (fwd_bytes[i], bwd_bytes[i]):
            n_bytes = int(rng.integers(1, max_bytes + 1))
            stream[:n_bytes] = rng.integers(1, 257, size=n_bytes)  # byte b -> b+1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_name": "CURV-TAIL-synthetic-demo",
        "num_samples": num_flows,
        "num_classes": num_classes,
        "max_packets": max_packets,
        "max_bytes_per_direction": max_bytes,
        "features": ["signed_log1p_packet_size", "log1p_inter_arrival_time"],
        "seed": 0,
    }
    (out / "cache_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.save(out / "sequences.npy", sequences)
    np.save(out / "lengths.npy", lengths)
    np.save(out / "fwd_bytes.npy", fwd_bytes)
    np.save(out / "bwd_bytes.npy", bwd_bytes)
    np.save(out / "labels.npy", labels)
    (out / "sample_ids.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")

    # Roughly 80 / 10 / 10 split, computed per class; every class keeps >=1
    # train flow, but the rarest ones may receive no val / test flow.
    train_ids, val_ids, test_ids = [], [], []
    for c in range(num_classes):
        members = [sid for sid, label in zip(sample_ids, labels) if int(label) == c]
        rng.shuffle(members)
        n_train = max(1, int(0.8 * len(members)))
        train_ids += members[:n_train]
        val_ids += members[n_train : n_train + max(1, (len(members) - n_train) // 2)]
        test_ids += members[n_train + max(1, (len(members) - n_train) // 2):]
    split = {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    split_of = {sid: name for name, ids in split.items() for sid in ids}
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for i, sid in enumerate(sample_ids):
            row = {
                "sample_id": sid,
                "label": int(labels[i]),
                "label_name": f"app_{int(labels[i]):02d}",
                "split": split_of[sid],
                "sequence_path": f"synth/{sid}.pcap",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote demo cache to {out.resolve()}  ({num_flows} flows, {num_classes} classes)")
    for name, ids in split.items():
        print(f"  {name:5s}: {len(ids):4d} flows")


if __name__ == "__main__":
    main()
