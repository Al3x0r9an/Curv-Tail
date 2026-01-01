#!/usr/bin/env python
"""Convert a sample CSV into a runnable CURV-TAIL "mini" cache.

The sample CSVs produced by :file:`export_mini_samples.py` (or shipped under
``data/samples/``) contain one row per flow with a single ``packets`` column of
the form ``"c0,c1;c0,c1;..."`` -- the raw cache channels
(``signed_log1p(packet_size)``, ``log1p(iat)``) per valid packet.  This script
re-assembles them into the on-disk cache schema documented in ``README.md``
(sequences / lengths / labels / sample_ids / split / manifest), so that the
model can be trained and evaluated on them.

The shipped samples carry the bidirectional payload-byte columns
(``fwd_bytes`` / ``bwd_bytes``); when present they are copied into the cache
byte arrays and the byte branch is exercised with real payloads.  A CSV
exported without ``--include-bytes`` (see ``export_mini_samples.py``) has no
byte columns; those arrays are then zero-filled and the byte branch
contributes no signal, though its code path is still exercised.  Full-fidelity
runs use the complete caches distributed from Google Drive (see README).

Usage
-----
    python scripts/sample_csv_to_cache.py \\
        --csv data/samples/datacon_website_mini.csv \\
        --out data/mini/datacon_website --name datacon_website_mini

The resulting cache directory is ready for ``configs/mini_datacon_website.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Sample CSV (with or without fwd_bytes/bwd_bytes columns).")
    parser.add_argument("--out", required=True, help="Output cache directory.")
    parser.add_argument("--name", default="mini", help="Dataset name for cache_manifest.json.")
    parser.add_argument("--max-packets", type=int, default=64)
    parser.add_argument("--max-bytes", type=int, default=256)
    args = parser.parse_args()

    rows: list[dict] = []
    with Path(args.csv).open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            record["label"] = int(record["label"])
            record["num_packets"] = int(record["num_packets"])
            rows.append(record)

    n = len(rows)
    if n == 0:
        raise SystemExit("empty sample CSV")
    sample_ids = [r["sample_id"] for r in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise SystemExit("duplicate sample_id in CSV")

    has_bytes = "fwd_bytes" in rows[0] and "bwd_bytes" in rows[0]
    sequences = np.zeros((n, args.max_packets, 2), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    fwd_bytes = np.zeros((n, args.max_bytes), dtype=np.uint16)
    bwd_bytes = np.zeros((n, args.max_bytes), dtype=np.uint16)
    for i, row in enumerate(rows):
        num = row["num_packets"]
        if not 1 <= num <= args.max_packets:
            raise SystemExit(f"row {row['sample_id']}: bad num_packets {num}")
        lengths[i] = num
        tokens = [p for p in (row["packets"] or "").split(";") if p]
        if len(tokens) != num:
            raise SystemExit(f"row {row['sample_id']}: num_packets mismatch with packets")
        for t, token in enumerate(tokens):
            c0, c1 = token.split(",")
            sequences[i, t, 0] = np.float32(float(c0))
            sequences[i, t, 1] = np.float32(float(c1))
        if has_bytes:
            for field, dest in (("fwd_bytes", fwd_bytes[i]), ("bwd_bytes", bwd_bytes[i])):
                values = [int(v) for v in (row[field] or "").split(",") if v]
                if len(values) > args.max_bytes:
                    raise SystemExit(f"row {row['sample_id']}: {field} longer than {args.max_bytes}")
                if values and min(values) < 1:
                    raise SystemExit(f"row {row['sample_id']}: byte tokens must be 1..256 (0 is padding)")
                dest[: len(values)] = np.asarray(values, dtype=np.uint16)

    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    labels_set = sorted(set(int(x) for x in labels))
    if labels_set != list(range(labels_set[-1] + 1)):
        raise SystemExit("labels must form a contiguous 0..C-1 range")

    split_sets = {name: [r["sample_id"] for r in rows if r["split"] == name]
                  for name in ("train", "val", "test")}
    for name in split_sets:
        if name == "train" and not split_sets[name]:
            raise SystemExit("sample has no training rows")
    union = [sid for name in ("train", "val", "test") for sid in split_sets[name]]
    if sorted(union) != sorted(sample_ids):
        raise SystemExit("split assignment must cover every row exactly once")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(split_sets[a]) & set(split_sets[b]):
            raise SystemExit(f"{a}/{b} splits overlap")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_name": args.name,
        "kind": "mini_sample",
        "note": ("contains payload bytes" if has_bytes
                 else "payload bytes omitted; byte arrays zero-filled"),
        "num_samples": n,
        "num_classes": int(labels.max()) + 1,
        "max_packets": args.max_packets,
        "max_bytes_per_direction": args.max_bytes,
        "features": ["signed_log1p_packet_size", "log1p_inter_arrival_time"],
    }
    (out / "cache_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "split.json").write_text(json.dumps(split_sets, indent=2), encoding="utf-8")
    np.save(out / "sequences.npy", sequences)
    np.save(out / "lengths.npy", lengths)
    np.save(out / "labels.npy", labels)
    np.save(out / "fwd_bytes.npy", fwd_bytes)
    np.save(out / "bwd_bytes.npy", bwd_bytes)
    (out / "sample_ids.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "sample_id": row["sample_id"],
                "label": row["label"],
                "label_name": row["label_name"],
                "sequence_path": "mini-sample",
            }, ensure_ascii=False) + "\n")

    print(f"wrote mini cache to {out.resolve()}")
    print(f"  flows={n} classes={metadata['num_classes']} "
          f"per-split={ {k: len(v) for k, v in split_sets.items()} }")
    print("  expected_num_samples for the mini config:", n)


if __name__ == "__main__":
    main()
