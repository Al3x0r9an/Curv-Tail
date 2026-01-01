#!/usr/bin/env python
"""Export a small, privacy-safe sample of a full CURV-TAIL data cache as CSV.

The sample keeps exactly the features CURV-TAIL consumes: per packet the two
cache channels (``signed_log1p(packet_size)``, ``log1p(iat)``), the bidirectional
payload-byte token streams (when ``--include-bytes`` is given), and the label.
IPs, ports and timestamps are never exported.

* With ``--include-bytes`` the CSV is a faithful slice of the cache and the
  resulting mini run exercises the byte branch with real payloads.  Those bytes
  are raw application data: they contain no IPs, ports or timestamps, but they
  can carry hostnames, HTTP headers and TLS certificate fields, so redistribute
  the CSV only under the source dataset's licence.
* Without it (default) the byte arrays are zero-filled on conversion; the byte
  branch still runs but contributes no signal.  This keeps the sample free of
  any payload in case you distribute it under a stricter dataset license.

The CSV is used by ``sample_csv_to_cache.py`` to build a small runnable "mini"
cache for checking that the pipeline / configuration runs.

Sampling strategy
-----------------
Stratified per class, preserving the original dataset splits:

* train:  ``--train-per-class-high`` flows for the most frequent ~40% classes
  (by original train count) and ``--train-per-class-low`` for the rest,
* val / test: up to 1 flow per class where the original split has one.

Every class that appears in the source ``train`` split keeps at least one
training flow, as the closed-set training requirement demands; a class missing
from that split is dropped from the sample entirely.

Usage
-----
Requires ``curv_tail`` to be importable, since the script reads the cache
directly -- install the package (``pip install -e .``) or run with
``PYTHONPATH=.``::

    python scripts/export_mini_samples.py \
        --cache-dir  <full cache dir> \
        --split      <split.json> \
        --out        data/samples/<dataset>_mini.csv \
        [--include-bytes] [--label-map <label_map.json>]

``--label-map`` is optional: it maps integer labels to human-readable names for
the ``label_name`` column.  Without it ``label_name`` falls back to the numeric
label.  ``--include-bytes`` writes the bidirectional payload-byte columns;
without it those columns are omitted and are zero-filled on conversion.

The float channels are written with enough precision that converting the CSV
back (``sample_csv_to_cache.py``) reproduces the cache rows byte-exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from curv_tail.data import load_multiview_cache


def _label_names(label_map_path: str | None) -> dict[int, str]:
    if not label_map_path or not Path(label_map_path).exists():
        return {}
    payload = json.loads(Path(label_map_path).read_text(encoding="utf-8"))
    names = {}
    for key, value in payload.items():
        try:
            names[int(key)] = str(value)
        except (TypeError, ValueError):
            pass
    return names


def _encode_row(cache, row: int, num_packets: int) -> str:
    sequence = np.asarray(cache.sequences[row], dtype=np.float32)[:num_packets]
    cells = []
    for p in range(num_packets):
        c0 = repr(float(sequence[p, 0]))
        c1 = repr(float(sequence[p, 1]))
        cells.append(f"{c0},{c1}")
    return ";".join(cells)


def _encode_bytes(cache, row: int, field: str) -> str:
    """Serialize the non-padding byte tokens of one direction as comma ints."""
    tokens = np.asarray(getattr(cache, field)[row], dtype=np.int64)
    prefix = int(np.count_nonzero(tokens))
    if prefix == 0:
        return ""
    return ",".join(str(int(t)) for t in tokens[:prefix])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Full multiview cache directory.")
    parser.add_argument("--split", required=True, help="Original split.json.")
    parser.add_argument("--label-map", help="Optional label_map.json (int label -> name).")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--head-fraction", type=float, default=0.4,
                        help="Top-frequency share treated as 'head' classes.")
    parser.add_argument("--train-per-class-high", type=int, default=4)
    parser.add_argument("--train-per-class-low", type=int, default=1)
    parser.add_argument("--include-bytes", action="store_true",
                        help="Also export bidirectional payload-byte tokens.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache = load_multiview_cache(args.cache_dir)
    split_payload = json.loads(Path(args.split).read_text(encoding="utf-8"))
    split_of = {name: set(ids) for name, ids in split_payload.items()}
    labels = np.asarray(cache.labels, dtype=np.int64)
    lengths = np.asarray(cache.lengths, dtype=np.int64)
    sample_ids = list(cache.sample_ids)
    names = _label_names(args.label_map)
    n = len(sample_ids)

    # Per-class row indices, split by original split.
    classes = set(int(x) for x in labels)
    train_rows: dict[int, list[int]] = {c: [] for c in classes}
    val_rows: dict[int, list[int]] = {c: [] for c in classes}
    test_rows: dict[int, list[int]] = {c: [] for c in classes}
    for row in range(n):
        c = int(labels[row])
        sid = sample_ids[row]
        if sid in split_of["train"]:
            train_rows[c].append(row)
        elif sid in split_of["val"]:
            val_rows[c].append(row)
        elif sid in split_of["test"]:
            test_rows[c].append(row)
    del split_of

    train_counts = np.array([len(train_rows[c]) for c in sorted(classes)], dtype=np.int64)
    cutoff = sorted(train_counts, reverse=True)[max(int(args.head_fraction * len(classes)) - 1, 0)]
    rng = np.random.default_rng(args.seed)

    chosen: list[tuple[str, str, int, str, int, str, str, str]] = []
    # (split, id, label, label_name, num_packets, packets, fwd_bytes, bwd_bytes)
    order = {"train": 0, "val": 1, "test": 2}
    for c in sorted(classes):
        k_train = args.train_per_class_high if train_counts[sorted(classes).index(c)] >= cutoff else args.train_per_class_low
        for name, rows, k in (("train", train_rows[c], k_train),
                              ("val", val_rows[c], 1),
                              ("test", test_rows[c], 1)):
            if not rows:
                continue
            rng.shuffle(rows)
            take = rows[: min(k, len(rows))]
            for row in take:
                sid = sample_ids[row]
                num = int(lengths[row])
                label_name = names.get(c, str(c))
                if args.include_bytes:
                    fwd = _encode_bytes(cache, row, "fwd_bytes")
                    bwd = _encode_bytes(cache, row, "bwd_bytes")
                else:
                    fwd = bwd = ""
                chosen.append((name, sid, c, label_name, num,
                               _encode_row(cache, row, num), fwd, bwd))
    del train_rows, val_rows, test_rows

    chosen.sort(key=lambda entry: (order[entry[0]], entry[1]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["sample_id", "split", "label", "label_name", "num_packets", "packets"]
    if args.include_bytes:
        header += ["fwd_bytes", "bwd_bytes"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for split_name, sid, c, label_name, num, encoded, fwd, bwd in chosen:
            writer.writerow([sid, split_name, c, label_name, num, encoded, fwd, bwd]
                            if args.include_bytes
                            else [sid, split_name, c, label_name, num, encoded])

    per_split = {}
    for split_name, _, _, _, _, _, _, _ in chosen:
        per_split[split_name] = per_split.get(split_name, 0) + 1
    print(f"wrote {len(chosen)} sample flows to {out.resolve()}")
    print(f"  classes={len(classes)} per-split={per_split}")
    print(f"  columns={header}")


if __name__ == "__main__":
    main()
