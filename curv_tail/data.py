"""Data loading: preprocessed traffic caches, datasets, and long-tail grouping.

On-disk cache layout
--------------------
A *multiview cache directory* holds the following files (one row per flow):

``cache_manifest.json``
    Cache metadata: ``num_samples``, ``num_classes``, ``max_packets``,
    ``max_bytes_per_direction``, ``features``.
``manifest.jsonl``
    One JSON object per flow: ``sample_id``, ``label`` (int), ``sequence_path``,
    ``label_name``, ``extra`` (optional).
``split.json``
    ``{"train": [ids...], "val": [ids...], "test": [ids...]}`` (no overlap).
``sequences.npy``
    Float32 ``[N, T, 2]`` packet features (channel 0 = signed log1p packet
    size, channel 1 = log1p inter-arrival time); padded rows are zero.
``lengths.npy``
    Int64 ``[N]`` number of valid packets per flow.
``fwd_bytes.npy`` / ``bwd_bytes.npy``
    UInt16 ``[N, B]`` payload byte tokens per direction.  Encoding: ``0`` =
    padding, raw byte ``b`` stored as ``b + 1`` (vocabulary 1..256).
``labels.npy``
    Int64 ``[N]`` integer class of each flow.
``sample_ids.txt``
    One sample id per line (ordered as the arrays above).

Everything else in this module is derived from these files:

* :class:`MultiViewTrafficDataset` -- a ``torch.utils.data.Dataset`` whose items
  are dicts keyed by ``x``, ``mask``, ``fwd_bytes``, ``bwd_bytes``, ``label``
  and ``sample_id``.
* :func:`compute_train_normalization` -- train-only feature statistics (kept
  for compatibility; CURV-TAIL uses *unstandardized* input because the packet
  size must stay byte-exact for discrete tokenization).
* :func:`cumulative_frequency_groups` -- the long-tail head/body/tail grouping:
  classes are ranked by training frequency and split into the most frequent
  50% (head), middle 30% (body) and least frequent 20% (tail).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "Record",
    "SequenceCache",
    "MultiViewCache",
    "load_manifest",
    "load_split",
    "records_by_split",
    "load_multiview_cache",
    "compute_train_normalization",
    "TrafficSequenceDataset",
    "MultiViewTrafficDataset",
    "training_class_counts",
    "cumulative_frequency_groups",
]


@dataclass(frozen=True)
class Record:
    """One labeled flow described by the manifest.

    Attributes:
        sample_id: Unique flow identifier.
        label: Integer class id.
        label_name: Human readable label.
        sequence_path: Where the flow came from (informational).
        uid: Optional extra identifier.
    """

    sample_id: str
    label: int
    label_name: str
    sequence_path: str
    uid: str | None


@dataclass
class SequenceCache:
    """Handles for a single-view packet cache (arrays are mmap-backed).

    Attributes:
        root: Cache directory.
        sequences: Float32 ``[N, T, 2]``.
        lengths: Int64 ``[N]``.
        sample_ids: ``[N]``.
        index: ``{sample_id: row}``.
        metadata: Parsed ``cache_manifest.json``.
    """

    root: Path
    sequences: np.ndarray
    lengths: np.ndarray
    sample_ids: list[str]
    index: dict[str, int]
    metadata: dict[str, Any]


@dataclass
class MultiViewCache(SequenceCache):
    """Cache extended with bidirectional byte payloads.

    Attributes:
        fwd_bytes: UInt16 ``[N, B]``.
        bwd_bytes: UInt16 ``[N, B]``.
        labels: Int64 ``[N]``.
    """

    fwd_bytes: np.ndarray
    bwd_bytes: np.ndarray
    labels: np.ndarray


def load_manifest(path: str | Path) -> list[Record]:
    """Read a JSON-lines manifest into :class:`Record` objects.

    Args:
        path: Path to ``manifest.jsonl``.

    Returns:
        List of records (one per non-empty line).

    Raises:
        ValueError: If the manifest contains duplicate sample ids.
    """
    records: list[Record] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            extra = row.get("extra", {})
            records.append(
                Record(
                    sample_id=str(row["sample_id"]),
                    label=int(row["label"]),
                    label_name=str(row.get("label_name", row["label"])),
                    sequence_path=str(row["sequence_path"]),
                    uid=str(extra["uid"]) if extra.get("uid") else None,
                )
            )
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample IDs in manifest: {path}")
    return records


def load_split(path: str | Path) -> dict[str, list[str]]:
    """Read the train / val / test split file.

    Args:
        path: Path to ``split.json``.

    Returns:
        ``{"train": ids, "val": ids, "test": ids}``.

    Raises:
        ValueError: If the three splits overlap.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {name: [str(x) for x in payload[name]] for name in ("train", "val", "test")}
    if sum(map(len, result.values())) != len(set().union(*map(set, result.values()))):
        raise ValueError("train/val/test sample IDs overlap")
    return result


def records_by_split(records: Iterable[Record], split: dict[str, list[str]]) -> dict[str, list[Record]]:
    """Group manifest records by split.

    Args:
        records: All manifest records.
        split: ``{"train": ids, ...}``.

    Returns:
        ``{"train": records, "val": records, "test": records}``.

    Raises:
        ValueError: If split ids and manifest ids do not match exactly.
    """
    by_id = {record.sample_id: record for record in records}
    expected = set(by_id)
    actual = set().union(*[set(v) for v in split.values()])
    if expected != actual:
        raise ValueError(
            f"split/manifest mismatch: missing={len(expected-actual)} unknown={len(actual-expected)}"
        )
    return {name: [by_id[sample_id] for sample_id in ids] for name, ids in split.items()}


def load_multiview_cache(path: str | Path) -> MultiViewCache:
    """Load a multiview cache directory (see module docstring for the layout).

    Args:
        path: Cache directory path.

    Returns:
        A :class:`MultiViewCache`.

    Raises:
        ValueError: If array shapes, dtypes or sample counts are inconsistent, if
            sample IDs are duplicated, or if the shifted byte tokens exceed the
            ``1..256`` vocabulary.
    """
    root = Path(path).resolve()
    sequences = np.load(root / "sequences.npy", mmap_mode="r")
    lengths = np.load(root / "lengths.npy", mmap_mode="r")
    sample_ids = (root / "sample_ids.txt").read_text(encoding="utf-8").splitlines()
    metadata = json.loads((root / "cache_manifest.json").read_text(encoding="utf-8"))
    if len(sample_ids) != sequences.shape[0] or len(sample_ids) != lengths.shape[0]:
        raise ValueError("inconsistent cache arrays and sample IDs")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample IDs in cache")
    if sequences.ndim != 3 or sequences.shape[2] != 2:
        raise ValueError(f"expected cache [N,L,2], got {sequences.shape}")
    fwd_bytes = np.load(root / "fwd_bytes.npy", mmap_mode="r")
    bwd_bytes = np.load(root / "bwd_bytes.npy", mmap_mode="r")
    labels = np.load(root / "labels.npy", mmap_mode="r")
    n = len(sample_ids)
    if fwd_bytes.ndim != 2 or bwd_bytes.shape != fwd_bytes.shape:
        raise ValueError(f"invalid bidirectional byte arrays: {fwd_bytes.shape}, {bwd_bytes.shape}")
    if any(array.shape[0] != n for array in (fwd_bytes, bwd_bytes, labels)):
        raise ValueError("multiview cache arrays have inconsistent sample counts")
    if fwd_bytes.dtype != np.uint16 or bwd_bytes.dtype != np.uint16:
        raise ValueError("byte caches must use uint16")
    if int(max(fwd_bytes.max(), bwd_bytes.max())) > 256:
        raise ValueError("shifted byte token exceeds vocabulary")
    return MultiViewCache(
        root=root,
        sequences=sequences,
        lengths=lengths,
        sample_ids=sample_ids,
        index={sid: i for i, sid in enumerate(sample_ids)},
        metadata=metadata,
        fwd_bytes=fwd_bytes,
        bwd_bytes=bwd_bytes,
        labels=labels,
    )


def compute_train_normalization(records: list[Record], cache: SequenceCache) -> dict[str, Any]:
    """Compute per-channel mean / std over the *training* packets only.

    Kept for interface compatibility and for users who train on raw standardized
    features.  CURV-TAIL itself uses unstandardized input (see the note on
    ``standardize=False`` in :class:`MultiViewTrafficDataset`).

    Args:
        records: Training records.
        cache: The (multi-view) cache.

    Returns:
        Dict with ``feature_names``, ``mean``, ``std`` (per feature) and the
        packet / flow counts.

    Raises:
        ValueError: If the training split is empty.
    """
    sums = np.zeros(2, dtype=np.float64)
    squares = np.zeros(2, dtype=np.float64)
    count = 0
    for start in range(0, len(records), 8192):
        indices = np.asarray([cache.index[r.sample_id] for r in records[start : start + 8192]], dtype=np.int64)
        batch = np.asarray(cache.sequences[indices], dtype=np.float32)
        lengths = np.asarray(cache.lengths[indices], dtype=np.int64)
        mask = np.arange(batch.shape[1])[None, :] < lengths[:, None]
        values = batch[mask]
        sums += values.sum(axis=0, dtype=np.float64)
        squares += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += int(values.shape[0])
    if count == 0:
        raise ValueError("empty training split")
    mean = sums / count
    var = np.maximum(squares / count - mean * mean, 1e-12)
    return {
        "feature_names": ["signed_log1p_packet_size", "log1p_inter_arrival_time"],
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
        "num_train_packets": count,
        "num_train_flows": len(records),
    }


class TrafficSequenceDataset(Dataset):
    """Base dataset yielding standardized packet sequences (single view).

    Args:
        records: Split records.
        cache: A :class:`SequenceCache`.
        normalization: Dict returned by :func:`compute_train_normalization`.
        standardize: Whether to standardize features with ``(x - mean) / std``.
    """

    def __init__(
        self, records: list[Record], cache: SequenceCache, normalization: dict[str, Any], *, standardize: bool = True
    ) -> None:
        super().__init__()
        self.records = records
        self.cache = cache
        self.indices = np.asarray([cache.index[r.sample_id] for r in records], dtype=np.int64)
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.maximum(np.asarray(normalization["std"], dtype=np.float32), 1e-6)
        # Discrete length-token models re-derive the exact packet-length token
        # from the RAW signed_log1p channel, so per-feature standardization must
        # be skipped: standardization collapses sizes to a few tokens and
        # destroys the discrete sequence.
        self.standardize = standardize

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        cache_index = int(self.indices[index])
        length = int(self.cache.lengths[cache_index])
        sequence = np.array(self.cache.sequences[cache_index], dtype=np.float32, copy=True)
        if self.standardize:
            sequence[:length] = (sequence[:length] - self.mean) / self.std
        sequence[length:] = 0.0
        mask = np.arange(sequence.shape[0]) < length
        record = self.records[index]
        return {
            "x": torch.from_numpy(sequence),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(record.label, dtype=torch.long),
            "sample_id": record.sample_id,
        }


class MultiViewTrafficDataset(TrafficSequenceDataset):
    """Dataset for CURV-TAIL: packet sequence + bidirectional byte payloads.

    Args:
        records: Split records.
        cache: A :class:`MultiViewCache`.
        normalization: Dict returned by :func:`compute_train_normalization`.  Its
            ``mean`` / ``std`` are not applied when ``standardize=False``
            (CURV-TAIL's setting), but both keys are still read at construction.
        standardize: See :class:`TrafficSequenceDataset`.

    Raises:
        ValueError: If the cache label and the manifest label disagree for any
            flow.
    """

    def __init__(
        self, records: list[Record], cache: MultiViewCache, normalization: dict[str, Any], *, standardize: bool = True
    ) -> None:
        super().__init__(records, cache, normalization, standardize=standardize)
        self.cache = cache
        for record, cache_index in zip(records, self.indices):
            if int(cache.labels[int(cache_index)]) != int(record.label):
                raise ValueError(f"cache/manifest label mismatch for {record.sample_id}")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = super().__getitem__(index)
        cache_index = int(self.indices[index])
        item.update(
            fwd_bytes=torch.from_numpy(np.array(self.cache.fwd_bytes[cache_index], dtype=np.int64, copy=True)),
            bwd_bytes=torch.from_numpy(np.array(self.cache.bwd_bytes[cache_index], dtype=np.int64, copy=True)),
        )
        return item


def training_class_counts(records: list[Record], num_classes: int) -> np.ndarray:
    """Per-class training counts (closed-set, every class must appear).

    Args:
        records: Training records.
        num_classes: Total number of classes.

    Returns:
        Int64 array of length ``num_classes``.

    Raises:
        ValueError: If a class has no training sample.
    """
    counts = np.bincount([r.label for r in records], minlength=num_classes).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("closed-set training split contains an empty class")
    return counts


def cumulative_frequency_groups(counts: np.ndarray) -> np.ndarray:
    """Assign each class a head / body / tail group by training frequency.

    Classes are sorted by training count (descending).  The most frequent 50%
    of classes are *head*, the middle 30% are *body*, and the least frequent
    20% are *tail* -- the long-tail definition used throughout this codebase.

    Args:
        counts: Per-class training counts.

    Returns:
        Int64 array with ``0`` = head, ``1`` = body, ``2`` = tail.
    """
    n = int(len(counts))
    head_count = max(1, int(0.5 * n + 0.5))
    tail_count = max(1, int(0.2 * n + 0.5))
    order = np.argsort(-counts, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    groups = np.full(n, 1, dtype=np.int64)  # body
    groups[ranks < head_count] = 0          # head
    groups[ranks >= n - tail_count] = 2     # tail
    return groups
