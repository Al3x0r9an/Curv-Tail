"""Metric computation, including long-tail (head / body / tail) aggregates.

The long-tail groupings are produced by
:func:`curv_tail.data.cumulative_frequency_groups`; the functions here consume
them to report overall accuracy / macro-F1 and per-group metrics, which are the
primary reported results of this release.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

__all__ = ["compute_metrics", "per_class_table", "json_ready"]


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    topk_predictions: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """Compute overall and long-tail-aware metrics for one evaluation pass.

    Args:
        labels: Ground-truth class ids.
        predictions: Argmax class predictions.
        topk_predictions: ``[N, k]`` top-k class predictions.
        groups: Head/body/tail group id per class (see
            :func:`curv_tail.data.cumulative_frequency_groups`).

    Returns:
        A flat dict of scalar metrics: ``acc``, ``balanced_acc``, ``macro_f1``,
        ``weighted_f1``, per-group (``head|body|tail``) macro-F1/acc and class
        counts, tail precision/recall, tail-to-head/body/frequent error rates,
        the converse ``frequent_to_tail_rate``, and -- for every ``k`` in
        {1, 3, 5, 10} no larger than the given top-k width -- ``top{k}_hit``,
        ``tail_top{k}_hit`` and ``top{k}_predicted_class_coverage``.
    """
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    topk_predictions = np.asarray(topk_predictions, dtype=np.int64)
    classes = np.arange(len(groups), dtype=np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=classes, zero_division=0
    )
    output: dict[str, float] = {
        "acc": float(accuracy_score(labels, predictions)),
        "balanced_acc": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=np.bincount(labels, minlength=len(groups)))),
    }
    for group_id, group_name in enumerate(("head", "body", "tail")):
        class_ids = np.flatnonzero(groups == group_id)
        true_mask = np.isin(labels, class_ids)
        pred_mask = np.isin(predictions, class_ids)
        output[f"{group_name}_num_classes"] = float(len(class_ids))
        output[f"{group_name}_macro_f1"] = float(f1[class_ids].mean()) if len(class_ids) else 0.0
        output[f"{group_name}_acc"] = float((labels[true_mask] == predictions[true_mask]).mean()) if true_mask.any() else 0.0
        if group_name == "tail":
            true_positive = int(np.sum(true_mask & pred_mask & (labels == predictions)))
            output["tail_precision"] = float(true_positive / max(int(pred_mask.sum()), 1))
            output["tail_recall"] = float(true_positive / max(int(true_mask.sum()), 1))

    true_tail = groups[labels] == 2
    pred_group = groups[predictions]
    output["tail_to_head_rate"] = float(np.mean(pred_group[true_tail] == 0)) if true_tail.any() else 0.0
    output["tail_to_body_rate"] = float(np.mean(pred_group[true_tail] == 1)) if true_tail.any() else 0.0
    output["tail_to_frequent_rate"] = float(np.mean(pred_group[true_tail] != 2)) if true_tail.any() else 0.0
    frequent = ~true_tail
    output["frequent_to_tail_rate"] = float(np.mean(pred_group[frequent] == 2)) if frequent.any() else 0.0

    max_k = topk_predictions.shape[1]
    for k in (1, 3, 5, 10):
        if k > max_k:
            continue
        hit = np.any(topk_predictions[:, :k] == labels[:, None], axis=1)
        output[f"top{k}_hit"] = float(hit.mean())
        output[f"tail_top{k}_hit"] = float(hit[true_tail].mean()) if true_tail.any() else 0.0
        output[f"top{k}_predicted_class_coverage"] = float(
            len(np.unique(topk_predictions[:, :k])) / len(groups)
        )
    return output


def per_class_table(labels: np.ndarray, predictions: np.ndarray, groups: np.ndarray, counts: np.ndarray) -> pd.DataFrame:
    """Per-class precision / recall / F1 with group and support columns.

    Args:
        labels: Ground-truth class ids.
        predictions: Argmax class predictions.
        groups: Head/body/tail group per class.
        counts: Per-class training counts.

    Returns:
        A ``pandas.DataFrame`` with one row per class and the columns
        ``class_id``, ``train_count``, ``group_id``, ``support``, ``precision``,
        ``recall`` and ``f1``.
    """
    classes = np.arange(len(groups), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=classes, zero_division=0
    )
    return pd.DataFrame(
        {
            "class_id": classes,
            "train_count": counts,
            "group_id": groups,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def json_ready(row: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy scalars inside a dict into native Python types.

    Args:
        row: A dict possibly containing ``np.integer`` / ``np.floating`` values.

    Returns:
        A copy of the dict with NumPy integer / floating scalars converted to
        native Python types.  Other value types are passed through unchanged, so
        the result is JSON-serializable only if the dict held no other
        non-serializable values.
    """
    output = {}
    for key, value in row.items():
        if isinstance(value, (np.integer,)):
            output[key] = int(value)
        elif isinstance(value, (np.floating,)):
            output[key] = float(value)
        else:
            output[key] = value
    return output
