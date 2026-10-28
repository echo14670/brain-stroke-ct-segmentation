from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def binary_case_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float | int]:
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": int(prediction.sum()),
        "ground_truth_positive": int(target.sum()),
        "pixels": int(target.size),
    }


def _mean(rows: list[dict[str, float | int]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row[key]) for row in rows]))


def aggregate_case_metrics(
    rows: Iterable[dict[str, object]],
) -> dict[str, float | int]:
    rows = list(rows)
    positive_rows = [
        row for row in rows if int(row["ground_truth_positive"]) > 0
    ]
    tp = sum(int(row["tp"]) for row in positive_rows)
    fp = sum(int(row["fp"]) for row in positive_rows)
    fn = sum(int(row["fn"]) for row in positive_rows)
    normal_rows = [row for row in rows if row["group"] == "Normal"]
    normal_fp_ratios = [
        int(row["predicted_positive"]) / max(int(row["pixels"]), 1)
        for row in normal_rows
    ]
    return {
        "case_count": len(rows),
        "positive_case_count": len(positive_rows),
        "normal_case_count": len(normal_rows),
        "positive_mean_dice": _mean(positive_rows, "dice"),
        "positive_mean_iou": _mean(positive_rows, "iou"),
        "positive_pixel_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "positive_pixel_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "normal_false_positive_pixel_ratio": (
            float(np.mean(normal_fp_ratios)) if normal_fp_ratios else 0.0
        ),
    }


def detection_metrics(
    labels: Iterable[int], predictions: Iterable[int]
) -> dict[str, float | int]:
    labels = np.asarray(list(labels), dtype=np.int32)
    predictions = np.asarray(list(predictions), dtype=np.int32)
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same shape")
    tp = int(np.logical_and(labels == 1, predictions == 1).sum())
    fp = int(np.logical_and(labels == 0, predictions == 1).sum())
    fn = int(np.logical_and(labels == 1, predictions == 0).sum())
    tn = int(np.logical_and(labels == 0, predictions == 0).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def group_case_metrics(
    rows: Iterable[dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)
    return {group: aggregate_case_metrics(items) for group, items in grouped.items()}
