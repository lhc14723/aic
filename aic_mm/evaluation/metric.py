"""101-point interpolated mAP for local, training-only validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aic_mm.constants import CLASS_NAMES, NUM_CLASSES
from aic_mm.io_utils import write_json


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = boxes.astype(np.float64, copy=True)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return result


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    if many.size == 0:
        return np.zeros(0, dtype=np.float64)
    top_left = np.maximum(one[:2], many[:, :2])
    bottom_right = np.minimum(one[2:], many[:, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0.0, None), axis=1)
    one_area = max(0.0, one[2] - one[0]) * max(0.0, one[3] - one[1])
    many_area = np.clip(many[:, 2] - many[:, 0], 0.0, None) * np.clip(
        many[:, 3] - many[:, 1], 0.0, None
    )
    return intersection / np.maximum(one_area + many_area - intersection, 1e-12)


def _read_boxes(path: Path, prediction: bool) -> list[tuple[int, np.ndarray, float]]:
    rows: list[tuple[int, np.ndarray, float]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.split()
            if not fields:
                continue
            expected = 6 if prediction else 5
            if len(fields) != expected:
                raise ValueError(f"{path}:{line_number}: expected {expected} fields")
            class_id = int(fields[0])
            box = np.asarray([float(value) for value in fields[1:5]], dtype=np.float64)
            confidence = float(fields[5]) if prediction else 1.0
            if not 0 <= class_id < NUM_CLASSES or not np.isfinite(box).all():
                raise ValueError(f"{path}:{line_number}: invalid class or box")
            if prediction and not np.isfinite(confidence):
                raise ValueError(f"{path}:{line_number}: invalid confidence")
            rows.append((class_id, xywh_to_xyxy(box[None, :])[0], confidence))
    return rows


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO/VOC-style 101-point interpolated precision."""
    if recall.size == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_points = np.linspace(0.0, 1.0, 101)
    sampled = [
        float(envelope[recall >= point].max()) if np.any(recall >= point) else 0.0
        for point in recall_points
    ]
    return float(np.mean(sampled))


def evaluate_directories(
    ground_truth_dir: str | Path,
    prediction_dir: str | Path,
    *,
    sample_ids: list[str] | None = None,
    output_path: str | Path | None = None,
    iou_thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate predictions without ever reading or annotating test data."""
    ground_truth_dir, prediction_dir = Path(ground_truth_dir), Path(prediction_dir)
    if sample_ids is None:
        sample_ids = sorted(path.stem for path in ground_truth_dir.glob("*.txt"))
    thresholds = (
        np.asarray(iou_thresholds, dtype=np.float64)
        if iou_thresholds is not None
        else np.linspace(0.50, 0.95, 10)
    )

    gt_by_class: list[dict[str, np.ndarray]] = [dict() for _ in range(NUM_CLASSES)]
    pred_by_class: list[list[tuple[str, np.ndarray, float]]] = [[] for _ in range(NUM_CLASSES)]
    for sample_id in sample_ids:
        for class_id, box, _ in _read_boxes(ground_truth_dir / f"{sample_id}.txt", False):
            current = gt_by_class[class_id].setdefault(sample_id, np.empty((0, 4), dtype=np.float64))
            gt_by_class[class_id][sample_id] = np.vstack((current, box))
        for class_id, box, confidence in _read_boxes(
            prediction_dir / f"{sample_id}.txt", True
        ):
            pred_by_class[class_id].append((sample_id, box, confidence))

    per_class: dict[str, Any] = {}
    all_aps = np.zeros((NUM_CLASSES, len(thresholds)), dtype=np.float64)
    for class_id, class_name in enumerate(CLASS_NAMES):
        predictions = sorted(pred_by_class[class_id], key=lambda row: row[2], reverse=True)
        gt_count = sum(len(boxes) for boxes in gt_by_class[class_id].values())
        for threshold_index, threshold in enumerate(thresholds):
            matched = {
                sample_id: np.zeros(len(boxes), dtype=bool)
                for sample_id, boxes in gt_by_class[class_id].items()
            }
            true_positive = np.zeros(len(predictions), dtype=np.float64)
            false_positive = np.zeros(len(predictions), dtype=np.float64)
            for prediction_index, (sample_id, box, _) in enumerate(predictions):
                gt_boxes = gt_by_class[class_id].get(sample_id, np.empty((0, 4)))
                ious = box_iou(box, gt_boxes)
                if ious.size:
                    best = int(np.argmax(ious))
                    if ious[best] >= threshold and not matched[sample_id][best]:
                        true_positive[prediction_index] = 1.0
                        matched[sample_id][best] = True
                        continue
                false_positive[prediction_index] = 1.0
            cumulative_tp = np.cumsum(true_positive)
            cumulative_fp = np.cumsum(false_positive)
            recall = cumulative_tp / max(gt_count, 1)
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
            all_aps[class_id, threshold_index] = (
                _average_precision(recall, precision) if gt_count else 0.0
            )

        per_class[class_name] = {
            "ground_truth_objects": gt_count,
            "predictions": len(predictions),
            "ap50": float(all_aps[class_id, 0]),
            "map50_95": float(all_aps[class_id].mean()),
            "ap_by_iou": {
                f"{threshold:.2f}": float(all_aps[class_id, index])
                for index, threshold in enumerate(thresholds)
            },
        }

    result = {
        "samples": len(sample_ids),
        "map50": float(all_aps[:, 0].mean()),
        "map50_95": float(all_aps.mean()),
        "iou_thresholds": thresholds.tolist(),
        "per_class": per_class,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
