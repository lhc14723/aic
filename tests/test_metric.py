from pathlib import Path

import numpy as np

from aic_mm.evaluation.metric import evaluate_directories


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_perfect_prediction_has_perfect_ap_for_present_class(tmp_path: Path) -> None:
    ground_truth, predictions = tmp_path / "gt", tmp_path / "pred"
    _write(ground_truth / "one.txt", "0 0.5 0.5 0.2 0.2\n")
    _write(predictions / "one.txt", "0 0.5 0.5 0.2 0.2 0.9\n")
    result = evaluate_directories(
        ground_truth,
        predictions,
        sample_ids=["one"],
        iou_thresholds=np.asarray([0.5]),
    )
    assert result["per_class"]["person"]["ap50"] == 1.0


def test_wrong_location_has_zero_ap(tmp_path: Path) -> None:
    ground_truth, predictions = tmp_path / "gt", tmp_path / "pred"
    _write(ground_truth / "one.txt", "0 0.2 0.2 0.1 0.1\n")
    _write(predictions / "one.txt", "0 0.8 0.8 0.1 0.1 0.9\n")
    result = evaluate_directories(
        ground_truth,
        predictions,
        sample_ids=["one"],
        iou_thresholds=np.asarray([0.5]),
    )
    assert result["per_class"]["person"]["ap50"] == 0.0
