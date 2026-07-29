import numpy as np
import pytest

from aic_mm.inference.predict import _tta_views, _weighted_box_fusion


def test_tta_views_are_unique_and_flip_largest_scale() -> None:
    assert _tta_views(960, [960, 1280, 960], True) == [
        (960, False),
        (1280, False),
        (1280, True),
    ]


def test_tta_fusion_combines_consistent_boxes_and_downweights_single_view() -> None:
    candidates = [
        (0, np.asarray([0.10, 0.10, 0.30, 0.30]), 0.90, 0),
        (0, np.asarray([0.11, 0.10, 0.31, 0.30]), 0.75, 1),
        (0, np.asarray([0.10, 0.11, 0.30, 0.31]), 0.60, 2),
        (1, np.asarray([0.60, 0.60, 0.80, 0.80]), 0.90, 0),
    ]
    fused = _weighted_box_fusion(candidates, views=3, iou_threshold=0.65)
    assert len(fused) == 2
    class_zero = next(item for item in fused if item[0] == 0)
    class_one = next(item for item in fused if item[0] == 1)
    assert class_zero[2] == pytest.approx(0.75)
    assert class_one[2] == pytest.approx(0.30)
    assert np.allclose(class_zero[1], [0.10333333, 0.10266667, 0.30333333, 0.30266667])


def test_tta_fusion_validates_parameters() -> None:
    with pytest.raises(ValueError, match="views"):
        _weighted_box_fusion([], views=0, iou_threshold=0.65)
    with pytest.raises(ValueError, match="IoU"):
        _weighted_box_fusion([], views=1, iou_threshold=1.0)
