import pytest
import torch

from aic_mm.evaluation.ablation import ABLATION_MODES, mask_modalities


@pytest.mark.parametrize(
    ("mode", "zero_channels"),
    [
        ("rgb", range(3, 8)),
        ("rgb_ir", range(5, 8)),
        ("rgb_depth", range(3, 5)),
    ],
)
def test_mask_modalities_only_zeros_requested_channels(
    mode: str, zero_channels: range
) -> None:
    inputs = torch.arange(16, dtype=torch.float32).reshape(1, 8, 1, 2)
    masked = mask_modalities(inputs, mode)
    assert masked.data_ptr() != inputs.data_ptr()
    for channel in range(8):
        if channel in zero_channels:
            assert torch.count_nonzero(masked[:, channel]) == 0
        else:
            assert torch.equal(masked[:, channel], inputs[:, channel])


def test_full_ablation_returns_original_tensor() -> None:
    inputs = torch.ones((1, 8, 2, 2))
    assert mask_modalities(inputs, "full") is inputs


def test_mask_modalities_validates_mode_and_shape() -> None:
    assert set(ABLATION_MODES) == {"full", "rgb", "rgb_ir", "rgb_depth"}
    with pytest.raises(ValueError, match="Unknown ablation mode"):
        mask_modalities(torch.ones((1, 8, 2, 2)), "unknown")
    with pytest.raises(ValueError, match="8 channels"):
        mask_modalities(torch.ones((1, 3, 2, 2)), "rgb")
