import pytest
import torch
from torch import nn
from ultralytics.nn.modules import Conv

from aic_mm.models.fusion import (
    TriModalStem,
    TriModalStemV2,
    fusion_stem_type,
    initialize_stem_from_rgb_conv,
    replace_first_layer_with_tri_modal_stem,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleList((Conv(8, 16, 3, 2),))


@pytest.mark.parametrize("stem_type", (TriModalStem, TriModalStemV2))
def test_fusion_stem_preserves_expected_shape(stem_type: type[TriModalStem]) -> None:
    stem = stem_type(Conv(8, 16, 3, 2))
    output = stem(torch.rand(2, 8, 64, 96))
    assert output.shape == (2, 16, 32, 48)
    assert torch.isfinite(output).all()


def test_v2_uses_channel_wise_auxiliary_gates() -> None:
    stem = TriModalStemV2(Conv(8, 16, 3, 2))
    assert stem.quality_gate.out_channels == 32
    assert stem.output_channels == 16


def test_replace_is_variant_strict() -> None:
    model = TinyModel()
    installed = replace_first_layer_with_tri_modal_stem(model, variant="v2")
    assert isinstance(installed, TriModalStemV2)
    assert replace_first_layer_with_tri_modal_stem(model, variant="v2") is installed
    with pytest.raises(ValueError, match="already uses"):
        replace_first_layer_with_tri_modal_stem(model, variant="v1")


def test_fusion_variant_validation() -> None:
    assert fusion_stem_type("v1") is TriModalStem
    assert fusion_stem_type("v2") is TriModalStemV2
    with pytest.raises(ValueError, match="Unknown fusion variant"):
        fusion_stem_type("v3")  # type: ignore[arg-type]


def test_public_rgb_stem_weights_are_transferred_explicitly() -> None:
    source = Conv(3, 16, 3, 2)
    with torch.no_grad():
        source.conv.weight.fill_(0.25)
        source.bn.weight.fill_(0.75)
    stem = TriModalStemV2(Conv(8, 16, 3, 2))
    initialize_stem_from_rgb_conv(stem, source)
    assert torch.all(stem.rgb.conv.weight == 0.25)
    assert torch.all(stem.infrared.conv.weight == 0.125)
    assert torch.allclose(stem.depth.conv.weight, torch.full_like(stem.depth.conv.weight, 1 / 12))
    assert torch.all(stem.rgb.bn.weight == 0.75)
