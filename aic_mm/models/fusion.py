"""Quality-gated visible/infrared/depth fusion stem."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import torch
from torch import nn
from ultralytics.nn.modules import Conv


def _new_conv_like(source: Conv, input_channels: int) -> Conv:
    convolution = source.conv

    def scalar_if_equal(value: Any) -> Any:
        if isinstance(value, tuple) and len(value) == 2 and value[0] == value[1]:
            return value[0]
        return value

    return Conv(
        input_channels,
        convolution.out_channels,
        k=scalar_if_equal(convolution.kernel_size),
        s=scalar_if_equal(convolution.stride),
        p=scalar_if_equal(convolution.padding),
        g=convolution.groups,
        d=scalar_if_equal(convolution.dilation),
        act=deepcopy(source.act),
    )


def _copy_batch_norm(source: Conv, destination: Conv) -> None:
    destination.bn.load_state_dict(deepcopy(source.bn.state_dict()))


@torch.no_grad()
def initialize_stem_from_rgb_conv(stem: "TriModalStem", source: Conv) -> None:
    """Initialize all three branches from a public three-channel RGB stem.

    Ultralytics cannot directly transfer a 3-channel checkpoint tensor into
    the temporary 8-channel model stem because their shapes differ. Calling
    this explicitly preserves the valuable pretrained first-layer filters
    instead of silently leaving that layer random.
    """
    if not isinstance(source, Conv):
        raise TypeError(f"Expected an Ultralytics Conv source, got {type(source).__name__}")
    source_weight = source.conv.weight.detach()
    if source_weight.shape[0] != stem.rgb.conv.weight.shape[0] or source_weight.shape[1] < 3:
        raise ValueError(
            "RGB source is incompatible with fusion stem: "
            f"source={tuple(source_weight.shape)}, target={tuple(stem.rgb.conv.weight.shape)}"
        )
    rgb_weight = source_weight[:, :3].clone()
    stem.rgb.conv.weight.copy_(rgb_weight)
    mean_weight = rgb_weight.mean(dim=1, keepdim=True)
    stem.infrared.conv.weight.copy_(mean_weight.repeat(1, 2, 1, 1) / 2.0)
    stem.depth.conv.weight.copy_(mean_weight.repeat(1, 3, 1, 1) / 3.0)
    for branch in (stem.rgb, stem.infrared, stem.depth):
        _copy_batch_norm(source, branch)


class TriModalStem(nn.Module):
    """Fuse RGB, two IR features and three depth-quality features at P1.

    Gates start close to zero, so initialization behaves like the pretrained
    RGB detector. During training the model can raise either auxiliary branch
    only where that sensor adds useful information.
    """

    is_tri_modal_stem = True

    def __init__(self, source: Conv) -> None:
        super().__init__()
        if not isinstance(source, Conv):
            raise TypeError(f"Expected an Ultralytics Conv stem, got {type(source).__name__}")
        if source.conv.in_channels < 3:
            raise ValueError("Source stem must expose at least three input channels")

        self.rgb = _new_conv_like(source, 3)
        self.infrared = _new_conv_like(source, 2)
        self.depth = _new_conv_like(source, 3)
        output_channels = source.conv.out_channels
        self.quality_gate = nn.Conv2d(output_channels * 3, 2, kernel_size=1)
        self._initialize_from_source(source)

        # Ultralytics graph metadata is assigned to parsed layers.
        for attribute in ("i", "f", "type"):
            if hasattr(source, attribute):
                setattr(self, attribute, getattr(source, attribute))
        self.np = sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def _initialize_from_source(self, source: Conv) -> None:
        initialize_stem_from_rgb_conv(self, source)
        nn.init.zeros_(self.quality_gate.weight)
        nn.init.constant_(self.quality_gate.bias, -2.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 8:
            raise ValueError(f"TriModalStem expects BCHW with 8 channels, got {tuple(inputs.shape)}")
        rgb = self.rgb(inputs[:, 0:3])
        infrared = self.infrared(inputs[:, 3:5])
        depth = self.depth(inputs[:, 5:8])
        gates = torch.sigmoid(self.quality_gate(torch.cat((rgb, infrared, depth), dim=1)))
        return rgb + gates[:, 0:1] * infrared + gates[:, 1:2] * depth


class TriModalStemV2(TriModalStem):
    """Fuse modalities with spatially varying, channel-wise residual gates.

    V1 predicts one gate per auxiliary modality at every pixel. That is safe
    and compact, but all feature channels must accept the same infrared/depth
    contribution. V2 predicts one gate per feature channel, allowing features
    useful for thermal targets (for example lights) to use infrared strongly
    without forcing the same weight onto texture or geometry features.

    Branch names intentionally match V1. A V1 checkpoint can therefore
    initialize the RGB/infrared/depth convolutions while the wider V2 gate is
    learned during an explicit V2 fine-tune or retraining run.
    """

    def __init__(self, source: Conv) -> None:
        nn.Module.__init__(self)
        if not isinstance(source, Conv):
            raise TypeError(f"Expected an Ultralytics Conv stem, got {type(source).__name__}")
        if source.conv.in_channels < 3:
            raise ValueError("Source stem must expose at least three input channels")

        self.rgb = _new_conv_like(source, 3)
        self.infrared = _new_conv_like(source, 2)
        self.depth = _new_conv_like(source, 3)
        output_channels = source.conv.out_channels
        self.output_channels = output_channels
        self.quality_gate = nn.Conv2d(
            output_channels * 3,
            output_channels * 2,
            kernel_size=1,
        )
        self._initialize_from_source(source)
        # A slightly less conservative prior than V1 is justified by the
        # measured validation ablation, while RGB remains the dominant path.
        nn.init.constant_(self.quality_gate.bias, -2.0)

        for attribute in ("i", "f", "type"):
            if hasattr(source, attribute):
                setattr(self, attribute, getattr(source, attribute))
        self.np = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 8:
            raise ValueError(f"TriModalStemV2 expects BCHW with 8 channels, got {tuple(inputs.shape)}")
        rgb = self.rgb(inputs[:, 0:3])
        infrared = self.infrared(inputs[:, 3:5])
        depth = self.depth(inputs[:, 5:8])
        gates = torch.sigmoid(self.quality_gate(torch.cat((rgb, infrared, depth), dim=1)))
        infrared_gate, depth_gate = gates.split(self.output_channels, dim=1)
        return rgb + infrared_gate * infrared + depth_gate * depth


FusionVariant = Literal["v1", "v2"]


def fusion_stem_type(variant: FusionVariant) -> type[TriModalStem]:
    """Return the requested fusion module class with strict validation."""
    if variant == "v1":
        return TriModalStem
    if variant == "v2":
        return TriModalStemV2
    raise ValueError(f"Unknown fusion variant: {variant!r}; expected 'v1' or 'v2'")


def replace_first_layer_with_tri_modal_stem(
    model: nn.Module,
    *,
    variant: FusionVariant = "v1",
) -> TriModalStem:
    """Replace one Ultralytics DetectionModel's first Conv exactly once."""
    layers: Any = getattr(model, "model", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Model does not expose an Ultralytics layer sequence")
    if isinstance(layers[0], TriModalStem):
        expected_type = fusion_stem_type(variant)
        if type(layers[0]) is not expected_type:
            raise ValueError(
                f"Model already uses {type(layers[0]).__name__}, cannot replace it "
                f"in-place with {expected_type.__name__}"
            )
        return layers[0]
    if not isinstance(layers[0], Conv):
        raise TypeError(f"Unsupported first model layer: {type(layers[0]).__name__}")
    stem = fusion_stem_type(variant)(layers[0])
    layers[0] = stem
    return stem
