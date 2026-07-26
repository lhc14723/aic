"""Quality-gated visible/infrared/depth fusion stem."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

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
        source_weight = source.conv.weight.detach()
        rgb_weight = source_weight[:, :3].clone()
        if rgb_weight.shape[1] < 3:
            rgb_weight = rgb_weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        self.rgb.conv.weight.copy_(rgb_weight)
        mean_weight = rgb_weight.mean(dim=1, keepdim=True)
        self.infrared.conv.weight.copy_(mean_weight.repeat(1, 2, 1, 1) / 2.0)
        self.depth.conv.weight.copy_(mean_weight.repeat(1, 3, 1, 1) / 3.0)
        for branch in (self.rgb, self.infrared, self.depth):
            _copy_batch_norm(source, branch)
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


def replace_first_layer_with_tri_modal_stem(model: nn.Module) -> TriModalStem:
    """Replace one Ultralytics DetectionModel's first Conv exactly once."""
    layers: Any = getattr(model, "model", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Model does not expose an Ultralytics layer sequence")
    if isinstance(layers[0], TriModalStem):
        return layers[0]
    if not isinstance(layers[0], Conv):
        raise TypeError(f"Unsupported first model layer: {type(layers[0]).__name__}")
    stem = TriModalStem(layers[0])
    layers[0] = stem
    return stem
