"""Project-wide constants."""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = (
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbagecan",
    "uav",
    "tricycle",
)

NUM_CLASSES = len(CLASS_NAMES)
MULTISPECTRAL_CHANNELS = 8

# Multi-page TIFF channel order. This order is part of the model contract and
# must stay identical in preprocessing, training, validation and inference.
CHANNEL_NAMES: tuple[str, ...] = (
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "infrared_raw",
    "infrared_contrast",
    "depth_near",
    "depth_valid",
    "depth_is_metric",
)

SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
