"""Ultralytics dataset adapter with modality-aware photometric augmentation."""

from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics.data.dataset import YOLODataset


class TriModalPhotometric:
    """Apply independent sensor perturbations after shared geometric transforms."""

    def __init__(
        self,
        *,
        probability: float = 0.8,
        infrared_gain: float = 0.15,
        depth_hole_probability: float = 0.25,
        modality_dropout_probability: float = 0.08,
    ) -> None:
        self.probability = float(probability)
        self.infrared_gain = float(infrared_gain)
        self.depth_hole_probability = float(depth_hole_probability)
        self.modality_dropout_probability = float(modality_dropout_probability)

    @staticmethod
    def _clip(values: np.ndarray) -> np.ndarray:
        return np.clip(values, 0, 255).astype(np.uint8)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        image = labels["img"]
        if image.ndim != 3 or image.shape[2] != 8:
            raise ValueError(f"Expected an HWC 8-channel image, got {image.shape}")
        if np.random.random() >= self.probability:
            return labels
        image = image.copy()

        # The RGB branch sees moderate gain/gamma changes. Geometry remains untouched.
        rgb = image[..., :3].astype(np.float32) / 255.0
        gamma = np.random.uniform(0.85, 1.18)
        gain = np.random.uniform(0.88, 1.12)
        image[..., :3] = self._clip(255.0 * gain * np.power(rgb, gamma))

        ir_gain = np.random.uniform(1.0 - self.infrared_gain, 1.0 + self.infrared_gain)
        ir_offset = np.random.uniform(-12.0, 12.0)
        image[..., 3:5] = self._clip(image[..., 3:5].astype(np.float32) * ir_gain + ir_offset)

        if np.random.random() < self.depth_hole_probability:
            height, width = image.shape[:2]
            hole_width = max(1, int(width * np.random.uniform(0.03, 0.15)))
            hole_height = max(1, int(height * np.random.uniform(0.03, 0.15)))
            x1 = np.random.randint(0, max(1, width - hole_width + 1))
            y1 = np.random.randint(0, max(1, height - hole_height + 1))
            image[y1 : y1 + hole_height, x1 : x1 + hole_width, 5:7] = 0

        if np.random.random() < self.modality_dropout_probability:
            image[..., 3:5] = 0
        if np.random.random() < self.modality_dropout_probability:
            image[..., 5:8] = 0

        labels["img"] = np.ascontiguousarray(image)
        return labels


class TriModalYOLODataset(YOLODataset):
    """YOLO detection dataset that understands the project's 8-channel contract."""

    def __init__(self, *args: Any, tri_augment: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.tri_augment = dict(tri_augment or {})
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp: Any = None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            # The final Ultralytics Format step converts HWC to BCHW, so sensor
            # perturbations must be inserted immediately before it.
            transforms.insert(len(transforms.transforms) - 1, TriModalPhotometric(**self.tri_augment))
        return transforms
