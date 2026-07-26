"""Deterministic test-set inference for one competition detector."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO
from ultralytics.utils import TQDM

# Import is required before loading a checkpoint that contains this module.
from aic_mm.models.fusion import TriModalStem  # noqa: F401


def _atomic_prediction(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _clip_xywh(box: np.ndarray) -> tuple[float, float, float, float] | None:
    x, y, width, height = (float(value) for value in box)
    x1, y1 = max(0.0, x - width / 2.0), max(0.0, y - height / 2.0)
    x2, y2 = min(1.0, x + width / 2.0), min(1.0, y + height / 2.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


def predict_test_set(
    weights: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    *,
    imgsz: int = 960,
    confidence: float = 0.001,
    iou: float = 0.70,
    max_det: int = 100,
    device: str | int = 0,
    half: bool = True,
    batch: int = 1,
) -> dict[str, Any]:
    """Run automatic inference and create one result TXT per test image."""
    weights, image_dir, output_dir = Path(weights), Path(image_dir), Path(output_dir)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    image_paths = sorted(image_dir.glob("*.tiff"))
    if not image_paths:
        raise FileNotFoundError(f"No processed TIFF images found: {image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights), task="detect")
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        conf=confidence,
        iou=iou,
        max_det=max_det,
        device=device,
        half=half,
        batch=batch,
        stream=True,
        verbose=False,
    )
    written: set[str] = set()
    total_detections = 0
    for result in TQDM(results, total=len(image_paths), desc="Predicting"):
        sample_id = Path(result.path).stem
        boxes = result.boxes
        rows: list[tuple[float, str]] = []
        if boxes is not None and len(boxes):
            xywhn = boxes.xywhn.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)
            confidences = boxes.conf.detach().cpu().numpy()
            for class_id, box, score in zip(class_ids, xywhn, confidences):
                clipped = _clip_xywh(box)
                if clipped is None:
                    continue
                x, y, width, height = clipped
                line = (
                    f"{int(class_id)} {x:.8f} {y:.8f} "
                    f"{width:.8f} {height:.8f} {float(score):.8f}"
                )
                rows.append((float(score), line))
        rows.sort(key=lambda pair: pair[0], reverse=True)
        _atomic_prediction(output_dir / f"{sample_id}.txt", [line for _, line in rows[:max_det]])
        written.add(sample_id)
        total_detections += min(len(rows), max_det)

    # Empty prediction files are legitimate and required for a complete submission.
    for image_path in image_paths:
        if image_path.stem not in written:
            _atomic_prediction(output_dir / f"{image_path.stem}.txt", [])
    return {"images": len(image_paths), "detections": total_detections, "output": str(output_dir)}
