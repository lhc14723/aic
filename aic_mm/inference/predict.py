"""Deterministic test-set inference for one competition detector."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ultralytics import YOLO
from ultralytics.utils import TQDM
from ultralytics.utils.patches import imread

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


def _iou_xyxy(one: np.ndarray, two: np.ndarray) -> float:
    top_left = np.maximum(one[:2], two[:2])
    bottom_right = np.minimum(one[2:], two[2:])
    intersection = float(np.prod(np.clip(bottom_right - top_left, 0.0, None)))
    one_area = float(np.prod(np.clip(one[2:] - one[:2], 0.0, None)))
    two_area = float(np.prod(np.clip(two[2:] - two[:2], 0.0, None)))
    return intersection / max(one_area + two_area - intersection, 1e-12)


def _weighted_box_fusion(
    candidates: Iterable[tuple[int, np.ndarray, float, int]],
    *,
    views: int,
    iou_threshold: float,
) -> list[tuple[int, np.ndarray, float]]:
    """Fuse same-class TTA boxes and reward agreement between independent views."""
    if views < 1:
        raise ValueError("views must be positive")
    if not 0.0 < iou_threshold < 1.0:
        raise ValueError("TTA fusion IoU must be between 0 and 1")
    clusters: list[dict[str, Any]] = []
    ordered = sorted(candidates, key=lambda item: item[2], reverse=True)
    for class_id, box, score, view_id in ordered:
        best_cluster: dict[str, Any] | None = None
        best_iou = iou_threshold
        for cluster in clusters:
            if cluster["class_id"] != class_id:
                continue
            overlap = _iou_xyxy(box, cluster["box"])
            if overlap >= best_iou:
                best_iou = overlap
                best_cluster = cluster
        if best_cluster is None:
            clusters.append(
                {
                    "class_id": class_id,
                    "boxes": [box.astype(np.float64, copy=True)],
                    "scores": [float(score)],
                    "view_ids": {int(view_id)},
                    "box": box.astype(np.float64, copy=True),
                }
            )
            continue
        best_cluster["boxes"].append(box.astype(np.float64, copy=True))
        best_cluster["scores"].append(float(score))
        best_cluster["view_ids"].add(int(view_id))
        scores = np.asarray(best_cluster["scores"], dtype=np.float64)
        boxes = np.stack(best_cluster["boxes"])
        best_cluster["box"] = np.average(boxes, axis=0, weights=scores)

    fused: list[tuple[int, np.ndarray, float]] = []
    for cluster in clusters:
        # Mean confidence is multiplied by view coverage. A box supported by
        # all views keeps its calibration; one-view noise is down-weighted.
        score = float(np.mean(cluster["scores"])) * len(cluster["view_ids"]) / views
        fused.append((int(cluster["class_id"]), cluster["box"], score))
    return sorted(fused, key=lambda item: item[2], reverse=True)


def _tta_views(
    imgsz: int,
    scales: Iterable[int] | None,
    horizontal_flip: bool,
) -> list[tuple[int, bool]]:
    normalized = [int(value) for value in (scales or (imgsz,))]
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("TTA scales must contain positive image sizes")
    views = [(value, False) for value in dict.fromkeys(normalized)]
    if horizontal_flip:
        views.append((max(normalized), True))
    return views


def _read_tta_image(path: Path, horizontal_flip: bool) -> np.ndarray:
    image = imread(str(path))
    if image is None:
        raise ValueError(f"Could not read processed image: {path}")
    if image.ndim != 3 or image.shape[2] != 8:
        raise ValueError(f"Expected HWC 8-channel TIFF at {path}, got {image.shape}")
    if horizontal_flip:
        image = image[:, ::-1]
    return np.ascontiguousarray(image)


def _collect_tta_predictions(
    model: YOLO,
    image_paths: list[Path],
    *,
    imgsz: int,
    horizontal_flip: bool,
    view_id: int,
    confidence: float,
    iou: float,
    max_det: int,
    device: str | int,
    half: bool,
    batch: int,
    destination: dict[str, list[tuple[int, np.ndarray, float, int]]],
) -> None:
    for offset in TQDM(
        range(0, len(image_paths), batch),
        desc=f"TTA {imgsz}{'-flip' if horizontal_flip else ''}",
    ):
        chunk = image_paths[offset : offset + batch]
        images = [_read_tta_image(path, horizontal_flip) for path in chunk]
        results = model.predict(
            source=images,
            imgsz=imgsz,
            conf=confidence,
            iou=iou,
            max_det=max_det,
            device=device,
            half=half,
            batch=len(images),
            augment=False,
            verbose=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(f"TTA prediction count mismatch: {len(results)} != {len(chunk)}")
        for path, result in zip(chunk, results):
            boxes = result.boxes
            if boxes is None or not len(boxes):
                continue
            xyxy = boxes.xyxyn.detach().cpu().numpy().astype(np.float64)
            if horizontal_flip:
                old_x1 = xyxy[:, 0].copy()
                xyxy[:, 0] = 1.0 - xyxy[:, 2]
                xyxy[:, 2] = 1.0 - old_x1
            class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)
            scores = boxes.conf.detach().cpu().numpy()
            destination[path.stem].extend(
                (int(class_id), box, float(score), view_id)
                for class_id, box, score in zip(class_ids, xyxy, scores)
            )


def _write_tta_predictions(
    model: YOLO,
    image_paths: list[Path],
    output_dir: Path,
    *,
    views: list[tuple[int, bool]],
    confidence: float,
    iou: float,
    fusion_iou: float,
    max_det: int,
    device: str | int,
    half: bool,
    batch: int,
) -> int:
    candidates: dict[str, list[tuple[int, np.ndarray, float, int]]] = {
        path.stem: [] for path in image_paths
    }
    for view_id, (view_imgsz, horizontal_flip) in enumerate(views):
        _collect_tta_predictions(
            model,
            image_paths,
            imgsz=view_imgsz,
            horizontal_flip=horizontal_flip,
            view_id=view_id,
            confidence=confidence,
            iou=iou,
            max_det=max_det,
            device=device,
            half=half,
            batch=batch,
            destination=candidates,
        )

    total_detections = 0
    for image_path in image_paths:
        fused = _weighted_box_fusion(
            candidates[image_path.stem],
            views=len(views),
            iou_threshold=fusion_iou,
        )
        lines: list[str] = []
        for class_id, box, score in fused[:max_det]:
            x1, y1, x2, y2 = np.clip(box, 0.0, 1.0)
            width, height = x2 - x1, y2 - y1
            if width <= 0.0 or height <= 0.0:
                continue
            x, y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            lines.append(
                f"{class_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f} {score:.8f}"
            )
        _atomic_prediction(output_dir / f"{image_path.stem}.txt", lines)
        total_detections += len(lines)
    return total_detections


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
    tta_scales: Iterable[int] | None = None,
    tta_flip: bool = False,
    tta_fusion_iou: float = 0.65,
) -> dict[str, Any]:
    """Run automatic inference and create one result TXT per test image."""
    weights, image_dir, output_dir = Path(weights), Path(image_dir), Path(output_dir)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if batch < 1:
        raise ValueError("batch must be positive")
    image_paths = sorted(image_dir.glob("*.tiff"))
    if not image_paths:
        raise FileNotFoundError(f"No processed TIFF images found: {image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights), task="detect")
    views = _tta_views(imgsz, tta_scales, tta_flip)
    if len(views) > 1:
        total_detections = _write_tta_predictions(
            model,
            image_paths,
            output_dir,
            views=views,
            confidence=confidence,
            iou=iou,
            fusion_iou=tta_fusion_iou,
            max_det=max_det,
            device=device,
            half=half,
            batch=batch,
        )
        return {
            "images": len(image_paths),
            "detections": total_detections,
            "output": str(output_dir),
            "tta_views": [{"imgsz": size, "horizontal_flip": flip} for size, flip in views],
        }

    results = model.predict(
        source=str(image_dir),
        imgsz=views[0][0],
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
