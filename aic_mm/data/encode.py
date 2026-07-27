"""Encode aligned visible/infrared/depth samples as 8-channel TIFF images."""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from ultralytics.utils import TQDM

from aic_mm.constants import CHANNEL_NAMES, CLASS_NAMES, MULTISPECTRAL_CHANNELS, NUM_CLASSES
from aic_mm.io_utils import read_json, read_jsonl, write_json


class EncodingError(RuntimeError):
    """Raised when one official multi-modal sample cannot be encoded safely."""


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise EncodingError(f"OpenCV could not read image: {path}")
    return image


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        if image.shape[2] < 3:
            image = image[..., 0]
        else:
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image
    values = image.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(values[finite], (0.5, 99.5))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    values = (values - low) * (255.0 / (high - low))
    return np.clip(values, 0, 255).astype(np.uint8)


def _contrast_channel(gray: np.ndarray) -> np.ndarray:
    values = gray.astype(np.float32)
    nonzero = values[values > 0]
    source = nonzero if nonzero.size >= 32 else values.reshape(-1)
    low, high = np.percentile(source, (1.0, 99.0))
    if high <= low:
        return gray.copy()
    return np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def _encode_depth(depth: np.ndarray, suffix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return near-emphasis, validity and metric-domain channels."""
    is_metric = depth.ndim == 2 and depth.dtype == np.uint16 and suffix.lower() == ".png"
    if is_metric:
        values = depth.astype(np.float32)
        valid = (values >= 300.0) & (values <= 20000.0)
        near = np.zeros(values.shape, dtype=np.uint8)
        if valid.any():
            log_min, log_max = np.log(300.0), np.log(20000.0)
            normalized = (np.log(np.clip(values, 300.0, 20000.0)) - log_min) / (log_max - log_min)
            near[valid] = np.clip((1.0 - normalized[valid]) * 255.0, 0, 255).astype(np.uint8)
        return (
            near,
            valid.astype(np.uint8) * 255,
            np.full(values.shape, 255, dtype=np.uint8),
        )

    gray = _to_gray_u8(depth)
    return gray, (gray > 0).astype(np.uint8) * 255, np.zeros(gray.shape, dtype=np.uint8)


def encode_modalities(
    visible_path: str | Path,
    infrared_path: str | Path,
    depth_path: str | Path,
) -> np.ndarray:
    """Build the fixed HWC uint8 tensor used by both training and inference."""
    visible_path, infrared_path, depth_path = map(Path, (visible_path, infrared_path, depth_path))
    visible = _read_image(visible_path)
    if visible.ndim != 3 or visible.shape[2] < 3:
        raise EncodingError(f"Visible image is not three-channel: {visible_path}, shape={visible.shape}")
    rgb = cv2.cvtColor(visible[..., :3], cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = _to_gray_u8(rgb)
        rgb = np.repeat(rgb[..., None], 3, axis=2)

    infrared = _to_gray_u8(_read_image(infrared_path))
    infrared_contrast = _contrast_channel(infrared)
    depth_near, depth_valid, depth_metric = _encode_depth(_read_image(depth_path), depth_path.suffix)

    shapes = {
        "visible": rgb.shape[:2],
        "infrared": infrared.shape,
        "depth": depth_near.shape,
    }
    if len(set(shapes.values())) != 1:
        raise EncodingError(f"Modalities are not spatially aligned: {shapes}")

    encoded = np.dstack(
        (
            rgb[..., 0],
            rgb[..., 1],
            rgb[..., 2],
            infrared,
            infrared_contrast,
            depth_near,
            depth_valid,
            depth_metric,
        )
    )
    if encoded.dtype != np.uint8 or encoded.shape[2] != MULTISPECTRAL_CHANNELS:
        raise EncodingError(f"Unexpected encoded tensor: shape={encoded.shape}, dtype={encoded.dtype}")
    return np.ascontiguousarray(encoded)


def sanitize_label_file(path: str | Path) -> tuple[str, dict[str, int]]:
    """Clip official boxes to the image while preserving the originals read-only."""
    lines: list[str] = []
    stats: Counter[str] = Counter()
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            fields = text.split()
            if len(fields) != 5:
                raise EncodingError(f"{path}:{line_number}: expected five label fields")
            try:
                class_id = int(fields[0])
                x, y, width, height = (float(value) for value in fields[1:])
            except ValueError as exc:
                raise EncodingError(f"{path}:{line_number}: invalid label numeric value") from exc
            values = np.asarray([x, y, width, height], dtype=np.float64)
            if not 0 <= class_id < NUM_CLASSES or not np.isfinite(values).all():
                raise EncodingError(f"{path}:{line_number}: invalid class or non-finite box")
            if width <= 0 or height <= 0:
                stats["dropped_degenerate"] += 1
                continue

            original = np.asarray(
                [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0]
            )
            clipped = np.clip(original, 0.0, 1.0)
            if not np.allclose(original, clipped, rtol=0.0, atol=1e-12):
                stats["clipped"] += 1
            x1, y1, x2, y2 = clipped.tolist()
            if x2 <= x1 or y2 <= y1:
                stats["dropped_degenerate"] += 1
                continue
            new_x, new_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            new_width, new_height = x2 - x1, y2 - y1
            lines.append(
                f"{class_id} {new_x:.8f} {new_y:.8f} {new_width:.8f} {new_height:.8f}"
            )
            stats["objects_written"] += 1
    return ("\n".join(lines) + ("\n" if lines else "")), dict(stats)


def write_multipage_tiff(path: str | Path, encoded: np.ndarray) -> None:
    """Atomically write HWC channels as ordered TIFF pages."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.tiff")
    pages = [np.ascontiguousarray(encoded[..., index]) for index in range(encoded.shape[2])]
    try:
        if not cv2.imwritemulti(str(temporary), pages):
            raise EncodingError(f"OpenCV failed to write TIFF: {destination}")
        ok, reloaded = cv2.imreadmulti(str(temporary), flags=cv2.IMREAD_UNCHANGED)
        if not ok or len(reloaded) != MULTISPECTRAL_CHANNELS:
            raise EncodingError(f"TIFF verification failed: {temporary}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _encode_one(
    row: dict[str, Any],
    subset: str,
    project_root: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, int]:
    sample_id = str(row["sample_id"])
    output_image = output_root / "images" / subset / f"{sample_id}.tiff"
    output_label = output_root / "labels" / subset / f"{sample_id}.txt"
    stats: Counter[str] = Counter()

    if overwrite or not output_image.is_file():
        encoded = encode_modalities(
            _resolve(project_root, row["visible"]),
            _resolve(project_root, row["infrared"]),
            _resolve(project_root, row["depth"]),
        )
        write_multipage_tiff(output_image, encoded)
        stats["images_written"] += 1
    else:
        stats["images_skipped"] += 1

    if subset != "test":
        label_path = row.get("label")
        if not label_path:
            raise EncodingError(f"Training sample has no label path: {sample_id}")
        label_text, label_stats = sanitize_label_file(_resolve(project_root, label_path))
        if overwrite or not output_label.is_file():
            _write_text_atomic(output_label, label_text)
            stats["labels_written"] += 1
        else:
            stats["labels_skipped"] += 1
        stats.update(label_stats)
    return dict(stats)


def _dataset_yaml(output_root: Path, *, all_training_images: bool = False) -> dict[str, Any]:
    config = {
        "path": str(output_root.resolve()),
        "train": ["images/train", "images/val"] if all_training_images else "images/train",
        "val": "images/val",
        "test": "images/test",
        "channels": MULTISPECTRAL_CHANNELS,
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "channel_names": list(CHANNEL_NAMES),
    }
    return config


def prepare_processed_dataset(
    train_manifest: str | Path,
    test_manifest: str | Path,
    split_path: str | Path,
    output_root: str | Path,
    project_root: str | Path,
    *,
    workers: int = 2,
    overwrite: bool = False,
    subsets: Iterable[str] = ("train", "val", "test"),
) -> dict[str, Any]:
    """Build all requested processed subsets without touching official data."""
    train_rows = read_jsonl(train_manifest)
    test_rows = read_jsonl(test_manifest)
    split = read_json(split_path)
    output_root, project_root = Path(output_root).resolve(), Path(project_root).resolve()
    requested = tuple(dict.fromkeys(subsets))
    unknown = set(requested) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"Unknown subsets: {sorted(unknown)}")

    train_by_id = {str(row["sample_id"]): row for row in train_rows}
    jobs: list[tuple[dict[str, Any], str]] = []
    for subset in ("train", "val"):
        if subset not in requested:
            continue
        for sample_id in split[subset]:
            if sample_id not in train_by_id:
                raise EncodingError(f"Split references missing sample: {sample_id}")
            jobs.append((train_by_id[sample_id], subset))
    if "test" in requested:
        jobs.extend((row, "test") for row in test_rows)

    totals: Counter[str] = Counter()
    failures: list[str] = []
    if workers <= 1:
        iterator = (
            _encode_one(row, subset, project_root, output_root, overwrite)
            for row, subset in jobs
        )
        for stats in TQDM(iterator, total=len(jobs), desc="Encoding"):
            totals.update(stats)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _encode_one, row, subset, project_root, output_root, overwrite
                ): (row["sample_id"], subset)
                for row, subset in jobs
            }
            for future in TQDM(as_completed(futures), total=len(futures), desc="Encoding"):
                sample_id, subset = futures[future]
                try:
                    totals.update(future.result())
                except Exception as exc:
                    failures.append(f"{subset}/{sample_id}: {exc}")
    if failures:
        raise EncodingError(
            f"{len(failures)} samples failed; first failures:\n" + "\n".join(failures[:20])
        )

    output_root.mkdir(parents=True, exist_ok=True)
    yaml_path = output_root / "aic_multispectral.yaml"
    _write_text_atomic(
        yaml_path,
        yaml.safe_dump(_dataset_yaml(output_root), allow_unicode=True, sort_keys=False),
    )
    _write_text_atomic(
        output_root / "aic_multispectral_all.yaml",
        yaml.safe_dump(
            _dataset_yaml(output_root, all_training_images=True),
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    report = {
        "jobs": len(jobs),
        "subsets": list(requested),
        "workers": workers,
        "overwrite": overwrite,
        **dict(totals),
    }
    write_json(output_root / "encoding_report.json", report)
    return report
