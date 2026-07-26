"""Validate and package competition result files without manual intervention."""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any

from aic_mm.constants import NUM_CLASSES


def validate_prediction_directory(
    prediction_dir: str | Path,
    expected_ids: list[str],
    *,
    max_det: int = 100,
) -> dict[str, Any]:
    prediction_dir = Path(prediction_dir)
    expected = set(expected_ids)
    present = {path.stem for path in prediction_dir.glob("*.txt")}
    missing, extra = sorted(expected - present), sorted(present - expected)
    errors: list[str] = []
    detections = 0
    for sample_id in sorted(expected & present):
        path = prediction_dir / f"{sample_id}.txt"
        nonempty = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(nonempty) > max_det:
            errors.append(f"{path.name}: {len(nonempty)} detections exceeds max_det={max_det}")
        previous_confidence = math.inf
        for line_number, line in enumerate(nonempty, start=1):
            fields = line.split()
            if len(fields) != 6:
                errors.append(f"{path.name}:{line_number}: expected 6 fields")
                continue
            try:
                class_id = int(fields[0])
                x, y, width, height, confidence = map(float, fields[1:])
            except ValueError:
                errors.append(f"{path.name}:{line_number}: invalid numeric field")
                continue
            values = (x, y, width, height, confidence)
            if not all(math.isfinite(value) for value in values):
                errors.append(f"{path.name}:{line_number}: non-finite value")
            elif not 0 <= class_id < NUM_CLASSES:
                errors.append(f"{path.name}:{line_number}: class out of range")
            elif not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                errors.append(f"{path.name}:{line_number}: invalid normalized box")
            elif not 0 <= confidence <= 1:
                errors.append(f"{path.name}:{line_number}: invalid confidence")
            elif confidence > previous_confidence + 1e-12:
                errors.append(f"{path.name}:{line_number}: confidence order is not descending")
            previous_confidence = confidence
        detections += len(nonempty)
    if missing or extra or errors:
        details = []
        if missing:
            details.append(f"missing files ({len(missing)}): {missing[:20]}")
        if extra:
            details.append(f"extra files ({len(extra)}): {extra[:20]}")
        if errors:
            details.append("format errors:\n" + "\n".join(errors[:30]))
        raise ValueError("Submission validation failed:\n" + "\n".join(details))
    return {"files": len(expected), "detections": detections}


def package_submission(
    prediction_dir: str | Path,
    expected_ids: list[str],
    archive_path: str | Path,
    *,
    max_det: int = 100,
) -> dict[str, Any]:
    """Create a deterministic ZIP whose root directly contains prediction TXTs."""
    prediction_dir, archive_path = Path(prediction_dir), Path(archive_path)
    report = validate_prediction_directory(prediction_dir, expected_ids, max_det=max_det)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for sample_id in sorted(expected_ids):
            source = prediction_dir / f"{sample_id}.txt"
            info = zipfile.ZipInfo(f"{sample_id}.txt", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())
    return {**report, "archive": str(archive_path)}
