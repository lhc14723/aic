#!/usr/bin/env python3
"""Independently audit an AIC submission ZIP against its manifest and source TXT files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    archive_path: Path,
    manifest_path: Path,
    prediction_dir: Path,
    *,
    num_classes: int,
    max_det: int,
) -> dict[str, Any]:
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_names = [f"{str(row['sample_id'])}.txt" for row in manifest_rows]
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("Manifest contains duplicate sample IDs")
    expected = set(expected_names)

    prediction_files = {path.name: path for path in prediction_dir.glob("*.txt")}
    if set(prediction_files) != expected:
        missing = sorted(expected - set(prediction_files))
        extra = sorted(set(prediction_files) - expected)
        raise ValueError(f"Prediction directory mismatch: missing={missing[:20]} extra={extra[:20]}")

    class_counts: Counter[int] = Counter()
    confidences: list[float] = []
    detections = 0
    empty_files = 0
    max_detections_in_file = 0
    source_digest = hashlib.sha256()

    with zipfile.ZipFile(archive_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"ZIP CRC failure: {corrupt_member}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        if set(names) != expected:
            missing = sorted(expected - set(names))
            extra = sorted(set(names) - expected)
            raise ValueError(f"ZIP member mismatch: missing={missing[:20]} extra={extra[:20]}")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or len(pure.parts) != 1 or pure.name != name or not name.endswith(".txt"):
                raise ValueError(f"ZIP member is not a root TXT file: {name!r}")

        for name in sorted(expected):
            payload = archive.read(name)
            source = prediction_files[name].read_bytes()
            if payload != source:
                raise ValueError(f"ZIP member differs from source prediction: {name}")
            source_digest.update(name.encode("utf-8"))
            source_digest.update(b"\0")
            source_digest.update(source)

            lines = [
                line
                for line in payload.decode("utf-8-sig").splitlines()
                if line.strip()
            ]
            if len(lines) > max_det:
                raise ValueError(f"{name}: {len(lines)} detections exceeds max_det={max_det}")
            empty_files += int(not lines)
            max_detections_in_file = max(max_detections_in_file, len(lines))
            previous_confidence = math.inf

            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                if len(fields) != 6:
                    raise ValueError(f"{name}:{line_number}: expected 6 fields")
                try:
                    class_id = int(fields[0])
                    x, y, width, height, confidence = map(float, fields[1:])
                except ValueError as exc:
                    raise ValueError(f"{name}:{line_number}: invalid numeric field") from exc
                values = (x, y, width, height, confidence)
                if not 0 <= class_id < num_classes:
                    raise ValueError(f"{name}:{line_number}: class out of range")
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"{name}:{line_number}: non-finite value")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError(f"{name}:{line_number}: invalid normalized box")
                if not 0 <= confidence <= 1:
                    raise ValueError(f"{name}:{line_number}: invalid confidence")
                if confidence > previous_confidence + 1e-12:
                    raise ValueError(f"{name}:{line_number}: confidence order is not descending")
                previous_confidence = confidence
                detections += 1
                class_counts[class_id] += 1
                confidences.append(confidence)

    return {
        "status": "PASS",
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "prediction_dir": str(prediction_dir),
        "prediction_content_sha256": source_digest.hexdigest(),
        "files": len(expected),
        "detections": detections,
        "empty_files": empty_files,
        "max_detections_in_file": max_detections_in_file,
        "class_counts": {str(key): class_counts[key] for key in range(num_classes)},
        "confidence_min": min(confidences) if confidences else None,
        "confidence_median": statistics.median(confidences) if confidences else None,
        "confidence_max": max(confidences) if confidences else None,
        "checks": {
            "zip_crc": "PASS",
            "manifest_coverage": "PASS",
            "no_duplicate_members": "PASS",
            "root_txt_members_only": "PASS",
            "source_bytes_match": "PASS",
            "six_fields": "PASS",
            "class_range": "PASS",
            "finite_values": "PASS",
            "normalized_boxes": "PASS",
            "confidence_range": "PASS",
            "confidence_descending_per_image": "PASS",
            "max_detections": "PASS",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()

    report = audit(
        args.archive,
        args.manifest,
        args.predictions,
        num_classes=args.num_classes,
        max_det=args.max_det,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
