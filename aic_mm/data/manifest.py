"""Discover and audit the official AIC tri-modal dataset."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aic_mm.constants import NUM_CLASSES, SUPPORTED_IMAGE_SUFFIXES
from aic_mm.io_utils import write_json, write_jsonl


class DatasetIntegrityError(RuntimeError):
    """Raised when official modalities or labels do not match."""


def _files_by_stem(directory: Path, suffixes: frozenset[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required dataset directory not found: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in result:
            raise DatasetIntegrityError(
                f"Duplicate sample stem '{path.stem}' in {directory}: {result[path.stem].name}, {path.name}"
            )
        result[path.stem] = path.resolve()
    return result


def infer_group_id(stem: str, image_suffix: str) -> str:
    """Infer a conservative scene group from the heterogeneous source names."""
    parts = stem.split("_")
    if stem.startswith("shuming_") and len(parts) >= 3:
        try:
            return f"shuming_bucket_{int(parts[1]) // 20:04d}"
        except ValueError:
            return "shuming_" + "_".join(parts[1:-1])
    if stem.startswith("hehe_") and len(parts) >= 5:
        return "hehe_" + "_".join(parts[2:-1])
    if stem.startswith("hehe_") and len(parts) >= 3:
        try:
            return f"hehe_bucket_{int(parts[1]) // 20:04d}"
        except ValueError:
            return "hehe_" + "_".join(parts[1:-1])
    if len(parts) >= 3:
        return "sequence_" + "_".join(parts[:-1])
    if stem.isdigit():
        domain = "jpg" if image_suffix.lower() in {".jpg", ".jpeg"} else "png"
        return f"{domain}_numeric_{int(stem) // 100:05d}"
    return f"singleton_{stem}"


def audit_label_file(path: Path) -> tuple[list[int], list[dict[str, Any]]]:
    """Parse one official training label and report, but do not modify, anomalies."""
    class_counts = [0] * NUM_CLASSES
    issues: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                issues.append({"line": line_number, "kind": "field_count", "value": line})
                continue
            try:
                class_id = int(fields[0])
                x, y, width, height = (float(value) for value in fields[1:])
            except ValueError:
                issues.append({"line": line_number, "kind": "parse_error", "value": line})
                continue
            values = np.asarray([x, y, width, height], dtype=np.float64)
            if class_id < 0 or class_id >= NUM_CLASSES:
                issues.append({"line": line_number, "kind": "class_range", "value": line})
                continue
            if not np.isfinite(values).all() or width <= 0 or height <= 0:
                issues.append({"line": line_number, "kind": "invalid_numeric", "value": line})
                continue
            class_counts[class_id] += 1
            x1, y1 = x - width / 2, y - height / 2
            x2, y2 = x + width / 2, y + height / 2
            overflow = max(0.0, -x1, -y1, x2 - 1.0, y2 - 1.0)
            if overflow > 0:
                issues.append(
                    {
                        "line": line_number,
                        "kind": "box_crosses_image",
                        "overflow": overflow,
                        "value": line,
                    }
                )
    return class_counts, issues


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _perceptual_hash(path: Path) -> str:
    try:
        import imagehash
    except ImportError as exc:
        raise RuntimeError("ImageHash is required for --compute-phash") from exc
    with Image.open(path) as image:
        return str(imagehash.phash(image.convert("RGB")))


def scan_dataset(
    dataset_root: str | Path,
    project_root: str | Path,
    *,
    require_labels: bool,
    compute_phash: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan one extracted dataset and return manifest rows plus audit summary."""
    root = Path(dataset_root).resolve()
    project = Path(project_root).resolve()
    modalities = {
        name: _files_by_stem(root / name, SUPPORTED_IMAGE_SUFFIXES)
        for name in ("visible", "infrared", "depth")
    }
    labels = _files_by_stem(root / "labels", frozenset({".txt"})) if require_labels else {}

    stem_sets = {name: set(paths) for name, paths in modalities.items()}
    all_stems = set.union(*stem_sets.values())
    common_stems = set.intersection(*stem_sets.values())
    if all_stems != common_stems:
        missing = {name: sorted(all_stems - stems)[:20] for name, stems in stem_sets.items()}
        raise DatasetIntegrityError(f"Modalities are not one-to-one under {root}: {missing}")
    if require_labels and set(labels) != common_stems:
        raise DatasetIntegrityError(
            f"Image/label stems differ under {root}: "
            f"missing labels={sorted(common_stems - set(labels))[:20]}, "
            f"orphan labels={sorted(set(labels) - common_stems)[:20]}"
        )

    rows: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    issue_examples: list[dict[str, Any]] = []
    format_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    class_totals = [0] * NUM_CLASSES

    for stem in sorted(common_stems):
        paths = {name: modalities[name][stem] for name in modalities}
        metadata: dict[str, tuple[int, int, str]] = {}
        for name, path in paths.items():
            with Image.open(path) as image:
                metadata[name] = (image.width, image.height, image.mode)
        dimensions = {(width, height) for width, height, _ in metadata.values()}
        if len(dimensions) != 1:
            raise DatasetIntegrityError(f"Spatially mismatched modalities for '{stem}': {metadata}")
        width, height = next(iter(dimensions))

        class_counts = [0] * NUM_CLASSES
        if require_labels:
            class_counts, issues = audit_label_file(labels[stem])
            for issue in issues:
                issue_counts[issue["kind"]] += 1
                if len(issue_examples) < 50:
                    issue_examples.append({"sample_id": stem, **issue})
            class_totals = [a + b for a, b in zip(class_totals, class_counts)]

        image_suffix = paths["visible"].suffix.lower()
        format_counts[image_suffix] += 1
        dimension_counts[f"{width}x{height}"] += 1
        row = {
            "sample_id": stem,
            "visible": _relative(paths["visible"], project),
            "infrared": _relative(paths["infrared"], project),
            "depth": _relative(paths["depth"], project),
            "label": _relative(labels[stem], project) if require_labels else None,
            "width": width,
            "height": height,
            "image_suffix": image_suffix,
            "depth_mode": metadata["depth"][2],
            "group_id": infer_group_id(stem, image_suffix),
            "class_counts": class_counts,
        }
        if compute_phash:
            row["visible_phash"] = _perceptual_hash(paths["visible"])
        rows.append(row)

    summary = {
        "dataset_root": _relative(root, project),
        "samples": len(rows),
        "formats": dict(sorted(format_counts.items())),
        "dimensions": dict(sorted(dimension_counts.items())),
        "class_totals": class_totals,
        "label_issue_counts": dict(sorted(issue_counts.items())),
        "label_issue_examples": issue_examples,
        "phash_computed": compute_phash,
    }
    return rows, summary


def build_manifests(
    train_root: str | Path,
    test_root: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    *,
    compute_phash: bool = False,
) -> dict[str, Any]:
    """Build deterministic train/test JSONL manifests and an audit report."""
    destination = Path(output_dir)
    train_rows, train_summary = scan_dataset(
        train_root, project_root, require_labels=True, compute_phash=compute_phash
    )
    test_rows, test_summary = scan_dataset(
        test_root, project_root, require_labels=False, compute_phash=False
    )
    write_jsonl(destination / "train_manifest.jsonl", train_rows)
    write_jsonl(destination / "test_manifest.jsonl", test_rows)
    report = {"train": train_summary, "test": test_summary}
    write_json(destination / "audit_report.json", report)
    return report

