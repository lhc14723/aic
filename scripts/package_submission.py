#!/usr/bin/env python3
"""Strictly validate and ZIP the automatic test predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from aic_mm.inference.submission import package_submission
from aic_mm.io_utils import read_jsonl


def _expected_ids(manifest: str | None, test_images: str) -> list[str]:
    """Resolve the immutable test sample list from a manifest or TIFF directory."""
    if manifest:
        manifest_path = Path(manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Test manifest not found: {manifest_path}")
        sample_ids = [str(row["sample_id"]) for row in read_jsonl(manifest_path)]
    else:
        image_dir = Path(test_images)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Processed test image directory not found: {image_dir}")
        sample_ids = sorted(path.stem for path in image_dir.glob("*.tiff"))
    if not sample_ids:
        raise ValueError("No expected test sample IDs were found")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Expected test sample IDs contain duplicates")
    return sample_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/test_predictions")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSONL manifest. When omitted, IDs come from --test-images.",
    )
    parser.add_argument("--test-images", default="data/processed/images/test")
    parser.add_argument("--output", default="outputs/submission.zip")
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()
    expected_ids = _expected_ids(args.manifest, args.test_images)
    print(
        package_submission(
            args.predictions,
            expected_ids,
            args.output,
            max_det=args.max_det,
        )
    )


if __name__ == "__main__":
    main()
