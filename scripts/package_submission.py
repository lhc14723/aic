#!/usr/bin/env python3
"""Strictly validate and ZIP the automatic test predictions."""

from __future__ import annotations

import argparse

from aic_mm.inference.submission import package_submission
from aic_mm.io_utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/test_predictions")
    parser.add_argument("--manifest", default="artifacts/test_manifest.jsonl")
    parser.add_argument("--output", default="outputs/submission.zip")
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()
    expected_ids = [str(row["sample_id"]) for row in read_jsonl(args.manifest)]
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
