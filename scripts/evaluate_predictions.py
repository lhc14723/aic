#!/usr/bin/env python3
"""Evaluate validation predictions with the competition-style metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_mm.evaluation.metric import evaluate_directories
from aic_mm.io_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="data/processed/labels/val")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default="artifacts/split.json")
    parser.add_argument("--output", default="outputs/validation_metric.json")
    args = parser.parse_args()
    split = read_json(args.split)
    result = evaluate_directories(
        args.ground_truth,
        args.predictions,
        sample_ids=list(split["val"]),
        output_path=args.output,
    )
    print(json.dumps({"mAP50": result["map50"], "mAP50-95": result["map50_95"]}, indent=2))
    print(f"Detailed report: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
