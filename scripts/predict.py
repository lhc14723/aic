#!/usr/bin/env python3
"""Run automatic inference over all processed test samples."""

from __future__ import annotations

import argparse

from aic_mm.config import load_config, project_path
from aic_mm.inference.predict import predict_test_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/predict.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    report = predict_test_set(
        project_path(config, config["weights"]),
        project_path(config, config["image_dir"]),
        project_path(config, config["output_dir"]),
        **dict(config.get("predict") or {}),
    )
    print(report)


if __name__ == "__main__":
    main()
