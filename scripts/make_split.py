#!/usr/bin/env python3
"""Create the leakage-resistant train/validation split."""

from __future__ import annotations

import argparse

from aic_mm.config import load_config, project_path
from aic_mm.data.split import create_group_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    options = dict(config.get("split_options") or {})
    result = create_group_split(
        project_path(config, config["train_manifest"]),
        project_path(config, config["split"]),
        **options,
    )
    print(f"Train: {result['stats']['train']['images']} images")
    print(f"Val:   {result['stats']['val']['images']} images")
    print(f"Split: {project_path(config, config['split'])}")


if __name__ == "__main__":
    main()
