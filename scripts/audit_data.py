#!/usr/bin/env python3
"""Create official-data manifests and a read-only integrity audit."""

from __future__ import annotations

import argparse

from aic_mm.config import load_config, project_path
from aic_mm.data.manifest import build_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--compute-phash",
        action="store_true",
        help="Compute train-only perceptual hashes (requires ImageHash and is slower).",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    report = build_manifests(
        project_path(config, config["train_root"]),
        project_path(config, config["test_root"]),
        project_path(config, config["artifacts_dir"]),
        config["_project_root"],
        compute_phash=args.compute_phash,
    )
    print(f"Train samples: {report['train']['samples']}")
    print(f"Test samples:  {report['test']['samples']}")
    print(f"Audit report:  {project_path(config, config['audit_report'])}")


if __name__ == "__main__":
    main()
