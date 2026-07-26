#!/usr/bin/env python3
"""Build the derived 8-channel training/test dataset."""

from __future__ import annotations

import argparse

from aic_mm.config import load_config, project_path
from aic_mm.data.encode import prepare_processed_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--subsets", nargs="+", choices=("train", "val", "test"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Acknowledge that full TIFF encoding is time-consuming and uses substantial disk space.",
    )
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit(
            "This command will process 3000 full-resolution samples and create a large derived dataset. "
            "Re-run with --yes after checking available disk space."
        )
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    config = load_config(args.config)
    report = prepare_processed_dataset(
        project_path(config, config["train_manifest"]),
        project_path(config, config["test_manifest"]),
        project_path(config, config["split"]),
        project_path(config, config["processed_root"]),
        config["_project_root"],
        workers=args.workers,
        overwrite=args.overwrite,
        subsets=args.subsets or ("train", "val", "test"),
    )
    print(report)


if __name__ == "__main__":
    main()
