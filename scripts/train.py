#!/usr/bin/env python3
"""Train the quality-gated AIC detector."""

from __future__ import annotations

import argparse

from aic_mm.training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_fusion.yaml")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume this config's interrupted run from a compatible last.pt.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Acknowledge that this starts a long, GPU-intensive training run.",
    )
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit("Training is a long GPU task. Re-run with --yes when ready.")
    run_training(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
