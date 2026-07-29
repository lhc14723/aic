#!/usr/bin/env python3
"""Measure how much infrared and depth contribute to one trained model."""

from __future__ import annotations

import argparse
import json

from aic_mm.evaluation.ablation import ABLATION_MODES, evaluate_modality_ablations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="weights/final_all_4090d_clean/best.pt",
    )
    parser.add_argument(
        "--data",
        default="data/processed/aic_multispectral.yaml",
    )
    parser.add_argument(
        "--output",
        default="outputs/modality_ablation_final_all_960.json",
    )
    parser.add_argument(
        "--project",
        default="outputs/modality_ablation_final_all_960",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Acknowledge that this runs four complete GPU validations.",
    )
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit(
            "Modality ablation runs four GPU validations. Re-run with --yes when ready."
        )
    report = evaluate_modality_ablations(
        args.weights,
        args.data,
        output=args.output,
        project=args.project,
        modes=ABLATION_MODES,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        half=args.half,
        confidence=args.confidence,
        iou=args.iou,
        max_det=args.max_det,
    )
    summary = {
        mode: {
            "map50": result["map50"],
            "map50_95": result["map50_95"],
            "map50_95_delta_vs_full": result["map50_95_delta_vs_full"],
        }
        for mode, result in report["results"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
