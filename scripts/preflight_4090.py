#!/usr/bin/env python3
"""Fail-fast readiness checks for the YOLO26m-V2 single-RTX-4090 pipeline."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml

from aic_mm.config import load_config, project_path, require_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {
    "images/train": 1600,
    "images/val": 400,
    "images/test": 1000,
    "labels/train": 1600,
    "labels/val": 400,
}
CONFIG_CONTRACTS: dict[str, dict[str, Any]] = {
    "configs/smoke_fusion_v2_m_4090.yaml": {
        "data": "data/processed/aic_multispectral.yaml",
        "pretrained": "weights/yolo26m.pt",
        "name": "smoke_fusion_v2_m_4090_do_not_submit",
        "epochs": 1,
        "batch": 8,
        "imgsz": 960,
    },
    "configs/train_fusion_v2_m_4090.yaml": {
        "data": "data/processed/aic_multispectral.yaml",
        "pretrained": "weights/yolo26m.pt",
        "name": "aic_fusion_v2_m_stage1_4090",
        "epochs": 180,
        "batch": 8,
        "imgsz": 960,
    },
    "configs/smoke_fusion_v2_m_highres_4090.yaml": {
        "data": "data/processed/aic_multispectral.yaml",
        "pretrained": "outputs/aic_fusion_v2_m_stage1_4090/weights/best.pt",
        "name": "smoke_fusion_v2_m_highres_4090_do_not_submit",
        "epochs": 1,
        "batch": 4,
        "imgsz": 1280,
    },
    "configs/finetune_fusion_v2_m_highres_4090.yaml": {
        "data": "data/processed/aic_multispectral.yaml",
        "pretrained": "outputs/aic_fusion_v2_m_stage1_4090/weights/best.pt",
        "name": "aic_fusion_v2_m_highres_4090",
        "epochs": 40,
        "batch": 4,
        "imgsz": 1280,
    },
    "configs/final_all_fusion_v2_m_4090.yaml": {
        "data": "data/processed/aic_multispectral_all.yaml",
        "pretrained": "outputs/aic_fusion_v2_m_highres_4090/weights/best.pt",
        "name": "aic_fusion_v2_m_final_all_4090",
        "epochs": 20,
        "batch": 4,
        "imgsz": 1280,
    },
}


def _count_files(relative: str) -> int:
    directory = PROJECT_ROOT / "data" / "processed" / relative
    suffix = "*.tiff" if relative.startswith("images/") else "*.txt"
    return sum(1 for path in directory.glob(suffix) if path.is_file())


def _check_dataset_yaml(relative: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Dataset YAML must be a mapping: {path}")
    if "path" in content:
        raise ValueError(
            f"{path} contains a fixed 'path'. Remove it so splits resolve relative to the YAML."
        )
    if content.get("channels") != 8 or len(content.get("names", {})) != 12:
        raise ValueError(f"Dataset contract mismatch in {path}: expected 8 channels and 12 classes")
    return content


def _check_config(relative: str, contract: dict[str, Any]) -> dict[str, Any]:
    config = load_config(PROJECT_ROOT / relative)
    train = require_mapping(config, "train")
    if config.get("model") != "yolo26m-p2.yaml":
        raise ValueError(f"{relative}: model must be yolo26m-p2.yaml")
    if config.get("fusion_variant") != "v2":
        raise ValueError(f"{relative}: fusion_variant must be v2")
    for key in ("epochs", "batch", "imgsz"):
        if train.get(key) != contract[key]:
            raise ValueError(f"{relative}: {key}={train.get(key)!r}, expected {contract[key]!r}")
    if str(config.get("data")) != contract["data"]:
        raise ValueError(f"{relative}: unexpected data path {config.get('data')!r}")
    if str(config.get("pretrained")) != contract["pretrained"]:
        raise ValueError(f"{relative}: unexpected pretrained path {config.get('pretrained')!r}")
    if train.get("name") != contract["name"]:
        raise ValueError(f"{relative}: unexpected output name {train.get('name')!r}")
    if train.get("resume") != contract.get("resume"):
        raise ValueError(f"{relative}: unexpected resume path {train.get('resume')!r}")
    if not project_path(config, config["data"]).is_file():
        raise FileNotFoundError(f"{relative}: dataset YAML is missing")
    return {
        "model": config["model"],
        "fusion_variant": config["fusion_variant"],
        "epochs": train["epochs"],
        "batch": train["batch"],
        "imgsz": train["imgsz"],
        "resume": train.get("resume"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Validate files and configs without requiring a CUDA GPU (for local CI only).",
    )
    args = parser.parse_args()

    if Path.cwd().resolve() != PROJECT_ROOT:
        raise SystemExit(f"Run from the project root: cd {PROJECT_ROOT}")

    weight = PROJECT_ROOT / "weights" / "yolo26m.pt"
    if not weight.is_file() or weight.stat().st_size < 10_000_000:
        raise SystemExit(f"Official YOLO26m checkpoint is missing or incomplete: {weight}")

    datasets = {
        relative: _check_dataset_yaml(relative)
        for relative in (
            "data/processed/aic_multispectral.yaml",
            "data/processed/aic_multispectral_all.yaml",
        )
    }
    counts = {relative: _count_files(relative) for relative in EXPECTED_COUNTS}
    wrong_counts = {
        relative: {"actual": counts[relative], "expected": expected}
        for relative, expected in EXPECTED_COUNTS.items()
        if counts[relative] != expected
    }
    if wrong_counts:
        raise SystemExit(f"Processed dataset counts are incorrect: {wrong_counts}")

    configs = {
        relative: _check_config(relative, contract)
        for relative, contract in CONFIG_CONTRACTS.items()
    }
    predict_config = load_config(PROJECT_ROOT / "configs/predict_fusion_v2_m_tta_4090.yaml")
    predict = dict(predict_config.get("predict") or {})
    expected_prediction = {
        "weights": "outputs/aic_fusion_v2_m_final_all_4090/weights/last.pt",
        "image_dir": "data/processed/images/test",
        "output_dir": "outputs/test_predictions_fusion_v2_m_tta_4090",
        "imgsz": 1280,
        "batch": 4,
        "tta_scales": [960, 1280],
        "tta_flip": True,
    }
    actual_prediction = {
        "weights": predict_config.get("weights"),
        "image_dir": predict_config.get("image_dir"),
        "output_dir": predict_config.get("output_dir"),
        "imgsz": predict.get("imgsz"),
        "batch": predict.get("batch"),
        "tta_scales": predict.get("tta_scales"),
        "tta_flip": predict.get("tta_flip"),
    }
    if actual_prediction != expected_prediction:
        raise SystemExit(
            f"Final prediction config contract mismatch: "
            f"actual={actual_prediction}, expected={expected_prediction}"
        )

    gpu: dict[str, Any] | None = None
    if not args.skip_gpu:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable in the active Python environment")
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_gib < 20:
            raise SystemExit(
                f"GPU 0 has only {total_gib:.1f} GiB; the supplied batch sizes target a 24 GB RTX 4090"
            )
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_gib": round(total_gib, 2),
            "torch_cuda": torch.version.cuda,
        }

    report = {
        "status": "READY",
        "project_root": str(PROJECT_ROOT),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": gpu,
        "weight_bytes": weight.stat().st_size,
        "counts": counts,
        "datasets": {
            relative: {
                "portable": "path" not in content,
                "channels": content["channels"],
                "classes": len(content["names"]),
            }
            for relative, content in datasets.items()
        },
        "configs": configs,
        "prediction": actual_prediction,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
