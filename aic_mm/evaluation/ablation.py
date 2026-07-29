"""Automatic modality ablation for one trained tri-modal detector."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from ultralytics import YOLO

from aic_mm.io_utils import write_json
from aic_mm.models.fusion import TriModalStem

ABLATION_MODES = ("full", "rgb", "rgb_ir", "rgb_depth")

_ZERO_CHANNELS: dict[str, tuple[slice, ...]] = {
    "full": (),
    "rgb": (slice(3, 8),),
    "rgb_ir": (slice(5, 8),),
    "rgb_depth": (slice(3, 5),),
}


def mask_modalities(inputs: torch.Tensor, mode: str) -> torch.Tensor:
    """Return one input tensor with the requested sensor branches disabled."""
    if mode not in _ZERO_CHANNELS:
        raise ValueError(f"Unknown ablation mode {mode!r}; expected one of {ABLATION_MODES}")
    if inputs.ndim != 4 or inputs.shape[1] != 8:
        raise ValueError(f"Expected BCHW input with 8 channels, got {tuple(inputs.shape)}")
    if mode == "full":
        return inputs
    masked = inputs.clone()
    for channels in _ZERO_CHANNELS[mode]:
        masked[:, channels] = 0
    return masked


def _find_tri_modal_stem(model: nn.Module) -> TriModalStem:
    stems = [module for module in model.modules() if isinstance(module, TriModalStem)]
    if len(stems) != 1:
        raise RuntimeError(f"Expected exactly one TriModalStem, found {len(stems)}")
    return stems[0]


def _register_mask(stem: TriModalStem, mode: str) -> torch.utils.hooks.RemovableHandle | None:
    if mode == "full":
        return None

    def pre_hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("TriModalStem did not receive a tensor as its first argument")
        return (mask_modalities(args[0], mode), *args[1:])

    return stem.register_forward_pre_hook(pre_hook)


def _serialize_metrics(metrics: Any, names: dict[int, str]) -> dict[str, Any]:
    box = metrics.box
    class_indices = [int(index) for index in box.ap_class_index]
    per_class: dict[str, dict[str, float | int]] = {}
    for position, class_id in enumerate(class_indices):
        per_class[names[class_id]] = {
            "class_id": class_id,
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "map50": float(box.ap50[position]),
            "map50_95": float(box.maps[position]),
        }
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
        "per_class": per_class,
    }


def evaluate_modality_ablations(
    weights: str | Path,
    data: str | Path,
    *,
    output: str | Path,
    project: str | Path,
    modes: Iterable[str] = ABLATION_MODES,
    imgsz: int = 960,
    batch: int = 4,
    workers: int = 1,
    device: str | int = 0,
    half: bool = True,
    confidence: float = 0.001,
    iou: float = 0.70,
    max_det: int = 100,
) -> dict[str, Any]:
    """Evaluate full and sensor-masked variants without changing checkpoint weights."""
    weights = Path(weights).resolve()
    data = Path(data).resolve()
    output = Path(output)
    project = Path(project)
    selected_modes = tuple(modes)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if not selected_modes:
        raise ValueError("At least one ablation mode is required")
    invalid = [mode for mode in selected_modes if mode not in ABLATION_MODES]
    if invalid:
        raise ValueError(f"Unknown ablation modes: {invalid}")
    if len(set(selected_modes)) != len(selected_modes):
        raise ValueError(f"Duplicate ablation modes: {selected_modes}")

    report: dict[str, Any] = {
        "weights": str(weights),
        "data": str(data),
        "settings": {
            "imgsz": int(imgsz),
            "batch": int(batch),
            "workers": int(workers),
            "device": str(device),
            "half": bool(half),
            "confidence": float(confidence),
            "iou": float(iou),
            "max_det": int(max_det),
        },
        "results": {},
    }

    for mode in selected_modes:
        started = time.perf_counter()
        model = YOLO(str(weights), task="detect")
        stem = _find_tri_modal_stem(model.model)
        handle = _register_mask(stem, mode)
        try:
            metrics = model.val(
                data=str(data),
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                device=device,
                half=half,
                conf=confidence,
                iou=iou,
                max_det=max_det,
                plots=False,
                save_json=False,
                project=str(project),
                name=mode,
                exist_ok=True,
                verbose=False,
            )
            result = _serialize_metrics(metrics, model.names)
            result["elapsed_seconds"] = time.perf_counter() - started
            report["results"][mode] = result
            write_json(output, report)
        finally:
            if handle is not None:
                handle.remove()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    full_result = report["results"].get("full")
    if full_result is not None:
        full_map = float(full_result["map50_95"])
        for result in report["results"].values():
            result["map50_95_delta_vs_full"] = float(result["map50_95"]) - full_map
    write_json(output, report)
    return report
