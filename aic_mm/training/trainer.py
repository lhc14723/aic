"""Custom Ultralytics trainer for the AIC 8-channel detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from aic_mm.config import load_config, project_path, require_mapping
from aic_mm.data.dataset import TriModalYOLODataset
from aic_mm.models.fusion import (
    FusionVariant,
    TriModalStem,
    TriModalStemV2,
    initialize_stem_from_rgb_conv,
    replace_first_layer_with_tri_modal_stem,
)


def _weights_have_custom_stem(weights: Any) -> bool:
    source = weights.get("model") if isinstance(weights, dict) else weights
    layers = getattr(source, "model", None)
    return bool(layers is not None and len(layers) and isinstance(layers[0], TriModalStem))


def _weights_fusion_variant(weights: Any) -> FusionVariant | None:
    source = weights.get("model") if isinstance(weights, dict) else weights
    layers = getattr(source, "model", None)
    if layers is None or not len(layers):
        return None
    if isinstance(layers[0], TriModalStemV2):
        return "v2"
    if isinstance(layers[0], TriModalStem):
        return "v1"
    return None


def _weights_first_conv(weights: Any) -> Any:
    source = weights.get("model") if isinstance(weights, dict) else weights
    layers = getattr(source, "model", None)
    return layers[0] if layers is not None and len(layers) else None


def _assert_loss_integrity(trainer: DetectionTrainer) -> None:
    """Abort immediately when a loss tensor is negative or non-finite.

    A failed process can then be resumed by the external supervisor from the
    previous epoch checkpoint instead of allowing invalid arithmetic to
    contaminate subsequent checkpoints.
    """
    loss_items = getattr(trainer, "loss_items", None)
    if isinstance(loss_items, dict) and loss_items:
        named_values = list(loss_items.items())
    else:
        loss = getattr(trainer, "loss", None)
        if loss is None:
            return
        named_values = [("loss", loss)]

    tensors = [
        value.detach().reshape(-1)
        for _, value in named_values
        if isinstance(value, torch.Tensor) and value.numel()
    ]
    if not tensors:
        return
    values = torch.cat(tensors).to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(values).all()):
        snapshot = {name: value.detach().cpu().tolist() for name, value in named_values}
        raise FloatingPointError(f"QUALITY_GUARD_NONFINITE_LOSS {snapshot}")
    if float(values.min()) < -1e-7:
        snapshot = {name: value.detach().cpu().tolist() for name, value in named_values}
        raise FloatingPointError(f"QUALITY_GUARD_NEGATIVE_LOSS {snapshot}")


class TriModalDetectionTrainer(DetectionTrainer):
    """Detection trainer that installs the fusion stem and sensor augmentation."""

    def __init__(
        self,
        *args: Any,
        tri_augment: dict[str, Any] | None = None,
        fusion_variant: FusionVariant | None = None,
        **kwargs: Any,
    ) -> None:
        self.tri_augment = dict(tri_augment or {})
        if fusion_variant not in (None, "v1", "v2"):
            raise ValueError("fusion_variant must be one of: null, 'v1', 'v2'")
        self.fusion_variant = fusion_variant
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: str | None = None, weights: Any = None, verbose: bool = True):
        model = self.set_model_names_for_load(
            DetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
            )
        )
        checkpoint_variant = _weights_fusion_variant(weights) if weights is not None else None
        target_variant: FusionVariant = self.fusion_variant or checkpoint_variant or "v1"
        # A custom checkpoint needs a compatible branch layout before transfer.
        # When V2 is explicitly requested from V1 weights, common branch
        # tensors transfer and the wider gate is initialized safely.
        if weights is not None and _weights_have_custom_stem(weights):
            replace_first_layer_with_tri_modal_stem(model, variant=target_variant)
            model.load(weights)
        else:
            if weights is not None:
                model.load(weights)
            stem = replace_first_layer_with_tri_modal_stem(model, variant=target_variant)
            source_conv = _weights_first_conv(weights) if weights is not None else None
            if isinstance(source_conv, Conv):
                initialize_stem_from_rgb_conv(stem, source_conv)
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        padding = 0.0 if mode == "train" else 0.5
        return TriModalYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=padding,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            tri_augment=self.tri_augment if mode == "train" else None,
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        # Ultralytics' plotting code assumes three channels. Keep diagnostics
        # useful by visualizing the visible branch only.
        visible_batch = dict(batch)
        visible_batch["img"] = batch["img"][:, :3]
        super().plot_training_samples(visible_batch, ni)


def run_training(config_path: str | Path, *, resume: str | Path | None = None) -> None:
    """Validate project configuration and start an explicitly requested run."""
    config = load_config(config_path)
    train_settings = dict(require_mapping(config, "train"))
    data_path = project_path(config, config["data"])
    pretrained = project_path(config, config["pretrained"])
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset YAML not found: {data_path}. Run scripts/build_multispectral.py first."
        )
    if not pretrained.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {pretrained}. "
            f"Download or transfer the configured {pretrained.name} checkpoint before training."
        )

    project_value = train_settings.get("project")
    if project_value:
        train_settings["project"] = str(project_path(config, project_value))
    resume_value = resume if resume is not None else train_settings.get("resume")
    if isinstance(resume_value, (str, Path)):
        resume_path = project_path(config, resume_value)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        run_name = train_settings.get("name")
        if project_value and run_name:
            expected_run = (project_path(config, project_value) / str(run_name)).resolve()
            if resume_path.parent.parent.resolve() != expected_run:
                raise ValueError(
                    f"Resume checkpoint belongs to {resume_path.parent.parent}, "
                    f"but config output is {expected_run}"
                )
        train_settings["resume"] = str(resume_path)
        train_settings["exist_ok"] = True
    train_settings.update(
        {
            "data": str(data_path),
            "model": config.get("model", "yolo26s-p2.yaml"),
            "pretrained": str(pretrained),
            "task": "detect",
            "mode": "train",
        }
    )
    trainer = TriModalDetectionTrainer(
        overrides=train_settings,
        tri_augment=dict(config.get("tri_augment") or {}),
        fusion_variant=config.get("fusion_variant"),
    )
    trainer.add_callback("on_train_batch_end", _assert_loss_integrity)
    trainer.train()
