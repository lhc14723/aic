"""Custom Ultralytics trainer for the AIC 8-channel detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from aic_mm.config import load_config, project_path, require_mapping
from aic_mm.data.dataset import TriModalYOLODataset
from aic_mm.models.fusion import TriModalStem, replace_first_layer_with_tri_modal_stem


def _weights_have_custom_stem(weights: Any) -> bool:
    source = weights.get("model") if isinstance(weights, dict) else weights
    layers = getattr(source, "model", None)
    return bool(layers is not None and len(layers) and isinstance(layers[0], TriModalStem))


class TriModalDetectionTrainer(DetectionTrainer):
    """Detection trainer that installs the fusion stem and sensor augmentation."""

    def __init__(self, *args: Any, tri_augment: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.tri_augment = dict(tri_augment or {})
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
        # A custom checkpoint already contains branch-specific parameters and
        # therefore needs the same module layout before state-dict transfer.
        if weights is not None and _weights_have_custom_stem(weights):
            replace_first_layer_with_tri_modal_stem(model)
            model.load(weights)
        else:
            if weights is not None:
                model.load(weights)
            replace_first_layer_with_tri_modal_stem(model)
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


def run_training(config_path: str | Path) -> None:
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
            "Download the official YOLO26s checkpoint before training."
        )

    project_value = train_settings.get("project")
    if project_value:
        train_settings["project"] = str(project_path(config, project_value))
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
    )
    trainer.train()
