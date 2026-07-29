from pathlib import Path

import pytest
import yaml

import aic_mm.training.trainer as trainer_module


def _training_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data.yaml"
    weights = tmp_path / "weights" / "yolo26m.pt"
    resume = tmp_path / "outputs" / "stage1" / "weights" / "last.pt"
    data.write_text("names: {0: object}\ntrain: images\nval: images\n", encoding="utf-8")
    weights.parent.mkdir()
    weights.touch()
    resume.parent.mkdir(parents=True)
    resume.touch()
    config = {
        "project_root": ".",
        "data": "data.yaml",
        "model": "yolo26m-p2.yaml",
        "pretrained": "weights/yolo26m.pt",
        "fusion_variant": "v2",
        "train": {"project": "outputs", "name": "stage1", "epochs": 180},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, resume


def test_cli_resume_is_resolved_and_keeps_the_same_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, resume = _training_fixture(tmp_path)
    captured: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, *, overrides, tri_augment, fusion_variant):
            captured.update(
                overrides=overrides,
                tri_augment=tri_augment,
                fusion_variant=fusion_variant,
            )

        def add_callback(self, _event, _callback):
            return None

        def train(self):
            captured["trained"] = True

    monkeypatch.setattr(trainer_module, "TriModalDetectionTrainer", FakeTrainer)
    trainer_module.run_training(config_path, resume=resume)

    overrides = captured["overrides"]
    assert isinstance(overrides, dict)
    assert overrides["resume"] == str(resume.resolve())
    assert overrides["exist_ok"] is True
    assert captured["fusion_variant"] == "v2"
    assert captured["trained"] is True


def test_cli_resume_rejects_a_checkpoint_from_another_stage(tmp_path: Path) -> None:
    config_path, _resume = _training_fixture(tmp_path)
    wrong = tmp_path / "outputs" / "highres" / "weights" / "last.pt"
    wrong.parent.mkdir(parents=True)
    wrong.touch()

    with pytest.raises(ValueError, match="config output"):
        trainer_module.run_training(config_path, resume=wrong)
