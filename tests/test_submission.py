from pathlib import Path
from zipfile import ZipFile

import pytest

from aic_mm.inference.submission import package_submission, validate_prediction_directory
from scripts.package_submission import _expected_ids


def test_submission_validation_and_root_layout(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "a.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")
    (predictions / "b.txt").write_text("", encoding="utf-8")
    archive = tmp_path / "submission.zip"
    report = package_submission(predictions, ["a", "b"], archive)
    assert report["files"] == 2
    with ZipFile(archive) as handle:
        assert handle.namelist() == ["a.txt", "b.txt"]


def test_submission_rejects_missing_file(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    with pytest.raises(ValueError, match="missing files"):
        validate_prediction_directory(predictions, ["a"])


def test_expected_ids_fall_back_to_processed_test_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "b.tiff").touch()
    (images / "a.tiff").touch()
    (images / "ignored.png").touch()
    assert _expected_ids(None, str(images)) == ["a", "b"]
