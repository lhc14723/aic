from pathlib import Path

import cv2
import numpy as np

from aic_mm.data.encode import (
    _dataset_yaml,
    encode_modalities,
    sanitize_label_file,
    write_multipage_tiff,
)


def test_sanitize_clips_without_touching_source(tmp_path: Path) -> None:
    source = tmp_path / "label.txt"
    original = "0 0.01 0.50 0.10 0.20\n1 0.50 0.50 0.20 0.20\n"
    source.write_text(original, encoding="utf-8")
    derived, stats = sanitize_label_file(source)
    assert source.read_text(encoding="utf-8") == original
    assert stats["clipped"] == 1
    assert len(derived.strip().splitlines()) == 2
    first = list(map(float, derived.splitlines()[0].split()[1:]))
    assert np.allclose(first, [0.03, 0.5, 0.06, 0.2])


def test_encode_and_tiff_roundtrip(tmp_path: Path) -> None:
    visible = np.zeros((12, 16, 3), dtype=np.uint8)
    visible[..., 2] = 200  # BGR red
    infrared = np.tile(np.arange(16, dtype=np.uint8), (12, 1))
    depth = np.full((12, 16), 1000, dtype=np.uint16)
    paths = [tmp_path / name for name in ("visible.png", "infrared.png", "depth.png")]
    assert cv2.imwrite(str(paths[0]), visible)
    assert cv2.imwrite(str(paths[1]), infrared)
    assert cv2.imwrite(str(paths[2]), depth)

    encoded = encode_modalities(*paths)
    assert encoded.shape == (12, 16, 8)
    assert encoded.dtype == np.uint8
    assert np.all(encoded[..., 0] == 200)
    assert np.all(encoded[..., 6] == 255)
    assert np.all(encoded[..., 7] == 255)

    output = tmp_path / "sample.tiff"
    write_multipage_tiff(output, encoded)
    ok, pages = cv2.imreadmulti(str(output), flags=cv2.IMREAD_UNCHANGED)
    assert ok and len(pages) == 8
    assert np.array_equal(np.dstack(pages), encoded)


def test_dataset_yaml_is_portable_between_project_roots(tmp_path: Path) -> None:
    clean = _dataset_yaml(tmp_path)
    all_data = _dataset_yaml(tmp_path, all_training_images=True)
    assert "path" not in clean
    assert clean["train"] == "images/train"
    assert all_data["train"] == ["images/train", "images/val"]
    assert clean["val"] == all_data["val"] == "images/val"
