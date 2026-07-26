from pathlib import Path

from aic_mm.data.split import create_group_split
from aic_mm.io_utils import write_jsonl


def test_milp_split_preserves_groups_and_classes(tmp_path: Path) -> None:
    rows = []
    for group_index in range(20):
        for frame_index in range(3):
            counts = [1] * 12
            # Add a dominant common class, mimicking the real long-tail dataset.
            counts[0] += 4
            rows.append(
                {
                    "sample_id": f"{group_index:02d}_{frame_index:02d}",
                    "group_id": f"group_{group_index:02d}",
                    "class_counts": counts,
                    "image_suffix": ".jpg" if group_index < 4 else ".png",
                    "width": 640 if group_index < 4 else 1920,
                    "height": 360 if group_index < 4 else 1080,
                }
            )
    manifest, output = tmp_path / "manifest.jsonl", tmp_path / "split.json"
    write_jsonl(manifest, rows)
    split = create_group_split(manifest, output, val_fraction=0.2)
    train_groups = {sample_id.split("_")[0] for sample_id in split["train"]}
    val_groups = {sample_id.split("_")[0] for sample_id in split["val"]}
    assert train_groups.isdisjoint(val_groups)
    assert all(count > 0 for count in split["stats"]["val"]["class_counts"])
