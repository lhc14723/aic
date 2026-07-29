#!/usr/bin/env python3
"""Build a labeled, leaderboard-like probe split without touching test labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aic_mm.constants import CHANNEL_NAMES, CLASS_NAMES, NUM_CLASSES


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _hash_int(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _choose_group_query(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row["sample_id"]))
    selected = [
        row for row in ordered if _hash_int(seed, str(row["sample_id"])) % 3 == 0
    ]
    if not selected:
        selected = [ordered[_hash_int(seed, str(ordered[0]["group_id"])) % len(ordered)]]
    if len(selected) == len(ordered):
        selected = selected[:-1]
    return selected


def _class_counts(rows: list[dict[str, Any]]) -> np.ndarray:
    return sum(
        (np.asarray(row["class_counts"], dtype=np.int64) for row in rows),
        np.zeros(NUM_CLASSES, dtype=np.int64),
    )


def _find_processed_image(processed_root: Path, sample_id: str) -> Path:
    candidates = [
        processed_root / "images" / "train" / f"{sample_id}.tiff",
        processed_root / "images" / "val" / f"{sample_id}.tiff",
    ]
    found = [path.resolve() for path in candidates if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one processed image for {sample_id}: {found}")
    label = Path(str(found[0]).replace("/images/", "/labels/")).with_suffix(".txt")
    if not label.is_file():
        raise FileNotFoundError(label)
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/train_manifest.jsonl")
    parser.add_argument("--strict-split", default="artifacts/split.json")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--output-dir", default="artifacts/intrascene_probe")
    parser.add_argument("--candidate-seeds", type=int, default=10000)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    strict_split_path = Path(args.strict_split)
    rows = _read_jsonl(manifest_path)
    row_by_id = {str(row["sample_id"]): row for row in rows}
    strict_split = json.loads(strict_split_path.read_text(encoding="utf-8"))
    strict_val_ids = set(map(str, strict_split["val"]))
    strict_val_rows = [row_by_id[sample_id] for sample_id in sorted(strict_val_ids)]

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_val_rows:
        by_group[str(row["group_id"])].append(row)
    eligible_groups = {
        group_id: group_rows
        for group_id, group_rows in by_group.items()
        if len(group_rows) >= 2
    }
    eligible_rows = [row for group_rows in eligible_groups.values() for row in group_rows]
    eligible_totals = _class_counts(eligible_rows)
    target_images = len(eligible_rows) / 3.0
    target_classes = eligible_totals.astype(np.float64) / 3.0

    best: tuple[float, int, list[dict[str, Any]], np.ndarray] | None = None
    for seed in range(args.candidate_seeds):
        query = [
            row
            for group_rows in eligible_groups.values()
            for row in _choose_group_query(group_rows, seed)
        ]
        counts = _class_counts(query)
        image_error = abs(len(query) - target_images) / max(target_images, 1.0)
        class_error = float(
            np.mean(np.abs(counts - target_classes) / np.maximum(target_classes, 1.0))
        )
        missing = int(np.count_nonzero((counts == 0) & (eligible_totals > 0)))
        score = 4.0 * image_error + 2.0 * class_error + 50.0 * missing
        candidate = (score, seed, query, counts)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("Could not construct an in-scene probe split")

    score, seed, query_rows, query_counts = best
    query_ids = sorted(str(row["sample_id"]) for row in query_rows)
    query_set = set(query_ids)
    train_ids = sorted(sample_id for sample_id in row_by_id if sample_id not in query_set)
    train_groups = {str(row_by_id[sample_id]["group_id"]) for sample_id in train_ids}
    query_groups = {str(row_by_id[sample_id]["group_id"]) for sample_id in query_ids}
    if not query_groups <= train_groups:
        raise RuntimeError("Probe query contains a group with no training support")

    processed_root = Path(args.processed_root).resolve()
    train_images = [_find_processed_image(processed_root, sample_id) for sample_id in train_ids]
    query_images = [_find_processed_image(processed_root, sample_id) for sample_id in query_ids]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_list = output_dir / "train.txt"
    val_list = output_dir / "val.txt"
    train_list.write_text("\n".join(map(str, train_images)) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(map(str, query_images)) + "\n", encoding="utf-8")

    data_yaml = {
        "path": str(processed_root),
        "train": str(train_list.resolve()),
        "val": str(val_list.resolve()),
        "channels": 8,
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "channel_names": list(CHANNEL_NAMES),
    }
    yaml_path = output_dir / "aic_multispectral_intrascene_probe.yaml"
    yaml_path.write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    report = {
        "purpose": "leaderboard-like labeled probe; never a final all-data training split",
        "selection_score": score,
        "seed": seed,
        "strict_train_images_already_seen_by_checkpoint": len(strict_split["train"]),
        "additional_support_images": len(train_ids) - len(strict_split["train"]),
        "train_images": len(train_ids),
        "query_images": len(query_ids),
        "query_groups": len(query_groups),
        "all_query_groups_have_train_support": True,
        "train_class_counts": _class_counts([row_by_id[x] for x in train_ids]).tolist(),
        "query_class_counts": query_counts.tolist(),
        "train": train_ids,
        "val": query_ids,
        "sha256": {
            "source_manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source_strict_split": hashlib.sha256(strict_split_path.read_bytes()).hexdigest(),
            "train_list": hashlib.sha256(train_list.read_bytes()).hexdigest(),
            "val_list": hashlib.sha256(val_list.read_bytes()).hexdigest(),
            "data_yaml": hashlib.sha256(yaml_path.read_bytes()).hexdigest(),
        },
    }
    report_path = output_dir / "split.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seed": seed,
                "train_images": len(train_ids),
                "query_images": len(query_ids),
                "query_class_counts": query_counts.tolist(),
                "data": str(yaml_path.resolve()),
                "report": str(report_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
