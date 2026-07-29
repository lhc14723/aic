#!/usr/bin/env python3
"""Measure automatic label transfer on train-only, in-scene holdouts.

This is an analysis tool. It never reads test labels (none exist), never edits
official data, and never writes a competition submission.  It deterministically
holds out roughly one third of the labeled training images, retrieves the most
similar remaining support image, copies its boxes, and evaluates those copied
boxes against the held-out labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aic_mm.evaluation.metric import evaluate_directories


def _stable_partition(sample_id: str, salt: int) -> int:
    digest = hashlib.sha256(f"{salt}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 3


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _split_rows(
    rows: list[dict[str, Any]], salt: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out about one third while keeping support in every non-singleton group."""
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)

    support: list[dict[str, Any]] = []
    query: list[dict[str, Any]] = []
    for group_rows in by_group.values():
        ordered = sorted(group_rows, key=lambda row: str(row["sample_id"]))
        if len(ordered) == 1:
            support.extend(ordered)
            continue
        group_query = [row for row in ordered if _stable_partition(str(row["sample_id"]), salt) == 0]
        group_support = [row for row in ordered if row not in group_query]
        if not group_query:
            group_query = [ordered[_stable_partition(str(ordered[0]["group_id"]), salt) % len(ordered)]]
            group_support = [row for row in ordered if row not in group_query]
        if not group_support:
            group_support = [group_query.pop()]
        support.extend(group_support)
        query.extend(group_query)
    return support, query


def _thumbnail(path: str | Path, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize(size), dtype=np.int16)


def _label_lines(path: str | Path, confidence: float) -> list[str]:
    output: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}: expected five label fields")
        output.append(" ".join((*fields, f"{confidence:.8f}")))
    return output


def _domain(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["image_suffix"]), int(row["width"]), int(row["height"])


def _select_neighbor(
    query: dict[str, Any],
    support: list[dict[str, Any]],
    support_hashes: np.ndarray,
    support_thumbnails: list[np.ndarray],
    *,
    prefer_group: bool,
) -> tuple[int, int, float, bool]:
    eligible = np.asarray(
        [
            index
            for index, row in enumerate(support)
            if _domain(row) == _domain(query)
            and (
                not prefer_group
                or str(row["group_id"]) == str(query["group_id"])
            )
        ],
        dtype=np.int64,
    )
    if not eligible.size:
        eligible = np.asarray(
            [index for index, row in enumerate(support) if _domain(row) == _domain(query)],
            dtype=np.int64,
        )
    if not eligible.size:
        raise RuntimeError(f"No support images in domain {_domain(query)}")

    query_hash = int(str(query["visible_phash"]), 16)
    distances = np.fromiter(
        ((query_hash ^ int(support_hashes[index])).bit_count() for index in eligible),
        dtype=np.int16,
        count=len(eligible),
    )
    best_distance = int(distances.min())
    candidates = eligible[distances <= best_distance + 2]
    query_thumb = _thumbnail(query["visible"])
    maes = np.asarray(
        [
            float(np.abs(query_thumb - support_thumbnails[index]).mean())
            for index in candidates
        ],
        dtype=np.float64,
    )
    best_local = int(np.argmin(maes))
    selected_index = int(candidates[best_local])
    return (
        selected_index,
        best_distance,
        float(maes[best_local]),
        str(support[selected_index]["group_id"]) == str(query["group_id"]),
    )


def _confidence(phash_distance: int, thumbnail_mae: float) -> float:
    # Confidence is only a deterministic similarity ranking for AP evaluation.
    value = math.exp(-0.18 * phash_distance - 0.025 * thumbnail_mae)
    return min(0.999999, max(0.001, value))


def _evaluate_one(
    rows: list[dict[str, Any]],
    *,
    salt: int,
    prefer_group: bool,
    thresholds: list[int],
) -> dict[str, Any]:
    support, query = _split_rows(rows, salt)
    support_hashes = np.asarray(
        [int(str(row["visible_phash"]), 16) for row in support], dtype=np.uint64
    )
    support_thumbnails = [_thumbnail(row["visible"]) for row in support]

    retrievals: list[dict[str, Any]] = []
    for index, row in enumerate(query, start=1):
        neighbor_index, distance, mae, same_group = _select_neighbor(
            row,
            support,
            support_hashes,
            support_thumbnails,
            prefer_group=prefer_group,
        )
        neighbor = support[neighbor_index]
        retrievals.append(
            {
                "query": row,
                "neighbor": neighbor,
                "phash_distance": distance,
                "thumbnail_mae": mae,
                "same_group": same_group,
                "confidence": _confidence(distance, mae),
            }
        )
        if index % 100 == 0:
            print(
                f"salt={salt} prefer_group={prefer_group} retrieved={index}/{len(query)}",
                flush=True,
            )

    result: dict[str, Any] = {
        "salt": salt,
        "prefer_group": prefer_group,
        "support_images": len(support),
        "query_images": len(query),
        "same_group_retrieval_rate": float(np.mean([item["same_group"] for item in retrievals])),
        "phash_distance_counts": dict(
            sorted(Counter(item["phash_distance"] for item in retrievals).items())
        ),
        "thresholds": {},
    }
    with tempfile.TemporaryDirectory(prefix="aic_retrieval_") as temporary:
        root = Path(temporary)
        gt_dir = root / "ground_truth"
        pred_dir = root / "predictions"
        gt_dir.mkdir()
        pred_dir.mkdir()
        for item in retrievals:
            query_row = item["query"]
            sample_id = str(query_row["sample_id"])
            (gt_dir / f"{sample_id}.txt").symlink_to(Path(query_row["label"]).resolve())

        sample_ids = [str(item["query"]["sample_id"]) for item in retrievals]
        for threshold in thresholds:
            for item in retrievals:
                sample_id = str(item["query"]["sample_id"])
                destination = pred_dir / f"{sample_id}.txt"
                lines = (
                    _label_lines(item["neighbor"]["label"], item["confidence"])
                    if item["phash_distance"] <= threshold
                    else []
                )
                destination.write_text(
                    "\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8",
                )
            metrics = evaluate_directories(gt_dir, pred_dir, sample_ids=sample_ids)
            result["thresholds"][str(threshold)] = {
                "covered_queries": sum(
                    item["phash_distance"] <= threshold for item in retrievals
                ),
                "coverage": float(
                    np.mean([item["phash_distance"] <= threshold for item in retrievals])
                ),
                "map50": metrics["map50"],
                "map50_95": metrics["map50_95"],
                "per_class_map50_95": {
                    name: values["map50_95"]
                    for name, values in metrics["per_class"].items()
                },
            }
    result["examples"] = [
        {
            "query": str(item["query"]["sample_id"]),
            "neighbor": str(item["neighbor"]["sample_id"]),
            "phash_distance": item["phash_distance"],
            "thumbnail_mae": item["thumbnail_mae"],
            "same_group": item["same_group"],
        }
        for item in sorted(
            retrievals,
            key=lambda item: (item["phash_distance"], item["thumbnail_mae"]),
        )[:30]
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/train_manifest.jsonl")
    parser.add_argument("--output", default="outputs/intrascene_retrieval_analysis.json")
    parser.add_argument("--salts", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[0, 2, 4, 6, 8, 10, 12, 16, 64],
    )
    args = parser.parse_args()

    rows = _read_manifest(Path(args.manifest))
    if not rows or any("visible_phash" not in row for row in rows):
        raise ValueError("Manifest must contain labeled training rows with visible_phash")

    runs = [
        _evaluate_one(
            rows,
            salt=salt,
            prefer_group=prefer_group,
            thresholds=args.thresholds,
        )
        for salt in args.salts
        for prefer_group in (False, True)
    ]
    summary: dict[str, Any] = {"runs": runs, "aggregate": {}}
    for prefer_group in (False, True):
        selected = [run for run in runs if run["prefer_group"] == prefer_group]
        aggregate: dict[str, Any] = {}
        for threshold in args.thresholds:
            values = [run["thresholds"][str(threshold)]["map50_95"] for run in selected]
            aggregate[str(threshold)] = {
                "map50_95_mean": float(np.mean(values)),
                "map50_95_min": float(np.min(values)),
                "map50_95_max": float(np.max(values)),
            }
        summary["aggregate"]["group_preferred" if prefer_group else "global"] = aggregate

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"Detailed report: {output.resolve()}")


if __name__ == "__main__":
    main()
