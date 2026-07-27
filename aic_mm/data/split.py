"""Leakage-resistant, group-aware train/validation splitting."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from aic_mm.constants import NUM_CLASSES
from aic_mm.io_utils import read_jsonl, write_json


class _DisjointSet:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _merge_phash_groups(rows: list[dict[str, Any]], max_distance: int) -> dict[str, str]:
    """Merge filename groups when train-only perceptual hashes are nearly identical."""
    group_ids = sorted({str(row["group_id"]) for row in rows})
    disjoint = _DisjointSet(group_ids)
    hashed = [(row, int(row["visible_phash"], 16)) for row in rows if row.get("visible_phash")]
    for index, (left, left_hash) in enumerate(hashed):
        for right, right_hash in hashed[index + 1 :]:
            if left["image_suffix"] != right["image_suffix"]:
                continue
            if (left_hash ^ right_hash).bit_count() <= max_distance:
                disjoint.union(str(left["group_id"]), str(right["group_id"]))
    return {group: disjoint.find(group) for group in group_ids}


def _aggregate_groups(rows: list[dict[str, Any]], aliases: dict[str, str]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        group_id = aliases.get(str(row["group_id"]), str(row["group_id"]))
        aggregate = groups.setdefault(
            group_id,
            {
                "sample_ids": [],
                "class_counts": np.zeros(NUM_CLASSES, dtype=np.float64),
                "domains": defaultdict(int),
            },
        )
        aggregate["sample_ids"].append(str(row["sample_id"]))
        aggregate["class_counts"] += np.asarray(row["class_counts"], dtype=np.float64)
        domain = f"{row['image_suffix']}:{row['width']}x{row['height']}"
        aggregate["domains"][domain] += 1
    return groups


def _selection_score(
    selected: set[str],
    groups: dict[str, dict[str, Any]],
    total_images: int,
    total_classes: np.ndarray,
    total_domains: dict[str, int],
    val_fraction: float,
) -> float:
    val_images = sum(len(groups[group]["sample_ids"]) for group in selected)
    val_classes = sum((groups[group]["class_counts"] for group in selected), np.zeros(NUM_CLASSES))
    val_domains: defaultdict[str, int] = defaultdict(int)
    for group in selected:
        for domain, count in groups[group]["domains"].items():
            val_domains[domain] += count

    target_images = total_images * val_fraction
    target_classes = total_classes * val_fraction
    image_error = abs(val_images - target_images) / max(target_images, 1.0)
    class_error = np.mean(np.abs(val_classes - target_classes) / np.maximum(target_classes, 1.0))
    domain_error = np.mean(
        [
            abs(val_domains[domain] - count * val_fraction) / max(count * val_fraction, 1.0)
            for domain, count in total_domains.items()
        ]
    )
    missing_penalty = float(np.count_nonzero((val_classes == 0) & (total_classes > 0))) * 25.0
    train_missing_penalty = float(np.count_nonzero((total_classes - val_classes <= 0) & (total_classes > 0))) * 25.0
    return 4.0 * image_error + 2.0 * class_error + domain_error + missing_penalty + train_missing_penalty


def _milp_selection(
    groups: dict[str, dict[str, Any]],
    total_images: int,
    total_classes: np.ndarray,
    total_domains: dict[str, int],
    val_fraction: float,
) -> set[str]:
    """Solve group selection with absolute-deviation slack variables."""
    group_ids = sorted(groups)
    domain_ids = sorted(total_domains)
    features = ["images", *[f"class_{i}" for i in range(NUM_CLASSES)], *domain_ids]
    feature_totals = np.asarray(
        [total_images, *total_classes.tolist(), *[total_domains[name] for name in domain_ids]],
        dtype=np.float64,
    )
    feature_matrix = np.zeros((len(features), len(group_ids)), dtype=np.float64)
    for group_index, group_id in enumerate(group_ids):
        group = groups[group_id]
        feature_matrix[0, group_index] = len(group["sample_ids"])
        feature_matrix[1 : 1 + NUM_CLASSES, group_index] = group["class_counts"]
        for domain_index, domain in enumerate(domain_ids):
            feature_matrix[1 + NUM_CLASSES + domain_index, group_index] = group["domains"].get(
                domain, 0
            )

    feature_count, group_count = feature_matrix.shape
    variable_count = group_count + feature_count
    objective = np.zeros(variable_count, dtype=np.float64)
    # Penalize relative errors. Class balance is deliberately strongest because
    # rare-class AP otherwise becomes too noisy to guide model selection.
    feature_weights = np.asarray(
        [5.0, *([4.0] * NUM_CLASSES), *([1.5] * len(domain_ids))], dtype=np.float64
    )
    objective[group_count:] = feature_weights / np.maximum(feature_totals * val_fraction, 1.0)

    # |A*x - target| <= slack.
    upper_matrix = np.zeros((feature_count * 2, variable_count), dtype=np.float64)
    targets = feature_totals * val_fraction
    for index in range(feature_count):
        upper_matrix[index * 2, :group_count] = feature_matrix[index]
        upper_matrix[index * 2, group_count + index] = -1.0
        upper_matrix[index * 2 + 1, :group_count] = -feature_matrix[index]
        upper_matrix[index * 2 + 1, group_count + index] = -1.0
    upper_bounds = np.empty(feature_count * 2, dtype=np.float64)
    upper_bounds[0::2] = targets
    upper_bounds[1::2] = -targets
    constraints: list[LinearConstraint] = [
        LinearConstraint(upper_matrix, -np.inf, upper_bounds)
    ]

    # Hard guards prevent a low-object rare class or data domain from collapsing.
    guard_matrix = np.zeros((1 + NUM_CLASSES + len(domain_ids), variable_count), dtype=np.float64)
    guard_matrix[:, :group_count] = feature_matrix
    guard_lower = np.zeros(guard_matrix.shape[0], dtype=np.float64)
    guard_upper = np.zeros(guard_matrix.shape[0], dtype=np.float64)
    guard_lower[0], guard_upper[0] = total_images * 0.18, total_images * 0.22
    for class_id, total in enumerate(total_classes.astype(int)):
        minimum = max(2, int(math.floor(total * 0.10)))
        guard_lower[1 + class_id] = minimum
        guard_upper[1 + class_id] = total - minimum
    for domain_index, domain in enumerate(domain_ids):
        total = total_domains[domain]
        guard_lower[1 + NUM_CLASSES + domain_index] = max(1, total * 0.10)
        guard_upper[1 + NUM_CLASSES + domain_index] = total * 0.40
    constraints.append(LinearConstraint(guard_matrix, guard_lower, guard_upper))

    lower = np.zeros(variable_count, dtype=np.float64)
    upper = np.full(variable_count, np.inf, dtype=np.float64)
    upper[:group_count] = 1.0
    integrality = np.zeros(variable_count, dtype=np.int32)
    integrality[:group_count] = 1
    solution = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": 60.0, "mip_rel_gap": 0.001},
    )
    if solution.x is None or solution.status not in {0, 1}:
        raise RuntimeError(f"Group split integer optimization failed: {solution.message}")
    return {
        group_id
        for group_id, selected in zip(group_ids, solution.x[:group_count])
        if selected >= 0.5
    }


def create_group_split(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 2026,
    trials: int = 4000,
    phash_distance: int = 4,
) -> dict[str, Any]:
    """Create a deterministic group-aware split with multi-label/domain balancing."""
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be between 0.05 and 0.5")
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_path}")

    aliases = _merge_phash_groups(rows, phash_distance)
    groups = _aggregate_groups(rows, aliases)
    total_classes = sum(
        (np.asarray(row["class_counts"], dtype=np.float64) for row in rows),
        np.zeros(NUM_CLASSES, dtype=np.float64),
    )
    total_domains: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        total_domains[f"{row['image_suffix']}:{row['width']}x{row['height']}"] += 1

    best_selected = _milp_selection(
        groups, len(rows), total_classes, dict(total_domains), val_fraction
    )
    best_score = _selection_score(
        best_selected,
        groups,
        len(rows),
        total_classes,
        dict(total_domains),
        val_fraction,
    )
    val_ids = sorted(
        sample_id for group in best_selected for sample_id in groups[group]["sample_ids"]
    )
    val_set = set(val_ids)
    train_ids = sorted(str(row["sample_id"]) for row in rows if str(row["sample_id"]) not in val_set)
    if not train_ids or not val_ids:
        raise RuntimeError("Split unexpectedly produced an empty train or validation subset")

    row_by_id = {str(row["sample_id"]): row for row in rows}

    def subset_stats(sample_ids: list[str]) -> dict[str, Any]:
        class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        domains: defaultdict[str, int] = defaultdict(int)
        subset_groups: set[str] = set()
        for sample_id in sample_ids:
            row = row_by_id[sample_id]
            class_counts += np.asarray(row["class_counts"], dtype=np.int64)
            domains[f"{row['image_suffix']}:{row['width']}x{row['height']}"] += 1
            subset_groups.add(aliases.get(str(row["group_id"]), str(row["group_id"])))
        return {
            "images": len(sample_ids),
            "groups": len(subset_groups),
            "class_counts": class_counts.tolist(),
            "domains": dict(sorted(domains.items())),
        }

    train_groups = {
        aliases.get(str(row_by_id[sample_id]["group_id"]), str(row_by_id[sample_id]["group_id"]))
        for sample_id in train_ids
    }
    val_groups = {
        aliases.get(str(row_by_id[sample_id]["group_id"]), str(row_by_id[sample_id]["group_id"]))
        for sample_id in val_ids
    }
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"Group leakage detected: {sorted(overlap)[:20]}")

    result = {
        "seed": seed,
        "val_fraction_requested": val_fraction,
        "solver": "scipy.optimize.milp (HiGHS)",
        "legacy_trials_argument": trials,
        "score": best_score,
        "phash_distance": phash_distance,
        "train": train_ids,
        "val": val_ids,
        "stats": {
            "train": subset_stats(train_ids),
            "val": subset_stats(val_ids),
        },
    }
    write_json(output_path, result)
    return result
