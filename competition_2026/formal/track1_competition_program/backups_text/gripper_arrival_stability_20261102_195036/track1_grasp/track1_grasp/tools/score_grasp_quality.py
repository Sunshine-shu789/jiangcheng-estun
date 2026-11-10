#!/usr/bin/env python3
"""Offline ranking of parallel-jaw candidates with GraspNet-style checks.

Input is a JSON document or NPZ archive containing:

* ``candidates``: an N x 17 GraspNet array, or mappings/arrays with the same
  score, width, height, depth, rotation(9), translation(3), object_id layout;
* ``target_points``: an N x 3 target point cloud in the same point-cloud
  frame as the candidate translations/rotations;
* ``obstacle_points``: an optional N x 3 obstacle/support point cloud in that
  same frame;
* ``candidate_tolerances``: optional per-candidate tolerance in metres;
* ``candidate_ids``: optional stable identifiers.

This tool is read-only.  It never imports ROS, opens the gripper, or commands
the arm.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from track1_grasp.grasp_quality import (
    GraspCandidate,
    GraspQualityResult,
    GraspScoringConfig,
    load_scoring_config,
    score_candidates_with_nms,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "grasp_quality_ms42dc_estimated.yaml"
)
_SCENE_KEYS = (
    "candidates",
    "target_points",
    "obstacle_points",
    "candidate_tolerances",
    "candidate_ids",
)


def _point_array(value: Any, name: str, *, required: bool) -> np.ndarray:
    if value is None:
        if required:
            raise ValueError(f"input is missing {name}")
        return np.empty((0, 3), dtype=float)
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        if required:
            raise ValueError(f"{name} must not be empty")
        return np.empty((0, 3), dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape N x 3")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=float)


def _mapping_candidate(
    item: Mapping[str, Any],
    *,
    fallback_id: str,
    fallback_tolerance_m: float,
) -> GraspCandidate:
    if "graspnet_array" in item:
        grasp = np.asarray(item["graspnet_array"], dtype=float).reshape(-1)
        return GraspCandidate.from_graspnet_array(
            grasp,
            candidate_id=str(item.get("candidate_id", fallback_id)),
            tolerance_m=float(item.get("tolerance_m", fallback_tolerance_m)),
            metadata=item.get("metadata") or {},
        )

    translation = item.get("translation_m", item.get("translation"))
    rotation = item.get("rotation", item.get("rotation_matrix"))
    if translation is None or rotation is None:
        raise ValueError(
            f"candidate {fallback_id} must contain translation and rotation"
        )
    return GraspCandidate(
        candidate_id=str(item.get("candidate_id", fallback_id)),
        translation_m=tuple(float(value) for value in np.asarray(translation).reshape(-1)),
        rotation=tuple(float(value) for value in np.asarray(rotation).reshape(-1)),
        width_m=float(item.get("width_m", item.get("width"))),
        height_m=float(item.get("height_m", item.get("height"))),
        depth_m=float(item.get("depth_m", item.get("depth"))),
        tolerance_m=float(item.get("tolerance_m", fallback_tolerance_m)),
        network_score=float(item.get("network_score", item.get("score", 1.0))),
        object_id=int(item.get("object_id", -1)),
        metadata=item.get("metadata") or {},
    )


def parse_candidates(
    raw_candidates: Any,
    *,
    tolerance_values: Any = None,
    candidate_ids: Any = None,
) -> tuple[GraspCandidate, ...]:
    if raw_candidates is None:
        raise ValueError("input is missing candidates")
    raw = np.asarray(raw_candidates)
    if raw.size == 0:
        raise ValueError("candidates must not be empty")

    ids = (
        [str(value) for value in np.asarray(candidate_ids).reshape(-1)]
        if candidate_ids is not None
        else []
    )
    if ids and len(ids) != len(raw):
        raise ValueError("candidate_ids length must match candidates")

    if tolerance_values is None:
        tolerances = np.zeros(len(raw), dtype=float)
    else:
        tolerance_array = np.asarray(tolerance_values, dtype=float).reshape(-1)
        if tolerance_array.size == 1:
            tolerances = np.full(len(raw), float(tolerance_array[0]), dtype=float)
        elif tolerance_array.size == len(raw):
            tolerances = tolerance_array
        else:
            raise ValueError("candidate_tolerances length must be 1 or match candidates")

    candidates: list[GraspCandidate] = []
    for index, item in enumerate(raw):
        fallback_id = ids[index] if ids else f"grasp_{index:04d}"
        tolerance = float(tolerances[index])
        if isinstance(item, Mapping):
            candidate = _mapping_candidate(
                item,
                fallback_id=fallback_id,
                fallback_tolerance_m=tolerance,
            )
        else:
            values = np.asarray(item, dtype=float).reshape(-1)
            if values.size != 17:
                raise ValueError(
                    f"candidate {fallback_id} must contain 17 GraspNet floats"
                )
            candidate = GraspCandidate.from_graspnet_array(
                values,
                candidate_id=fallback_id,
                tolerance_m=tolerance,
            )
        candidates.append(candidate)
    return tuple(candidates)


def _load_scene(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("JSON input must contain one object")
        return dict(payload)
    raise ValueError("input must be a .json or .npz file")


def _ranked(results: Sequence[GraspQualityResult]) -> list[dict[str, Any]]:
    ordered = sorted(
        results,
        key=lambda item: (
            item.feasible,
            item.quality,
            item.candidate.network_score,
            item.candidate.candidate_id,
        ),
        reverse=True,
    )
    return [
        {"rank": rank, **result.to_dict()}
        for rank, result in enumerate(ordered, start=1)
    ]


def build_report(scene: Mapping[str, Any], config: GraspScoringConfig) -> dict[str, Any]:
    candidates = parse_candidates(
        scene.get("candidates"),
        tolerance_values=scene.get("candidate_tolerances"),
        candidate_ids=scene.get("candidate_ids"),
    )
    target_points = _point_array(
        scene.get("target_points"),
        "target_points",
        required=True,
    )
    obstacle_points = _point_array(
        scene.get("obstacle_points"),
        "obstacle_points",
        required=False,
    )

    all_results, kept, suppressed = score_candidates_with_nms(
        candidates,
        target_points,
        obstacle_points,
        config,
    )
    feasible_count = sum(item.feasible for item in all_results)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_read_only",
        "advisory": (
            "Ranking score is not a calibrated probability of grasp success. "
            "This report cannot authorize live robot motion."
        ),
        "config": config.to_dict(),
        "input_summary": {
            "candidate_count": len(candidates),
            "target_points": len(target_points),
            "obstacle_points": len(obstacle_points),
        },
        "counts": {
            "candidates": len(all_results),
            "feasible": feasible_count,
            "infeasible": len(all_results) - feasible_count,
            "selected_after_nms": len(kept),
            "suppressed_by_nms": len(suppressed),
        },
        "selected": [
            {"rank": rank, **result.to_dict()}
            for rank, result in enumerate(kept, start=1)
        ],
        "suppressed": [item.to_dict() for item in suppressed],
        "results": _ranked(all_results),
    }


def synthetic_demo_scene() -> dict[str, Any]:
    """Build a reproducible scene containing good and deliberately bad grasps."""

    center = np.asarray((0.025, 0.0, 0.0), dtype=float)
    half = np.asarray((0.020, 0.030, 0.030), dtype=float)
    axes = [
        np.arange(center[index] - half[index], center[index] + half[index] + 1e-9, 0.0025)
        for index in range(3)
    ]
    points: list[np.ndarray] = []
    for face_axis in range(3):
        first_axis, second_axis = [axis for axis in range(3) if axis != face_axis]
        for first in axes[first_axis]:
            for second in axes[second_axis]:
                for sign in (-1.0, 1.0):
                    point = np.zeros(3, dtype=float)
                    point[face_axis] = center[face_axis] + sign * half[face_axis]
                    point[first_axis] = first
                    point[second_axis] = second
                    points.append(point)
    target_points = np.asarray(points, dtype=float)

    identity = np.eye(3, dtype=float).reshape(-1)
    specs = (
        ("demo_good", 0.95, 0.064, (0.000, 0.000, 0.0), 0.020),
        ("demo_duplicate", 0.90, 0.066, (0.004, 0.000, 0.0), 0.015),
        ("demo_collision", 0.85, 0.0678, (0.000, -0.004, 0.0), 0.012),
        ("demo_narrow", 0.80, 0.055, (0.000, 0.000, 0.0), 0.010),
        ("demo_empty", 0.75, 0.064, (0.000, 0.120, 0.0), 0.010),
    )
    candidates = []
    for _, score, width, translation, _ in specs:
        candidates.append(
            np.concatenate(
                (
                    np.asarray((score, width, 0.060, 0.050), dtype=float),
                    identity,
                    np.asarray(translation, dtype=float),
                    np.asarray((-1.0,), dtype=float),
                )
            )
        )
    return {
        "candidates": np.asarray(candidates, dtype=float),
        "candidate_ids": np.asarray([item[0] for item in specs]),
        "candidate_tolerances": np.asarray([item[4] for item in specs], dtype=float),
        "target_points": target_points,
        # One scene point outside the good grasp but inside the shifted
        # collision demo's left finger.
        "obstacle_points": np.asarray(((0.025, -0.0488, 0.0),), dtype=float),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="score_grasp_quality",
        description=(
            "Offline GraspNet-style parallel-jaw scoring and 3-D NMS. "
            "Read-only: never moves the arm or gripper."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"scoring YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=".npz or .json scene file; omit when using --demo",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write report JSON here; otherwise print to stdout",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="score a built-in synthetic cube scene",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.demo == (args.input is not None):
        raise SystemExit("choose exactly one of --input or --demo")

    try:
        config = load_scoring_config(args.config)
        scene = synthetic_demo_scene() if args.demo else _load_scene(args.input)
        report = build_report(scene, config)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(report, indent=args.indent, ensure_ascii=False)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
