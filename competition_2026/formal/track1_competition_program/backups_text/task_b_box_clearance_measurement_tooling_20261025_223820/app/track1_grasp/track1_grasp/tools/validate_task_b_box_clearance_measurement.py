#!/usr/bin/env python3
"""Validate a static Task B box-clearance measurement record.

The validator only reads YAML and evidence paths.  It never constructs a ROS
node and never publishes robot or gripper commands.  On success it can emit the
reviewed ``box_clearance`` YAML block for deliberate manual insertion into the
Trial/Preflight/Live configurations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, pi
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from track1_grasp.collision import (
    GEOMETRY_SOURCE_FIELD_MEASURED,
    BoxClearanceConfig,
    BoxObstacleConfig,
    validate_box_clearance_geometry,
)


DEFAULT_MEASUREMENT = Path(
    "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/"
    "config/track1_task_b_box_clearance_measurement_20261025.yaml"
)
_ALLOWED_CATEGORIES = {"lower_corner", "edge_boss", "other"}


@dataclass(frozen=True)
class ValidatedMeasurement:
    source_path: Path
    config: BoxClearanceConfig
    evidence_count: int


def _mapping(
    value: Any,
    field_name: str,
    errors: list[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field_name} must be a mapping")
        return {}
    return value


def _text(
    data: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: list[str],
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_name} must be a non-empty string")
        return ""
    return value.strip()


def _number(
    data: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: list[str],
) -> float | None:
    value = data.get(key)
    if isinstance(value, bool):
        errors.append(f"{field_name} must be a finite number")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field_name} must be a finite number")
        return None
    if not isfinite(number):
        errors.append(f"{field_name} must be a finite number")
        return None
    return number


def _pair(
    data: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: list[str],
) -> tuple[float, float] | None:
    value = data.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        errors.append(f"{field_name} must contain two finite numbers")
        return None
    if len(value) != 2:
        errors.append(f"{field_name} must contain exactly two numbers")
        return None
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            errors.append(f"{field_name}[{index}] must be a finite number")
            return None
        try:
            number = float(item)
        except (TypeError, ValueError):
            errors.append(f"{field_name}[{index}] must be a finite number")
            return None
        if not isfinite(number):
            errors.append(f"{field_name}[{index}] must be a finite number")
            return None
        result.append(number)
    return (result[0], result[1])


def _boolean_true(
    data: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: list[str],
) -> bool:
    value = data.get(key)
    if value is not True:
        errors.append(f"{field_name} must be explicitly true")
        return False
    return True


def _path_list(
    value: Any,
    field_name: str,
    base_dir: Path,
    errors: list[str],
    *,
    minimum_count: int = 1,
) -> tuple[Path, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        errors.append(f"{field_name} must be a sequence of evidence paths")
        return ()
    paths: list[Path] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{field_name}[{index}] must be a non-empty path")
            continue
        path = Path(item.strip()).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if not path.is_file():
            errors.append(f"{field_name}[{index}] is not a readable file: {path}")
            continue
        paths.append(path)
    if len(paths) < minimum_count:
        errors.append(
            f"{field_name} must contain at least {minimum_count} existing file(s)"
        )
    return tuple(paths)


def _validate_evidence(
    value: Any,
    base_dir: Path,
    errors: list[str],
) -> tuple[Path, ...]:
    evidence = _mapping(value, "evidence", errors)
    collected: list[Path] = []
    collected.extend(
        _path_list(
            evidence.get("overview_photos"),
            "evidence.overview_photos",
            base_dir,
            errors,
        )
    )
    corner_photos = _mapping(
        evidence.get("corner_photos"),
        "evidence.corner_photos",
        errors,
    )
    for corner in ("northwest", "northeast", "southwest", "southeast"):
        collected.extend(
            _path_list(
                corner_photos.get(corner),
                f"evidence.corner_photos.{corner}",
                base_dir,
                errors,
            )
        )
    boss_photos = _mapping(
        evidence.get("boss_photos"),
        "evidence.boss_photos",
        errors,
    )
    for boss in ("north", "south"):
        collected.extend(
            _path_list(
                boss_photos.get(boss),
                f"evidence.boss_photos.{boss}",
                base_dir,
                errors,
            )
        )
    collected.extend(
        _path_list(
            evidence.get("gripper_geometry_sources"),
            "evidence.gripper_geometry_sources",
            base_dir,
            errors,
        )
    )
    return tuple(collected)


def _validate_obstacles(
    value: Any,
    errors: list[str],
) -> tuple[BoxObstacleConfig, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        errors.append("obstacles_m must be a sequence")
        return ()
    obstacles: list[BoxObstacleConfig] = []
    names: set[str] = set()
    lower_corner_count = 0
    edge_boss_count = 0
    for index, item in enumerate(value):
        obstacle_data = _mapping(item, f"obstacles_m[{index}]", errors)
        name = _text(
            obstacle_data,
            "name",
            f"obstacles_m[{index}].name",
            errors,
        )
        category = _text(
            obstacle_data,
            "category",
            f"obstacles_m[{index}].category",
            errors,
        )
        if name:
            if name in names:
                errors.append(f"duplicate obstacle name: {name!r}")
            names.add(name)
        if category and category not in _ALLOWED_CATEGORIES:
            errors.append(
                f"obstacles_m[{index}].category must be one of "
                f"{sorted(_ALLOWED_CATEGORIES)}"
            )
        if category == "lower_corner":
            lower_corner_count += 1
        elif category == "edge_boss":
            edge_boss_count += 1
        vertices = obstacle_data.get("vertices_m")
        try:
            obstacles.append(
                BoxObstacleConfig.from_dict(
                    {"name": name or f"obstacle_{index}", "vertices_m": vertices},
                    index=index,
                )
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"obstacles_m[{index}] is invalid: {exc}")
    if lower_corner_count < 8:
        errors.append(
            "obstacles_m must contain at least eight convex lower_corner pieces"
        )
    if edge_boss_count < 2:
        errors.append("obstacles_m must contain at least two edge_boss pieces")
    return tuple(obstacles)


def validate_measurement(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
) -> ValidatedMeasurement:
    """Validate a complete, reviewed field measurement or raise with all errors."""

    errors: list[str] = []
    if not isinstance(payload, Mapping):
        raise ValueError("measurement document root must be a mapping")

    schema_version = payload.get("schema_version")
    if schema_version != 1 or isinstance(schema_version, bool):
        errors.append("schema_version must be 1")
    status = payload.get("status")
    if status != "reviewed":
        errors.append("status must be 'reviewed' only after field review")

    metadata = _mapping(payload.get("metadata"), "metadata", errors)
    _text(metadata, "project", "metadata.project", errors)
    measured_at = _text(metadata, "measured_at", "metadata.measured_at", errors)
    if measured_at:
        try:
            parsed_time = datetime.fromisoformat(measured_at.replace("Z", "+00:00"))
        except ValueError:
            errors.append("metadata.measured_at must be valid ISO-8601")
        else:
            if parsed_time.tzinfo is None:
                errors.append("metadata.measured_at must include a timezone")
    _text(metadata, "measured_by", "metadata.measured_by", errors)
    _text(metadata, "reviewed_by", "metadata.reviewed_by", errors)
    _text(
        metadata,
        "measurement_location",
        "metadata.measurement_location",
        errors,
    )
    _text(
        metadata,
        "controller_base_frame",
        "metadata.controller_base_frame",
        errors,
    )
    _text(metadata, "instrument_id", "metadata.instrument_id", errors)
    _text(metadata, "measurement_method", "metadata.measurement_method", errors)
    _text(metadata, "bin_identifier", "metadata.bin_identifier", errors)
    uncertainty_m = _number(
        metadata,
        "uncertainty_m",
        "metadata.uncertainty_m",
        errors,
    )
    if uncertainty_m is not None and not (0.0 < uncertainty_m <= 0.10):
        errors.append("metadata.uncertainty_m must be in (0, 0.10] m")
    yaw_uncertainty_rad = _number(
        metadata,
        "yaw_uncertainty_rad",
        "metadata.yaw_uncertainty_rad",
        errors,
    )
    if yaw_uncertainty_rad is not None and not (
        0.0 < yaw_uncertainty_rad <= pi / 4.0
    ):
        errors.append("metadata.yaw_uncertainty_rad must be in (0, pi/4] rad")

    box = _mapping(payload.get("box"), "box", errors)
    center_xy_m = _pair(box, "center_xy_m", "box.center_xy_m", errors)
    if center_xy_m is not None and max(abs(value) for value in center_xy_m) > 2.0:
        errors.append("box.center_xy_m is implausibly far from the base origin")
    yaw_rad = _number(box, "yaw_rad", "box.yaw_rad", errors)
    if yaw_rad is not None and abs(yaw_rad) > 2.0 * pi:
        errors.append("box.yaw_rad is implausible; use radians")
    interior_size_m = _pair(
        box,
        "interior_size_m",
        "box.interior_size_m",
        errors,
    )
    if interior_size_m is not None and any(
        value < 0.05 or value > 5.0 for value in interior_size_m
    ):
        errors.append("box.interior_size_m values must be in [0.05, 5.0] m")

    clearances = _mapping(payload.get("clearances"), "clearances", errors)
    wall_clearance_m = _number(
        clearances,
        "wall_m",
        "clearances.wall_m",
        errors,
    )
    if wall_clearance_m is not None and not (0.0 < wall_clearance_m <= 0.50):
        errors.append("clearances.wall_m must be in (0, 0.50] m")
    obstacle_clearance_m = _number(
        clearances,
        "obstacle_m",
        "clearances.obstacle_m",
        errors,
    )
    if obstacle_clearance_m is not None and not (
        0.0 < obstacle_clearance_m <= 0.50
    ):
        errors.append("clearances.obstacle_m must be in (0, 0.50] m")
    if uncertainty_m is not None:
        if (
            wall_clearance_m is not None
            and wall_clearance_m + 1e-12 < uncertainty_m
        ):
            errors.append(
                "clearances.wall_m must be at least metadata.uncertainty_m"
            )
        if (
            obstacle_clearance_m is not None
            and obstacle_clearance_m + 1e-12 < uncertainty_m
        ):
            errors.append(
                "clearances.obstacle_m must be at least metadata.uncertainty_m"
            )

    footprint_data = _mapping(
        payload.get("gripper_footprint"),
        "gripper_footprint",
        errors,
    )
    _text(
        footprint_data,
        "source_description",
        "gripper_footprint.source_description",
        errors,
    )
    raw_footprint = footprint_data.get("flange_xy_m")
    if (
        isinstance(raw_footprint, (str, bytes))
        or not isinstance(raw_footprint, Sequence)
        or len(raw_footprint) < 3
    ):
        errors.append("gripper_footprint.flange_xy_m must contain at least 3 points")
        footprint: tuple[tuple[float, float], ...] = ()
    else:
        footprint = tuple(
            point
            for point in (raw_footprint or ())
            if isinstance(point, Sequence)
            and not isinstance(point, (str, bytes))
            and len(point) == 2
        )
        if len(footprint) != len(raw_footprint):
            errors.append("gripper_footprint.flange_xy_m points must be XY pairs")
    if footprint:
        footprint_array = np.asarray(footprint, dtype=float)
        spans = np.ptp(footprint_array, axis=0)
        if np.any(spans < 0.005) or np.any(spans > 1.0):
            errors.append(
                "gripper footprint spans must each be in [0.005, 1.0] m"
            )

    obstacles = _validate_obstacles(payload.get("obstacles_m"), errors)
    evidence = _validate_evidence(
        payload.get("evidence"),
        source_path.parent,
        errors,
    )

    review = _mapping(payload.get("review"), "review", errors)
    _boolean_true(
        review,
        "measurement_complete",
        "review.measurement_complete",
        errors,
    )
    _boolean_true(
        review,
        "dimensions_rechecked",
        "review.dimensions_rechecked",
        errors,
    )
    _boolean_true(
        review,
        "all_obstacles_inventoried",
        "review.all_obstacles_inventoried",
        errors,
    )
    _boolean_true(
        review,
        "gripper_footprint_reviewed",
        "review.gripper_footprint_reviewed",
        errors,
    )
    _boolean_true(
        review,
        "safety_approved",
        "review.safety_approved",
        errors,
    )
    _text(review, "review_notes", "review.review_notes", errors)

    if errors:
        formatted = "\n".join(f"- {message}" for message in errors)
        raise ValueError(f"measurement is not ready:\n{formatted}")

    assert center_xy_m is not None
    assert yaw_rad is not None
    assert interior_size_m is not None
    assert wall_clearance_m is not None
    assert obstacle_clearance_m is not None
    config = BoxClearanceConfig(
        required=True,
        geometry_validated=True,
        geometry_source=GEOMETRY_SOURCE_FIELD_MEASURED,
        center_xy_m=center_xy_m,
        yaw_rad=yaw_rad,
        interior_size_m=interior_size_m,
        wall_clearance_m=wall_clearance_m,
        obstacle_clearance_m=obstacle_clearance_m,
        gripper_footprint_flange_xy_m=footprint,
        obstacles_m=obstacles,
    )
    try:
        validate_box_clearance_geometry(config)
    except ValueError as exc:
        raise ValueError(f"measurement geometry is invalid: {exc}") from exc
    return ValidatedMeasurement(
        source_path=source_path.resolve(),
        config=config,
        evidence_count=len(evidence),
    )


def box_clearance_dict(config: BoxClearanceConfig) -> dict[str, Any]:
    """Return the reviewed YAML block; insertion into runtime configs is manual."""

    return {
        "required": True,
        "geometry_validated": True,
        "geometry_source": GEOMETRY_SOURCE_FIELD_MEASURED,
        "center_xy_m": [float(value) for value in config.center_xy_m or ()],
        "yaw_rad": float(config.yaw_rad),
        "interior_size_m": [
            float(value) for value in config.interior_size_m or ()
        ],
        "wall_clearance_m": float(config.wall_clearance_m),
        "obstacle_clearance_m": float(config.obstacle_clearance_m),
        "gripper_footprint_flange_xy_m": [
            [float(point[0]), float(point[1])]
            for point in config.gripper_footprint_flange_xy_m
        ],
        "obstacles_m": [
            {
                "name": obstacle.name,
                "vertices_m": [
                    [float(point[0]), float(point[1])]
                    for point in obstacle.vertices_m
                ],
            }
            for obstacle in config.obstacles_m
        ],
    }


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read measurement file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"measurement file {path} must contain a mapping root")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Task B box-clearance field measurement. "
            "This tool never moves the robot or gripper."
        )
    )
    parser.add_argument(
        "measurement_file",
        nargs="?",
        type=Path,
        default=DEFAULT_MEASUREMENT,
        help=f"measurement YAML (default: {DEFAULT_MEASUREMENT})",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="validate and print a summary (default)",
    )
    action.add_argument(
        "--emit-box-clearance",
        action="store_true",
        help="validate and print the box_clearance YAML block",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = _load_yaml(args.measurement_file)
        validated = validate_measurement(
            payload,
            source_path=args.measurement_file,
        )
    except ValueError as exc:
        print(f"MEASUREMENT REJECTED\n{exc}")
        return 2

    if args.emit_box_clearance:
        print(
            yaml.safe_dump(
                {"box_clearance": box_clearance_dict(validated.config)},
                sort_keys=False,
                allow_unicode=True,
            ).rstrip()
        )
        return 0

    print("MEASUREMENT VALID")
    print(f"source={validated.source_path}")
    print(f"evidence_files={validated.evidence_count}")
    print(f"obstacles={len(validated.config.obstacles_m)}")
    print(f"footprint_points={len(validated.config.gripper_footprint_flange_xy_m)}")
    print("No robot or gripper command was created or sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
