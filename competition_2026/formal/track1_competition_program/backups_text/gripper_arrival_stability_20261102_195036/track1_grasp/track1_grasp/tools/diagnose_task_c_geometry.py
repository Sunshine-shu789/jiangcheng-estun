#!/usr/bin/env python3
"""Read-only Task C RGB-D/PCA geometry diagnosis.

This tool is deliberately separated from ``run_task_c``.  It loads the formal
Task C configuration and vision source, but it never constructs ``PickPipeline``,
``PosePlanner``, a robot-motion backend, or a gripper backend.  It therefore has
no code path to publish ``RobotMove``, open/close the gripper, or connect to
``RobotInfo``.

The camera pose used for metric diagnostics comes from the audited fixed Task C
observation joint file and the calibrated flange-camera extrinsics:

    T_base_camera = controller_fk(observation_joints) @ T_flange_camera

The current physical arm is assumed to be at that observation posture.  This is
an explicit read-only assumption; it is not verified against robot feedback by
this tool and is not motion authorization.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml

from ..config import (
    TaskCDShortEdgeGraspConfig,
    Track1Config,
    VisionConfig,
    load_config,
    load_joint_pose_file,
)
from ..detection_stabilizer import box_iou
from ..domain import Detection, Frame, TaskMode
from ..kinematics import controller_fk, matrix_to_rpy, rpy_to_matrix
from ..planner import load_extrinsics
from ..vision import build_vision
from ..vision_runtime import render_preview


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "track1_task_c_v11_official_20261102.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs"
DEFAULT_FRAMES = 15
DEFAULT_MIN_CLUSTER_FRAMES = 5
DEFAULT_SPATIAL_MATCH_RADIUS_M = 0.025
DEFAULT_BBOX_MATCH_IOU = 0.65


class DiagnosisError(RuntimeError):
    """Raised when read-only diagnosis preconditions are not satisfied."""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists():
        raise DiagnosisError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _unique_default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = DEFAULT_OUTPUT_ROOT / f"task_c_geometry_diagnostic_{stamp}"
    candidate = base
    index = 1
    while candidate.exists():
        candidate = Path(f"{base}_{index:02d}")
        index += 1
    return candidate


def _track_id(detection: Detection) -> str:
    explicit = str(detection.instance_id or "").strip()
    if explicit:
        return explicit
    raw = detection.attributes.get("track_id")
    return "" if raw is None else str(raw)


def _finite_xyz(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    return tuple(float(item) for item in array)


def _transform_point(transform: np.ndarray, point: Sequence[float]) -> np.ndarray:
    value = np.asarray(point, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise DiagnosisError("camera point must be a finite 3-vector")
    return (transform @ np.concatenate((value, np.ones(1))))[:3]


def _normalise_xy(vector: Sequence[float]) -> np.ndarray | None:
    value = np.asarray(vector, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return None
    return value / norm


def _undirected_angle(angle: float) -> float:
    return float((float(angle) + math.pi * 0.5) % math.pi - math.pi * 0.5)


def _circular_axis_mean(angles: Iterable[float]) -> float | None:
    values = np.asarray(
        [float(value) for value in angles if math.isfinite(float(value))],
        dtype=float,
    )
    if values.size == 0:
        return None
    doubled = 2.0 * values
    mean = 0.5 * math.atan2(
        float(np.mean(np.sin(doubled))),
        float(np.mean(np.cos(doubled))),
    )
    return _undirected_angle(mean)


def _median(values: Iterable[Any]) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.median(clean)) if clean else None


def _median_triplet(values: Iterable[Any]) -> tuple[float, float, float] | None:
    clean: list[tuple[float, float, float]] = []
    for value in values:
        xyz = _finite_xyz(value)
        if xyz is not None:
            clean.append(xyz)
    if not clean:
        return None
    array = np.asarray(clean, dtype=float)
    median = np.median(array, axis=0)
    return tuple(float(item) for item in median)


def _distance(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    first = _finite_xyz(a)
    second = _finite_xyz(b)
    if first is None or second is None:
        return None
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


def _same_spatial_target(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    radius_m: float,
    min_iou: float,
) -> bool:
    if str(left.get("label", "")) != str(right.get("label", "")):
        return False
    for key in (
        "object_center_camera_xyz_m",
        "object_center_base_xyz_m",
        "detection_anchor_base_xyz_m",
    ):
        distance = _distance(left.get(key), right.get(key))
        if distance is not None and distance <= radius_m:
            return True
    left_bbox = left.get("bbox_xyxy_px")
    right_bbox = right.get("bbox_xyxy_px")
    if left_bbox is None or right_bbox is None:
        return False
    try:
        return box_iou(
            tuple(float(value) for value in left_bbox),
            tuple(float(value) for value in right_bbox),
        ) >= min_iou
    except (TypeError, ValueError):
        return False


def assert_read_only_task_c(config: Track1Config) -> None:
    """Fail closed unless this is the supported read-only Task C geometry path."""

    if config.task.mode is not TaskMode.C:
        raise DiagnosisError(
            f"read-only Task C diagnosis requires task.mode=C, got "
            f"{config.task.mode.value}"
        )
    if not config.task_cd.enabled:
        raise DiagnosisError("Task C diagnosis requires task_cd.enabled=true")
    short_edge = config.task_cd.short_edge_grasp
    if short_edge is None or not short_edge.enabled:
        raise DiagnosisError(
            "Task C diagnosis requires task_cd.short_edge_grasp.enabled=true"
        )
    if short_edge.geometry_policy.enabled:
        raise DiagnosisError(
            "geometry_policy is enabled; use the geometry-policy acceptance path "
            "instead of this legacy short-edge diagnosis"
        )
    if not config.task_cd.observation_pose_file:
        raise DiagnosisError("task_cd.observation_pose_file is missing")
    if not config.pose.extrinsics_file:
        raise DiagnosisError("pose.extrinsics_file is missing")
    if config.pose.camera_frame != "flange":
        raise DiagnosisError(
            "Task C diagnosis requires pose.camera_frame=flange"
        )
    if config.pose.use_detection_base_xyz:
        raise DiagnosisError(
            "Task C diagnosis requires pose.use_detection_base_xyz=false so the "
            "audited camera transform is the geometry source"
        )
    if config.pose.fallback_camera_to_base:
        raise DiagnosisError(
            "Task C diagnosis refuses pose.fallback_camera_to_base=true"
        )


def _load_observation_geometry(
    config: Track1Config,
) -> tuple[dict[str, Any], np.ndarray]:
    pose_path = Path(config.task_cd.observation_pose_file)
    joints = load_joint_pose_file(
        pose_path,
        expected_role="task_cd_observation",
    )
    base_flange = np.asarray(controller_fk(joints), dtype=float)
    flange_camera = np.asarray(load_extrinsics(config.pose.extrinsics_file), dtype=float)
    if base_flange.shape != (4, 4) or not np.all(np.isfinite(base_flange)):
        raise DiagnosisError("controller_fk did not return a finite 4x4 transform")
    if (
        flange_camera.shape != (4, 4)
        or not np.all(np.isfinite(flange_camera))
        or abs(float(np.linalg.det(flange_camera[:3, :3]))) <= 1e-12
    ):
        raise DiagnosisError("flange-camera extrinsics are invalid")
    base_camera = base_flange @ flange_camera
    if (
        base_camera.shape != (4, 4)
        or not np.all(np.isfinite(base_camera))
        or abs(float(np.linalg.det(base_camera[:3, :3]))) <= 1e-12
    ):
        raise DiagnosisError("composed base-camera transform is invalid")

    with pose_path.open("r", encoding="utf-8") as handle:
        capture = yaml.safe_load(handle) or {}
    recorded_flange_xyz = _finite_xyz(capture.get("flange_xyz_m"))
    recorded_flange_rpy = _finite_xyz(capture.get("flange_rpy_rad"))
    fk_flange_xyz = tuple(float(value) for value in base_flange[:3, 3])
    fk_flange_rpy = tuple(float(value) for value in matrix_to_rpy(base_flange[:3, :3]))
    position_consistency = _distance(recorded_flange_xyz, fk_flange_xyz)
    rotation_consistency_deg: float | None = None
    if recorded_flange_rpy is not None:
        recorded_rotation = rpy_to_matrix(recorded_flange_rpy)
        delta = recorded_rotation.T @ base_flange[:3, :3]
        cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
        rotation_consistency_deg = float(math.degrees(math.acos(cosine)))

    observation = {
        "pose_file": str(pose_path.resolve()),
        "role": str(capture.get("role", "")),
        "captured_at": capture.get("captured_at"),
        "joint_positions_rad": list(joints),
        "joint_positions_deg": [float(math.degrees(value)) for value in joints],
        "recorded_flange_xyz_m": recorded_flange_xyz,
        "recorded_flange_rpy_rad": recorded_flange_rpy,
        "fk_flange_xyz_m": fk_flange_xyz,
        "fk_flange_rpy_rad": fk_flange_rpy,
        "fk_vs_capture_position_error_m": position_consistency,
        "fk_vs_capture_rotation_error_deg": rotation_consistency_deg,
        "T_base_flange": _json_safe(base_flange),
        "T_flange_camera": _json_safe(flange_camera),
        "T_base_camera": _json_safe(base_camera),
        "camera_origin_base_xyz_m": tuple(float(value) for value in base_camera[:3, 3]),
    }
    return observation, base_camera


def _short_edge_heading(
    detection: Detection,
    base_camera: np.ndarray,
    short_edge: TaskCDShortEdgeGraspConfig,
) -> dict[str, Any]:
    """Compute the exact production short-edge RPY and its geometry evidence."""

    required = {
        "object_center_camera_xyz": detection.object_center_camera_xyz,
        "major_axis_camera": detection.major_axis_camera,
        "minor_axis_camera": detection.minor_axis_camera,
        "major_axis_length_m": detection.major_axis_length_m,
        "minor_axis_length_m": detection.minor_axis_length_m,
        "pose_confidence": detection.pose_confidence,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise DiagnosisError(
            "short-edge grasp requires complete PCA data; missing "
            + ", ".join(missing)
        )

    pose_confidence = float(detection.pose_confidence)
    if (
        not math.isfinite(pose_confidence)
        or pose_confidence < float(short_edge.min_pose_confidence)
    ):
        raise DiagnosisError(
            f"pose confidence {pose_confidence:.6f} is below "
            f"{float(short_edge.min_pose_confidence):.6f}"
        )
    major_length = float(detection.major_axis_length_m)
    minor_length = float(detection.minor_axis_length_m)
    if (
        not math.isfinite(major_length)
        or not math.isfinite(minor_length)
        or major_length <= 0.0
        or minor_length <= 0.0
    ):
        raise DiagnosisError("PCA axis lengths must be finite and positive")
    axis_ratio = major_length / minor_length
    if axis_ratio + 1e-12 < float(short_edge.pca.min_axis_ratio):
        raise DiagnosisError(
            f"PCA axis ratio {axis_ratio:.6f} is below "
            f"{float(short_edge.pca.min_axis_ratio):.6f}"
        )

    major_axis_camera = np.asarray(detection.major_axis_camera, dtype=float)
    minor_axis_camera = np.asarray(detection.minor_axis_camera, dtype=float)
    if (
        major_axis_camera.shape != (3,)
        or not np.all(np.isfinite(major_axis_camera))
        or float(np.linalg.norm(major_axis_camera)) <= 1e-12
    ):
        raise DiagnosisError("major_axis_camera must be finite and non-zero")
    if (
        minor_axis_camera.shape != (3,)
        or not np.all(np.isfinite(minor_axis_camera))
        or float(np.linalg.norm(minor_axis_camera)) <= 1e-12
    ):
        raise DiagnosisError("minor_axis_camera must be finite and non-zero")
    major_axis_camera = major_axis_camera / float(np.linalg.norm(major_axis_camera))
    minor_axis_camera = minor_axis_camera / float(np.linalg.norm(minor_axis_camera))

    object_center_camera = np.asarray(detection.object_center_camera_xyz, dtype=float)
    if (
        object_center_camera.shape != (3,)
        or not np.all(np.isfinite(object_center_camera))
    ):
        raise DiagnosisError("object_center_camera_xyz must be a finite 3-vector")
    object_center_base = _transform_point(base_camera, object_center_camera)

    major_axis_base = base_camera[:3, :3] @ major_axis_camera
    minor_axis_base = base_camera[:3, :3] @ minor_axis_camera
    major_norm = float(np.linalg.norm(major_axis_base))
    minor_norm = float(np.linalg.norm(minor_axis_base))
    if major_norm <= 1e-12 or minor_norm <= 1e-12:
        raise DiagnosisError("PCA axis became zero in the robot base frame")
    major_axis_base = major_axis_base / major_norm
    minor_axis_base = minor_axis_base / minor_norm
    major_horizontal_norm = float(np.linalg.norm(major_axis_base[:2]))
    minor_horizontal_norm = float(np.linalg.norm(minor_axis_base[:2]))
    if not math.isfinite(major_horizontal_norm) or major_horizontal_norm <= 1e-6:
        raise DiagnosisError(
            "PCA major axis has no usable horizontal projection in base frame"
        )
    if not math.isfinite(minor_horizontal_norm) or minor_horizontal_norm <= 1e-6:
        raise DiagnosisError(
            "PCA minor axis has no usable horizontal projection in base frame"
        )

    mx = float(major_axis_base[0])
    my = float(major_axis_base[1])
    minor_mx = float(minor_axis_base[0])
    minor_my = float(minor_axis_base[1])
    closure_axis_tool = np.asarray(short_edge.closure_axis_tool_xy, dtype=float)
    if (
        closure_axis_tool.shape != (2,)
        or not np.all(np.isfinite(closure_axis_tool))
        or float(np.linalg.norm(closure_axis_tool)) <= 1e-12
    ):
        raise DiagnosisError("closure_axis_tool_xy must be finite and non-zero")
    closure_axis_tool = closure_axis_tool / float(np.linalg.norm(closure_axis_tool))
    major_angle = float(math.atan2(my, mx))
    minor_angle = float(math.atan2(minor_my, minor_mx))
    closure_angle = float(
        math.atan2(float(closure_axis_tool[1]), float(closure_axis_tool[0]))
    )
    combined_yaw = minor_angle + closure_angle
    yaw = float(
        math.atan2(math.sin(combined_yaw), math.cos(combined_yaw))
    )
    grasp_rpy = (math.pi, 0.0, yaw)
    rotation = np.asarray(rpy_to_matrix(grasp_rpy), dtype=float)
    closure_axis_base = rotation[:, :2] @ closure_axis_tool
    major_horizontal_axis = np.asarray(
        [mx / major_horizontal_norm, my / major_horizontal_norm, 0.0],
        dtype=float,
    )
    minor_horizontal_axis = np.asarray(
        [
            minor_mx / minor_horizontal_norm,
            minor_my / minor_horizontal_norm,
            0.0,
        ],
        dtype=float,
    )
    alignment = float(abs(np.dot(closure_axis_base, minor_horizontal_axis)))
    if (
        not math.isfinite(alignment)
        or alignment < 0.999999
        or abs(float(closure_axis_base[2])) > 1e-9
    ):
        raise DiagnosisError(
            "configured closure-axis must align with the PCA minor axis: "
            f"absolute dot={alignment:.9f}"
        )
    closure_xy = _normalise_xy(closure_axis_base[:2])
    major_xy = _normalise_xy(major_horizontal_axis[:2])
    minor_xy = _normalise_xy(minor_horizontal_axis[:2])
    if closure_xy is None or major_xy is None or minor_xy is None:
        raise DiagnosisError(
            "closure, major, or minor axis has no horizontal direction"
        )
    alignment_dot_xy = float(abs(np.dot(closure_xy, minor_xy)))
    alignment_error_deg = float(
        math.degrees(
            math.acos(float(np.clip(alignment_dot_xy, -1.0, 1.0)))
        )
    )

    return {
        "status": "accepted",
        "reason": "",
        "object_center_camera_xyz_m": tuple(
            float(value) for value in object_center_camera
        ),
        "object_center_base_xyz_m": tuple(
            float(value) for value in object_center_base
        ),
        "major_axis_camera_unit": tuple(
            float(value) for value in major_axis_camera
        ),
        "minor_axis_camera_unit": tuple(
            float(value) for value in minor_axis_camera
        ),
        "major_axis_base_unit": tuple(float(value) for value in major_axis_base),
        "minor_axis_base_unit": tuple(float(value) for value in minor_axis_base),
        "major_axis_base_xy_unit": tuple(float(value) for value in major_xy),
        "minor_axis_base_xy_unit": tuple(float(value) for value in minor_xy),
        "major_axis_base_xy_angle_rad": _undirected_angle(major_angle),
        "major_axis_base_xy_angle_deg": math.degrees(
            _undirected_angle(major_angle)
        ),
        "minor_axis_base_xy_angle_rad": _undirected_angle(minor_angle),
        "minor_axis_base_xy_angle_deg": math.degrees(
            _undirected_angle(minor_angle)
        ),
        "major_axis_length_m": major_length,
        "minor_axis_length_m": minor_length,
        "estimated_width_m": min(major_length, minor_length),
        "axis_ratio": axis_ratio,
        "pose_confidence": pose_confidence,
        "major_angle_rad": major_angle,
        "minor_angle_rad": minor_angle,
        "closure_angle_rad": closure_angle,
        "yaw_rad": yaw,
        "grasp_rpy_rad": grasp_rpy,
        "closure_axis_tool_xy_unit": tuple(
            float(value) for value in closure_axis_tool
        ),
        "closure_axis_base": tuple(
            float(value) for value in closure_axis_base
        ),
        "closure_axis_base_xy_unit": tuple(float(value) for value in closure_xy),
        "alignment": alignment,
        "alignment_error_deg": alignment_error_deg,
    }


def _depth_metrics(depth_m: Any, short_edge: TaskCDShortEdgeGraspConfig) -> dict[str, Any]:
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2 or depth.size == 0:
        raise DiagnosisError(f"depth must be a non-empty 2-D array, got {depth.shape}")
    finite = np.isfinite(depth)
    positive = finite & (depth > 0.0)
    lower = float(short_edge.pca.min_depth_m)
    upper = float(short_edge.pca.max_depth_m)
    in_range = positive & (depth >= lower) & (depth <= upper)
    height, width = depth.shape
    central = in_range[height // 4 : height - height // 4, width // 4 : width - width // 4]
    values = depth[in_range]
    quantiles = {
        "p01": None,
        "p10": None,
        "p50": None,
        "p90": None,
        "p99": None,
    }
    if values.size:
        for name, percentile in (
            ("p01", 1.0),
            ("p10", 10.0),
            ("p50", 50.0),
            ("p90", 90.0),
            ("p99", 99.0),
        ):
            quantiles[name] = float(np.quantile(values, percentile / 100.0))
    return {
        "shape": [int(height), int(width)],
        "configured_range_m": [lower, upper],
        "in_range_pixels": int(in_range.sum()),
        "in_range_ratio": float(in_range.mean()),
        "central_in_range_ratio": float(central.mean()),
        "quantiles_m": quantiles,
    }


def _detection_record(
    detection: Detection,
    *,
    frame_index: int,
    detection_index: int,
    base_camera: np.ndarray,
    short_edge: TaskCDShortEdgeGraspConfig,
    known_labels: set[str],
) -> dict[str, Any]:
    anchor_camera = _finite_xyz(detection.camera_xyz)
    anchor_base: tuple[float, float, float] | None = None
    if anchor_camera is not None:
        anchor_base = tuple(
            float(value)
            for value in _transform_point(base_camera, anchor_camera)
        )

    record: dict[str, Any] = {
        "frame_index": int(frame_index),
        "detection_index": int(detection_index),
        "label": detection.label,
        "is_known_class": detection.label.strip().casefold() in known_labels,
        "confidence": float(detection.confidence),
        "stable_confidence": float(
            detection.attributes.get("stable_confidence", detection.confidence)
        ),
        "raw_confidence": float(
            detection.attributes.get("raw_confidence", detection.confidence)
        ),
        "bbox_xyxy_px": [float(value) for value in detection.bbox],
        "bbox_center_uv_px": [
            0.5 * (float(detection.bbox[0]) + float(detection.bbox[2])),
            0.5 * (float(detection.bbox[1]) + float(detection.bbox[3])),
        ],
        "depth_m": None if detection.depth_m is None else float(detection.depth_m),
        "detection_anchor_camera_xyz_m": anchor_camera,
        "detection_anchor_base_xyz_m": anchor_base,
        "track_id": _track_id(detection),
        "graspability": float(detection.graspability),
        "geometry_status": "rejected",
        "geometry_reason": "",
        "duplicate_of_detection_index": None,
        "cluster_id": None,
        "attributes": _json_safe(detection.attributes),
    }
    try:
        heading = _short_edge_heading(detection, base_camera, short_edge)
    except (DiagnosisError, TypeError, ValueError) as exc:
        record["geometry_reason"] = str(exc)
        center_camera = _finite_xyz(detection.object_center_camera_xyz)
        record["object_center_camera_xyz_m"] = center_camera
        record["object_center_base_xyz_m"] = (
            None
            if center_camera is None
            else tuple(
                float(value)
                for value in _transform_point(base_camera, center_camera)
            )
        )
        record["major_axis_length_m"] = detection.major_axis_length_m
        record["minor_axis_length_m"] = detection.minor_axis_length_m
        record["estimated_width_m"] = (
            None
            if detection.major_axis_length_m is None
            or detection.minor_axis_length_m is None
            else float(min(detection.major_axis_length_m, detection.minor_axis_length_m))
        )
        record["axis_ratio"] = (
            None
            if detection.major_axis_length_m is None
            or detection.minor_axis_length_m is None
            else float(detection.major_axis_length_m / detection.minor_axis_length_m)
        )
        record["pose_confidence"] = detection.pose_confidence
        record["yaw_rad"] = None
        record["grasp_rpy_rad"] = None
        record["closure_axis_base"] = None
        record["closure_axis_base_xy_unit"] = None
        record["alignment_error_deg"] = None
        record["jaw_tip_target_base_xyz_m"] = None
        record["xy_correction_xyz_m"] = None
        record["xy_correction_horizontal_norm_m"] = None
    else:
        record.update(heading)
        record["geometry_status"] = "accepted"
        record["geometry_reason"] = ""
        center_base = _finite_xyz(heading["object_center_base_xyz_m"])
        if center_base is not None:
            fixed_z = short_edge.jaw_tip_z_m
            target_z = float(fixed_z) if fixed_z is not None else center_base[2]
            record["jaw_tip_target_base_xyz_m"] = (
                float(center_base[0]),
                float(center_base[1]),
                target_z,
            )
        else:
            record["jaw_tip_target_base_xyz_m"] = None
        if record["object_center_base_xyz_m"] is not None and anchor_base is not None:
            center = np.asarray(record["object_center_base_xyz_m"], dtype=float)
            anchor = np.asarray(anchor_base, dtype=float)
            correction = center - anchor
            record["xy_correction_xyz_m"] = tuple(
                float(value) for value in correction
            )
            record["xy_correction_horizontal_norm_m"] = float(
                np.linalg.norm(correction[:2])
            )
        else:
            record["xy_correction_xyz_m"] = None
            record["xy_correction_horizontal_norm_m"] = None
    return record


def _record_quality(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        1.0 if record.get("geometry_status") == "accepted" else 0.0,
        1.0 if record.get("is_known_class") else 0.0,
        float(record.get("stable_confidence", record.get("confidence", 0.0))),
        float(record.get("graspability", 0.0)),
    )


def _deduplicate_frame(
    records: Sequence[dict[str, Any]],
    *,
    radius_m: float,
    min_iou: float,
) -> list[int]:
    groups: list[list[int]] = []
    for index, record in enumerate(records):
        match_group: list[int] | None = None
        for group in groups:
            representative = records[group[0]]
            if _same_spatial_target(
                representative,
                record,
                radius_m=radius_m,
                min_iou=min_iou,
            ):
                match_group = group
                break
        if match_group is None:
            groups.append([index])
            continue
        match_group.append(index)
        best_index = max(match_group, key=lambda item: _record_quality(records[item]))
        match_group.remove(best_index)
        match_group.insert(0, best_index)

    representative_indexes: list[int] = []
    for group in groups:
        representative = group[0]
        representative_indexes.append(representative)
        for index in group[1:]:
            records[index]["duplicate_of_detection_index"] = int(
                records[representative]["detection_index"]
            )
    return sorted(representative_indexes)


def _cluster_records(
    records: Sequence[dict[str, Any]],
    *,
    radius_m: float,
    min_iou: float,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for record in records:
        if record.get("duplicate_of_detection_index") is not None:
            continue
        match: dict[str, Any] | None = None
        for cluster in clusters:
            if any(
                _same_spatial_target(
                    member,
                    record,
                    radius_m=radius_m,
                    min_iou=min_iou,
                )
                for member in cluster["observations"]
            ):
                match = cluster
                break
        if match is None:
            match = {
                "cluster_id": "",
                "label": str(record["label"]),
                "observations": [],
            }
            clusters.append(match)
        match["observations"].append(record)

    clusters.sort(
        key=lambda cluster: (
            min(int(item["frame_index"]) for item in cluster["observations"]),
            cluster["label"],
            str(cluster["observations"][0]["track_id"]),
        )
    )
    for index, cluster in enumerate(clusters, start=1):
        cluster_id = f"C{index:03d}"
        cluster["cluster_id"] = cluster_id
        for observation in cluster["observations"]:
            observation["cluster_id"] = cluster_id
            frame_index = int(observation["frame_index"])
            detection_index = int(observation["detection_index"])
            for other in records:
                if (
                    int(other["frame_index"]) == frame_index
                    and other.get("duplicate_of_detection_index") == detection_index
                ):
                    other["cluster_id"] = cluster_id
    return clusters


def _triplet_jitter_m(values: Iterable[Any]) -> dict[str, float | None]:
    points = [
        xyz
        for xyz in (_finite_xyz(value) for value in values)
        if xyz is not None
    ]
    if not points:
        return {"max_from_median_m": None, "axis_range_m": None}
    array = np.asarray(points, dtype=float)
    median = np.median(array, axis=0)
    distances = np.linalg.norm(array - median, axis=1)
    axis_range = np.max(array, axis=0) - np.min(array, axis=0)
    return {
        "max_from_median_m": float(np.max(distances)),
        "axis_range_m": float(np.max(axis_range)),
    }


def _cluster_payload(
    cluster: Mapping[str, Any],
    *,
    min_cluster_frames: int,
) -> dict[str, Any]:
    observations = list(cluster["observations"])
    accepted = [
        item for item in observations if item.get("geometry_status") == "accepted"
    ]
    frame_indexes = sorted({int(item["frame_index"]) for item in observations})
    track_ids = sorted(
        {str(item["track_id"]) for item in observations if str(item["track_id"])}
    )
    rejection_reasons = Counter(
        str(item.get("geometry_reason", ""))
        for item in observations
        if item.get("geometry_status") != "accepted"
    )
    major_angles = [
        float(item["major_axis_base_xy_angle_rad"])
        for item in accepted
        if item.get("major_axis_base_xy_angle_rad") is not None
    ]
    minor_angles = [
        float(item["minor_axis_base_xy_angle_rad"])
        for item in accepted
        if item.get("minor_axis_base_xy_angle_rad") is not None
    ]
    yaw_values = [
        float(item["yaw_rad"])
        for item in accepted
        if item.get("yaw_rad") is not None
    ]
    frame_hits = len(frame_indexes)
    geometry_fraction = (
        float(len(accepted) / len(observations)) if observations else 0.0
    )
    stable = frame_hits >= min_cluster_frames
    geometry_stable = (
        stable
        and len(accepted) >= min_cluster_frames
        and geometry_fraction >= 0.80
    )
    return {
        "cluster_id": str(cluster["cluster_id"]),
        "label": str(cluster["label"]),
        "status": (
            "stable_geometry"
            if geometry_stable
            else ("stable_geometry_incomplete" if stable else "insufficient_frames")
        ),
        "frame_hits": frame_hits,
        "observation_count": len(observations),
        "geometry_accepted_count": len(accepted),
        "geometry_accepted_fraction": geometry_fraction,
        "stable_frame_threshold": int(min_cluster_frames),
        "stable": stable,
        "geometry_stable": geometry_stable,
        "frame_indexes": frame_indexes,
        "track_ids": track_ids,
        "track_id_count": len(track_ids),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "object_center_base_median_xyz_m": _median_triplet(
            item.get("object_center_base_xyz_m") for item in accepted
        ),
        "object_center_camera_median_xyz_m": _median_triplet(
            item.get("object_center_camera_xyz_m") for item in accepted
        ),
        "object_center_base_jitter": _triplet_jitter_m(
            item.get("object_center_base_xyz_m") for item in accepted
        ),
        "detection_anchor_base_median_xyz_m": _median_triplet(
            item.get("detection_anchor_base_xyz_m") for item in accepted
        ),
        "xy_correction_median_xyz_m": _median_triplet(
            item.get("xy_correction_xyz_m") for item in accepted
        ),
        "xy_correction_horizontal_norm_median_m": _median(
            item.get("xy_correction_horizontal_norm_m") for item in accepted
        ),
        "major_axis_base_xy_mean_angle_rad": _circular_axis_mean(major_angles),
        "major_axis_base_xy_mean_angle_deg": (
            None
            if _circular_axis_mean(major_angles) is None
            else float(math.degrees(_circular_axis_mean(major_angles)))
        ),
        "minor_axis_base_xy_mean_angle_rad": _circular_axis_mean(minor_angles),
        "minor_axis_base_xy_mean_angle_deg": (
            None
            if _circular_axis_mean(minor_angles) is None
            else float(math.degrees(_circular_axis_mean(minor_angles)))
        ),
        "yaw_mean_angle_rad": _circular_axis_mean(yaw_values),
        "yaw_mean_angle_deg": (
            None
            if _circular_axis_mean(yaw_values) is None
            else float(math.degrees(_circular_axis_mean(yaw_values)))
        ),
        "major_axis_length_median_m": _median(
            item.get("major_axis_length_m") for item in accepted
        ),
        "minor_axis_length_median_m": _median(
            item.get("minor_axis_length_m") for item in accepted
        ),
        "estimated_width_median_m": _median(
            item.get("estimated_width_m") for item in accepted
        ),
        "axis_ratio_median": _median(item.get("axis_ratio") for item in accepted),
        "pose_confidence_median": _median(
            item.get("pose_confidence") for item in accepted
        ),
        "closure_alignment_error_median_deg": _median(
            item.get("alignment_error_deg") for item in accepted
        ),
        "jaw_tip_target_base_median_xyz_m": _median_triplet(
            item.get("jaw_tip_target_base_xyz_m") for item in accepted
        ),
        "representative_detection": (
            None
            if not observations
            else {
                "frame_index": int(
                    max(
                        observations,
                        key=lambda item: (
                            *_record_quality(item),
                            -int(item["frame_index"]),
                            -int(item["detection_index"]),
                        ),
                    )["frame_index"]
                ),
                "detection_index": int(
                    max(
                        observations,
                        key=lambda item: (
                            *_record_quality(item),
                            -int(item["frame_index"]),
                            -int(item["detection_index"]),
                        ),
                    )["detection_index"]
                ),
            }
        ),
    }


def _bgr_u8(image: Any, *, field: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2).astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, :3].astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] == 3:
        return array.astype(np.uint8, copy=False)
    raise DiagnosisError(
        f"{field} must have shape (H,W), (H,W,3) or (H,W,4), got {array.shape}"
    )


def _depth_color(depth_m: Any, short_edge: TaskCDShortEdgeGraspConfig) -> np.ndarray:
    import cv2

    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2:
        raise DiagnosisError(f"depth must be 2-D, got {depth.shape}")
    lower = float(short_edge.pca.min_depth_m)
    upper = float(short_edge.pca.max_depth_m)
    valid = np.isfinite(depth) & (depth >= lower) & (depth <= upper)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if upper <= lower:
        raise DiagnosisError("PCA depth range must be increasing")
    scaled = (depth[valid] - lower) * (255.0 / (upper - lower))
    normalized[valid] = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)
    return colored


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    import cv2

    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise DiagnosisError(f"failed to write image: {path}")


def _project_camera_xyz(
    intrinsics: Any,
    xyz: Sequence[float] | None,
) -> tuple[int, int] | None:
    value = _finite_xyz(xyz)
    if intrinsics is None or value is None or value[2] <= 1e-6:
        return None
    return (
        int(round(float(intrinsics.fx) * value[0] / value[2] + float(intrinsics.cx))),
        int(round(float(intrinsics.fy) * value[1] / value[2] + float(intrinsics.cy))),
    )


def _draw_geometry_overlay(
    image: np.ndarray,
    frame: Frame,
    records: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    import cv2

    overlay = image.copy()
    for record in records:
        if record.get("duplicate_of_detection_index") is not None:
            continue
        track = record.get("track_id") or "?"
        cluster = record.get("cluster_id") or "unclustered"
        color = (30, 220, 30) if record.get("geometry_status") == "accepted" else (0, 165, 255)
        center = _project_camera_xyz(
            frame.camera_intrinsics,
            record.get("object_center_camera_xyz_m"),
        )
        if center is not None:
            cv2.drawMarker(
                overlay,
                center,
                (255, 120, 30),
                cv2.MARKER_CROSS,
                22,
                2,
                cv2.LINE_AA,
            )
        major_axis = record.get("major_axis_camera_unit")
        center_camera = record.get("object_center_camera_xyz_m")
        major_length = record.get("major_axis_length_m")
        if (
            major_axis is not None
            and center_camera is not None
            and major_length is not None
        ):
            axis = np.asarray(major_axis, dtype=float)
            center_xyz = np.asarray(center_camera, dtype=float)
            half = 0.5 * float(major_length)
            first = _project_camera_xyz(frame.camera_intrinsics, center_xyz - half * axis)
            second = _project_camera_xyz(frame.camera_intrinsics, center_xyz + half * axis)
            if first is not None and second is not None:
                cv2.line(overlay, first, second, (255, 0, 255), 2, cv2.LINE_AA)
        x0, y0, _x1, y1 = (int(round(float(value))) for value in record["bbox_xyxy_px"])
        center_base = record.get("object_center_base_xyz_m")
        correction = record.get("xy_correction_horizontal_norm_m")
        if center_base is None:
            text = f"{cluster} t={track} GEOMETRY REJECTED"
        else:
            text = (
                f"{cluster} t={track} XYZ=({center_base[0]:.3f},{center_base[1]:.3f},"
                f"{center_base[2]:.3f}) corr={0.0 if correction is None else correction:.3f}m"
            )
        text_y = min(overlay.shape[0] - 12, max(105, y1 + 18))
        cv2.putText(
            overlay,
            text,
            (max(2, x0), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _save_frame_artifacts(
    output_dir: Path,
    *,
    frame_index: int,
    frame: Frame,
    records: Sequence[dict[str, Any]],
    short_edge: TaskCDShortEdgeGraspConfig,
) -> dict[str, str]:
    image = _bgr_u8(frame.image, field="frame.image")
    if frame.depth_m is None:
        raise DiagnosisError(f"{frame.frame_id} has no aligned depth frame")
    stem = f"frame_{frame_index:03d}"
    rgb_path = output_dir / f"{stem}_rgb.jpg"
    depth_path = output_dir / f"{stem}_depth_color.jpg"
    depth_npz_path = output_dir / f"{stem}_depth_m.npz"
    overlay_path = output_dir / f"{stem}_detections_overlay.jpg"
    _write_jpeg(rgb_path, image)
    _write_jpeg(depth_path, _depth_color(frame.depth_m, short_edge))
    np.savez_compressed(
        depth_npz_path,
        depth_m=np.asarray(frame.depth_m, dtype=np.float32),
    )

    representatives = tuple(
        frame.detections[int(record["detection_index"])]
        for record in records
        if record.get("duplicate_of_detection_index") is None
    )
    deduped_frame = replace(frame, detections=representatives)
    known_count = sum(1 for record in records if record.get("is_known_class"))
    accepted_count = sum(
        1 for record in records if record.get("geometry_status") == "accepted"
    )
    overlay = render_preview(
        deduped_frame,
        status=(
            f"TASK C READ ONLY | frame {frame_index} | "
            f"known={known_count} geometry_ok={accepted_count}"
        ),
        target_class="",
        stale_after_s=1e9,
    )
    overlay = _draw_geometry_overlay(overlay, frame, records)
    _write_jpeg(overlay_path, overlay)
    return {
        "rgb": rgb_path.name,
        "depth_color": depth_path.name,
        "depth_m_npz": depth_npz_path.name,
        "detections_overlay": overlay_path.name,
    }


def _copy_representative_files(
    output_dir: Path,
    frame_record: Mapping[str, Any],
) -> dict[str, str]:
    copied: dict[str, str] = {}
    for key, filename in dict(frame_record.get("images", {})).items():
        source = output_dir / str(filename)
        target = output_dir / f"representative_{key}{source.suffix}"
        shutil.copy2(source, target)
        copied[key] = target.name
    return copied


def run_diagnosis(
    config: Track1Config,
    *,
    output_dir: Path,
    frame_count: int = DEFAULT_FRAMES,
    show: bool = False,
    require_known: bool = False,
    require_stable_geometry: bool = False,
    min_cluster_frames: int = DEFAULT_MIN_CLUSTER_FRAMES,
    spatial_match_radius_m: float = DEFAULT_SPATIAL_MATCH_RADIUS_M,
    bbox_match_iou: float = DEFAULT_BBOX_MATCH_IOU,
    config_path: Path | None = None,
    build_vision_fn: Callable[..., Any] = build_vision,
) -> dict[str, Any]:
    """Capture and diagnose Task C geometry without constructing actuators."""

    assert_read_only_task_c(config)
    if frame_count <= 0:
        raise DiagnosisError("frame_count must be positive")
    if min_cluster_frames <= 0:
        raise DiagnosisError("min_cluster_frames must be positive")
    if spatial_match_radius_m <= 0.0:
        raise DiagnosisError("spatial_match_radius_m must be positive")
    if not 0.0 <= bbox_match_iou <= 1.0:
        raise DiagnosisError("bbox_match_iou must be in [0, 1]")

    observation, base_camera = _load_observation_geometry(config)
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    known_labels = {
        label.strip().casefold()
        for label in config.target.known_classes
        if label.strip()
    }
    if not known_labels:
        known_labels = {
            label.strip().casefold()
            for label in config.vision.classes
            if label.strip()
        }

    output_dir = _prepare_output_dir(output_dir)
    vision_config = replace(
        config.vision,
        show=bool(show),
        save_annotated=False,
    )
    source = None
    capture_error = ""
    close_error = ""
    captured: list[tuple[Frame, list[dict[str, Any]]]] = []
    capture_times_ms: list[float] = []
    started_s = time.monotonic()
    first_capture_s: float | None = None
    log = lambda message: print(message, flush=True)  # noqa: E731

    try:
        source = build_vision_fn(
            vision_config,
            log=log,
            object_pose_config=short_edge.pca,
        )
        for frame_index in range(1, frame_count + 1):
            capture_started_s = time.monotonic()
            frame = source.capture()
            elapsed_ms = 1000.0 * (time.monotonic() - capture_started_s)
            capture_times_ms.append(float(elapsed_ms))
            if first_capture_s is None:
                first_capture_s = time.monotonic() - started_s
            records = [
                _detection_record(
                    detection,
                    frame_index=frame_index,
                    detection_index=detection_index,
                    base_camera=base_camera,
                    short_edge=short_edge,
                    known_labels=known_labels,
                )
                for detection_index, detection in enumerate(frame.detections)
            ]
            representative_indexes = _deduplicate_frame(
                records,
                radius_m=spatial_match_radius_m,
                min_iou=bbox_match_iou,
            )
            accepted = sum(
                1
                for index in representative_indexes
                if records[index].get("geometry_status") == "accepted"
            )
            duplicates = len(records) - len(representative_indexes)
            captured.append((frame, records))
            print(
                f"[diagnose] frame {frame_index}/{frame_count} {frame.frame_id}: "
                f"raw={len(records)} dedup={len(representative_indexes)} "
                f"geometry_ok={accepted} duplicates={duplicates} "
                f"capture={elapsed_ms:.0f}ms",
                flush=True,
            )
    except Exception as exc:
        capture_error = f"{type(exc).__name__}: {exc}"
        print(f"[diagnose] capture failed: {capture_error}", file=sys.stderr, flush=True)
    finally:
        elapsed_s = time.monotonic() - started_s
        if source is not None:
            close = getattr(source, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # pragma: no cover - hardware dependent
                    close_error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[diagnose] vision close failed: {close_error}",
                        file=sys.stderr,
                        flush=True,
                    )

    all_records = [
        record
        for _frame, records in captured
        for record in records
    ]
    clusters = _cluster_records(
        all_records,
        radius_m=spatial_match_radius_m,
        min_iou=bbox_match_iou,
    )
    cluster_payloads = [
        _cluster_payload(cluster, min_cluster_frames=min_cluster_frames)
        for cluster in clusters
    ]

    frames_payload: list[dict[str, Any]] = []
    for frame_index, (frame, records) in enumerate(captured, start=1):
        images = _save_frame_artifacts(
            output_dir,
            frame_index=frame_index,
            frame=frame,
            records=records,
            short_edge=short_edge,
        )
        representative_count = sum(
            1
            for record in records
            if record.get("duplicate_of_detection_index") is None
        )
        frames_payload.append(
            {
                "index": int(frame_index),
                "frame_id": frame.frame_id,
                "stamp_s": float(frame.stamp_s),
                "capture_elapsed_ms": capture_times_ms[frame_index - 1],
                "raw_detection_count": len(records),
                "deduplicated_detection_count": representative_count,
                "duplicate_count": len(records) - representative_count,
                "known_detection_count": sum(
                    1 for record in records if record.get("is_known_class")
                ),
                "geometry_accepted_count": sum(
                    1 for record in records if record.get("geometry_status") == "accepted"
                ),
                "depth": _depth_metrics(frame.depth_m, short_edge),
                "camera_intrinsics": (
                    None
                    if frame.camera_intrinsics is None
                    else {
                        "width": int(frame.camera_intrinsics.width),
                        "height": int(frame.camera_intrinsics.height),
                        "fx": float(frame.camera_intrinsics.fx),
                        "fy": float(frame.camera_intrinsics.fy),
                        "cx": float(frame.camera_intrinsics.cx),
                        "cy": float(frame.camera_intrinsics.cy),
                    }
                ),
                "images": images,
                "detections": records,
            }
        )

    representative_frame: dict[str, Any] | None = None
    representative_images: dict[str, str] = {}
    if frames_payload:
        best = max(
            frames_payload,
            key=lambda item: (
                int(item["geometry_accepted_count"]),
                int(item["deduplicated_detection_count"]),
                int(item["known_detection_count"]),
                -int(item["index"]),
            ),
        )
        representative_frame = {
            "index": int(best["index"]),
            "frame_id": best["frame_id"],
        }
        representative_images = _copy_representative_files(output_dir, best)
        representative_frame["images"] = representative_images

    key_observations = []
    for cluster in cluster_payloads:
        summary = {
            "cluster_id": cluster["cluster_id"],
            "label": cluster["label"],
            "status": cluster["status"],
            "frame_hits": cluster["frame_hits"],
            "track_ids": cluster["track_ids"],
            "object_center_base_median_xyz_m": cluster[
                "object_center_base_median_xyz_m"
            ],
            "object_center_base_jitter": cluster["object_center_base_jitter"],
            "major_axis_base_xy_mean_angle_deg": cluster[
                "major_axis_base_xy_mean_angle_deg"
            ],
            "axis_ratio_median": cluster["axis_ratio_median"],
            "pose_confidence_median": cluster["pose_confidence_median"],
            "estimated_width_median_m": cluster["estimated_width_median_m"],
            "xy_correction_horizontal_norm_median_m": cluster[
                "xy_correction_horizontal_norm_median_m"
            ],
            "closure_alignment_error_median_deg": cluster[
                "closure_alignment_error_median_deg"
            ],
            "jaw_tip_target_base_median_xyz_m": cluster[
                "jaw_tip_target_base_median_xyz_m"
            ],
            "geometry_accepted_fraction": cluster[
                "geometry_accepted_fraction"
            ],
            "rejection_reasons": cluster["rejection_reasons"],
        }
        key_observations.append(summary)
        center = summary["object_center_base_median_xyz_m"]
        center_text = (
            "unavailable"
            if center is None
            else f"({center[0]:.4f},{center[1]:.4f},{center[2]:.4f})"
        )
        print(
            f"[diagnose] {summary['cluster_id']} {summary['label']} "
            f"status={summary['status']} frames={summary['frame_hits']} "
            f"tracks={','.join(summary['track_ids']) or '-'} "
            f"center_base={center_text} "
            f"axis_ratio={summary['axis_ratio_median']} "
            f"confidence={summary['pose_confidence_median']} "
            f"alignment_err_deg={summary['closure_alignment_error_median_deg']} "
            f"correction_m={summary['xy_correction_horizontal_norm_median_m']}",
            flush=True,
        )

    failures: list[str] = []
    if capture_error:
        failures.append(f"capture failed: {capture_error}")
    if close_error:
        failures.append(f"vision close failed: {close_error}")
    if len(captured) != frame_count:
        failures.append(f"captured {len(captured)} of {frame_count} requested frames")
    # Only deduplicated observations count as known positives.
    dedup_records = [
        record
        for record in all_records
        if record.get("duplicate_of_detection_index") is None
    ]
    known_observations = sum(
        1 for record in dedup_records if record.get("is_known_class")
    )
    accepted_observations = sum(
        1 for record in dedup_records if record.get("geometry_status") == "accepted"
    )
    stable_clusters = sum(1 for cluster in cluster_payloads if cluster["stable"])
    stable_geometry_clusters = sum(
        1 for cluster in cluster_payloads if cluster["geometry_stable"]
    )
    if require_known and known_observations == 0:
        failures.append("--require-known was set but no known-class detection was observed")
    if require_stable_geometry and stable_geometry_clusters == 0:
        failures.append(
            "--require-stable-geometry was set but no physical cluster reached "
            f"{min_cluster_frames} stable accepted-geometry frames"
        )

    status = "passed" if not failures else "failed"
    depth_ratios = [
        float(frame["depth"]["in_range_ratio"]) for frame in frames_payload
    ]
    central_depth_ratios = [
        float(frame["depth"]["central_in_range_ratio"]) for frame in frames_payload
    ]
    aggregate = {
        "requested_frames": int(frame_count),
        "captured_frames": len(captured),
        "raw_observations": len(all_records),
        "duplicate_observations_removed": len(all_records) - len(dedup_records),
        "known_observations_deduplicated": known_observations,
        "geometry_accepted_observations_deduplicated": accepted_observations,
        "physical_clusters": len(cluster_payloads),
        "stable_clusters": stable_clusters,
        "stable_geometry_clusters": stable_geometry_clusters,
        "unique_labels": sorted(
            {str(cluster["label"]) for cluster in cluster_payloads}
        ),
        "mean_frame_depth_in_range_ratio": (
            float(sum(depth_ratios) / len(depth_ratios)) if depth_ratios else None
        ),
        "minimum_frame_depth_in_range_ratio": (
            min(depth_ratios) if depth_ratios else None
        ),
        "mean_frame_central_depth_in_range_ratio": (
            float(sum(central_depth_ratios) / len(central_depth_ratios))
            if central_depth_ratios
            else None
        ),
        "minimum_frame_central_depth_in_range_ratio": (
            min(central_depth_ratios) if central_depth_ratios else None
        ),
    }
    subsequent_times = capture_times_ms[1:]
    notes = [
        "No RobotInfo is required and no robot bridge is started.",
        "PickPipeline and PosePlanner were not constructed.",
        "No motion backend or gripper backend was constructed.",
        "No motion or gripper command was published by this tool.",
        "The camera pose is derived from the fixed observation joint file using "
        "controller_fk and the calibrated flange-camera extrinsics.",
        "The current physical arm posture is assumed to match the fixed "
        "observation pose; this report is not motion authorization.",
        "Physical clusters use same-label 25 mm spatial matching or 0.65 bbox "
        "IoU by default, so detector track re-IDs at the same location are "
        "aggregated for stability review.",
    ]
    if not captured:
        notes.append("No frame was captured, so this report is a failure baseline.")
    if known_observations == 0:
        notes.append("No known-class positive sample was observed.")
    if accepted_observations == 0:
        notes.append(
            "No detection had complete accepted PCA short-edge geometry; inspect "
            "per-detection geometry_reason before changing any grasp parameter."
        )
    if stable_geometry_clusters == 0:
        notes.append(
            "No physical cluster reached the configured multi-frame geometry "
            "stability threshold; do not tune or move from a single frame."
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "task_c_read_only_geometry_diagnostic",
        "generated_at": _now_iso(),
        "status": status,
        "scope": "vision_only_no_robot_info_no_motion_no_gripper",
        "safety": {
            "robot_info_required": False,
            "robot_bridge_started": False,
            "pick_pipeline_constructed": False,
            "pose_planner_constructed": False,
            "motion_backend_constructed": False,
            "gripper_backend_constructed": False,
            "motion_commands_published": False,
            "gripper_commands_sent": False,
            "execution_authorized": False,
        },
        "config": {
            "path": "" if config_path is None else str(config_path),
            "task_mode": config.task.mode.value,
            "max_picks": int(config.task.max_picks),
            "vision_backend": config.vision.backend,
            "weights": config.vision.weights,
            "weights_sha256": config.vision.weights_sha256,
            "width": int(config.vision.width),
            "height": int(config.vision.height),
            "fps": int(config.vision.fps),
            "rotate_180": bool(config.vision.rotate_180),
            "stabilizer_enabled": bool(config.vision.stabilize),
            "known_class_count": len(known_labels),
            "short_edge_grasp": _json_safe(asdict(short_edge)),
            "gripper_tip_offset_flange_m": config.pose.gripper_tip_offset_flange_m,
            "configured_motion_dry_run": bool(config.motion.dry_run),
            "configured_gripper_backend": config.gripper.backend,
            "show_window": bool(show),
        },
        "observation_geometry": observation,
        "capture": {
            "elapsed_s": float(elapsed_s),
            "startup_to_first_capture_s": first_capture_s,
            "first_capture_ms": capture_times_ms[0] if capture_times_ms else None,
            "mean_capture_ms": (
                float(sum(capture_times_ms) / len(capture_times_ms))
                if capture_times_ms
                else None
            ),
            "mean_subsequent_capture_ms": (
                float(sum(subsequent_times) / len(subsequent_times))
                if subsequent_times
                else None
            ),
            "capture_error": capture_error,
            "close_error": close_error,
        },
        "diagnosis": {
            "require_known": bool(require_known),
            "require_stable_geometry": bool(require_stable_geometry),
            "min_cluster_frames": int(min_cluster_frames),
            "spatial_match_radius_m": float(spatial_match_radius_m),
            "bbox_match_iou": float(bbox_match_iou),
            "passed": not failures,
            "failures": failures,
        },
        "aggregate": aggregate,
        "key_observations": key_observations,
        "clusters": cluster_payloads,
        "representative_frame": representative_frame,
        "frames": frames_payload,
        "notes": notes,
    }
    _write_json(output_dir / "summary.json", payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diagnose_task_c_geometry",
        description=(
            "Read-only Task C D405/V11/PCA short-edge geometry diagnosis. "
            "Never connects to RobotInfo or commands the arm/gripper."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="formal Task C YAML",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"number of frames to capture (default: {DEFAULT_FRAMES})",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "new output directory; defaults to "
            "logs/task_c_geometry_diagnostic_<time>"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show the observation-only OpenCV preview window",
    )
    parser.add_argument(
        "--require-known",
        action="store_true",
        help="fail unless at least one known-class detection is observed",
    )
    parser.add_argument(
        "--require-stable-geometry",
        action="store_true",
        help=(
            "fail unless one physical cluster has enough accepted-geometry "
            "frames for a multi-frame conclusion"
        ),
    )
    parser.add_argument(
        "--min-cluster-frames",
        type=int,
        default=DEFAULT_MIN_CLUSTER_FRAMES,
        help=(
            "accepted multi-frame observations required for a stable cluster "
            f"(default: {DEFAULT_MIN_CLUSTER_FRAMES})"
        ),
    )
    parser.add_argument(
        "--spatial-match-radius-mm",
        type=float,
        default=25.0,
        help="same-label physical cluster radius in millimetres (default: 25)",
    )
    parser.add_argument(
        "--bbox-match-iou",
        type=float,
        default=DEFAULT_BBOX_MATCH_IOU,
        help=f"same-label bbox dedup IoU (default: {DEFAULT_BBOX_MATCH_IOU})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    try:
        config = load_config(config_path)
        assert_read_only_task_c(config)
        config.validate()
    except Exception as exc:
        print(f"[diagnose] configuration rejected: {exc}", file=sys.stderr)
        return 2

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _unique_default_output_dir()
    )
    print(
        "[diagnose] READ ONLY: no RobotInfo, no robot bridge, no arm motion, "
        "no gripper command",
        flush=True,
    )
    print(
        f"[diagnose] config={config_path} frames={args.frames} output={output_dir}",
        flush=True,
    )
    try:
        payload = run_diagnosis(
            config,
            output_dir=output_dir,
            frame_count=args.frames,
            show=args.show,
            require_known=args.require_known,
            require_stable_geometry=args.require_stable_geometry,
            min_cluster_frames=args.min_cluster_frames,
            spatial_match_radius_m=args.spatial_match_radius_mm / 1000.0,
            bbox_match_iou=args.bbox_match_iou,
            config_path=config_path,
        )
    except KeyboardInterrupt:
        print("[diagnose] interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[diagnose] failed: {exc}", file=sys.stderr)
        return 1

    aggregate = payload["aggregate"]
    print(
        "[diagnose] "
        f"status={payload['status']} frames={aggregate['captured_frames']} "
        f"known={aggregate['known_observations_deduplicated']} "
        f"geometry_ok={aggregate['geometry_accepted_observations_deduplicated']} "
        f"clusters={aggregate['physical_clusters']} "
        f"stable={aggregate['stable_clusters']} "
        f"stable_geometry={aggregate['stable_geometry_clusters']} "
        f"depth_mean={aggregate['mean_frame_depth_in_range_ratio']}",
        flush=True,
    )
    print(f"[diagnose] summary: {output_dir / 'summary.json'}", flush=True)
    if payload["status"] != "passed":
        for failure in payload["diagnosis"]["failures"]:
            print(f"[diagnose] FAIL: {failure}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
