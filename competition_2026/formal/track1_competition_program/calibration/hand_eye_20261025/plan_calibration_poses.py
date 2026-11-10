#!/usr/bin/env python3
"""Plan and validate an eye-in-hand chessboard capture route.

This tool is read-only with respect to the robot: it subscribes to
``/RobotInfo`` and never imports or publishes ``/RobotMove``.  It produces a
reviewable pose plan only.  A separate, explicitly authorised capture tool is
required before any physical motion can occur.

The planned transform convention is the standard OpenCV eye-in-hand one:
``T_flange_camera``.  The nominal camera translation is used solely for image
visibility prediction; its rotation remains unvalidated and is not written
back into the production configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml

from track1_grasp.kinematics import (
    axis_angle_matrix,
    controller_fk,
    joints_within_limits,
    matrix_to_rpy,
    rotation_angle_error,
    rotation_matrix_to_vector,
    solve_pose_ik_multistart,
    weighted_jacobian_singular_values,
)


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    rx_deg: float = 0.0
    ry_deg: float = 0.0
    rz_deg: float = 0.0

    @property
    def description(self) -> str:
        parts: list[str] = []
        for axis, value in (("X", self.rx_deg), ("Y", self.ry_deg), ("Z", self.rz_deg)):
            if abs(value) > 1e-9:
                parts.append(f"法兰{axis}轴 {value:+.1f}°")
        return "，".join(parts) if parts else "初始观察姿态"


@dataclass(frozen=True)
class VisibilityMetrics:
    min_margin_px: float
    left_margin_px: float
    right_margin_px: float
    top_margin_px: float
    bottom_margin_px: float
    center_px: tuple[float, float]
    all_corners_visible: bool


# The negative RX side approaches a controller singularity near -13°.  Keep
# that side at -8° and take the larger tilt on the safer positive side.
CANDIDATE_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec("rx_m08", rx_deg=-8.0),
    CandidateSpec("rx_p08", rx_deg=8.0),
    CandidateSpec("rx_p16", rx_deg=16.0),
    CandidateSpec("ry_m08", ry_deg=-8.0),
    CandidateSpec("ry_p08", ry_deg=8.0),
    CandidateSpec("ry_m16", ry_deg=-16.0),
    CandidateSpec("ry_p16", ry_deg=16.0),
    CandidateSpec("rz_m08", rz_deg=-8.0),
    CandidateSpec("rz_p08", rz_deg=8.0),
    CandidateSpec("rz_m16", rz_deg=-16.0),
    CandidateSpec("rz_p16", rz_deg=16.0),
    CandidateSpec("xy_mm", rx_deg=-8.0, ry_deg=-10.0),
    CandidateSpec("xy_mp", rx_deg=-8.0, ry_deg=10.0),
    CandidateSpec("xy_pm", rx_deg=10.0, ry_deg=-10.0),
    CandidateSpec("xy_pp", rx_deg=10.0, ry_deg=10.0),
    CandidateSpec("xz_mm", rx_deg=-8.0, rz_deg=-10.0),
    CandidateSpec("xz_mp", rx_deg=-8.0, rz_deg=10.0),
    CandidateSpec("xz_pm", rx_deg=10.0, rz_deg=-10.0),
    CandidateSpec("xz_pp", rx_deg=10.0, rz_deg=10.0),
    CandidateSpec("yz_mm", ry_deg=-10.0, rz_deg=-10.0),
    CandidateSpec("yz_mp", ry_deg=-10.0, rz_deg=10.0),
    CandidateSpec("yz_pm", ry_deg=10.0, rz_deg=-10.0),
    CandidateSpec("yz_pp", ry_deg=10.0, rz_deg=10.0),
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML 顶层不是字典: {path}")
    return payload


def _rotation_from_candidate(spec: CandidateSpec) -> np.ndarray:
    return (
        axis_angle_matrix([0.0, 0.0, 1.0], math.radians(spec.rz_deg))
        @ axis_angle_matrix([0.0, 1.0, 0.0], math.radians(spec.ry_deg))
        @ axis_angle_matrix([1.0, 0.0, 0.0], math.radians(spec.rx_deg))
    )


def _target_transform(start: np.ndarray, spec: CandidateSpec) -> np.ndarray:
    target = start.copy()
    target[:3, :3] = start[:3, :3] @ _rotation_from_candidate(spec)
    return target


def _object_points(board_columns: int, board_rows: int, square_size_m: float) -> np.ndarray:
    points = np.zeros((board_columns * board_rows, 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0:board_columns, 0:board_rows].T.reshape(-1, 2)
        * float(square_size_m)
    )
    return points


def _average_transform(observations: Sequence[dict[str, Any]]) -> np.ndarray:
    transforms = [np.asarray(item["transform_camera_board"], dtype=float) for item in observations]
    if not transforms:
        raise RuntimeError("检测文件中没有 observations")
    translation = np.mean([item[:3, 3] for item in transforms], axis=0)
    rotation_sum = np.sum([item[:3, :3] for item in transforms], axis=0)
    u, _, vt = np.linalg.svd(rotation_sum)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _read_robot_state(timeout_s: float) -> dict[str, Any]:
    try:
        import rclpy
        from codroid_msgs.msg import RobotInfo
    except ImportError as exc:
        raise RuntimeError(
            "缺少 ROS 2 Python 环境或 codroid_msgs。请先运行 ensure_runtime、"
            "source /opt/ros/humble/setup.bash，再 source 运行区 setup.bash"
        ) from exc

    rclpy.init(args=[])
    node = rclpy.create_node("hand_eye_pose_plan_read_only")
    latest: dict[str, Any] | None = None

    def callback(msg: Any) -> None:
        nonlocal latest
        latest = {
            "state": str(getattr(msg, "state", "")),
            "fault_flag": int(getattr(msg, "fault_flag", 0)),
            "joint_positions_rad": [float(value) for value in getattr(msg, "joint_positions", [])],
            "end_positions": [float(value) for value in getattr(msg, "end_positions", [])],
        }

    subscription = node.create_subscription(RobotInfo, "/RobotInfo", callback, 10)
    deadline = node.get_clock().now().nanoseconds / 1e9 + float(timeout_s)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now_s = node.get_clock().now().nanoseconds / 1e9
            if latest is not None and now_s >= deadline - float(timeout_s) + 0.3:
                break
            if now_s >= deadline:
                break
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()

    if latest is None:
        raise RuntimeError(f"在 {timeout_s:.1f}s 内没有收到 /RobotInfo")
    if len(latest["joint_positions_rad"]) != 6:
        raise RuntimeError("RobotInfo.joint_positions 不是 6 轴数据")
    return latest


def _interpolate_pose_path(start: np.ndarray, target: np.ndarray, *, translation_step_m: float, rotation_step_deg: float) -> list[np.ndarray]:
    translation_delta = target[:3, 3] - start[:3, 3]
    relative_rotation = start[:3, :3].T @ target[:3, :3]
    rotation_vector = rotation_matrix_to_vector(relative_rotation)
    rotation_angle = float(np.linalg.norm(rotation_vector))
    count_translation = int(math.ceil(float(np.linalg.norm(translation_delta)) / translation_step_m)) if translation_step_m > 0 else 0
    count_rotation = int(math.ceil(math.degrees(rotation_angle) / rotation_step_deg)) if rotation_step_deg > 0 else 0
    count = max(1, count_translation, count_rotation)
    result: list[np.ndarray] = []
    for index in range(1, count + 1):
        fraction = index / count
        transform = np.eye(4)
        transform[:3, 3] = start[:3, 3] + fraction * translation_delta
        if rotation_angle <= 1e-12:
            transform[:3, :3] = start[:3, :3]
        else:
            transform[:3, :3] = start[:3, :3] @ axis_angle_matrix(
                rotation_vector / rotation_angle,
                fraction * rotation_angle,
            )
        result.append(transform)
    return result


def _solve_transform_path(
    start_q: np.ndarray,
    start_transform: np.ndarray,
    target_transform: np.ndarray,
    *,
    translation_step_m: float,
    rotation_step_deg: float,
) -> tuple[np.ndarray, dict[str, float]]:
    q = start_q.copy()
    max_step_deg = 0.0
    min_sigma = float("inf")
    max_position_error_m = 0.0
    max_rotation_error_rad = 0.0
    waypoints = _interpolate_pose_path(
        start_transform,
        target_transform,
        translation_step_m=translation_step_m,
        rotation_step_deg=rotation_step_deg,
    )
    for waypoint in waypoints:
        q_next = solve_pose_ik_multistart(
            q,
            waypoint[:3, 3],
            waypoint[:3, :3],
            max_iterations=1200,
        )
        max_step_deg = max(max_step_deg, float(np.degrees(np.max(np.abs(q_next - q)))))
        min_sigma = min(min_sigma, float(np.min(weighted_jacobian_singular_values(q_next))))
        q = q_next
    solved = controller_fk(q)
    max_position_error_m = float(np.linalg.norm(solved[:3, 3] - target_transform[:3, 3]))
    max_rotation_error_rad = rotation_angle_error(solved[:3, :3], target_transform[:3, :3])
    return q, {
        "pose_waypoint_count": float(len(waypoints)),
        "max_joint_step_deg": max_step_deg,
        "min_jacobian_singular_value": min_sigma,
        "endpoint_position_error_m": max_position_error_m,
        "endpoint_rotation_error_rad": max_rotation_error_rad,
    }


def _visibility(
    base_flange: np.ndarray,
    base_board: np.ndarray,
    flange_camera: np.ndarray,
    board_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
) -> VisibilityMetrics:
    base_camera = base_flange @ flange_camera
    camera_board = np.linalg.inv(base_camera) @ base_board
    if camera_board[2, 3] <= 1e-6:
        return VisibilityMetrics(
            min_margin_px=-1e9,
            left_margin_px=-1e9,
            right_margin_px=-1e9,
            top_margin_px=-1e9,
            bottom_margin_px=-1e9,
            center_px=(float("nan"), float("nan")),
            all_corners_visible=False,
        )
    rotation_vector, _ = cv2.Rodrigues(camera_board[:3, :3])
    projected, _ = cv2.projectPoints(
        board_points,
        rotation_vector,
        camera_board[:3, 3].reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )
    points = projected.reshape(-1, 2)
    if not np.all(np.isfinite(points)):
        raise RuntimeError("标定板投影出现非有限值")
    width, height = image_size
    left = float(np.min(points[:, 0]))
    right = float(width - 1 - np.max(points[:, 0]))
    top = float(np.min(points[:, 1]))
    bottom = float(height - 1 - np.max(points[:, 1]))
    center = np.mean(points, axis=0)
    margins = (left, right, top, bottom)
    return VisibilityMetrics(
        min_margin_px=float(min(margins)),
        left_margin_px=left,
        right_margin_px=right,
        top_margin_px=top,
        bottom_margin_px=bottom,
        center_px=(float(center[0]), float(center[1])),
        all_corners_visible=bool(
            left >= 0.0 and right >= 0.0 and top >= 0.0 and bottom >= 0.0
        ),
    )


def _joint_space_path(start: np.ndarray, target: np.ndarray, max_step_deg: float) -> np.ndarray:
    largest = float(np.max(np.abs(target - start)))
    count = max(1, int(math.ceil(math.degrees(largest) / max_step_deg)))
    fractions = np.linspace(0.0, 1.0, count + 1)[1:]
    return np.asarray([start + fraction * (target - start) for fraction in fractions])


def _validate_joint_path(
    start_q: np.ndarray,
    points: np.ndarray,
    *,
    start_transform: np.ndarray,
    base_board: np.ndarray,
    flange_camera: np.ndarray,
    board_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    min_singular_value: float,
    visibility_margin_px: float,
    joint_limit_margin_deg: float,
    max_flange_displacement_m: float,
    route_joint_step_deg: float,
) -> dict[str, Any]:
    margin_rad = math.radians(joint_limit_margin_deg)
    min_sigma = float("inf")
    min_visibility = float("inf")
    max_step = 0.0
    max_flange_displacement = 0.0
    previous = start_q
    for point in points:
        if not joints_within_limits(point, margin_rad):
            return {
                "valid": False,
                "reason": "joint limit margin exceeded",
                "point_deg": np.degrees(point).tolist(),
            }
        max_step = max(max_step, float(np.degrees(np.max(np.abs(point - previous)))))
        transform = controller_fk(point)
        max_flange_displacement = max(
            max_flange_displacement,
            float(np.linalg.norm(transform[:3, 3] - start_transform[:3, 3])),
        )
        sigma = float(np.min(weighted_jacobian_singular_values(point)))
        min_sigma = min(min_sigma, sigma)
        if sigma < min_singular_value:
            return {
                "valid": False,
                "reason": f"near-singular posture: min sigma={sigma:.6f}",
                "point_deg": np.degrees(point).tolist(),
                "min_jacobian_singular_value": min_sigma,
            }
        visibility = _visibility(
            transform,
            base_board,
            flange_camera,
            board_points,
            camera_matrix,
            dist_coeffs,
            image_size,
        )
        min_visibility = min(min_visibility, visibility.min_margin_px)
        if visibility.min_margin_px < visibility_margin_px:
            return {
                "valid": False,
                "reason": (
                    "predicted board visibility margin too small: "
                    f"{visibility.min_margin_px:.1f}px < {visibility_margin_px:.1f}px"
                ),
                "point_deg": np.degrees(point).tolist(),
                "visibility": asdict(visibility),
            }
        if max_flange_displacement > max_flange_displacement_m:
            return {
                "valid": False,
                "reason": (
                    "flange displacement too large: "
                    f"{max_flange_displacement * 1000:.1f}mm > "
                    f"{max_flange_displacement_m * 1000:.1f}mm"
                ),
                "point_deg": np.degrees(point).tolist(),
            }
        previous = point
    return {
        "valid": True,
        "point_count": int(len(points)),
        "max_joint_step_deg": max_step,
        "min_jacobian_singular_value": min_sigma,
        "min_board_visibility_margin_px": min_visibility,
        "max_flange_displacement_m": max_flange_displacement,
        "configured_route_joint_step_deg": route_joint_step_deg,
    }


def _candidate_rows(
    start_q: np.ndarray,
    start_transform: np.ndarray,
    *,
    base_board: np.ndarray,
    flange_camera: np.ndarray,
    board_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    min_singular_value: float,
    visibility_margin_px: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(CANDIDATE_SPECS, start=1):
        target = _target_transform(start_transform, spec)
        try:
            target_q, solve_metrics = _solve_transform_path(
                start_q,
                start_transform,
                target,
                translation_step_m=0.002,
                rotation_step_deg=1.0,
            )
        except RuntimeError as exc:
            rows.append(
                {
                    "id": f"cal_{index:02d}_{spec.name}",
                    "name": spec.name,
                    "description": spec.description,
                    "relative_rotation_deg": [spec.rx_deg, spec.ry_deg, spec.rz_deg],
                    "valid": False,
                    "rejection_reason": str(exc),
                }
            )
            continue
        solved = controller_fk(target_q)
        sigma = float(np.min(weighted_jacobian_singular_values(target_q)))
        visibility = _visibility(
            solved,
            base_board,
            flange_camera,
            board_points,
            camera_matrix,
            dist_coeffs,
            image_size,
        )
        joint_delta_deg = np.degrees(target_q - start_q)
        valid = True
        reasons: list[str] = []
        if not joints_within_limits(target_q, math.radians(2.0)):
            valid = False
            reasons.append("关节限位余量不足 2°")
        if sigma < min_singular_value:
            valid = False
            reasons.append(
                f"最小加权 Jacobian 奇异值 {sigma:.6f} < {min_singular_value:.6f}"
            )
        if visibility.min_margin_px < visibility_margin_px:
            valid = False
            reasons.append(
                "预计标定板边距 "
                f"{visibility.min_margin_px:.1f}px < {visibility_margin_px:.1f}px"
            )
        if solve_metrics["endpoint_position_error_m"] > 0.0004:
            valid = False
            reasons.append("终点位置误差超过 0.4mm")
        if solve_metrics["endpoint_rotation_error_rad"] > 0.0004:
            valid = False
            reasons.append("终点姿态误差超过 0.4mrad")
        rows.append(
            {
                "id": f"cal_{index:02d}_{spec.name}",
                "name": spec.name,
                "description": spec.description,
                "relative_rotation_deg": [spec.rx_deg, spec.ry_deg, spec.rz_deg],
                "valid": valid,
                "rejection_reason": "；".join(reasons),
                "target_transform_base_flange": target.tolist(),
                "target_joints_rad": target_q.tolist(),
                "target_joints_deg": np.degrees(target_q).tolist(),
                "target_xyz_m": solved[:3, 3].tolist(),
                "target_rpy_rad": list(matrix_to_rpy(solved[:3, :3])),
                "target_rpy_deg": np.degrees(matrix_to_rpy(solved[:3, :3])).tolist(),
                "joint_delta_from_start_deg": joint_delta_deg.tolist(),
                "max_joint_delta_from_start_deg": float(np.max(np.abs(joint_delta_deg))),
                "min_jacobian_singular_value": sigma,
                "predicted_visibility": asdict(visibility),
                "start_to_target_solve": solve_metrics,
            }
        )
    return rows


def _plan_route(
    start_q: np.ndarray,
    start_transform: np.ndarray,
    candidates: Sequence[dict[str, Any]],
    *,
    base_board: np.ndarray,
    flange_camera: np.ndarray,
    board_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    min_singular_value: float,
    visibility_margin_px: float,
    route_joint_step_deg: float,
    joint_limit_margin_deg: float,
    max_flange_displacement_m: float,
    speed_rad_s: float,
    settle_s: float,
) -> dict[str, Any]:
    remaining = {row["id"]: row for row in candidates if row["valid"]}
    current_id = "start"
    current_q = start_q.copy()
    points: list[np.ndarray] = [start_q.copy()]
    targets: list[dict[str, Any]] = []
    edge_metrics: list[dict[str, Any]] = []

    while remaining:
        ordered = sorted(
            remaining.values(),
            key=lambda row: float(
                np.max(np.abs(np.asarray(row["target_joints_rad"]) - current_q))
            ),
        )
        selected: dict[str, Any] | None = None
        selected_points: np.ndarray | None = None
        selected_metrics: dict[str, Any] | None = None
        rejection_notes: list[str] = []
        for candidate in ordered[: min(8, len(ordered))]:
            target_q = np.asarray(candidate["target_joints_rad"], dtype=float)
            proposed = _joint_space_path(current_q, target_q, route_joint_step_deg)
            metrics = _validate_joint_path(
                current_q,
                proposed,
                start_transform=start_transform,
                base_board=base_board,
                flange_camera=flange_camera,
                board_points=board_points,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_size=image_size,
                min_singular_value=min_singular_value,
                visibility_margin_px=visibility_margin_px,
                joint_limit_margin_deg=joint_limit_margin_deg,
                max_flange_displacement_m=max_flange_displacement_m,
                route_joint_step_deg=route_joint_step_deg,
            )
            if metrics["valid"]:
                selected = candidate
                selected_points = proposed
                selected_metrics = metrics
                break
            rejection_notes.append(f"{candidate['id']}: {metrics.get('reason', 'invalid')}")
        if selected is None or selected_points is None or selected_metrics is None:
            return {
                "valid": False,
                "reason": "greedy route could not find a safe next segment after " + current_id,
                "rejections": rejection_notes,
                "targets": targets,
                "edge_metrics": edge_metrics,
            }
        q_before = current_q.copy()
        current_q = np.asarray(selected["target_joints_rad"], dtype=float)
        points.extend(selected_points)
        current_id = str(selected["id"])
        max_delta_rad = float(np.max(np.abs(current_q - q_before)))
        metrics = dict(selected_metrics)
        metrics.update(
            {
                "from": targets[-1]["id"] if targets else "start",
                "to": current_id,
                "travel_s_at_requested_speed": max_delta_rad / speed_rad_s,
            }
        )
        targets.append(
            {
                "id": current_id,
                "description": selected["description"],
                "target_joints_deg": selected["target_joints_deg"],
                "target_xyz_m": selected["target_xyz_m"],
                "target_rpy_deg": selected["target_rpy_deg"],
            }
        )
        edge_metrics.append(metrics)
        del remaining[current_id]

    # Return to the initial posture so the camera ends where it started.
    return_points = _joint_space_path(current_q, start_q, route_joint_step_deg)
    return_metrics = _validate_joint_path(
        current_q,
        return_points,
        start_transform=start_transform,
        base_board=base_board,
        flange_camera=flange_camera,
        board_points=board_points,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        min_singular_value=min_singular_value,
        visibility_margin_px=visibility_margin_px,
        joint_limit_margin_deg=joint_limit_margin_deg,
        max_flange_displacement_m=max_flange_displacement_m,
        route_joint_step_deg=route_joint_step_deg,
    )
    if not return_metrics["valid"]:
        return {
            "valid": False,
            "reason": "safe return-to-start path failed: " + str(return_metrics.get("reason")),
            "targets": targets,
            "edge_metrics": edge_metrics,
        }
    points.extend(return_points)
    travel_s = max(float(np.max(np.abs(start_q - current_q))) / speed_rad_s, 0.0)
    return_metrics = dict(return_metrics)
    return_metrics.update(
        {
            "from": targets[-1]["id"] if targets else "start",
            "to": "start",
            "travel_s_at_requested_speed": travel_s,
        }
    )
    edge_metrics.append(return_metrics)
    route_array = np.asarray(points)
    total_travel_s = float(sum(item["travel_s_at_requested_speed"] for item in edge_metrics))
    capture_dwell_s = float(len(targets) + 1) * settle_s
    return {
        "valid": True,
        "targets": targets,
        "edge_metrics": edge_metrics,
        "joint_points_rad": route_array.tolist(),
        "joint_points_deg": np.degrees(route_array).tolist(),
        "joint_point_count": int(len(route_array)),
        "estimated_travel_s": total_travel_s,
        "estimated_capture_dwell_s": capture_dwell_s,
        "estimated_total_s": total_travel_s + capture_dwell_s,
        "max_joint_step_deg": float(
            max(
                np.degrees(np.max(np.abs(np.diff(route_array, axis=0)))) if len(route_array) > 1 else 0.0,
                0.0,
            )
        ),
        "min_jacobian_singular_value": float(
            min(item["min_jacobian_singular_value"] for item in edge_metrics)
        ),
        "min_board_visibility_margin_px": float(
            min(item["min_board_visibility_margin_px"] for item in edge_metrics)
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observation",
        type=Path,
        default=ROOT / "logs/hand_eye_calibration_20261025/board_precheck_20261025/detection_20261025_131746.json",
    )
    parser.add_argument(
        "--board-config",
        type=Path,
        default=ROOT / "calibration/hand_eye_20261025/chessboard.yaml",
    )
    parser.add_argument(
        "--extrinsics",
        type=Path,
        default=ROOT / "config/camera_to_flange.d405_assumed.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs/hand_eye_calibration_20261025/pose_plan",
    )
    parser.add_argument("--robot-timeout-s", type=float, default=5.0)
    parser.add_argument("--speed-rad-s", type=float, default=0.25)
    parser.add_argument("--settle-s", type=float, default=1.2)
    parser.add_argument("--route-joint-step-deg", type=float, default=0.25)
    parser.add_argument("--min-singular-value", type=float, default=0.015)
    parser.add_argument("--visibility-margin-px", type=float, default=30.0)
    parser.add_argument("--joint-limit-margin-deg", type=float, default=2.0)
    parser.add_argument("--max-flange-displacement-m", type=float, default=0.08)
    args = parser.parse_args()

    if not (0.0 < args.speed_rad_s <= 0.30):
        raise RuntimeError("标定速度必须大于 0 且不超过 0.30 rad/s")
    if not (0.05 <= args.route_joint_step_deg <= 1.0):
        raise RuntimeError("route-joint-step-deg 必须在 0.05 到 1.0 之间")

    observation_path = args.observation.expanduser().resolve()
    board_path = args.board_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    board_config = _load_yaml(board_path)
    board_columns, board_rows = (int(value) for value in board_config["opencv_inner_corner_pattern"])
    square_grid = [int(value) for value in board_config["printed_square_grid"]]
    square_size_m = float(board_config["square_size_mm"]) / 1000.0
    camera_matrix = np.asarray(board_config["camera_matrix"], dtype=float)
    dist_coeffs = np.asarray(board_config["distortion_coefficients"], dtype=float).reshape(-1)
    image_size = (int(board_config["camera_width"]), int(board_config["camera_height"]))
    detection = json.loads(observation_path.read_text(encoding="utf-8"))
    observation_board = detection.get("board", {})
    if [int(observation_board.get("pattern_columns", -1)), int(observation_board.get("pattern_rows", -1))] != [board_columns, board_rows]:
        raise RuntimeError("检测文件与棋盘内角点配置不一致")
    if int(observation_board.get("square_columns", -1)) != square_grid[0] or int(observation_board.get("square_rows", -1)) != square_grid[1]:
        raise RuntimeError("检测文件与打印方格数量不一致")
    if abs(float(observation_board.get("square_size_mm", -1.0)) - square_size_m * 1000.0) > 1e-6:
        raise RuntimeError("检测文件与方格尺寸配置不一致")
    if int(detection.get("detected", 0)) != int(detection.get("frame_count", -1)):
        raise RuntimeError("标定板检测不是全部成功")
    if float(detection["reprojection_rms_px"]["max"]) > float(board_config["reprojection_rms_limit_px"]):
        raise RuntimeError("标定板重投影误差超过限制")

    camera_board = _average_transform(detection["observations"])
    robot_state = _read_robot_state(args.robot_timeout_s)
    if robot_state["state"] != "AutoReady" or robot_state["fault_flag"] != 0:
        raise RuntimeError(
            f"机械臂当前不满足只读规划前检查: state={robot_state['state']} fault={robot_state['fault_flag']}"
        )
    start_q = np.asarray(robot_state["joint_positions_rad"], dtype=float)
    start_transform = controller_fk(start_q)
    if robot_state["end_positions"] and len(robot_state["end_positions"]) >= 6:
        reported_position = np.asarray(robot_state["end_positions"][:3], dtype=float)
        fk_position = start_transform[:3, 3]
        fk_error_m = float(np.linalg.norm(reported_position - fk_position))
        robot_state["fk_position_error_m"] = fk_error_m
        robot_state["fk_position_check"] = "pass" if fk_error_m <= 0.002 else "fail"
        if fk_error_m > 0.002:
            raise RuntimeError(
                f"控制器末端位置与 FK 差 {fk_error_m * 1000:.2f} mm，超过 2 mm"
            )

    extrinsics_payload = _load_yaml(args.extrinsics.expanduser().resolve())
    flange_camera = np.asarray(extrinsics_payload["matrix"], dtype=float)
    if flange_camera.shape != (4, 4):
        raise RuntimeError("外参 matrix 不是 4x4")
    base_board = start_transform @ flange_camera @ camera_board
    board_points = _object_points(board_columns, board_rows, square_size_m)

    start_visibility = _visibility(
        start_transform,
        base_board,
        flange_camera,
        board_points,
        camera_matrix,
        dist_coeffs,
        image_size,
    )
    start_sigma = float(np.min(weighted_jacobian_singular_values(start_q)))
    candidates = _candidate_rows(
        start_q,
        start_transform,
        base_board=base_board,
        flange_camera=flange_camera,
        board_points=board_points,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        min_singular_value=args.min_singular_value,
        visibility_margin_px=args.visibility_margin_px,
    )
    route = _plan_route(
        start_q,
        start_transform,
        candidates,
        base_board=base_board,
        flange_camera=flange_camera,
        board_points=board_points,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        min_singular_value=args.min_singular_value,
        visibility_margin_px=args.visibility_margin_px,
        route_joint_step_deg=args.route_joint_step_deg,
        joint_limit_margin_deg=args.joint_limit_margin_deg,
        max_flange_displacement_m=args.max_flange_displacement_m,
        speed_rad_s=args.speed_rad_s,
        settle_s=args.settle_s,
    )
    valid_candidates = [row for row in candidates if row["valid"]]
    all_candidates_valid = len(valid_candidates) == len(candidates)
    report_valid = bool(
        all_candidates_valid
        and route.get("valid")
        and start_sigma >= args.min_singular_value
        and start_visibility.min_margin_px >= args.visibility_margin_px
    )
    stamp = datetime.now().astimezone()
    payload = {
        "created_at": stamp.isoformat(timespec="seconds"),
        "mode": "read_only_pose_plan",
        "physical_motion_command_published": False,
        "requires_explicit_operator_approval_before_motion": True,
        "eye_in_hand_transform_convention": "T_flange_camera",
        "inputs": {
            "observation": str(observation_path),
            "board_config": str(board_path),
            "extrinsics_nominal_only": str(args.extrinsics.expanduser().resolve()),
        },
        "board": {
            "printed_square_grid": square_grid,
            "opencv_inner_corner_pattern": [board_columns, board_rows],
            "square_size_mm": square_size_m * 1000.0,
            "detection_frames": detection.get("frame_count"),
            "detection_rms_mean_px": detection["reprojection_rms_px"]["mean"],
            "detection_rms_max_px": detection["reprojection_rms_px"]["max"],
            "transform_camera_board": camera_board.tolist(),
        },
        "robot_start": {
            **robot_state,
            "joint_positions_deg": np.degrees(start_q).tolist(),
            "fk_transform_base_flange": start_transform.tolist(),
            "fk_xyz_m": start_transform[:3, 3].tolist(),
            "fk_rpy_deg": np.degrees(matrix_to_rpy(start_transform[:3, :3])).tolist(),
            "min_jacobian_singular_value": start_sigma,
            "predicted_visibility": asdict(start_visibility),
        },
        "safety_thresholds": {
            "calibration_joint_speed_rad_s": args.speed_rad_s,
            "route_joint_step_deg": args.route_joint_step_deg,
            "min_jacobian_singular_value": args.min_singular_value,
            "min_board_visibility_margin_px": args.visibility_margin_px,
            "joint_limit_margin_deg": args.joint_limit_margin_deg,
            "max_flange_displacement_m": args.max_flange_displacement_m,
            "settle_s_per_capture": args.settle_s,
        },
        "extrinsics_note": (
            "visibility planning used the nominal translation only; "
            f"rotation validated={bool(extrinsics_payload.get('validated', False))}"
        ),
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid_candidates),
        "candidates": candidates,
        "route": route,
        "ready_for_live_capture_review": report_valid,
    }
    json_path = output_dir / f"pose_plan_{stamp.strftime('%Y%m%d_%H%M%S')}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    markdown_path = json_path.with_suffix(".md")
    lines = [
        "# 手眼标定姿态计划（只读）",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 机械臂状态：{robot_state['state']}，fault_flag={robot_state['fault_flag']}",
        f"- 棋盘：{square_grid[0]}×{square_grid[1]} 方格，内角点 {board_columns}×{board_rows}，方格 {square_size_m * 1000:.1f} mm",
        f"- 检测：{detection.get('detected')} 帧，RMS 平均 {detection['reprojection_rms_px']['mean']:.3f}px，最大 {detection['reprojection_rms_px']['max']:.3f}px",
        f"- 候选姿态：{len(valid_candidates)}/{len(candidates)} 通过",
        f"- 计划标定速度：{args.speed_rad_s:.3f} rad/s",
        f"- 预计运动与采集时间：{route.get('estimated_total_s', 0.0):.1f}s（含每点 {args.settle_s:.1f}s 停稳采集）",
        f"- 实机运动命令：未发布",
        f"- 是否可进入实机采集审核：{'是' if report_valid else '否'}",
        "",
        "## 起始状态",
        "",
        f"- XYZ：{np.round(start_transform[:3, 3], 6).tolist()} m",
        f"- RPY：{np.round(np.degrees(matrix_to_rpy(start_transform[:3, :3])), 3).tolist()}°",
        f"- 关节：{np.round(np.degrees(start_q), 3).tolist()}°",
        f"- 最小加权 Jacobian 奇异值：{start_sigma:.6f}",
        f"- 标定板预计最小边距：{start_visibility.min_margin_px:.1f}px",
        "",
        "## 候选姿态",
        "",
        "| # | 姿态 | 方向 | 最大关节变化(°) | 最小sigma | 板边距(px) | 结果 |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for index, row in enumerate(candidates, start=1):
        if row.get("valid"):
            lines.append(
                f"| {index} | {row['id']} | {row['description']} | "
                f"{row['max_joint_delta_from_start_deg']:.2f} | "
                f"{row['min_jacobian_singular_value']:.4f} | "
                f"{row['predicted_visibility']['min_margin_px']:.1f} | 通过 |"
            )
        else:
            lines.append(
                f"| {index} | {row['id']} | {row['description']} | - | - | - | 拒绝：{row.get('rejection_reason', '')} |"
            )
    lines += [
        "",
        "## 推荐采集顺序",
        "",
        "| 顺序 | 目标 | 说明 | 预计运动(s) | 段内最小sigma | 段内最小板边距(px) |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for index, (target, metrics) in enumerate(zip(route.get("targets", []), route.get("edge_metrics", [])), start=1):
        lines.append(
            f"| {index} | {target['id']} | {target['description']} | "
            f"{metrics.get('travel_s_at_requested_speed', 0.0):.2f} | "
            f"{metrics.get('min_jacobian_singular_value', float('nan')):.4f} | "
            f"{metrics.get('min_board_visibility_margin_px', float('nan')):.1f} |"
        )
    if route.get("valid"):
        lines += [
            "",
            f"完整采集路线共 {len(route['targets']) + 1} 个采集状态（含起始），"
            f"{route['joint_point_count']} 个关节轨迹点，最后返回起始姿态。",
            "",
            f"- 路线最大关节点间步长：{route['max_joint_step_deg']:.3f}°",
            f"- 路线最小sigma：{route['min_jacobian_singular_value']:.4f}",
            f"- 路线最小预计板边距：{route['min_board_visibility_margin_px']:.1f}px",
        ]
    else:
        lines += ["", f"路线规划失败：{route.get('reason', 'unknown')}"]
    lines += [
        "",
        "> 本报告仅用于审核。未向 `/RobotMove` 发布任何命令，标定板必须保持固定。",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(
        {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "ready_for_live_capture_review": report_valid,
            "candidates_valid": f"{len(valid_candidates)}/{len(candidates)}",
            "route_valid": bool(route.get("valid")),
            "capture_states": len(route.get("targets", [])) + 1,
            "route_joint_points": route.get("joint_point_count"),
            "estimated_total_s": route.get("estimated_total_s"),
            "min_jacobian_singular_value": route.get("min_jacobian_singular_value"),
            "min_board_visibility_margin_px": route.get("min_board_visibility_margin_px"),
            "physical_motion_command_published": False,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if report_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
