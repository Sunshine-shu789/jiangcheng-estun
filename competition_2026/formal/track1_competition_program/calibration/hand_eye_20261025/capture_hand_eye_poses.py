#!/usr/bin/env python3
"""Execute an approved eye-in-hand pose plan and capture calibration frames.

Physical motion requires the explicit ``--execute`` flag.  The tool refuses to
move unless the plan is complete, freshly pre-checked and the current robot
pose matches the planned start.  The gripper is never commanded.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from itertools import combinations
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Sequence
from urllib.request import Request, urlopen

import cv2
import numpy as np
import yaml

from track1_grasp.config import MotionConfig
from track1_grasp.kinematics import (
    joints_within_limits,
    matrix_to_rpy,
    rotation_angle_error,
    weighted_jacobian_singular_values,
)
from track1_grasp.motion import RobotMoveMotionBackend

from chessboard_detector import BoardSpec, estimate_board_pose, render_overlay


ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{stamp}] {message}", flush=True)


def fetch_snapshot(url: str, timeout_s: float = 5.0) -> np.ndarray:
    request = Request(url, headers={"Cache-Control": "no-store"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = response.read()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"快照接口返回的图像无法解码: {url}")
    return image


def average_transform(observations: Sequence[Any]) -> np.ndarray:
    transforms = [observation.transform_camera_board for observation in observations]
    translation = np.mean([transform[:3, 3] for transform in transforms], axis=0)
    rotation_sum = np.sum([transform[:3, :3] for transform in transforms], axis=0)
    u, _, vt = np.linalg.svd(rotation_sum)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def load_board_config(path: Path) -> tuple[BoardSpec, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise RuntimeError("棋盘配置不是 YAML 字典")
    columns, rows = (int(value) for value in config["opencv_inner_corner_pattern"])
    grid = [int(value) for value in config["printed_square_grid"]]
    if grid != [columns + 1, rows + 1]:
        raise RuntimeError("打印方格数量与 OpenCV 内角点数量不一致")
    board = BoardSpec(
        columns=columns,
        rows=rows,
        square_size_m=float(config["square_size_mm"]) / 1000.0,
    )
    return board, config


def capture_frames(
    *,
    snapshot_url: str,
    board: BoardSpec,
    rms_limit_px: float,
    output_dir: Path,
    label: str,
    frame_count: int,
    interval_s: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    max_attempts: int = 12,
    max_translation_spread_mm: float = 1.0,
    max_rotation_spread_deg: float = 0.5,
) -> tuple[np.ndarray, list[Any]]:
    """Capture several usable observations and keep the best static frames.

    The live preview service occasionally drops a chessboard detection or has a
    short-lived corner-localization outlier even while the arm is stationary.
    Failed attempts are retained for audit, but do not invalidate the pose when
    enough independently verified frames pass the unchanged RMS limit.  Frames
    from one static pose must also agree with each other before they are used.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    if frame_count < 1 or max_attempts < frame_count:
        raise ValueError("max_attempts must be at least frame_count")
    valid_observations: list[Any] = []
    attempts: list[dict[str, Any]] = []
    selected: list[Any] | None = None
    for attempt_index in range(max_attempts):
        image = fetch_snapshot(snapshot_url)
        raw_path = output_dir / f"{label}_f{attempt_index:02d}.jpg"
        if not cv2.imwrite(str(raw_path), image):
            raise RuntimeError(f"无法保存快照: {raw_path}")
        attempt: dict[str, Any] = {
            "attempt": attempt_index,
            "image_path": str(raw_path),
        }
        try:
            observation = estimate_board_pose(
                image,
                board,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_path=str(raw_path),
            )
        except RuntimeError as exc:
            attempt.update({"valid": False, "error": str(exc)})
            attempts.append(attempt)
            if attempt_index + 1 < max_attempts:
                time.sleep(interval_s)
            continue

        overlay = render_overlay(image, observation, board)
        overlay_path = output_dir / f"{label}_f{attempt_index:02d}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        attempt.update(
            {
                "overlay_path": str(overlay_path),
                "reprojection_rms_px": observation.reprojection_rms_px,
                "corner_refinement_used": observation.corner_refinement_used,
                "tvec_m": observation.tvec.reshape(3).tolist(),
            }
        )
        if observation.reprojection_rms_px <= rms_limit_px:
            attempt["valid"] = True
            valid_observations.append(observation)
        else:
            attempt.update(
                {
                    "valid": False,
                    "error": (
                        f"RMS {observation.reprojection_rms_px:.3f}px > "
                        f"{rms_limit_px:.3f}px"
                    ),
                }
            )
        attempts.append(attempt)
        if len(valid_observations) >= frame_count:
            best_consistency: tuple[float, float, float] | None = None
            for candidate in combinations(valid_observations, frame_count):
                translations = np.asarray(
                    [observation.tvec.reshape(3) for observation in candidate]
                )
                median_translation = np.median(translations, axis=0)
                translation_spread_mm = float(
                    np.max(
                        np.linalg.norm(
                            translations - median_translation, axis=1
                        )
                    )
                    * 1000.0
                )
                rotations = [
                    observation.rotation_matrix for observation in candidate
                ]
                rotation_sum = np.sum(rotations, axis=0)
                u, _, vt = np.linalg.svd(rotation_sum)
                mean_rotation = u @ vt
                if np.linalg.det(mean_rotation) < 0.0:
                    u[:, -1] *= -1.0
                    mean_rotation = u @ vt
                rotation_spread_deg = float(
                    max(
                        np.degrees(
                            rotation_angle_error(rotation, mean_rotation)
                        )
                        for rotation in rotations
                    )
                )
                mean_rms = float(
                    np.mean(
                        [
                            observation.reprojection_rms_px
                            for observation in candidate
                        ]
                    )
                )
                if (
                    translation_spread_mm <= max_translation_spread_mm
                    and rotation_spread_deg <= max_rotation_spread_deg
                    and (
                        best_consistency is None
                        or (mean_rms, translation_spread_mm, rotation_spread_deg)
                        < best_consistency
                    )
                ):
                    best_consistency = (
                        mean_rms,
                        translation_spread_mm,
                        rotation_spread_deg,
                    )
                    selected = list(candidate)
            if selected is not None:
                break
        if attempt_index + 1 < max_attempts:
            time.sleep(interval_s)

    quality_path = output_dir / f"{label}_attempts.json"
    quality_path.write_text(
        json.dumps(attempts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if len(valid_observations) < frame_count:
        reasons = [
            f"第{item['attempt']}次: {item.get('error', '未通过')}"
            for item in attempts
            if not item.get("valid", False)
        ]
        raise RuntimeError(
            f"{label}: 在 {len(attempts)} 次尝试中仅获得 "
            f"{len(valid_observations)}/{frame_count} 个合格帧"
            + ("；" + "；".join(reasons) if reasons else "")
        )
    if selected is None:
        raise RuntimeError(
            f"{label}: 获得 {len(valid_observations)} 个合格帧，但找不到 "
            f"{frame_count} 帧同时满足平移差 <= "
            f"{max_translation_spread_mm:.3f}mm 和旋转差 <= "
            f"{max_rotation_spread_deg:.3f}° 的静止帧组"
        )
    return average_transform(selected), selected


def read_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "read_only_pose_plan":
        raise RuntimeError("计划文件模式不正确")
    if payload.get("physical_motion_command_published") is not False:
        raise RuntimeError("计划文件不是未执行状态")
    if payload.get("ready_for_live_capture_review") is not True:
        raise RuntimeError("计划未通过实机采集审核")
    route = payload.get("route", {})
    if route.get("valid") is not True:
        raise RuntimeError("计划路线无效")
    if len(payload.get("candidates", [])) != payload.get("candidate_count"):
        raise RuntimeError("候选数量不一致")
    if payload.get("valid_candidate_count") != payload.get("candidate_count"):
        raise RuntimeError("存在未通过审核的候选姿态")
    return payload


def validate_plan_for_execution(plan: dict[str, Any]) -> dict[str, Any]:
    thresholds = plan["safety_thresholds"]
    speed = float(thresholds["calibration_joint_speed_rad_s"])
    route_step_deg = float(thresholds["route_joint_step_deg"])
    sigma_limit = float(thresholds["min_jacobian_singular_value"])
    limit_margin_deg = float(thresholds["joint_limit_margin_deg"])
    if not (0.0 < speed <= 0.30):
        raise RuntimeError(f"计划速度不安全: {speed:.3f} rad/s")
    if not (0.05 <= route_step_deg <= 1.0):
        raise RuntimeError(f"计划关节点间步长不合理: {route_step_deg:.3f}°")
    route = plan["route"]
    points = np.asarray(route["joint_points_rad"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 6 or len(points) < 2:
        raise RuntimeError("路线关节数组形状无效")
    if not np.all(np.isfinite(points)):
        raise RuntimeError("路线包含非有限关节值")
    if route.get("joint_point_count") != len(points):
        raise RuntimeError("路线关节点点数不一致")
    delta_deg = np.degrees(np.diff(points, axis=0))
    max_route_step = float(np.max(np.abs(delta_deg)))
    if max_route_step > route_step_deg + 0.01:
        raise RuntimeError(
            f"路线关节点间步长 {max_route_step:.4f}° 超过 {route_step_deg:.4f}°"
        )
    margin_rad = math.radians(limit_margin_deg)
    min_sigma = float("inf")
    for index, point in enumerate(points):
        if not joints_within_limits(point, margin_rad):
            raise RuntimeError(f"路线第 {index + 1} 点违反关节限位余量")
        minimum = float(np.min(weighted_jacobian_singular_values(point)))
        min_sigma = min(min_sigma, minimum)
        if minimum < sigma_limit:
            raise RuntimeError(
                f"路线第 {index + 1} 点最小 sigma={minimum:.6f} < {sigma_limit:.6f}"
            )
    edge_metrics = route.get("edge_metrics", [])
    targets = route.get("targets", [])
    if len(edge_metrics) != len(targets) + 1:
        raise RuntimeError("路线边数与目标数不一致（应包含回程）")
    if sum(int(edge["point_count"]) for edge in edge_metrics) != len(points) - 1:
        raise RuntimeError("路线各段点数与总点数不一致")
    start_q = np.asarray(plan["robot_start"]["joint_positions_rad"], dtype=float)
    if not np.allclose(points[0], start_q, atol=1e-10, rtol=0.0):
        raise RuntimeError("路线起点与计划机器人起点不一致")
    if not np.allclose(points[-1], start_q, atol=1e-7, rtol=0.0):
        raise RuntimeError("路线终点没有返回计划起点")
    return {
        "speed_rad_s": speed,
        "route_step_deg": route_step_deg,
        "sigma_limit": sigma_limit,
        "min_sigma": min_sigma,
        "max_route_step_deg": max_route_step,
        "point_count": len(points),
        "segment_point_counts": [int(edge["point_count"]) for edge in edge_metrics],
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist() if isinstance(value, np.ndarray) else None)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot-url", default="http://127.0.0.1:9100/snapshot.jpg")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frames-per-pose", type=int, default=3)
    parser.add_argument("--max-frame-attempts", type=int, default=12)
    parser.add_argument("--frame-interval-s", type=float, default=0.20)
    parser.add_argument("--settle-s", type=float, default=None)
    parser.add_argument("--board-position-tolerance-mm", type=float, default=10.0)
    parser.add_argument("--board-rotation-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--start-joint-tolerance-deg", type=float, default=0.50)
    args = parser.parse_args()
    if not args.execute:
        raise RuntimeError("实机采集必须显式提供 --execute")
    if args.frames_per_pose < 1:
        raise RuntimeError("--frames-per-pose 必须至少为 1")
    if args.max_frame_attempts < args.frames_per_pose:
        raise RuntimeError("--max-frame-attempts 不得少于 --frames-per-pose")

    plan_path = args.plan.expanduser().resolve()
    plan = read_plan(plan_path)
    execution_metrics = validate_plan_for_execution(plan)
    thresholds = plan["safety_thresholds"]
    settle_s = float(thresholds["settle_s_per_capture"] if args.settle_s is None else args.settle_s)
    if settle_s < 0.5:
        raise RuntimeError("每个姿态停稳时间不得少于 0.5s")

    board, board_config = load_board_config(ROOT / "calibration/hand_eye_20261025/chessboard.yaml")
    camera_matrix = np.asarray(board_config["camera_matrix"], dtype=float)
    dist_coeffs = np.asarray(board_config["distortion_coefficients"], dtype=float).reshape(-1)
    rms_limit = float(board_config["reprojection_rms_limit_px"])

    stamp = datetime.now().astimezone()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT / "logs/hand_eye_calibration_20261025" / f"capture_{stamp.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = output_dir / "frames"
    manifest: dict[str, Any] = {
        "created_at": stamp.isoformat(timespec="seconds"),
        "plan": str(plan_path),
        "snapshot_url": args.snapshot_url,
        "frames_per_pose": args.frames_per_pose,
        "settle_s": settle_s,
        "execution_validation": execution_metrics,
        "physical_motion_started": False,
        "completed": False,
        "pose_records": [],
    }
    manifest_path = output_dir / "capture_manifest.json"
    save_json(manifest_path, manifest)

    expected_start_board = np.asarray(plan["board"]["transform_camera_board"], dtype=float)
    expected_start_q = np.asarray(plan["robot_start"]["joint_positions_rad"], dtype=float)
    route_points = np.asarray(plan["route"]["joint_points_rad"], dtype=float)
    segment_counts = execution_metrics["segment_point_counts"]
    targets = plan["route"]["targets"]
    speed_rad_s = execution_metrics["speed_rad_s"]

    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init(args=[])
    node = rclpy.create_node("hand_eye_calibration_capture")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin,
        name="hand_eye_robot_feedback",
        daemon=True,
    )
    spin_thread.start()
    motion_config = MotionConfig(
        backend="robotmove",
        dry_run=False,
        joint_speed_rad_s=speed_rad_s,
        joint_speed_safety_limit_rad_s=0.30,
        move_timeout_s=60.0,
        joint_move_timeout_s=60.0,
        poll_period_s=0.05,
        joint_tolerance_deg=0.35,
        joint_start_tolerance_deg=0.05,
        joint_limit_margin_deg=float(thresholds["joint_limit_margin_deg"]),
        joint_reject_timeout_s=2.0,
    )
    backend = RobotMoveMotionBackend(motion_config, node=node, log=log)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if not spin_thread.is_alive():
                raise RuntimeError("ROS 反馈执行器意外停止")
            state = backend.get_state()
            if state.ready and backend._publisher.get_subscription_count() > 0:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                f"动作前检查失败: state={backend.get_state().message}, "
                f"RobotMove subscribers={backend._publisher.get_subscription_count()}"
            )
        if state.joints is None:
            raise RuntimeError("动作前 RobotInfo 缺少关节角")
        start_error_deg = float(np.degrees(np.max(np.abs(np.asarray(state.joints) - expected_start_q))))
        if start_error_deg > args.start_joint_tolerance_deg:
            raise RuntimeError(
                f"当前关节与计划起点相差 {start_error_deg:.3f}°，超过 {args.start_joint_tolerance_deg:.3f}°"
            )
        log(
            f"动作前检查通过: {state.controller_state}, fault_flag={state.fault_flag}, "
            f"起点误差={start_error_deg:.3f}°, RobotMove订阅={backend._publisher.get_subscription_count()}"
        )

        current_board, start_observations = capture_frames(
            snapshot_url=args.snapshot_url,
            board=board,
            rms_limit_px=rms_limit,
            output_dir=frames_dir,
            label="pose_00_start",
            frame_count=args.frames_per_pose,
            interval_s=args.frame_interval_s,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            max_attempts=args.max_frame_attempts,
        )
        start_translation_error_mm = float(
            np.linalg.norm(current_board[:3, 3] - expected_start_board[:3, 3]) * 1000.0
        )
        start_rotation_error_deg = math.degrees(
            rotation_angle_error(current_board[:3, :3], expected_start_board[:3, :3])
        )
        if start_translation_error_mm > args.board_position_tolerance_mm:
            raise RuntimeError(
                f"标定板位置相对计划变化 {start_translation_error_mm:.2f}mm，"
                f"超过 {args.board_position_tolerance_mm:.2f}mm"
            )
        if start_rotation_error_deg > args.board_rotation_tolerance_deg:
            raise RuntimeError(
                f"标定板姿态相对计划变化 {start_rotation_error_deg:.2f}°，"
                f"超过 {args.board_rotation_tolerance_deg:.2f}°"
            )
        log(
            f"起始标定板检查通过: 位置差={start_translation_error_mm:.2f}mm, "
            f"姿态差={start_rotation_error_deg:.2f}°, "
            f"RMS={max(obs.reprojection_rms_px for obs in start_observations):.3f}px"
        )
        begin_state = backend.wait_for_state(1.0)
        manifest["start_state"] = {
            "joint_positions_rad": list(begin_state.joints or ()),
            "controller_state": begin_state.controller_state,
            "fault_flag": begin_state.fault_flag,
            "joint_error_from_plan_deg": start_error_deg,
            "board_transform_camera_board": current_board.tolist(),
            "board_position_error_mm": start_translation_error_mm,
            "board_rotation_error_deg": start_rotation_error_deg,
        }
        manifest["pose_records"].append(
            {
                "index": 0,
                "id": "start",
                "target_joints_rad": expected_start_q.tolist(),
                "actual_joints_rad": list(begin_state.joints or ()),
                "max_joint_error_deg": start_error_deg,
                "board_transform_camera_board": current_board.tolist(),
                "observation_count": len(start_observations),
                "reprojection_rms_px": [obs.reprojection_rms_px for obs in start_observations],
            }
        )
        save_json(manifest_path, manifest)

        manifest["physical_motion_started"] = True
        save_json(manifest_path, manifest)
        point_offset = 0
        for target_index, target in enumerate(targets, start=1):
            segment_count = segment_counts[target_index - 1]
            segment = route_points[point_offset + 1 : point_offset + 1 + segment_count]
            point_offset += segment_count
            target_q = np.asarray(target["target_joints_deg"], dtype=float) * math.pi / 180.0
            if not np.allclose(segment[-1], target_q, atol=2e-7, rtol=0.0):
                raise RuntimeError(f"{target['id']}: 计划段终点与目标关节不一致")
            state_before = backend.wait_for_state(1.0)
            if not state_before.ready:
                raise RuntimeError(f"{target['id']}: 运动前状态不可用: {state_before.message}")
            log(
                f"开始姿态 {target_index}/{len(targets)}: {target['id']} "
                f"({target['description']}), 轨迹点={len(segment)}"
            )
            actual_endpoint = backend._send_joint_trajectory(
                segment,
                name=f"hand_eye_{target_index:02d}_{target['id']}",
                velocity_rad_s=speed_rad_s,
                timeout_s=max(60.0, 10.0 + 0.8 * len(segment)),
                reject_timeout_s=min(30.0, 2.0 + 0.3 * max(0, len(segment) - 16)),
            )
            time.sleep(settle_s)
            state_after = backend.wait_for_state(1.0)
            if not state_after.ready or state_after.joints is None:
                raise RuntimeError(f"{target['id']}: 到达后状态不可用: {state_after.message}")
            endpoint_error_deg = float(
                np.degrees(np.max(np.abs(np.asarray(state_after.joints) - target_q)))
            )
            if endpoint_error_deg > motion_config.joint_tolerance_deg:
                raise RuntimeError(
                    f"{target['id']}: 关节到位误差 {endpoint_error_deg:.3f}° 超过 "
                    f"{motion_config.joint_tolerance_deg:.3f}°"
                )
            board_transform, observations = capture_frames(
                snapshot_url=args.snapshot_url,
                board=board,
                rms_limit_px=rms_limit,
                output_dir=frames_dir,
                label=f"pose_{target_index:02d}_{target['id']}",
                frame_count=args.frames_per_pose,
                interval_s=args.frame_interval_s,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                max_attempts=args.max_frame_attempts,
            )
            record = {
                "index": target_index,
                "id": target["id"],
                "description": target["description"],
                "target_joints_rad": target_q.tolist(),
                "actual_joints_rad": list(state_after.joints),
                "max_joint_error_deg": endpoint_error_deg,
                "command_endpoint_error_deg": float(
                    np.degrees(np.max(np.abs(np.asarray(actual_endpoint) - target_q)))
                ),
                "controller_state": state_after.controller_state,
                "fault_flag": state_after.fault_flag,
                "board_transform_camera_board": board_transform.tolist(),
                "board_tvec_m": board_transform[:3, 3].tolist(),
                "observation_count": len(observations),
                "reprojection_rms_px": [obs.reprojection_rms_px for obs in observations],
            }
            manifest["pose_records"].append(record)
            save_json(manifest_path, manifest)
            log(
                f"采集完成 {target_index}/{len(targets)}: {target['id']}, "
                f"到位误差={endpoint_error_deg:.3f}°, "
                f"RMS最大={max(obs.reprojection_rms_px for obs in observations):.3f}px"
            )

        return_count = segment_counts[-1]
        return_segment = route_points[point_offset + 1 : point_offset + 1 + return_count]
        point_offset += return_count
        if point_offset != len(route_points) - 1:
            raise RuntimeError("执行路线没有完整消费全部计划点")
        if not np.allclose(return_segment[-1], expected_start_q, atol=2e-7, rtol=0.0):
            raise RuntimeError("回程终点与计划起点不一致")
        log(f"开始返回初始姿态，轨迹点={len(return_segment)}")
        backend._send_joint_trajectory(
            return_segment,
            name="hand_eye_return_start",
            velocity_rad_s=speed_rad_s,
            timeout_s=max(60.0, 10.0 + 0.8 * len(return_segment)),
            reject_timeout_s=min(30.0, 2.0 + 0.3 * max(0, len(return_segment) - 16)),
        )
        time.sleep(settle_s)
        final_state = backend.wait_for_state(1.0)
        if not final_state.ready or final_state.joints is None:
            raise RuntimeError(f"返回后状态不可用: {final_state.message}")
        return_error_deg = float(
            np.degrees(np.max(np.abs(np.asarray(final_state.joints) - expected_start_q)))
        )
        if return_error_deg > motion_config.joint_tolerance_deg:
            raise RuntimeError(
                f"返回初始姿态误差 {return_error_deg:.3f}° 超过 "
                f"{motion_config.joint_tolerance_deg:.3f}°"
            )
        manifest["return_to_start"] = {
            "actual_joints_rad": list(final_state.joints),
            "max_joint_error_deg": return_error_deg,
            "controller_state": final_state.controller_state,
            "fault_flag": final_state.fault_flag,
        }
        manifest["completed"] = True
        manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        save_json(manifest_path, manifest)
        log(f"全部采集完成并返回起始姿态，回程误差={return_error_deg:.3f}°")
        print(str(manifest_path), flush=True)
        return 0
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["failed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        save_json(manifest_path, manifest)
        log(f"采集失败，已停止发送新运动命令: {type(exc).__name__}: {exc}")
        raise
    finally:
        backend.close()
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        if spin_thread.is_alive():
            log("ROS 反馈执行器未在 2 秒内退出")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
