#!/usr/bin/env python3
"""Zero-motion closed-loop validation of the calibrated eye-in-hand transform.

The script reads robot joint feedback from ``/RobotInfo`` and captures color
frames directly from the Intel RealSense D405.  For every accepted frame it
detects the 11x8-square / 14 mm chessboard, computes

    T_base_board = controller_fk(q) @ T_flange_camera @ T_camera_board

and compares the result with the fixed-board pose obtained by the 24-pose
hand-eye solve.  It never imports or publishes ``/RobotMove`` and never sends a
gripper command.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "app/track1_grasp"))

from chessboard_detector import (  # noqa: E402
    BoardObservation,
    BoardSpec,
    estimate_board_pose,
    render_overlay,
)
from track1_grasp.kinematics import (  # noqa: E402
    controller_fk,
    matrix_to_rpy,
    rotation_angle_error,
)


DEFAULT_RESULT = (
    SCRIPT_DIR
    / "results/hand_eye_d405_20261025_182144_6438b05191/hand_eye_result.json"
)
DEFAULT_EXTRINSICS = ROOT / "config/camera_to_flange.d405_calibrated_20261025.yaml"
DEFAULT_BOARD_CONFIG = SCRIPT_DIR / "chessboard.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proper_rotation(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def average_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("at least one transform is required")
    translation = np.mean([np.asarray(item)[:3, 3] for item in transforms], axis=0)
    rotation_sum = np.sum(
        [np.asarray(item, dtype=float)[:3, :3] for item in transforms], axis=0
    )
    result = np.eye(4, dtype=float)
    result[:3, :3] = proper_rotation(rotation_sum)
    result[:3, 3] = translation
    return result


def max_pair_rotation_deg(rotations: Sequence[np.ndarray]) -> float:
    maximum = 0.0
    for first in range(len(rotations)):
        for second in range(first + 1, len(rotations)):
            maximum = max(
                maximum,
                math.degrees(
                    rotation_angle_error(rotations[first], rotations[second])
                ),
            )
    return maximum


def read_robot_state(timeout_s: float) -> dict[str, Any]:
    """Collect fresh read-only /RobotInfo samples and summarize their stability."""
    try:
        import rclpy
        from codroid_msgs.msg import RobotInfo
    except ImportError as exc:
        raise RuntimeError(
            "缺少 ROS 2 Python 环境或 codroid_msgs。请先 source "
            "/opt/ros/humble/setup.bash 和运行区 setup.bash"
        ) from exc

    rclpy.init(args=[])
    node = rclpy.create_node(f"static_hand_eye_validation_{time.time_ns()}")
    samples: list[dict[str, Any]] = []

    def callback(msg: Any) -> None:
        joints = [float(value) for value in getattr(msg, "joint_positions", [])]
        if len(joints) != 6:
            return
        samples.append(
            {
                "received_monotonic_s": time.monotonic(),
                "state": str(getattr(msg, "state", "")),
                "fault_flag": int(getattr(msg, "fault_flag", 0)),
                "joint_positions_rad": joints,
            }
        )

    subscription = node.create_subscription(RobotInfo, "/RobotInfo", callback, 10)
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    try:
        while rclpy.ok() and time.monotonic() < deadline and len(samples) < 12:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()

    if not samples:
        raise RuntimeError(f"在 {timeout_s:.1f}s 内没有收到有效的 /RobotInfo")
    joints = np.asarray([sample["joint_positions_rad"] for sample in samples])
    stability_deg = float(np.degrees(np.ptp(joints, axis=0)).max())
    states = sorted({str(sample["state"]) for sample in samples})
    faults = [int(sample["fault_flag"]) for sample in samples]
    return {
        "sample_count": len(samples),
        "states": states,
        "fault_flags": faults,
        "joint_positions_rad_median": np.median(joints, axis=0).tolist(),
        "joint_stability_max_deg": stability_deg,
        "first_sample": samples[0],
        "last_sample": samples[-1],
    }


def load_board_config(path: Path) -> tuple[BoardSpec, np.ndarray, np.ndarray]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    pattern = data["opencv_inner_corner_pattern"]
    board = BoardSpec(
        columns=int(pattern[0]),
        rows=int(pattern[1]),
        square_size_m=float(data["square_size_mm"]) / 1000.0,
    )
    camera_matrix = np.asarray(data["camera_matrix"], dtype=float)
    dist_coeffs = np.asarray(data["distortion_coefficients"], dtype=float)
    if camera_matrix.shape != (3, 3):
        raise RuntimeError("chessboard.yaml 的 camera_matrix 不是 3x3")
    if dist_coeffs.size != 5:
        raise RuntimeError("chessboard.yaml 的畸变系数不是 5 项")
    if int(data.get("camera_view_rotation_deg", 180)) != 180:
        raise RuntimeError("当前静态验收仅支持 180° 应用视图旋转")
    return board, camera_matrix, dist_coeffs


def intrinsics_payload(intr: Any, image_rotation_deg: int) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
    }
    app = dict(raw)
    if image_rotation_deg == 180:
        app["cx"] = float((intr.width - 1) - intr.ppx)
        app["cy"] = float((intr.height - 1) - intr.ppy)
    return raw, app


def capture_frames(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "缺少 pyrealsense2，请使用 taskA 虚拟环境中的 Python 运行"
        ) from exc

    board, camera_matrix, dist_coeffs = load_board_config(args.board_config)
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )
    profile = pipeline.start(config)
    records: list[dict[str, Any]] = []
    try:
        serial = str(profile.get_device().get_info(rs.camera_info.serial_number))
        for _ in range(max(0, args.warmup)):
            pipeline.wait_for_frames(3000)

        for frame_index in range(args.frames):
            accepted = False
            for attempt in range(1, args.max_attempts + 1):
                frames = pipeline.wait_for_frames(5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                raw_image = np.asanyarray(color_frame.get_data()).copy()
                image = cv2.rotate(raw_image, cv2.ROTATE_180)
                intr = (
                    profile.get_stream(rs.stream.color)
                    .as_video_stream_profile()
                    .get_intrinsics()
                )
                raw_intr, app_intr = intrinsics_payload(intr, 180)
                image_rms_limit = float(args.rms_limit_px)
                observation = estimate_board_pose(
                    image,
                    board,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    image_path="",
                )
                stem = f"frame_{frame_index:02d}_attempt_{attempt:02d}"
                raw_path = output_dir / f"{stem}_raw.png"
                app_path = output_dir / f"{stem}_app.png"
                overlay_path = output_dir / f"{stem}_overlay.png"
                cv2.imwrite(str(raw_path), raw_image)
                cv2.imwrite(str(app_path), image)
                cv2.imwrite(str(overlay_path), render_overlay(image, observation, board))
                record = {
                    "frame_index": frame_index,
                    "attempt": attempt,
                    "accepted": observation.reprojection_rms_px <= image_rms_limit,
                    "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "serial": serial,
                    "raw_intrinsics": raw_intr,
                    "app_intrinsics": app_intr,
                    "raw_image": raw_path.name,
                    "app_image": app_path.name,
                    "overlay_image": overlay_path.name,
                    "reprojection_rms_px": observation.reprojection_rms_px,
                    "reprojection_mean_px": observation.reprojection_mean_px,
                    "reprojection_max_px": observation.reprojection_max_px,
                    "corner_refinement_used": observation.corner_refinement_used,
                    "transform_camera_board": observation.transform_camera_board.tolist(),
                    "board_center_camera_m": observation.board_center_camera_m.tolist(),
                    "board_normal_camera": observation.board_normal_camera.tolist(),
                }
                records.append(record)
                if record["accepted"]:
                    accepted = True
                    break
            if not accepted:
                raise RuntimeError(
                    f"第 {frame_index} 帧在 {args.max_attempts} 次尝试内未通过 "
                    f"RMS <= {args.rms_limit_px:.3f}px"
                )
            if frame_index + 1 < args.frames:
                time.sleep(max(0.0, args.interval_s))
    finally:
        pipeline.stop()

    accepted_records = [record for record in records if record["accepted"]]
    if len(accepted_records) != args.frames:
        raise RuntimeError("有效帧数量不足")
    return records


def build_report(payload: dict[str, Any]) -> str:
    checks = payload["checks"]
    lines = [
        "# D405 手眼静态闭环验收报告",
        "",
        f"- 时间：{payload['created_at']}",
        f"- 结论：**{'通过' if payload['validated'] else '未通过'}**",
        f"- 方法：`T_base_board = controller_fk(q) @ T_flange_camera @ T_camera_board`",
        f"- 手眼标定：`{payload['calibration_id']}`",
        f"- 标定板：11 × 8 方格，方格边长 14 mm（10 × 7 内角点）",
        f"- 帧数：{payload['accepted_frame_count']} 帧",
        f"- 机械臂状态：{', '.join(payload['robot']['pre_capture']['states'])}",
        "- 机器人运动：无；夹爪命令：无",
        "",
        "## 结果",
        "",
        "| 指标 | 实测 | 阈值 | 结果 |",
        "|---|---:|---:|---|",
    ]
    metrics = payload["metrics"]
    threshold_rows = [
        ("重投影 RMS 最大值", metrics["reprojection_rms_max_px"], payload["thresholds"]["rms_limit_px"], "px"),
        ("相机系板中心稳定跨度", metrics["camera_board_translation_span_mm"], payload["thresholds"]["frame_translation_stability_mm"], "mm"),
        ("相机系板姿态稳定跨度", metrics["camera_board_rotation_span_deg"], payload["thresholds"]["frame_rotation_stability_deg"], "°"),
        ("基座系板位姿平均平移误差", metrics["base_board_translation_error_mm"], payload["thresholds"]["base_translation_limit_mm"], "mm"),
        ("基座系板位姿平均旋转误差", metrics["base_board_rotation_error_deg"], payload["thresholds"]["base_rotation_limit_deg"], "°"),
        ("关节角稳定跨度", metrics["joint_stability_max_deg"], payload["thresholds"]["joint_stability_limit_deg"], "°"),
    ]
    for label, value, limit, unit in threshold_rows:
        passed = float(value) <= float(limit)
        lines.append(
            f"| {label} | {float(value):.6f} {unit} | ≤ {float(limit):.3f} {unit} | "
            f"{'通过' if passed else '未通过'} |"
        )

    lines.extend(
        [
            "",
            "## 每帧结果",
            "",
            "| 帧 | RMS (px) | 板中心相机系 (mm) | 基座系平移误差 (mm) | 基座系旋转误差 (°) |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for item in payload["accepted_frames"]:
        center = ", ".join(f"{float(value):.3f}" for value in item["board_center_camera_mm"])
        lines.append(
            f"| {item['frame_index']} | {float(item['reprojection_rms_px']):.4f} | "
            f"[{center}] | {float(item['base_board_translation_error_mm']):.4f} | "
            f"{float(item['base_board_rotation_error_deg']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## 基座系标定板位姿",
            "",
            "实测平均矩阵：",
            "```text",
            np.array2string(
                np.asarray(payload["base_board_transform_mean"]),
                precision=9,
                suppress_small=False,
            ),
            "```",
            "",
            f"- 平移：[{', '.join(f'{float(v):.9f}' for v in metrics['base_board_translation_mean_m'])}] m",
            f"- RPY：[{', '.join(f'{float(v):.6f}' for v in metrics['base_board_rpy_mean_deg'])}]°",
            f"- 期望平移：[{', '.join(f'{float(v):.9f}' for v in payload['expected_base_board_translation_m'])}] m",
            "",
            "## 质量门",
            "",
        ]
    )
    for name, passed in checks.items():
        lines.append(f"- `{name}`: {'通过' if passed else '未通过'}")
    lines.extend(
        [
            "",
            "## 安全说明",
            "",
            "本次只读取 `/RobotInfo` 并采集 D405 彩色图像；未导入或发布 `/RobotMove`，"
            "未发送夹爪命令。机械臂处于掉电状态不影响零运动静态闭环验收。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    parser.add_argument("--board-config", type=Path, default=DEFAULT_BOARD_CONFIG)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--interval-s", type=float, default=0.35)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--serial", default="")
    parser.add_argument("--robotinfo-timeout-s", type=float, default=5.0)
    parser.add_argument("--rms-limit-px", type=float, default=1.0)
    parser.add_argument("--frame-translation-stability-mm", type=float, default=1.0)
    parser.add_argument("--frame-rotation-stability-deg", type=float, default=0.5)
    parser.add_argument("--base-translation-limit-mm", type=float, default=3.0)
    parser.add_argument("--base-rotation-limit-deg", type=float, default=1.0)
    parser.add_argument("--joint-stability-limit-deg", type=float, default=0.01)
    args = parser.parse_args()

    if args.frames < 3:
        raise SystemExit("--frames 至少为 3")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts 至少为 1")

    result_path = args.result_json.expanduser().resolve()
    extrinsics_path = args.extrinsics.expanduser().resolve()
    board_config_path = args.board_config.expanduser().resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    extrinsics_data = yaml.safe_load(extrinsics_path.read_text(encoding="utf-8"))
    transform_flange_camera = np.asarray(extrinsics_data["matrix"], dtype=float)
    if transform_flange_camera.shape != (4, 4):
        raise SystemExit("外参文件 matrix 不是 4x4")
    expected_transform_base_board = np.asarray(
        result["fixed_board_consistency"]["average_board_transform_base_board"],
        dtype=float,
    )
    if expected_transform_base_board.shape != (4, 4):
        raise SystemExit("手眼结果中的固定板位姿不是 4x4")

    created_at = datetime.now().astimezone()
    output_dir = (
        args.output_root.expanduser().resolve()
        / f"static_validation_{created_at.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    pre_robot = read_robot_state(args.robotinfo_timeout_s)
    frame_records = capture_frames(args, output_dir)
    post_robot = read_robot_state(args.robotinfo_timeout_s)

    pre_q = np.asarray(pre_robot["joint_positions_rad_median"], dtype=float)
    post_q = np.asarray(post_robot["joint_positions_rad_median"], dtype=float)
    joints_unchanged_deg = float(np.degrees(np.max(np.abs(post_q - pre_q))))
    transform_base_flange = controller_fk(pre_q)

    accepted_records = [record for record in frame_records if record["accepted"]]
    camera_board_transforms = [
        np.asarray(record["transform_camera_board"], dtype=float)
        for record in accepted_records
    ]
    transform_camera_board_mean = average_transform(camera_board_transforms)
    base_board_transforms = [
        transform_base_flange @ transform_flange_camera @ transform
        for transform in camera_board_transforms
    ]
    transform_base_board_mean = average_transform(base_board_transforms)

    frame_results: list[dict[str, Any]] = []
    centers = []
    rotations = []
    for record, transform_base_board in zip(accepted_records, base_board_transforms, strict=True):
        translation_error_mm = float(
            np.linalg.norm(transform_base_board[:3, 3] - expected_transform_base_board[:3, 3])
            * 1000.0
        )
        rotation_error_deg = math.degrees(
            rotation_angle_error(
                transform_base_board[:3, :3],
                expected_transform_base_board[:3, :3],
            )
        )
        center = np.asarray(record["board_center_camera_m"], dtype=float)
        centers.append(center)
        rotations.append(transform_base_board[:3, :3])
        frame_results.append(
            {
                **record,
                "board_center_camera_mm": (center * 1000.0).tolist(),
                "base_board_transform": transform_base_board.tolist(),
                "base_board_translation_mm": (transform_base_board[:3, 3] * 1000.0).tolist(),
                "base_board_rpy_deg": list(
                    np.degrees(matrix_to_rpy(transform_base_board[:3, :3]))
                ),
                "base_board_translation_error_mm": translation_error_mm,
                "base_board_rotation_error_deg": rotation_error_deg,
            }
        )

    centers_array = np.asarray(centers, dtype=float)
    camera_translation_span_mm = float(
        np.linalg.norm(np.ptp(centers_array, axis=0)) * 1000.0
    )
    camera_rotation_span_deg = max_pair_rotation_deg(
        [transform[:3, :3] for transform in camera_board_transforms]
    )
    base_translations = np.asarray([transform[:3, 3] for transform in base_board_transforms])
    base_rotation_span_deg = max_pair_rotation_deg(rotations)
    base_translation_error_mm = float(
        np.linalg.norm(transform_base_board_mean[:3, 3] - expected_transform_base_board[:3, 3])
        * 1000.0
    )
    base_rotation_error_deg = math.degrees(
        rotation_angle_error(
            transform_base_board_mean[:3, :3],
            expected_transform_base_board[:3, :3],
        )
    )

    solver_matrix = np.asarray(result["camera_to_flange_matrix"], dtype=float)
    manifest_path_value = result.get("source_manifest", result.get("manifest"))
    manifest_sha256_value = result.get(
        "source_manifest_sha256", result.get("manifest_sha256")
    )
    if not manifest_path_value or not manifest_sha256_value:
        raise SystemExit("手眼结果缺少 source_manifest/manifest 或对应 SHA-256")
    source_manifest = Path(str(manifest_path_value))
    source_manifest_sha_ok = (
        source_manifest.is_file()
        and sha256_file(source_manifest) == str(manifest_sha256_value)
    )
    mount_quality = result["quality"]
    checks = {
        "solver_validated": bool(result.get("validated") is True),
        "extrinsics_validated": bool(extrinsics_data.get("validated") is True),
        "extrinsics_matches_solver": bool(
            np.allclose(transform_flange_camera, solver_matrix, atol=1e-9, rtol=0.0)
        ),
        "calibration_id_matches": bool(
            str(extrinsics_data.get("calibration_id", ""))
            == str(result.get("calibration_id", ""))
        ),
        "mount_distance_within_2mm": bool(
            float(mount_quality["mount_distance_error_mm"])
            <= float(mount_quality["mount_distance_warning_mm"])
        ),
        "source_manifest_sha256_matches": source_manifest_sha_ok,
        "robot_fault_flags_zero": bool(
            max(pre_robot["fault_flags"] + post_robot["fault_flags"]) == 0
        ),
        "robot_joints_stable": bool(
            max(
                pre_robot["joint_stability_max_deg"],
                post_robot["joint_stability_max_deg"],
            )
            <= args.joint_stability_limit_deg
        ),
        "robot_joints_unchanged_during_capture": bool(
            joints_unchanged_deg <= args.joint_stability_limit_deg
        ),
        "all_frames_detected": bool(len(accepted_records) == args.frames),
        "reprojection_rms_within_limit": bool(
            max(record["reprojection_rms_px"] for record in accepted_records)
            <= args.rms_limit_px
        ),
        "camera_board_translation_stable": bool(
            camera_translation_span_mm <= args.frame_translation_stability_mm
        ),
        "camera_board_rotation_stable": bool(
            camera_rotation_span_deg <= args.frame_rotation_stability_deg
        ),
        "base_board_translation_error_within_limit": bool(
            base_translation_error_mm <= args.base_translation_limit_mm
        ),
        "base_board_rotation_error_within_limit": bool(
            base_rotation_error_deg <= args.base_rotation_limit_deg
        ),
    }
    validated = all(checks.values())

    payload: dict[str, Any] = {
        "created_at": created_at.isoformat(timespec="seconds"),
        "validated": validated,
        "purpose": "zero-motion D405 eye-in-hand closed-loop validation",
        "calibration_id": str(result["calibration_id"]),
        "result_json": str(result_path),
        "result_json_sha256": sha256_file(result_path),
        "extrinsics_file": str(extrinsics_path),
        "extrinsics_sha256": sha256_file(extrinsics_path),
        "board_config_file": str(board_config_path),
        "board_config_sha256": sha256_file(board_config_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": str(manifest_sha256_value),
        "accepted_frame_count": len(accepted_records),
        "total_capture_attempts": len(frame_records),
        "thresholds": {
            "rms_limit_px": args.rms_limit_px,
            "frame_translation_stability_mm": args.frame_translation_stability_mm,
            "frame_rotation_stability_deg": args.frame_rotation_stability_deg,
            "base_translation_limit_mm": args.base_translation_limit_mm,
            "base_rotation_limit_deg": args.base_rotation_limit_deg,
            "joint_stability_limit_deg": args.joint_stability_limit_deg,
        },
        "robot": {
            "motion_commands_sent": False,
            "gripper_commands_sent": False,
            "pre_capture": pre_robot,
            "post_capture": post_robot,
            "joints_unchanged_during_capture_deg": joints_unchanged_deg,
            "base_flange_transform": transform_base_flange.tolist(),
        },
        "extrinsics": {
            "flange_camera_matrix": transform_flange_camera.tolist(),
            "translation_mm": (transform_flange_camera[:3, 3] * 1000.0).tolist(),
            "rpy_deg": list(np.degrees(matrix_to_rpy(transform_flange_camera[:3, :3]))),
            "mount_distance_error_mm": float(mount_quality["mount_distance_error_mm"]),
            "ready_for_task_b_activation": bool(
                result.get("ready_for_task_b_activation", False)
            ),
        },
        "camera_board_transform_mean": transform_camera_board_mean.tolist(),
        "base_board_transform_mean": transform_base_board_mean.tolist(),
        "expected_base_board_transform": expected_transform_base_board.tolist(),
        "expected_base_board_translation_m": expected_transform_base_board[:3, 3].tolist(),
        "metrics": {
            "reprojection_rms_mean_px": float(
                np.mean([record["reprojection_rms_px"] for record in accepted_records])
            ),
            "reprojection_rms_max_px": float(
                max(record["reprojection_rms_px"] for record in accepted_records)
            ),
            "camera_board_translation_span_mm": camera_translation_span_mm,
            "camera_board_rotation_span_deg": camera_rotation_span_deg,
            "joint_stability_max_deg": float(
                max(
                    pre_robot["joint_stability_max_deg"],
                    post_robot["joint_stability_max_deg"],
                )
            ),
            "joints_unchanged_during_capture_deg": joints_unchanged_deg,
            "base_board_translation_mean_m": transform_base_board_mean[:3, 3].tolist(),
            "base_board_rpy_mean_deg": list(
                np.degrees(matrix_to_rpy(transform_base_board_mean[:3, :3]))
            ),
            "base_board_translation_error_mm": base_translation_error_mm,
            "base_board_rotation_error_deg": base_rotation_error_deg,
            "base_board_translation_span_mm": float(
                np.linalg.norm(np.ptp(base_translations, axis=0)) * 1000.0
            ),
            "base_board_rotation_span_deg": base_rotation_span_deg,
        },
        "accepted_frames": frame_results,
        "all_capture_attempts": frame_records,
        "checks": checks,
    }

    json_path = output_dir / "static_validation_result.json"
    report_path = output_dir / "REPORT_zh.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(build_report(payload), encoding="utf-8")
    print(json.dumps(
        {
            "validated": validated,
            "json": str(json_path),
            "report": str(report_path),
            "base_translation_error_mm": base_translation_error_mm,
            "base_rotation_error_deg": base_rotation_error_deg,
            "rms_max_px": payload["metrics"]["reprojection_rms_max_px"],
            "camera_translation_span_mm": camera_translation_span_mm,
            "camera_rotation_span_deg": camera_rotation_span_deg,
            "failed_checks": [name for name, passed in checks.items() if not passed],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if validated else 2


if __name__ == "__main__":
    raise SystemExit(main())
