#!/usr/bin/env python3
"""Resume an interrupted eye-in-hand capture at a stopped calibration pose.

This is a narrow recovery tool.  It only accepts a failed capture manifest whose
pose records form an exact successful prefix of the reviewed route, verifies
that the robot is still at the next target pose, captures that pose, and only
then continues the remaining pre-checked route.  The gripper is never touched.

Physical motion requires the explicit ``--execute`` flag.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from track1_grasp.config import MotionConfig
from track1_grasp.motion import RobotMoveMotionBackend

from capture_hand_eye_poses import (
    ROOT,
    capture_frames,
    load_board_config,
    log,
    read_plan,
    validate_plan_for_execution,
)


DEFAULT_SOURCE_MANIFEST = (
    ROOT
    / "logs/hand_eye_calibration_20261025/capture_20261025_132635/capture_manifest.json"
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=(
                lambda value: value.item()
                if isinstance(value, np.generic)
                else value.tolist()
                if isinstance(value, np.ndarray)
                else None
            ),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_manifest(
    *,
    source_path: Path,
    plan_path: Path,
    plan: dict[str, Any],
    resume_target_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require the source records to be an exact successful route prefix."""

    if not source_path.is_file():
        raise RuntimeError(f"找不到中断采集 manifest: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("mode") not in (None, "capture"):
        raise RuntimeError(f"不支持的源 manifest 模式: {source.get('mode')!r}")
    if source.get("completed") is not False:
        raise RuntimeError("源 manifest 不是未完成状态")
    if source.get("physical_motion_started") is not True:
        raise RuntimeError("源 manifest 未记录已开始的物理运动，拒绝续采")
    if source.get("error") is None:
        raise RuntimeError("源 manifest 没有失败记录，拒绝续采")
    if Path(str(source.get("plan", ""))).expanduser().resolve() != plan_path:
        raise RuntimeError("源 manifest 与当前标定计划不一致")

    targets = plan["route"]["targets"]
    if not 1 <= resume_target_index <= len(targets):
        raise RuntimeError(
            f"续采目标序号 {resume_target_index} 超出 1..{len(targets)}"
        )
    records = source.get("pose_records")
    if not isinstance(records, list):
        raise RuntimeError("源 manifest 缺少 pose_records")
    if len(records) != resume_target_index:
        raise RuntimeError(
            f"源 manifest 有 {len(records)} 条姿态记录；从目标 "
            f"{resume_target_index} 续采应恰有 {resume_target_index} 条"
        )

    expected = [("start", 0)] + [
        (target["id"], index)
        for index, target in enumerate(targets[: resume_target_index - 1], start=1)
    ]
    for position, (record, (expected_id, expected_index)) in enumerate(
        zip(records, expected)
    ):
        if record.get("id") != expected_id or record.get("index") != expected_index:
            raise RuntimeError(
                f"源 manifest 第 {position} 条记录为 "
                f"{record.get('id')!r}/{record.get('index')!r}，应为 "
                f"{expected_id!r}/{expected_index!r}"
            )
        transform = np.asarray(
            record.get("board_transform_camera_board", []), dtype=float
        )
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise RuntimeError(f"源 manifest 记录 {expected_id} 的棋盘位姿无效")
        rms_values = record.get("reprojection_rms_px")
        if not isinstance(rms_values, list) or not rms_values:
            raise RuntimeError(f"源 manifest 记录 {expected_id} 缺少 RMS 审计数据")

    return source, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--resume-target-index", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot-url", default="http://127.0.0.1:9100/snapshot.jpg")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frames-per-pose", type=int, default=3)
    parser.add_argument("--max-frame-attempts", type=int, default=12)
    parser.add_argument("--frame-interval-s", type=float, default=0.20)
    parser.add_argument("--settle-s", type=float, default=None)
    parser.add_argument("--current-tolerance-deg", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("续采实机运动必须显式提供 --execute")
    if args.frames_per_pose < 1:
        raise RuntimeError("--frames-per-pose 必须至少为 1")
    if args.max_frame_attempts < args.frames_per_pose:
        raise RuntimeError("--max-frame-attempts 不得少于 --frames-per-pose")
    if not 0.0 < args.current_tolerance_deg <= 0.35:
        raise RuntimeError("--current-tolerance-deg 必须在 (0, 0.35] 内")

    plan_path = args.plan.expanduser().resolve()
    source_path = args.source_manifest.expanduser().resolve()
    plan = read_plan(plan_path)
    execution_metrics = validate_plan_for_execution(plan)
    source, source_records = validate_source_manifest(
        source_path=source_path,
        plan_path=plan_path,
        plan=plan,
        resume_target_index=args.resume_target_index,
    )

    thresholds = plan["safety_thresholds"]
    settle_s = float(
        thresholds["settle_s_per_capture"]
        if args.settle_s is None
        else args.settle_s
    )
    if settle_s < 0.5:
        raise RuntimeError("每个姿态停稳时间不得少于 0.5 s")

    board, board_config = load_board_config(
        ROOT / "calibration/hand_eye_20261025/chessboard.yaml"
    )
    camera_matrix = np.asarray(board_config["camera_matrix"], dtype=float)
    dist_coeffs = np.asarray(
        board_config["distortion_coefficients"], dtype=float
    ).reshape(-1)
    rms_limit = float(board_config["reprojection_rms_limit_px"])

    now = datetime.now().astimezone()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT
        / "logs/hand_eye_calibration_20261025"
        / f"capture_resume_{now.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = output_dir / "frames"
    manifest_path = output_dir / "capture_manifest.json"

    targets = plan["route"]["targets"]
    resume_index = args.resume_target_index
    resume_target = targets[resume_index - 1]
    manifest: dict[str, Any] = {
        "created_at": now.isoformat(timespec="seconds"),
        "mode": "resume_capture",
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_error": source.get("error"),
        "plan": str(plan_path),
        "snapshot_url": args.snapshot_url,
        "frames_per_pose": args.frames_per_pose,
        "max_frame_attempts": args.max_frame_attempts,
        "frame_interval_s": args.frame_interval_s,
        "settle_s": settle_s,
        "execution_validation": execution_metrics,
        "physical_motion_started": True,
        "new_physical_motion_started": False,
        "completed": False,
        "resume": {
            "target_index": resume_index,
            "target_id": resume_target["id"],
            "description": resume_target["description"],
            "source_pose_record_count": len(source_records),
            "current_target_tolerance_deg": args.current_tolerance_deg,
        },
        "pose_records": json.loads(json.dumps(source_records)),
    }
    atomic_write_json(manifest_path, manifest)

    expected_start_q = np.asarray(
        plan["robot_start"]["joint_positions_rad"], dtype=float
    )
    route_points = np.asarray(plan["route"]["joint_points_rad"], dtype=float)
    segment_counts = execution_metrics["segment_point_counts"]
    speed_rad_s = execution_metrics["speed_rad_s"]

    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init(args=[])
    node = rclpy.create_node("hand_eye_calibration_resume")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin,
        name="hand_eye_resume_robot_feedback",
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
                f"RobotMove订阅={backend._publisher.get_subscription_count()}"
            )
        if state.joints is None:
            raise RuntimeError("动作前 RobotInfo 缺少关节角")
        if state.fault:
            raise RuntimeError(
                f"动作前控制器有故障: {state.controller_state}, "
                f"fault_flag={state.fault_flag}"
            )

        target_q = np.radians(
            np.asarray(resume_target["target_joints_deg"], dtype=float)
        )
        current_q = np.asarray(state.joints, dtype=float)
        current_error_deg = float(
            np.degrees(np.max(np.abs(current_q - target_q)))
        )
        manifest["resume"]["robot_state_before_capture"] = {
            "controller_state": state.controller_state,
            "fault_flag": state.fault_flag,
            "joint_positions_rad": current_q.tolist(),
            "current_target_error_deg": current_error_deg,
        }
        atomic_write_json(manifest_path, manifest)
        if current_error_deg > args.current_tolerance_deg:
            raise RuntimeError(
                f"当前关节距续采目标 {resume_target['id']} "
                f"{current_error_deg:.3f}°，超过 "
                f"{args.current_tolerance_deg:.3f}°，拒绝续采"
            )
        log(
            f"续采前检查通过: 当前位于 {resume_target['id']}，误差="
            f"{current_error_deg:.3f}°，控制器={state.controller_state}，"
            f"fault_flag={state.fault_flag}"
        )

        board_transform, observations = capture_frames(
            snapshot_url=args.snapshot_url,
            board=board,
            rms_limit_px=rms_limit,
            output_dir=frames_dir,
            label=f"pose_{resume_index:02d}_{resume_target['id']}_resume",
            frame_count=args.frames_per_pose,
            interval_s=args.frame_interval_s,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            max_attempts=args.max_frame_attempts,
        )
        manifest["pose_records"].append(
            {
                "index": resume_index,
                "id": resume_target["id"],
                "description": resume_target["description"],
                "target_joints_rad": target_q.tolist(),
                "actual_joints_rad": current_q.tolist(),
                "max_joint_error_deg": current_error_deg,
                "command_endpoint_error_deg": None,
                "controller_state": state.controller_state,
                "fault_flag": state.fault_flag,
                "board_transform_camera_board": board_transform.tolist(),
                "board_tvec_m": board_transform[:3, 3].tolist(),
                "observation_count": len(observations),
                "reprojection_rms_px": [
                    obs.reprojection_rms_px for obs in observations
                ],
                "capture_origin": "resume_current_pose",
                "segment_command_sent": False,
            }
        )
        atomic_write_json(manifest_path, manifest)
        log(
            f"续采当前姿态完成 {resume_index}/{len(targets)}: "
            f"{resume_target['id']}, RMS最大="
            f"{max(obs.reprojection_rms_px for obs in observations):.3f}px"
        )

        point_offset = sum(segment_counts[:resume_index])
        for target_index in range(resume_index + 1, len(targets) + 1):
            target = targets[target_index - 1]
            segment_count = segment_counts[target_index - 1]
            segment = route_points[
                point_offset + 1 : point_offset + 1 + segment_count
            ]
            point_offset += segment_count
            target_q = np.radians(
                np.asarray(target["target_joints_deg"], dtype=float)
            )
            if not np.allclose(segment[-1], target_q, atol=2e-7, rtol=0.0):
                raise RuntimeError(f"{target['id']}: 计划段终点与目标关节不一致")

            state_before = backend.wait_for_state(1.0)
            if not state_before.ready or state_before.joints is None:
                raise RuntimeError(
                    f"{target['id']}: 运动前状态不可用: {state_before.message}"
                )
            previous_q = np.radians(
                np.asarray(targets[target_index - 2]["target_joints_deg"], dtype=float)
            )
            previous_error_deg = float(
                np.degrees(
                    np.max(np.abs(np.asarray(state_before.joints) - previous_q))
                )
            )
            if previous_error_deg > motion_config.joint_tolerance_deg:
                raise RuntimeError(
                    f"{target['id']}: 运动前不在上一姿态，误差 "
                    f"{previous_error_deg:.3f}° 超过 "
                    f"{motion_config.joint_tolerance_deg:.3f}°"
                )

            manifest["new_physical_motion_started"] = True
            atomic_write_json(manifest_path, manifest)
            log(
                f"开始姿态 {target_index}/{len(targets)}: {target['id']} "
                f"({target['description']}), 轨迹点={len(segment)}"
            )
            actual_endpoint = backend._send_joint_trajectory(
                segment,
                name=f"hand_eye_resume_{target_index:02d}_{target['id']}",
                velocity_rad_s=speed_rad_s,
                timeout_s=max(60.0, 10.0 + 0.8 * len(segment)),
                reject_timeout_s=min(
                    30.0, 2.0 + 0.3 * max(0, len(segment) - 16)
                ),
            )
            time.sleep(settle_s)
            state_after = backend.wait_for_state(1.0)
            if not state_after.ready or state_after.joints is None:
                raise RuntimeError(
                    f"{target['id']}: 到达后状态不可用: {state_after.message}"
                )
            endpoint_error_deg = float(
                np.degrees(
                    np.max(np.abs(np.asarray(state_after.joints) - target_q))
                )
            )
            if endpoint_error_deg > motion_config.joint_tolerance_deg:
                raise RuntimeError(
                    f"{target['id']}: 关节到位误差 {endpoint_error_deg:.3f}° "
                    f"超过 {motion_config.joint_tolerance_deg:.3f}°"
                )
            command_error_deg = float(
                np.degrees(
                    np.max(np.abs(np.asarray(actual_endpoint) - target_q))
                )
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
            manifest["pose_records"].append(
                {
                    "index": target_index,
                    "id": target["id"],
                    "description": target["description"],
                    "target_joints_rad": target_q.tolist(),
                    "actual_joints_rad": list(state_after.joints),
                    "max_joint_error_deg": endpoint_error_deg,
                    "command_endpoint_error_deg": command_error_deg,
                    "controller_state": state_after.controller_state,
                    "fault_flag": state_after.fault_flag,
                    "board_transform_camera_board": board_transform.tolist(),
                    "board_tvec_m": board_transform[:3, 3].tolist(),
                    "observation_count": len(observations),
                    "reprojection_rms_px": [
                        obs.reprojection_rms_px for obs in observations
                    ],
                    "capture_origin": "resumed_route",
                    "segment_command_sent": True,
                }
            )
            atomic_write_json(manifest_path, manifest)
            log(
                f"采集完成 {target_index}/{len(targets)}: {target['id']}, "
                f"到位误差={endpoint_error_deg:.3f}°, "
                f"RMS最大={max(obs.reprojection_rms_px for obs in observations):.3f}px"
            )

        return_count = segment_counts[-1]
        return_segment = route_points[
            point_offset + 1 : point_offset + 1 + return_count
        ]
        point_offset += return_count
        if point_offset != len(route_points) - 1:
            raise RuntimeError("续采路线没有完整消费全部计划点")
        if not np.allclose(return_segment[-1], expected_start_q, atol=2e-7, rtol=0.0):
            raise RuntimeError("回程终点与计划起点不一致")

        log(f"开始返回初始姿态，轨迹点={len(return_segment)}")
        backend._send_joint_trajectory(
            return_segment,
            name="hand_eye_resume_return_start",
            velocity_rad_s=speed_rad_s,
            timeout_s=max(60.0, 10.0 + 0.8 * len(return_segment)),
            reject_timeout_s=min(
                30.0, 2.0 + 0.3 * max(0, len(return_segment) - 16)
            ),
        )
        time.sleep(settle_s)
        final_state = backend.wait_for_state(1.0)
        if not final_state.ready or final_state.joints is None:
            raise RuntimeError(f"返回后状态不可用: {final_state.message}")
        return_error_deg = float(
            np.degrees(
                np.max(np.abs(np.asarray(final_state.joints) - expected_start_q))
            )
        )
        if return_error_deg > motion_config.joint_tolerance_deg:
            raise RuntimeError(
                f"返回初始姿态误差 {return_error_deg:.3f}° 超过 "
                f"{motion_config.joint_tolerance_deg:.3f}°"
            )

        expected_ids = ["start"] + [target["id"] for target in targets]
        actual_ids = [record.get("id") for record in manifest["pose_records"]]
        expected_indices = list(range(len(targets) + 1))
        actual_indices = [record.get("index") for record in manifest["pose_records"]]
        if actual_ids != expected_ids or actual_indices != expected_indices:
            raise RuntimeError(
                "最终标定记录顺序或数量不完整，拒绝标记采集完成"
            )

        manifest["return_to_start"] = {
            "actual_joints_rad": list(final_state.joints),
            "max_joint_error_deg": return_error_deg,
            "controller_state": final_state.controller_state,
            "fault_flag": final_state.fault_flag,
        }
        manifest["dataset_integrity"] = {
            "pose_record_count": len(manifest["pose_records"]),
            "expected_pose_record_count": len(targets) + 1,
            "unique_target_count": len(
                {record["id"] for record in manifest["pose_records"][1:]}
            ),
            "expected_target_count": len(targets),
            "route_order_matches_plan": True,
            "returned_to_start": True,
        }
        manifest["completed"] = True
        manifest["completed_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        atomic_write_json(manifest_path, manifest)
        log(
            f"续采全部完成并返回起始姿态，共 {len(manifest['pose_records'])} 条记录，"
            f"回程误差={return_error_deg:.3f}°"
        )
        print(str(manifest_path), flush=True)
        return 0
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["failed_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        atomic_write_json(manifest_path, manifest)
        log(f"续采失败，已停止发送新运动命令: {type(exc).__name__}: {exc}")
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
