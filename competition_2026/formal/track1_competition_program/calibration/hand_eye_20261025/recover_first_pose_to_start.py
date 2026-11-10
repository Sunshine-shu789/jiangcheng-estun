#!/usr/bin/env python3
"""Return the robot from the first hand-eye pose to the planned start pose.

This recovery is deliberately narrow: it accepts only the current position at
the first pose in the reviewed hand-eye plan and sends the same first segment in
reverse.  The gripper is never commanded.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from track1_grasp.config import MotionConfig
from track1_grasp.kinematics import joints_within_limits
from track1_grasp.motion import RobotMoveMotionBackend


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    ROOT
    / "logs/hand_eye_calibration_20261025/pose_plan/pose_plan_20261025_132014.json"
)


def log(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{stamp}] {message}", flush=True)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--current-tolerance-deg", type=float, default=0.35)
    parser.add_argument("--endpoint-tolerance-deg", type=float, default=0.35)
    parser.add_argument("--settle-s", type=float, default=1.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = args.plan.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    route = plan["route"]
    if not bool(route.get("valid")):
        raise RuntimeError("计划路线未通过原审核，拒绝恢复运动")

    thresholds = plan["safety_thresholds"]
    speed_rad_s = float(thresholds["calibration_joint_speed_rad_s"])
    route_step_deg = float(thresholds["route_joint_step_deg"])
    joint_limit_margin_deg = float(thresholds["joint_limit_margin_deg"])
    if not (0.0 < speed_rad_s <= 0.25):
        raise RuntimeError(f"拒绝异常标定速度 {speed_rad_s:.3f} rad/s")
    if not (0.0 < args.current_tolerance_deg <= 0.5):
        raise RuntimeError("当前位置容差必须在 (0, 0.5]° 内")
    if not (0.0 < args.endpoint_tolerance_deg <= 0.35):
        raise RuntimeError("返回终点容差必须在 (0, 0.35]° 内")
    if args.settle_s < 0.5:
        raise RuntimeError("停稳时间不得少于 0.5 s")

    points = np.asarray(route["joint_points_rad"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 6 or len(points) < 2:
        raise RuntimeError("计划路线关节数据形状异常")
    if not np.all(np.isfinite(points)):
        raise RuntimeError("计划路线包含非有限关节值")

    start_q = np.asarray(plan["robot_start"]["joint_positions_rad"], dtype=float)
    first_target_deg = np.asarray(route["targets"][0]["target_joints_deg"], dtype=float)
    first_target_q = np.radians(first_target_deg)
    if np.max(np.abs(points[0] - start_q)) > 1e-8:
        raise RuntimeError("计划路线首点与机器人起始关节不一致")

    endpoint_matches = np.max(np.abs(points - first_target_q), axis=1) <= 1e-7
    endpoint_indices = np.flatnonzero(endpoint_matches)
    if len(endpoint_indices) == 0:
        raise RuntimeError("计划路线中找不到第 1 个标定姿态")
    first_endpoint_index = int(endpoint_indices[0])
    if first_endpoint_index < 2:
        raise RuntimeError("第 1 个标定段点数异常")
    if np.max(np.abs(points[first_endpoint_index + 1] - first_target_q)) <= 1e-7:
        raise RuntimeError("第 1 个标定姿态在路线中不唯一")

    reverse_points = points[first_endpoint_index - 1 :: -1].copy()
    if not np.allclose(reverse_points[-1], start_q, atol=1e-8, rtol=0.0):
        raise RuntimeError("恢复路线终点不是计划起始姿态")
    if not all(
        joints_within_limits(point, math.radians(joint_limit_margin_deg))
        for point in reverse_points
    ):
        raise RuntimeError("恢复路线违反关节限位安全余量")
    step_deg = np.degrees(
        np.max(np.abs(np.diff(np.vstack((points[first_endpoint_index], reverse_points)), axis=0)), axis=1)
    )
    max_step_deg = float(np.max(step_deg))
    allowed_step_deg = max(route_step_deg + 1e-6, 0.3)
    if max_step_deg > allowed_step_deg:
        raise RuntimeError(
            f"恢复路线最大关节步长 {max_step_deg:.6f}° 超过 {allowed_step_deg:.6f}°"
        )

    stamp = datetime.now().astimezone()
    record_path = (
        ROOT
        / "logs/hand_eye_calibration_20261025"
        / f"recovery_first_pose_{stamp.strftime('%Y%m%d_%H%M%S')}.json"
    )
    record = {
        "created_at": stamp.isoformat(timespec="seconds"),
        "plan": str(plan_path),
        "mode": "reverse_first_calibration_segment_to_start",
        "physical_motion_started": False,
        "completed": False,
        "gripper_commanded": False,
        "speed_rad_s": speed_rad_s,
        "first_target_id": route["targets"][0]["id"],
        "first_target_joints_deg": first_target_deg.tolist(),
        "start_joints_deg": np.degrees(start_q).tolist(),
        "route_point_count": len(reverse_points),
        "max_route_step_deg": max_step_deg,
        "joint_limit_margin_deg": joint_limit_margin_deg,
    }
    log(
        f"恢复检查通过: 第1姿态={route['targets'][0]['id']}, "
        f"反向量={len(reverse_points)}, 最大步长={max_step_deg:.6f}°, "
        f"速度={speed_rad_s:.3f} rad/s"
    )
    if not args.execute:
        log("未指定 --execute，仅完成离线检查，不连接机械臂、不发送运动命令")
        return 0

    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init(args=[])
    node = rclpy.create_node("hand_eye_first_pose_recovery")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin, name="hand_eye_recovery_feedback", daemon=True
    )
    spin_thread.start()
    backend = None
    try:
        motion_config = MotionConfig(
            backend="robotmove",
            dry_run=False,
            joint_speed_rad_s=speed_rad_s,
            joint_speed_safety_limit_rad_s=0.25,
            move_timeout_s=60.0,
            joint_move_timeout_s=60.0,
            poll_period_s=0.05,
            joint_tolerance_deg=args.endpoint_tolerance_deg,
            joint_start_tolerance_deg=0.05,
            joint_limit_margin_deg=joint_limit_margin_deg,
            joint_reject_timeout_s=2.0,
        )
        backend = RobotMoveMotionBackend(motion_config, node=node, log=log)

        deadline = time.monotonic() + 10.0
        state = backend.get_state()
        while time.monotonic() < deadline:
            if not spin_thread.is_alive():
                raise RuntimeError("ROS 反馈执行器意外停止")
            state = backend.get_state()
            if state.ready and backend._publisher.get_subscription_count() > 0:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                f"动作前检查失败: {state.message}, "
                f"RobotMove订阅={backend._publisher.get_subscription_count()}"
            )
        if state.joints is None:
            raise RuntimeError("当前 RobotInfo 缺少关节角")
        before_q = np.asarray(state.joints, dtype=float)
        current_error_deg = float(
            np.degrees(np.max(np.abs(before_q - first_target_q)))
        )
        record["before"] = {
            "controller_state": state.controller_state,
            "fault_flag": state.fault_flag,
            "joints_rad": before_q.tolist(),
            "joints_deg": np.degrees(before_q).tolist(),
            "first_target_error_deg": current_error_deg,
        }
        atomic_write_json(record_path, record)
        if current_error_deg > args.current_tolerance_deg:
            raise RuntimeError(
                f"当前位置距第1标定姿态 {current_error_deg:.3f}° 超过 "
                f"{args.current_tolerance_deg:.3f}°，拒绝恢复"
            )

        record["physical_motion_started"] = True
        atomic_write_json(record_path, record)
        log(f"开始沿原路径反向返回，目标关节={np.round(np.degrees(start_q), 3).tolist()}")
        backend._send_joint_trajectory(
            reverse_points,
            name="hand_eye_recover_first_pose_to_start",
            velocity_rad_s=speed_rad_s,
            timeout_s=60.0,
            reject_timeout_s=30.0,
        )
        time.sleep(args.settle_s)
        final_state = backend.wait_for_state(1.0)
        if not final_state.ready or final_state.joints is None:
            raise RuntimeError(f"恢复后状态不可用: {final_state.message}")
        final_q = np.asarray(final_state.joints, dtype=float)
        final_error_deg = float(
            np.degrees(np.max(np.abs(final_q - start_q)))
        )
        record["after"] = {
            "controller_state": final_state.controller_state,
            "fault_flag": final_state.fault_flag,
            "joints_rad": final_q.tolist(),
            "joints_deg": np.degrees(final_q).tolist(),
            "start_error_deg": final_error_deg,
        }
        if final_error_deg > args.endpoint_tolerance_deg:
            raise RuntimeError(
                f"返回起始姿态误差 {final_error_deg:.3f}° 超过 "
                f"{args.endpoint_tolerance_deg:.3f}°"
            )
        record["completed"] = True
        record["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        atomic_write_json(record_path, record)
        log(f"已返回初始姿态，误差={final_error_deg:.3f}°")
        print(str(record_path), flush=True)
        return 0
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["failed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        atomic_write_json(record_path, record)
        log(f"恢复失败，已停止发送新运动命令: {type(exc).__name__}: {exc}")
        raise
    finally:
        if backend is not None:
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
