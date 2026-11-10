"""Safely return the banana from Task B point 4's lift pose.

This recovery is intentionally narrow.  It only accepts the known point-4
lift posture, verifies that the gripper still reports a held/closed object,
pre-checks the complete descent and retreat route, then performs exactly one
in-place release:

    current lift -> release (80 mm down) -> open -> retreat (80 mm up)

No pick/search/retry logic is included.  ``--preflight-only`` performs only
read-only state checks and IK planning; real motion requires ``--execute``.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MISSION_CONFIG = ROOT / "config/track1_task_b_point4_live_once.yaml"

# Targets recorded immediately before and during the interrupted point-4 run.
RELEASE_XYZ = (0.049884856, 0.445201479, 0.164424007)
GRASP_RPY = (3.057380772, 0.107412163, -0.625207713)
LIFT_XYZ = (0.049884856, 0.445201479, 0.244424007)
RETREAT_XYZ = LIFT_XYZ

EXPECTED_DESCENT_M = 0.080
DESCENT_DISTANCE_TOLERANCE_M = 0.001
CURRENT_POSITION_TOLERANCE_M = 0.003
RELEASE_POSITION_TOLERANCE_M = 0.003
RETREAT_POSITION_TOLERANCE_M = 0.003
ORIENTATION_TOLERANCE_RAD = 0.02
HELD_CLOSED_ANGLE_MIN_DEG = 1000.0
MAX_PREOPEN_ANGLE_DRIFT_DEG = 10.0
MIN_OPEN_ANGLE_CHANGE_DEG = 5.0
GRIPPER_SPEED_STOP_RAD_S = 0.2


def _pose_errors(actual, expected_xyz, expected_rpy) -> tuple[float, float]:
    from track1_grasp.kinematics import rotation_angle_error, rpy_to_matrix

    position_error = float(
        np.linalg.norm(np.asarray(actual.xyz, dtype=float) - np.asarray(expected_xyz))
    )
    orientation_error = float(
        rotation_angle_error(
            rpy_to_matrix(np.asarray(actual.rpy, dtype=float)),
            rpy_to_matrix(np.asarray(expected_rpy, dtype=float)),
        )
    )
    return position_error, orientation_error


def _state_fields(state) -> str:
    joints = (
        "None"
        if state.joints is None
        else str(np.round(np.degrees(np.asarray(state.joints)), 4).tolist())
    )
    xyz = (
        "None"
        if state.pose is None
        else repr(tuple(round(float(value), 6) for value in state.pose.xyz))
    )
    rpy = (
        "None"
        if state.pose is None
        else repr(tuple(round(float(value), 6) for value in state.pose.rpy))
    )
    return (
        f"{state.message}; fault={state.fault}; joints_deg={joints}; "
        f"tcp_xyz_m={xyz}; tcp_rpy_rad={rpy}"
    )


def _status_fields(status) -> str:
    if status is None:
        return "status=None"
    return (
        f"id={status.device_id} arrived={status.arrived} "
        f"angle={status.angle_deg:.1f} deg "
        f"speed={status.speed_rad_s:.1f} rad/s"
    )


def _validate_held_closed_status(status, *, label: str) -> None:
    if status is None:
        raise RuntimeError(f"{label}: 夹爪状态读取失败")
    if not status.arrived:
        raise RuntimeError(f"{label}: 夹爪未报告到位")
    if status.angle_deg < HELD_CLOSED_ANGLE_MIN_DEG:
        raise RuntimeError(
            f"{label}: 夹爪角度 {status.angle_deg:.1f}° 不符合夹持香蕉状态"
        )
    if abs(status.speed_rad_s) > GRIPPER_SPEED_STOP_RAD_S:
        raise RuntimeError(
            f"{label}: 夹爪仍在运动，速度 {status.speed_rad_s:.1f} rad/s"
        )


def _validate_open_status(status, *, pre_open_angle_deg: float) -> None:
    if status is None:
        raise RuntimeError("张开后夹爪状态读取失败")
    if not status.arrived:
        raise RuntimeError("张开后夹爪未报告到位")
    angle_change = abs(status.angle_deg - pre_open_angle_deg)
    if angle_change < MIN_OPEN_ANGLE_CHANGE_DEG:
        raise RuntimeError(
            "夹爪张开未获得角度明显变化："
            f"{pre_open_angle_deg:.1f}° -> {status.angle_deg:.1f}°"
        )
    if abs(status.speed_rad_s) > GRIPPER_SPEED_STOP_RAD_S:
        raise RuntimeError(
            f"张开后夹爪速度未归零：{status.speed_rad_s:.1f} rad/s"
        )


def _validate_robot_pose(
    state,
    *,
    label: str,
    expected_xyz,
    position_tolerance_m: float,
) -> tuple[float, float]:
    if not state.ready or state.fault or state.pose is None or state.joints is None:
        raise RuntimeError(f"{label}: 机械臂状态异常：{state.message}")
    position_error, orientation_error = _pose_errors(
        state.pose,
        expected_xyz,
        GRASP_RPY,
    )
    if position_error > position_tolerance_m:
        raise RuntimeError(
            f"{label}: 机械臂位置偏差 {position_error * 1000:.3f} mm，"
            f"超过 {position_tolerance_m * 1000:.1f} mm"
        )
    if orientation_error > ORIENTATION_TOLERANCE_RAD:
        raise RuntimeError(
            f"{label}: 机械臂姿态偏差 {orientation_error:.5f} rad，"
            f"超过 {ORIENTATION_TOLERANCE_RAD:.3f} rad"
        )
    return position_error, orientation_error


def _route_metrics(route) -> tuple[int, float, float, float]:
    waypoints = sum(max(0, len(segment.waypoints) - 1) for segment in route.segments)
    min_sigma = min(segment.min_jacobian_singular_value for segment in route.segments)
    max_step = max(segment.max_joint_step_deg for segment in route.segments)
    max_accel = max(segment.max_joint_acceleration_deg for segment in route.segments)
    return waypoints, min_sigma, max_step, max_accel


def run(
    *,
    mission_config_path: Path,
    log_dir: Path,
    execute: bool,
) -> int:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    from track1_grasp.config import load_config
    from track1_grasp.domain import Pose
    from track1_grasp.gripper import build_gripper
    from track1_grasp.motion import build_motion

    mode = "execute" if execute else "preflight"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"task_b_point4_recovery_{mode}_{stamp}.log"
    summary_path = log_dir / f"task_b_point4_recovery_{mode}_{stamp}.md"
    report: list[str] = []
    result_text = "执行失败"
    return_code = 1

    def log(message: str) -> None:
        text = str(message)
        print(text, flush=True)
        report.append(text)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    log(f"[start] mode={mode}; log={log_path}")
    if execute:
        log(
            "[authorization] 用户明确允许从当前4号点抬升位执行一次原位放回："
            "下降80 mm、张开夹爪并撤回，速度0.01 m/s，异常立即停止"
        )
    else:
        log("[authorization] 只读预检；不会发布机械臂运动或夹爪动作命令")

    mission_config = load_config(mission_config_path)
    motion_config = replace(mission_config.motion, dry_run=not execute)
    gripper_config = replace(mission_config.gripper, speed_rad_s=10.0)

    rclpy.init()
    node = Node("task_b_point4_recovery")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin,
        name="task-b-point4-recovery-ros",
        daemon=True,
    )
    spin_thread.start()

    motion = None
    gripper = None
    try:
        motion = build_motion(motion_config, node=node, log=log)
        gripper = build_gripper(gripper_config, node=node, log=log)

        state = motion.wait_for_state(5.0)
        position_error, orientation_error = _validate_robot_pose(
            state,
            label="初始姿态",
            expected_xyz=LIFT_XYZ,
            position_tolerance_m=CURRENT_POSITION_TOLERANCE_M,
        )
        log(
            f"[preflight] robot: {_state_fields(state)} "
            f"lift_position_error={position_error * 1000:.3f} mm "
            f"lift_orientation_error={orientation_error:.6f} rad"
        )

        pre_status = gripper.read_status(timeout_s=3.0)
        _validate_held_closed_status(pre_status, label="初始夹持状态")
        log(f"[preflight] gripper: {_status_fields(pre_status)}")

        publisher = getattr(motion, "_publisher", None)
        if execute:
            waiter = getattr(publisher, "wait_for_matching_subscriptions", None)
            if callable(waiter):
                if not waiter(3.0):
                    raise RuntimeError("/RobotMove 没有匹配的桥接订阅者")
            else:
                count_getter = getattr(publisher, "get_subscription_count", None)
                if callable(count_getter) and int(count_getter()) < 1:
                    raise RuntimeError("/RobotMove 没有订阅者")
            log("[preflight] /RobotMove bridge subscriber confirmed")

        release = Pose(RELEASE_XYZ, GRASP_RPY)
        retreat = Pose(RETREAT_XYZ, GRASP_RPY)
        targets = (release, retreat)
        names = ("descend_to_point4_release", "retreat_after_release")
        log(
            "[targets] "
            f"release_xyz={release.xyz} release_rpy={release.rpy}; "
            f"retreat_xyz={retreat.xyz} retreat_rpy={retreat.rpy}"
        )

        route = motion.precheck_linear_targets(
            targets,
            step_m=motion_config.cartesian_step_m,
        )
        if len(route.segments) != 2:
            raise RuntimeError(f"恢复路径应有2段，实际为{len(route.segments)}段")
        descent_distance = route.segments[0].distance_m
        retreat_distance = route.segments[1].distance_m
        for label, distance in (
            ("下降", descent_distance),
            ("撤回", retreat_distance),
        ):
            if abs(distance - EXPECTED_DESCENT_M) > DESCENT_DISTANCE_TOLERANCE_M:
                raise RuntimeError(
                    f"{label}距离 {distance * 1000:.3f} mm 不符合授权的 "
                    f"80 mm（容差 ±1 mm）"
                )
        waypoints, min_sigma, max_step, max_accel = _route_metrics(route)
        log(
            f"[preflight] IK PASS: segments=2 waypoints={waypoints} "
            f"descent={descent_distance * 1000:.3f} mm "
            f"retreat={retreat_distance * 1000:.3f} mm "
            f"min_sigma={min_sigma:.5f} max_joint_step={max_step:.3f} deg "
            f"max_joint_accel={max_accel:.3f} deg"
        )

        if not execute:
            result_text = "预检通过：当前姿态、夹持状态和两段恢复路径均安全；未发送运动命令"
            log("[RESULT] " + result_text)
            return_code = 0
            return return_code

        # Never trust state gathered before the CPU-heavy IK pre-check.
        state = motion.wait_for_state(5.0)
        position_error, orientation_error = _validate_robot_pose(
            state,
            label="IK预检后姿态",
            expected_xyz=LIFT_XYZ,
            position_tolerance_m=CURRENT_POSITION_TOLERANCE_M,
        )
        log(
            f"[preflight:post-ik] robot: {_state_fields(state)} "
            f"lift_position_error={position_error * 1000:.3f} mm "
            f"lift_orientation_error={orientation_error:.6f} rad"
        )

        post_pre_status = gripper.read_status(timeout_s=3.0)
        _validate_held_closed_status(post_pre_status, label="IK预检后夹持状态")
        angle_drift = abs(post_pre_status.angle_deg - pre_status.angle_deg)
        if angle_drift > MAX_PREOPEN_ANGLE_DRIFT_DEG:
            raise RuntimeError(
                f"IK预检期间夹爪角度变化 {angle_drift:.1f}°，疑似香蕉滑落或移动"
            )
        log(
            f"[preflight:post-ik] gripper: {_status_fields(post_pre_status)} "
            f"angle_drift={angle_drift:.1f} deg"
        )

        log("[motion] executing pre-checked 80 mm descent to point-4 release pose")
        motion.execute_linear_route(
            replace(route, segments=(route.segments[0],)),
            names=(names[0],),
            speed_mps=0.01,
        )

        time.sleep(0.5)
        at_release = motion.wait_for_state(3.0)
        release_position_error, release_orientation_error = _validate_robot_pose(
            at_release,
            label="释放位",
            expected_xyz=RELEASE_XYZ,
            position_tolerance_m=RELEASE_POSITION_TOLERANCE_M,
        )
        log(
            f"[release-check] robot: {_state_fields(at_release)} "
            f"position_error={release_position_error * 1000:.3f} mm "
            f"orientation_error={release_orientation_error:.6f} rad"
        )

        closed_status = gripper.read_status(timeout_s=3.0)
        _validate_held_closed_status(closed_status, label="释放前夹持状态")
        pre_open_angle = closed_status.angle_deg
        log(f"[release] opening gripper from {_status_fields(closed_status)}")
        gripper.open()

        open_status = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            open_status = gripper.read_status(timeout_s=1.0)
            try:
                _validate_open_status(
                    open_status,
                    pre_open_angle_deg=pre_open_angle,
                )
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.3)
        _validate_open_status(open_status, pre_open_angle_deg=pre_open_angle)
        log(
            f"[release] OPEN CONFIRMED: angle changed "
            f"{pre_open_angle:.1f} -> {open_status.angle_deg:.1f} deg"
        )

        time.sleep(0.5)
        post_open_state = motion.wait_for_state(3.0)
        post_open_position_error, _ = _validate_robot_pose(
            post_open_state,
            label="张开后释放位",
            expected_xyz=RELEASE_XYZ,
            position_tolerance_m=RELEASE_POSITION_TOLERANCE_M,
        )
        log(
            "[post-open-check] robot remains at release; "
            f"position_error={post_open_position_error * 1000:.3f} mm"
        )
        retreat_status = gripper.read_status(timeout_s=3.0)
        if retreat_status is None or not retreat_status.arrived:
            raise RuntimeError("撤回前夹爪未报告到位，未撤回")
        if abs(retreat_status.speed_rad_s) > GRIPPER_SPEED_STOP_RAD_S:
            raise RuntimeError(
                f"撤回前夹爪仍在运动：{retreat_status.speed_rad_s:.1f} rad/s"
            )

        log("[motion] executing pre-checked 80 mm upward retreat")
        motion.execute_linear_route(
            replace(route, segments=(route.segments[1],)),
            names=(names[1],),
            speed_mps=0.01,
        )

        final_state = motion.wait_for_state(3.0)
        final_position_error, final_orientation_error = _validate_robot_pose(
            final_state,
            label="最终撤回位",
            expected_xyz=LIFT_XYZ,
            position_tolerance_m=RETREAT_POSITION_TOLERANCE_M,
        )
        final_status = gripper.read_status(timeout_s=3.0)
        if final_status is None or not final_status.arrived:
            raise RuntimeError("最终夹爪状态读取失败或未报告到位")
        if abs(final_status.speed_rad_s) > GRIPPER_SPEED_STOP_RAD_S:
            raise RuntimeError(
                f"最终夹爪速度未归零：{final_status.speed_rad_s:.1f} rad/s"
            )
        log(f"[final] robot: {_state_fields(final_state)}")
        log(
            f"[final] gripper: {_status_fields(final_status)} "
            f"position_error={final_position_error * 1000:.3f} mm "
            f"orientation_error={final_orientation_error:.6f} rad"
        )
        result_text = "成功：已下降80 mm、张开夹爪并撤回80 mm，机械臂与夹爪状态正常"
        log("[RESULT] " + result_text)
        return_code = 0
        return return_code
    except BaseException as exc:
        result_text = f"失败并停止：{type(exc).__name__}: {exc}"
        log("[ABORT] " + result_text)
        return_code = 1
        return return_code
    finally:
        try:
            if gripper is not None:
                gripper.close_backend()
        except Exception:
            pass
        try:
            if motion is not None:
                motion.close()
        except Exception:
            pass
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        try:
            executor.remove_node(node)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

        finished = datetime.now().astimezone()
        summary = [
            "# Task B 4号点原位放回记录",
            "",
            f"- 模式：{mode}",
            f"- 开始时间：{started.isoformat(timespec='seconds')}",
            f"- 结束时间：{finished.isoformat(timespec='seconds')}",
            f"- 结果：{result_text}",
            "",
            "## 动作范围",
            "",
            "- 仅接受当前记录的4号点抬升位。",
            "- 下降80 mm到已记录的释放位。",
            "- 张开夹爪并确认角度变化及速度归零。",
            "- 垂直撤回80 mm。",
            "- 不复用1号点坐标，不执行搜索、抓取或重试。",
            "",
            "## 安全门",
            "",
            "- 初始和最终姿态必须与4号点记录位置相差不超过3 mm。",
            "- 夹爪必须报告到位、角度符合夹持状态且速度接近零。",
            "- 两段路径必须在首条运动命令前完成完整IK预检。",
            "- 每段实际距离必须为80 mm，容差±1 mm。",
            "- IK预检后再次读取机械臂和夹爪状态。",
            "- 到达释放位误差不超过3 mm才允许张开。",
            "- 任何检查或动作异常均停止，不自动重试。",
            "",
            "## 执行输出",
            "",
            "```text",
            *report,
            "```",
            "",
            f"执行日志：`{log_path}`",
            "",
        ]
        summary_path.write_text("\n".join(summary), encoding="utf-8")
        print(f"SUMMARY_WRITTEN={summary_path}", flush=True)
        print(f"EXECUTION_LOG={log_path}", flush=True)
        print(f"EXECUTION_EXIT_STATUS={return_code}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="read state and pre-check the complete route without any motion",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="execute the authorized one-time point-4 recovery",
    )
    parser.add_argument(
        "--mission-config",
        type=Path,
        default=DEFAULT_MISSION_CONFIG,
    )
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    args = parser.parse_args()
    os.chdir(ROOT)
    return run(
        mission_config_path=args.mission_config.resolve(),
        log_dir=args.log_dir.resolve(),
        execute=bool(args.execute),
    )


if __name__ == "__main__":
    raise SystemExit(main())
