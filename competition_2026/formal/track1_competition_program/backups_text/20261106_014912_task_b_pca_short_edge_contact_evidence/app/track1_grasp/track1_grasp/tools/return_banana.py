"""Supervised return of the banana to its recorded pick pose.

This tool is intentionally narrow: it assumes the robot is already holding the
banana in Task B's saved initial pose, pre-checks the complete Cartesian route,
returns to the recorded release pose, opens the jaws, and retreats vertically.
No route is published unless every IK and state-feedback gate passes first.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MISSION_CONFIG = ROOT / "config/track1_live.yaml"
DEFAULT_GRIPPER_CONFIG = ROOT / "config/ms42dc_ros_test.yaml"

# Recorded when the banana was picked at Task B point 1.
RELEASE_XYZ = (-0.063443253, 0.543732009, 0.161534092)
RELEASE_RPY = (3.082206718, 0.098641098, -0.072880619)
SAFE_Z_M = 0.300
RETREAT_M = 0.050
RELEASE_POSITION_TOLERANCE_M = 0.003
MIN_OPEN_ANGLE_CHANGE_DEG = 5.0
GRIPPER_SPEED_STOP_RAD_S = 0.2


def _state_fields(state) -> str:
    joints = (
        "None"
        if state.joints is None
        else str(np.round(np.degrees(np.asarray(state.joints)), 4).tolist())
    )
    xyz = (
        "None"
        if state.pose is None
        else repr(tuple(round(float(v), 6) for v in state.pose.xyz))
    )
    rpy = (
        "None"
        if state.pose is None
        else repr(tuple(round(float(v), 6) for v in state.pose.rpy))
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


def _pose_errors(actual, expected_xyz, expected_rpy) -> tuple[float, float]:
    from track1_grasp.kinematics import rotation_angle_error, rpy_to_matrix

    position_error = float(
        np.linalg.norm(np.asarray(actual.xyz) - np.asarray(expected_xyz))
    )
    orientation_error = float(
        rotation_angle_error(
            rpy_to_matrix(np.asarray(actual.rpy)),
            rpy_to_matrix(np.asarray(expected_rpy)),
        )
    )
    return position_error, orientation_error


def run(
    *,
    mission_config_path: Path,
    gripper_config_path: Path,
    log_dir: Path,
) -> int:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    from track1_grasp.config import load_config
    from track1_grasp.domain import Pose
    from track1_grasp.gripper import build_gripper
    from track1_grasp.motion import build_motion

    log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"task_b_banana_return_execute_{stamp}.log"
    summary_path = log_dir / f"task_b_banana_return_{stamp}.md"
    report: list[str] = []
    result_text = "执行失败"
    failure_text = ""

    def log(message: str) -> None:
        text = str(message)
        print(text, flush=True)
        report.append(text)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    log(f"[start] banana return; log={log_path}")
    log(f"[authorization] user explicitly requested 放回 after the pick/hold test")

    mission_config = load_config(mission_config_path)
    gripper_config = replace(
        load_config(gripper_config_path).gripper,
        speed_rad_s=10.0,
    )
    motion_config = replace(mission_config.motion, dry_run=False)

    rclpy.init()
    node = Node("task_b_banana_return_execute")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin,
        name="task-b-banana-return-ros",
        daemon=True,
    )
    spin_thread.start()

    motion = None
    gripper = None
    try:
        motion = build_motion(motion_config, node=node, log=log)
        gripper = build_gripper(gripper_config, node=node, log=log)

        state = motion.wait_for_state(5.0)
        if not state.ready or state.pose is None or state.joints is None:
            raise RuntimeError(f"机械臂启动条件不满足：{state.message}")
        log(f"[preflight] robot: {_state_fields(state)}")

        publisher = getattr(motion, "_publisher", None)
        waiter = getattr(publisher, "wait_for_matching_subscriptions", None)
        if callable(waiter):
            if not waiter(3.0):
                raise RuntimeError("/RobotMove 没有匹配的桥接订阅者")
        else:
            count_getter = getattr(publisher, "get_subscription_count", None)
            if callable(count_getter) and int(count_getter()) < 1:
                raise RuntimeError("/RobotMove 没有订阅者")
        log("[preflight] /RobotMove bridge subscriber confirmed")

        pre_status = gripper.read_status(timeout_s=3.0)
        if pre_status is None:
            raise RuntimeError("夹爪状态读取失败")
        if not pre_status.arrived or pre_status.angle_deg < 1000.0:
            raise RuntimeError(
                "夹爪不处于预期的夹持状态："
                f"arrived={pre_status.arrived}, angle={pre_status.angle_deg:.1f} deg"
            )
        log(f"[preflight] gripper: {_status_fields(pre_status)}")

        lower_safe = Pose((state.pose.xyz[0], state.pose.xyz[1], SAFE_Z_M), state.pose.rpy)
        align_low = Pose((RELEASE_XYZ[0], RELEASE_XYZ[1], SAFE_Z_M), state.pose.rpy)
        above_release = Pose(
            (RELEASE_XYZ[0], RELEASE_XYZ[1], RELEASE_XYZ[2] + RETREAT_M),
            RELEASE_RPY,
        )
        release = Pose(RELEASE_XYZ, RELEASE_RPY)
        targets = (lower_safe, align_low, above_release, release, above_release)
        names = (
            "lower_safe",
            "align_low",
            "descend_rotate_to_above_release",
            "release",
            "retreat",
        )
        log(
            "[targets] "
            + "; ".join(
                f"{name}={pose.xyz}/{pose.rpy}"
                for name, pose in zip(names, targets)
            )
        )

        route = motion.precheck_linear_targets(
            targets,
            step_m=motion_config.cartesian_step_m,
        )
        total_waypoints = sum(
            max(0, len(segment.waypoints) - 1) for segment in route.segments
        )
        min_sigma = min(
            segment.min_jacobian_singular_value for segment in route.segments
        )
        max_step = max(segment.max_joint_step_deg for segment in route.segments)
        max_accel = max(
            segment.max_joint_acceleration_deg for segment in route.segments
        )
        log(
            f"[preflight] PASS: segments={len(route.segments)} "
            f"waypoints={total_waypoints} min_sigma={min_sigma:.5f} "
            f"max_joint_step={max_step:.3f} deg "
            f"max_joint_accel={max_accel:.3f} deg"
        )

        # The IK pre-check is CPU intensive.  Require a newly received state
        # before publishing the first command instead of trusting stale data.
        state = motion.wait_for_state(5.0)
        if not state.ready or state.pose is None or state.joints is None:
            raise RuntimeError(f"预检后机械臂状态不满足：{state.message}")
        log(f"[preflight:post-ik] robot: {_state_fields(state)}")

        post_pre_status = gripper.read_status(timeout_s=3.0)
        if post_pre_status is None or not post_pre_status.arrived:
            raise RuntimeError("预检后夹爪未报告到位")
        angle_drift = abs(post_pre_status.angle_deg - pre_status.angle_deg)
        if angle_drift > 10.0:
            raise RuntimeError(
                f"预检期间夹爪角度变化 {angle_drift:.1f} deg，疑似香蕉滑落或移动"
            )
        log(
            "[preflight:post-ik] gripper: "
            f"{_status_fields(post_pre_status)} angle_drift={angle_drift:.1f} deg"
        )

        inbound_route = replace(route, segments=tuple(route.segments[:4]))
        log("[motion] executing four pre-checked inbound segments")
        motion.execute_linear_route(
            inbound_route,
            names=tuple(names[:4]),
            speed_mps=float(motion_config.speed_mps),
        )

        time.sleep(0.5)
        at_release = motion.wait_for_state(3.0)
        if (
            not at_release.ready
            or at_release.fault
            or at_release.pose is None
        ):
            raise RuntimeError(f"释放位状态异常，未张开夹爪：{at_release.message}")
        position_error, orientation_error = _pose_errors(
            at_release.pose,
            RELEASE_XYZ,
            RELEASE_RPY,
        )
        log(
            f"[release-check] position_error={position_error * 1000:.3f} mm "
            f"orientation_error={orientation_error:.5f} rad"
        )
        if position_error > RELEASE_POSITION_TOLERANCE_M:
            raise RuntimeError(
                f"机械臂偏离原释放位 {position_error * 1000:.1f} mm，未张开夹爪"
            )

        closed_status = gripper.read_status(timeout_s=3.0)
        if closed_status is None or not closed_status.arrived:
            raise RuntimeError("释放前夹爪未报告到位，未张开夹爪")
        pre_open_angle = closed_status.angle_deg
        log(f"[release] opening gripper from {_status_fields(closed_status)}")
        gripper.open()

        open_confirmed = False
        open_status = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            open_status = gripper.read_status(timeout_s=1.0)
            if (
                open_status is not None
                and open_status.arrived
                and abs(open_status.angle_deg - pre_open_angle)
                >= MIN_OPEN_ANGLE_CHANGE_DEG
                and abs(open_status.speed_rad_s) <= GRIPPER_SPEED_STOP_RAD_S
            ):
                open_confirmed = True
                break
            time.sleep(0.4)
        if not open_confirmed or open_status is None:
            raise RuntimeError(
                "夹爪张开未获得“到位且角度明显变化”的反馈，停止并保持机械臂位置"
            )
        log(
            f"[release] OPEN CONFIRMED: angle changed "
            f"{pre_open_angle:.1f} -> {open_status.angle_deg:.1f} deg"
        )

        time.sleep(0.5)
        post_open_state = motion.wait_for_state(3.0)
        if (
            not post_open_state.ready
            or post_open_state.fault
            or post_open_state.pose is None
        ):
            raise RuntimeError(f"张开后机械臂状态异常，未抬升：{post_open_state.message}")
        post_open_position_error, _ = _pose_errors(
            post_open_state.pose,
            RELEASE_XYZ,
            RELEASE_RPY,
        )
        log(
            "[post-open-check] max/current position deviation="
            f"{post_open_position_error * 1000:.3f} mm"
        )
        if post_open_position_error > RELEASE_POSITION_TOLERANCE_M:
            raise RuntimeError("张开后机械臂偏离释放位超过 3 mm，未抬升")

        retreat_status = gripper.read_status(timeout_s=3.0)
        if retreat_status is None or not retreat_status.arrived:
            raise RuntimeError("抬升前夹爪未报告到位，未抬升")
        retreat_route = replace(route, segments=(route.segments[4],))
        log("[motion] executing pre-checked 50 mm upward retreat")
        motion.execute_linear_route(
            retreat_route,
            names=(names[4],),
            speed_mps=float(motion_config.speed_mps),
        )

        final_state = motion.wait_for_state(3.0)
        if final_state.pose is None or final_state.joints is None:
            raise RuntimeError("最终机械臂状态反馈不完整")
        if not final_state.ready or final_state.fault:
            raise RuntimeError(f"最终状态异常：{final_state.message}")
        final_status = gripper.read_status(timeout_s=3.0)
        log(f"[final] robot: {_state_fields(final_state)}")
        log(f"[final] gripper: {_status_fields(final_status)}")
        result_text = "成功：香蕉已放回原抓取位，夹爪已张开，机械臂已抬升 50 mm"
        log("[RESULT] " + result_text)
        return_code = 0
    except BaseException as exc:
        failure_text = f"{type(exc).__name__}: {exc}"
        result_text = f"失败并停止：{failure_text}"
        log("[ABORT] " + result_text)
        return_code = 1
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
            "# Task B 香蕉原位放回记录",
            "",
            f"- 开始时间：{started.isoformat(timespec='seconds')}",
            f"- 结束时间：{finished.isoformat(timespec='seconds')}",
            "- 授权：用户明确回复“好的，放回”。",
            f"- 结果：{result_text}",
            "",
            "## 路径",
            "",
            "- 当前夹持位 → z=0.300 m 安全高度",
            "- → 水平对正原抓取位",
            "- → 下降到释放点上方 50 mm 并完成姿态调整",
            "- → 垂直下降到原释放点",
            "- → 夹爪张开并确认角度变化",
            "- → 垂直抬升 50 mm",
            "",
            "## 安全门",
            "",
            "- 首条运动命令前完成全部 5 段 Cartesian IK 预检。",
            "- IK 预检后重新等待新的 RobotInfo，不使用过期反馈。",
            "- 机械臂必须 AutoReady、fault_flag=0。",
            "- 夹爪必须在预检后仍报告到位，角度漂移不得超过 10°。",
            "- 到达释放位后位置误差不得超过 3 mm 才允许张开。",
            "- 夹爪报告到位且角度相对闭合值变化至少 5° 才允许抬升。",
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

    return return_code


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement that this command may move real hardware",
    )
    parser.add_argument("--mission-config", type=Path, default=DEFAULT_MISSION_CONFIG)
    parser.add_argument("--gripper-config", type=Path, default=DEFAULT_GRIPPER_CONFIG)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    args = parser.parse_args()
    if not args.execute:
        parser.error("refusing hardware motion without --execute")
    os.chdir(ROOT)
    return run(
        mission_config_path=args.mission_config.resolve(),
        gripper_config_path=args.gripper_config.resolve(),
        log_dir=args.log_dir.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
