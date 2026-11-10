"""One-shot, non-grasping alignment of the pick_4 observation pose.

The operator explicitly authorized exactly one Cartesian observation move:
25.243 mm toward the current banana waist centre at 0.12 m/s.  This tool never
constructs a gripper backend and never publishes anything except the single
pre-checked joint trajectory produced by ``RobotMoveMotionBackend``.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    ROOT / "config/track1_task_b_five_point_full_live_20261106.yaml"
)

EXPECTED_START_XYZ = (0.056669, 0.417884, 0.256339)
EXPECTED_START_RPY = (-3.112935, 0.045258, -0.347986)
TARGET_XYZ = (0.031712, 0.421674, 0.256339)
TARGET_RPY = EXPECTED_START_RPY
AUTHORIZED_SPEED_MPS = 0.12
EXPECTED_DISTANCE_M = 0.025243
START_POSITION_TOLERANCE_M = 0.003
START_ROTATION_TOLERANCE_RAD = 0.02
ENDPOINT_POSITION_TOLERANCE_M = 0.003
ENDPOINT_ROTATION_TOLERANCE_RAD = 0.02


def _pose_errors(actual, expected_xyz, expected_rpy) -> tuple[float, float]:
    from track1_grasp.kinematics import rotation_angle_error, rpy_to_matrix

    position_error = float(
        np.linalg.norm(np.asarray(actual.xyz) - np.asarray(expected_xyz))
    )
    rotation_error = float(
        rotation_angle_error(
            rpy_to_matrix(np.asarray(actual.rpy)),
            rpy_to_matrix(np.asarray(expected_rpy)),
        )
    )
    return position_error, rotation_error


def _pose_text(pose) -> str:
    return (
        f"xyz=({pose.xyz[0]:.6f}, {pose.xyz[1]:.6f}, {pose.xyz[2]:.6f}) "
        f"rpy=({pose.rpy[0]:.6f}, {pose.rpy[1]:.6f}, {pose.rpy[2]:.6f})"
    )


def run(config_path: Path, log_dir: Path) -> int:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    from track1_grasp.config import load_config
    from track1_grasp.domain import Pose
    from track1_grasp.motion import build_motion

    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"task_b_pick4_observation_align_{stamp}.log"

    def log(message: str) -> None:
        text = str(message)
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    log(f"[start] pick_4 observation alignment; log={log_path}")
    log("[authorization] one move only; no gripper, no grasp, no retry")

    cfg = load_config(config_path)
    if cfg.motion.backend != "robotmove":
        raise RuntimeError(f"unexpected motion backend: {cfg.motion.backend}")
    if bool(cfg.motion.dry_run):
        raise RuntimeError("live observation alignment requires motion.dry_run=false")
    if not math.isclose(
        float(cfg.motion.speed_mps),
        AUTHORIZED_SPEED_MPS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"configured speed {cfg.motion.speed_mps:.6f} m/s is not the "
            f"authorized {AUTHORIZED_SPEED_MPS:.3f} m/s"
        )

    target = Pose(TARGET_XYZ, TARGET_RPY)
    target_distance = float(
        np.linalg.norm(np.asarray(TARGET_XYZ) - np.asarray(EXPECTED_START_XYZ))
    )
    if abs(target_distance - EXPECTED_DISTANCE_M) > 1e-6:
        raise RuntimeError(
            f"authorized target distance mismatch: {target_distance:.9f} m"
        )
    log(
        f"[authorization-check] target={_pose_text(target)} "
        f"distance={target_distance:.6f} m speed={AUTHORIZED_SPEED_MPS:.3f} m/s"
    )

    rclpy.init()
    node = Node("task_b_pick4_observation_align_once")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin,
        name="task-b-pick4-observation-align-ros",
        daemon=True,
    )
    spin_thread.start()

    motion = None
    try:
        motion = build_motion(cfg.motion, node=node, log=log)
        start = motion.wait_for_feedback_state(5.0)
        if not start.ready or start.fault or start.pose is None or start.joints is None:
            raise RuntimeError(f"robot startup gate failed: {start.message}")
        start_position_error, start_rotation_error = _pose_errors(
            start.pose,
            EXPECTED_START_XYZ,
            EXPECTED_START_RPY,
        )
        log(
            f"[preflight] state={start.controller_state} fault={start.fault} "
            f"{_pose_text(start.pose)}"
        )
        log(
            f"[preflight] joints_deg="
            f"{np.round(np.degrees(np.asarray(start.joints)), 3).tolist()}"
        )
        log(
            f"[preflight] expected-start error position={start_position_error * 1000:.3f} mm "
            f"rotation={start_rotation_error:.6f} rad"
        )
        if start_position_error > START_POSITION_TOLERANCE_M:
            raise RuntimeError(
                f"current position differs from authorized pick_4 observation "
                f"by {start_position_error * 1000:.3f} mm"
            )
        if start_rotation_error > START_ROTATION_TOLERANCE_RAD:
            raise RuntimeError(
                f"current orientation differs by {start_rotation_error:.6f} rad"
            )

        publisher = getattr(motion, "_publisher", None)
        if publisher is None:
            raise RuntimeError("RobotMove publisher is unavailable")
        waiter = getattr(publisher, "wait_for_matching_subscriptions", None)
        if callable(waiter):
            if not waiter(5.0):
                raise RuntimeError("/RobotMove bridge subscriber not found")
        elif int(publisher.get_subscription_count()) < 1:
            raise RuntimeError("/RobotMove bridge subscriber not found")
        log("[preflight] /RobotMove bridge subscriber confirmed")

        log("[plan] full Cartesian IK/route/singularity precheck starting")
        route = motion.precheck_linear_targets(
            (target,),
            step_m=float(cfg.motion.cartesian_step_m),
        )
        if len(route.segments) != 1:
            raise RuntimeError("one-shot route did not produce exactly one segment")
        segment = route.segments[0]
        if len(segment.waypoints) < 2 or len(segment.joint_waypoints) != len(segment.waypoints):
            raise RuntimeError("prechecked segment has no complete joint plan")
        if not (0.020 <= segment.distance_m <= 0.030):
            raise RuntimeError(
                f"prechecked distance {segment.distance_m:.6f} m is outside "
                "the authorized 20-30 mm window"
            )
        log(
            f"[plan-pass] distance={segment.distance_m:.6f} m "
            f"waypoints={len(segment.waypoints) - 1} "
            f"max_joint_step={segment.max_joint_step_deg:.3f} deg "
            f"max_joint_accel={segment.max_joint_acceleration_deg:.3f} deg "
            f"min_sigma={segment.min_jacobian_singular_value:.6f}"
        )

        sequence = motion.get_feedback_sequence()
        fresh = motion.wait_for_feedback_state(3.0, newer_than=sequence)
        if not fresh.ready or fresh.fault or fresh.pose is None or fresh.joints is None:
            raise RuntimeError(f"post-precheck fresh state gate failed: {fresh.message}")
        drift_position, drift_rotation = _pose_errors(
            fresh.pose,
            segment.waypoints[0].xyz,
            segment.waypoints[0].rpy,
        )
        log(
            f"[post-ik] fresh feedback position drift={drift_position * 1000:.3f} mm "
            f"rotation drift={drift_rotation:.6f} rad"
        )
        if drift_position > START_POSITION_TOLERANCE_M:
            raise RuntimeError("robot moved after IK precheck; refusing stale route")
        if drift_rotation > START_ROTATION_TOLERANCE_RAD:
            raise RuntimeError("orientation changed after IK precheck; refusing stale route")

        if callable(waiter) and not waiter(2.0):
            raise RuntimeError("/RobotMove bridge subscriber disappeared")
        elif not callable(waiter) and int(publisher.get_subscription_count()) < 1:
            raise RuntimeError("/RobotMove bridge subscriber disappeared")

        log("[motion] sending the single authorized 0.12 m/s observation move")
        motion.execute_linear_route(
            route,
            names=("pick4_observation_align_once",),
            speed_mps=AUTHORIZED_SPEED_MPS,
        )

        sequence = motion.get_feedback_sequence()
        final = motion.wait_for_feedback_state(3.0, newer_than=sequence)
        if final.pose is None or final.joints is None:
            raise RuntimeError(f"final feedback incomplete: {final.message}")
        final_position_error, final_rotation_error = _pose_errors(
            final.pose,
            TARGET_XYZ,
            TARGET_RPY,
        )
        log(
            f"[final] state={final.controller_state} ready={final.ready} "
            f"fault={final.fault} {_pose_text(final.pose)}"
        )
        log(
            f"[final] joints_deg="
            f"{np.round(np.degrees(np.asarray(final.joints)), 3).tolist()}"
        )
        log(
            f"[final] target error position={final_position_error * 1000:.3f} mm "
            f"rotation={final_rotation_error:.6f} rad"
        )
        if not final.ready or final.fault:
            raise RuntimeError(f"final controller state gate failed: {final.message}")
        if final_position_error > ENDPOINT_POSITION_TOLERANCE_M:
            raise RuntimeError(
                f"final position error {final_position_error * 1000:.3f} mm "
                f"exceeds {ENDPOINT_POSITION_TOLERANCE_M * 1000:.3f} mm"
            )
        if final_rotation_error > ENDPOINT_ROTATION_TOLERANCE_RAD:
            raise RuntimeError(
                f"final rotation error {final_rotation_error:.6f} rad exceeds "
                f"{ENDPOINT_ROTATION_TOLERANCE_RAD:.6f} rad"
            )

        log("[success] pick_4 observation pose aligned; no grasp performed")
        log("VERDICT=OBSERVATION_ALIGNED")
        log(f"LOG={log_path}")
        return 0
    except BaseException as exc:
        log(f"[ABORT] {type(exc).__name__}: {exc}")
        log("VERDICT=ABORTED_NO_RETRY")
        log(f"LOG={log_path}")
        return 1
    finally:
        if motion is not None:
            try:
                motion.close()
            except Exception:
                pass
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        if spin_thread.is_alive():
            log("[cleanup] ROS executor thread did not stop within 2.0 s")
        try:
            executor.remove_node(node)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement for the one authorized real-arm move",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    args = parser.parse_args()
    if not args.execute:
        parser.error("refusing real-arm motion without --execute")
    return run(args.config.resolve(), args.log_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
