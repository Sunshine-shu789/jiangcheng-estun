"""One-shot J6-only view probe for the pick_4 observation pose.

The operator explicitly authorized exactly one real-arm move: rotate Joint6 by
+10 degrees while keeping Joint1..Joint5 fixed.  This changes the wrist-mounted
camera viewpoint without commanding a Cartesian translation.  The tool never
constructs a gripper backend, never closes the jaws, and never retries.
"""

from __future__ import annotations

import argparse
import math
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "config/track1_task_b_five_point_full_live_20261106.yaml"

# Feedback immediately before the confirmed one-shot view probe.
EXPECTED_START_JOINTS_DEG = (
    70.266,
    93.340,
    -80.361,
    -75.366,
    87.413,
    0.273,
)
EXPECTED_START_XYZ = (0.031706, 0.421677, 0.256342)
AUTHORIZED_J6_DELTA_DEG = 10.0
START_JOINT_TOLERANCE_DEG = 0.20
START_POSITION_TOLERANCE_M = 0.002
START_ROTATION_TOLERANCE_RAD = 0.02
ENDPOINT_JOINT_TOLERANCE_DEG = 0.05
ENDPOINT_POSITION_TOLERANCE_M = 0.001
ENDPOINT_ROTATION_TOLERANCE_RAD = 0.02
MAX_CAMERA_TANGENTIAL_SPEED_MPS = 0.12
CAMERA_MOUNT_RADIUS_M = 0.06519641930696288


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
    from track1_grasp.kinematics import (
        _controller_fk_and_jacobian,
        controller_fk,
        joints_within_limits,
        matrix_to_rpy,
        rotation_angle_error,
        rpy_to_matrix,
    )
    from track1_grasp.motion import build_motion

    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"task_b_pick4_j6_view_probe_{stamp}.log"

    def log(message: str) -> None:
        text = str(message)
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    log(f"[start] pick_4 J6-only view probe; log={log_path}")
    log(
        "[authorization] one J6-only move; J1-J5 fixed; no gripper; "
        "no grasp; no retry"
    )

    cfg = load_config(config_path)
    if cfg.motion.backend != "robotmove":
        raise RuntimeError(f"unexpected motion backend: {cfg.motion.backend}")
    if bool(cfg.motion.dry_run):
        raise RuntimeError("live J6 view probe requires motion.dry_run=false")
    joint_speed = float(cfg.motion.joint_speed_rad_s)
    if not math.isfinite(joint_speed) or joint_speed <= 0.0:
        raise RuntimeError("configured joint speed is invalid")
    camera_tangential_speed = joint_speed * CAMERA_MOUNT_RADIUS_M
    log(
        f"[authorization-check] J6 delta={AUTHORIZED_J6_DELTA_DEG:+.3f} deg "
        f"joint_speed={joint_speed:.3f} rad/s "
        f"camera_tangential_speed={camera_tangential_speed:.4f} m/s"
    )
    if camera_tangential_speed > MAX_CAMERA_TANGENTIAL_SPEED_MPS + 1e-12:
        raise RuntimeError(
            f"J6 camera tangential speed {camera_tangential_speed:.4f} m/s "
            f"exceeds the authorized {MAX_CAMERA_TANGENTIAL_SPEED_MPS:.3f} m/s"
        )

    rclpy.init()
    node = Node("task_b_pick4_j6_view_probe_once")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin,
        name="task-b-pick4-j6-view-probe-ros",
        daemon=True,
    )
    spin_thread.start()

    motion = None
    try:
        motion = build_motion(cfg.motion, node=node, log=log)
        start = motion.wait_for_feedback_state(5.0)
        if start.ready is False or start.fault or start.pose is None or start.joints is None:
            raise RuntimeError(f"robot startup gate failed: {start.message}")

        start_q = np.asarray(start.joints, dtype=float)
        expected_q = np.radians(np.asarray(EXPECTED_START_JOINTS_DEG, dtype=float))
        start_joint_error_deg = math.degrees(
            float(np.max(np.abs(start_q - expected_q)))
        )
        start_position_error, _ = _pose_errors(
            start.pose,
            EXPECTED_START_XYZ,
            start.pose.rpy,
        )
        start_fk = controller_fk(start_q)
        start_fk_rpy = matrix_to_rpy(start_fk[:3, :3])
        start_fk_rotation_error = float(
            rotation_angle_error(
                rpy_to_matrix(start.pose.rpy),
                rpy_to_matrix(start_fk_rpy),
            )
        )
        log(
            f"[preflight] state={start.controller_state} fault={start.fault} "
            f"{_pose_text(start.pose)}"
        )
        log(
            f"[preflight] joints_deg="
            f"{np.round(np.degrees(start_q), 3).tolist()} "
            f"start_joint_error={start_joint_error_deg:.3f} deg "
            f"start_position_error={start_position_error * 1000:.3f} mm "
            f"fk_rotation_error={start_fk_rotation_error:.6f} rad"
        )
        if start_joint_error_deg > START_JOINT_TOLERANCE_DEG:
            raise RuntimeError(
                f"current joints differ from the authorized start by "
                f"{start_joint_error_deg:.3f} deg"
            )
        if start_position_error > START_POSITION_TOLERANCE_M:
            raise RuntimeError(
                f"current TCP differs from the authorized start by "
                f"{start_position_error * 1000:.3f} mm"
            )
        if start_fk_rotation_error > START_ROTATION_TOLERANCE_RAD:
            raise RuntimeError(
                f"feedback/kinematics orientation mismatch is "
                f"{start_fk_rotation_error:.6f} rad"
            )

        target_q = start_q.copy()
        target_q[5] += math.radians(AUTHORIZED_J6_DELTA_DEG)
        delta_q = target_q - start_q
        non_j6_delta_deg = math.degrees(
            float(np.max(np.abs(np.delete(delta_q, 5))))
        )
        if non_j6_delta_deg > 1e-12:
            raise RuntimeError("J6 probe target unexpectedly changes J1-J5")
        if not joints_within_limits(
            target_q,
            math.radians(float(cfg.motion.joint_limit_margin_deg)),
        ):
            raise RuntimeError("J6 probe target violates the joint-limit margin")

        # Precheck the actual one-degree controller trajectory, including TCP
        # workspace, orientation continuity and Jacobian conditioning.
        steps = int(math.ceil(abs(AUTHORIZED_J6_DELTA_DEG) / 1.0))
        joint_waypoints = [
            start_q + delta_q * (index / steps) for index in range(1, steps + 1)
        ]
        low = np.asarray(cfg.motion.workspace_min, dtype=float) + float(
            cfg.motion.workspace_margin
        )
        high = np.asarray(cfg.motion.workspace_max, dtype=float) - float(
            cfg.motion.workspace_margin
        )
        tcp_positions = []
        jacobian_sigmas = []
        for index, waypoint in enumerate(joint_waypoints, start=1):
            transform, jacobian = _controller_fk_and_jacobian(waypoint)
            xyz = np.asarray(transform[:3, 3], dtype=float)
            tcp_positions.append(xyz)
            if np.any(xyz < low - 1e-9) or np.any(xyz > high + 1e-9):
                raise RuntimeError(
                    f"J6 probe waypoint {index}/{steps} leaves the workspace: "
                    f"xyz={tuple(round(float(v), 6) for v in xyz)}"
                )
            sigma = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
            jacobian_sigmas.append(sigma)
            if sigma < float(cfg.motion.cartesian_min_jacobian_singular_value):
                raise RuntimeError(
                    f"J6 probe waypoint {index}/{steps} is singular: "
                    f"sigma={sigma:.6f}"
                )
        tcp_positions_array = np.asarray(tcp_positions)
        tcp_translation = float(
            np.max(
                np.linalg.norm(
                    tcp_positions_array - np.asarray(start.pose.xyz, dtype=float),
                    axis=1,
                )
            )
        )
        endpoint_t = controller_fk(target_q)
        endpoint_xyz = tuple(float(v) for v in endpoint_t[:3, 3])
        endpoint_rpy = tuple(float(v) for v in matrix_to_rpy(endpoint_t[:3, :3]))
        if tcp_translation > ENDPOINT_POSITION_TOLERANCE_M:
            raise RuntimeError(
                f"J6-only route would translate TCP by {tcp_translation * 1000:.3f} mm"
            )
        log(
            f"[plan-pass] steps={steps} "
            f"j6_start_deg={math.degrees(start_q[5]):.3f} "
            f"j6_target_deg={math.degrees(target_q[5]):.3f} "
            f"J1-J5_change_deg={non_j6_delta_deg:.9f} "
            f"max_tcp_translation={tcp_translation * 1000:.6f} mm "
            f"min_sigma={min(jacobian_sigmas):.6f} "
            f"endpoint_rpy_deg={np.round(np.degrees(endpoint_rpy), 3).tolist()}"
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

        sequence = motion.get_feedback_sequence()
        fresh = motion.wait_for_feedback_state(3.0, newer_than=sequence)
        if fresh.ready is False or fresh.fault or fresh.joints is None:
            raise RuntimeError(f"fresh feedback gate failed: {fresh.message}")
        fresh_q = np.asarray(fresh.joints, dtype=float)
        fresh_error_deg = math.degrees(float(np.max(np.abs(fresh_q - start_q))))
        if fresh_error_deg > START_JOINT_TOLERANCE_DEG:
            raise RuntimeError(
                f"robot moved after precheck by {fresh_error_deg:.3f} deg"
            )
        if callable(waiter) and not waiter(2.0):
            raise RuntimeError("/RobotMove bridge subscriber disappeared")
        elif not callable(waiter) and int(publisher.get_subscription_count()) < 1:
            raise RuntimeError("/RobotMove bridge subscriber disappeared")

        log("[motion] sending the single authorized J6 +10 degree view probe")
        motion.move_joints(
            target_q,
            name="pick4_j6_view_probe_once",
            max_step_deg=1.0,
        )

        sequence = motion.get_feedback_sequence()
        final = motion.wait_for_feedback_state(3.0, newer_than=sequence)
        if final.pose is None or final.joints is None:
            raise RuntimeError(f"final feedback incomplete: {final.message}")
        final_q = np.asarray(final.joints, dtype=float)
        target_error_deg = math.degrees(float(np.max(np.abs(final_q - target_q))))
        non_j6_final_change_deg = math.degrees(
            float(np.max(np.abs((final_q - start_q)[:5])))
        )
        actual_j6_delta_deg = math.degrees(float(final_q[5] - start_q[5]))
        final_position_error, final_rotation_error = _pose_errors(
            final.pose,
            endpoint_xyz,
            endpoint_rpy,
        )
        log(
            f"[final] state={final.controller_state} ready={final.ready} "
            f"fault={final.fault} {_pose_text(final.pose)}"
        )
        log(
            f"[final] joints_deg={np.round(np.degrees(final_q), 3).tolist()} "
            f"target_error={target_error_deg:.3f} deg "
            f"j6_delta={actual_j6_delta_deg:.3f} deg "
            f"J1-J5_change={non_j6_final_change_deg:.3f} deg "
            f"tcp_error={final_position_error * 1000:.3f} mm "
            f"orientation_error={final_rotation_error:.6f} rad"
        )
        if not final.ready or final.fault:
            raise RuntimeError(f"final controller state gate failed: {final.message}")
        if target_error_deg > ENDPOINT_JOINT_TOLERANCE_DEG:
            raise RuntimeError(
                f"final joint error {target_error_deg:.3f} deg exceeds "
                f"{ENDPOINT_JOINT_TOLERANCE_DEG:.3f} deg"
            )
        if non_j6_final_change_deg > ENDPOINT_JOINT_TOLERANCE_DEG:
            raise RuntimeError(
                f"J1-J5 moved by {non_j6_final_change_deg:.3f} deg"
            )
        if abs(actual_j6_delta_deg - AUTHORIZED_J6_DELTA_DEG) > ENDPOINT_JOINT_TOLERANCE_DEG:
            raise RuntimeError(
                f"actual J6 delta {actual_j6_delta_deg:.3f} deg does not match "
                f"the authorized {AUTHORIZED_J6_DELTA_DEG:.3f} deg"
            )
        if final_position_error > ENDPOINT_POSITION_TOLERANCE_M:
            raise RuntimeError(
                f"final TCP translation {final_position_error * 1000:.3f} mm "
                f"exceeds {ENDPOINT_POSITION_TOLERANCE_M * 1000:.3f} mm"
            )
        if final_rotation_error > ENDPOINT_ROTATION_TOLERANCE_RAD:
            raise RuntimeError(
                f"final orientation error {final_rotation_error:.6f} rad exceeds "
                f"{ENDPOINT_ROTATION_TOLERANCE_RAD:.6f} rad"
            )

        log("[success] J6-only view probe completed; no gripper or grasp command")
        log("VERDICT=J6_VIEW_PROBE_COMPLETED")
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
        help="required acknowledgement for the one authorized J6-only move",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    args = parser.parse_args()
    if not args.execute:
        parser.error("refusing real-arm motion without --execute")
    return run(args.config.resolve(), args.log_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
