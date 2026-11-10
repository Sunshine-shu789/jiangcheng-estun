#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAMP="$(date '+%Y%m%d_%H%M%S')"
MAIN_LOG="$ROOT/logs/task_b_recovery_direct_joint_home_${STAMP}.log"
BRIDGE_LOG="$ROOT/logs/codroid_bridge_recovery_direct_joint_home_${STAMP}.log"
CONFIG="${TRACK1_CONFIG:-$ROOT/config/track1_task_b_pick1_single_low_speed_live_20261106.yaml}"
EXPECTED_CONFIG="${TRACK1_EXPECTED_POSE_CONFIG:-$ROOT/config/track1_task_b_five_point_full_live_20261106.yaml}"
POINT_INDEX="${TRACK1_RECOVERY_POINT_INDEX:-1}"
CONFIRM="${TRACK1_CONFIRM:-0}"
PY="${TRACK1_PYTHON:-/usr/bin/python3}"
SITE="${TRACK1_SITE_PACKAGES:-/media/sunshine/新加卷/taskA/venv/lib/python3.10/site-packages}"
RUNTIME="${TRACK1_RUNTIME_MOUNT:-/media/sunshine/ESTUN_RUNTIME}"
BRIDGE_PID=""

cleanup() {
    if [ -n "$BRIDGE_PID" ] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill -TERM "$BRIDGE_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$BRIDGE_PID" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL "$BRIDGE_PID" 2>/dev/null || true
        wait "$BRIDGE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"$ROOT/bin/ensure_runtime"
source /opt/ros/humble/setup.bash
source "$RUNTIME/ros2_ws/install/setup.bash"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export PYTHONPATH="$ROOT/app/track1_grasp:$SITE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

printf 'direct-joint recovery log: %s\nbridge log: %s\nconfirm=%s\n' \
    "$MAIN_LOG" "$BRIDGE_LOG" "$CONFIRM" | tee "$MAIN_LOG"

"$PY" -m taskA_grasp.nodes.udp_bridge --ros-args \
    -p "listen_port:=${TRACK1_BRIDGE_PORT:-9006}" \
    -p "controller_host:=${TRACK1_CONTROLLER_HOST:-192.168.101.100}" \
    -p "controller_port:=${TRACK1_CONTROLLER_PORT:-9005}" \
    >"$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!
sleep 0.6
kill -0 "$BRIDGE_PID" 2>/dev/null || {
    tail -n 80 "$BRIDGE_LOG" | tee -a "$MAIN_LOG"
    exit 1
}

timeout 180s "$PY" - "$CONFIG" "$EXPECTED_CONFIG" "$POINT_INDEX" "$CONFIRM" <<'PY' 2>&1 | tee -a "$MAIN_LOG"
import math
import sys
import threading

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from track1_grasp.config import load_config, load_joint_pose_file
from track1_grasp.gripper import build_gripper
from track1_grasp.kinematics import controller_fk, joints_within_limits
from track1_grasp.motion import build_motion

config_path, expected_config_path = sys.argv[1], sys.argv[2]
point_index = int(sys.argv[3])
confirm = sys.argv[4].strip().lower() in {"1", "true", "yes", "on"}

cfg = load_config(config_path)
expected_cfg = load_config(expected_config_path)
if cfg.motion.backend != "robotmove" or cfg.motion.dry_run:
    raise RuntimeError("recovery requires live robotmove motion")
if cfg.gripper.backend != "ms42dc":
    raise RuntimeError(f"unexpected gripper backend: {cfg.gripper.backend}")
if not (0 <= point_index < len(expected_cfg.task_b.point_observation_joints_deg)):
    raise RuntimeError(f"missing observation posture index {point_index}")
expected_deg = expected_cfg.task_b.point_observation_joints_deg[point_index]
if expected_deg is None:
    raise RuntimeError(f"observation posture index {point_index} is empty")
expected_q = np.radians(np.asarray(expected_deg, dtype=float))

target_q = np.asarray(
    load_joint_pose_file(
        cfg.task_b.initial_pose_file,
        expected_role="task_b_initial_pose",
    ),
    dtype=float,
)
if target_q.shape != (6,) or not np.all(np.isfinite(target_q)):
    raise RuntimeError("invalid initial joint target")
margin = math.radians(float(cfg.motion.joint_limit_margin_deg))
if not joints_within_limits(target_q, margin):
    raise RuntimeError("initial pose violates joint limits")


def log(message: str) -> None:
    print(message, flush=True)


def ptext(pose) -> str:
    return (
        f"xyz=({pose.xyz[0]:.6f}, {pose.xyz[1]:.6f}, {pose.xyz[2]:.6f}) "
        f"rpy=({pose.rpy[0]:.6f}, {pose.rpy[1]:.6f}, {pose.rpy[2]:.6f})"
    )


rclpy.init()
node = Node("task_b_recovery_direct_joint_home")
executor = SingleThreadedExecutor()
executor.add_node(node)
spin_thread = threading.Thread(
    target=executor.spin,
    name="task-b-direct-recovery-ros",
    daemon=True,
)
spin_thread.start()
motion = None
gripper = None
try:
    motion = build_motion(cfg.motion, node=node, log=log)
    gripper = build_gripper(cfg.gripper, log=log)

    gs = gripper.read_status()
    if gs is None:
        raise RuntimeError("gripper status read failed")
    log(
        f"[preflight] gripper arrived={gs.arrived} angle={gs.angle_deg:.1f} deg "
        f"speed={gs.speed_rad_s:.2f} rad/s"
    )
    if (
        not gs.arrived
        or gs.angle_deg > 100.0
        or abs(float(gs.speed_rad_s)) > 0.2
    ):
        raise RuntimeError("gripper not confirmed open and stationary; refusing arm motion")

    state = motion.wait_for_state(5.0)
    if not state.ready or state.pose is None or state.joints is None:
        raise RuntimeError(f"robot preflight failed: {state.message}")
    current_q = np.asarray(state.joints, dtype=float)
    current_pose = state.pose
    if current_q.shape != (6,) or not np.all(np.isfinite(current_q)):
        raise RuntimeError("invalid current joint feedback")
    if not joints_within_limits(current_q, margin):
        raise RuntimeError("current joints violate configured limits")
    if float(current_pose.xyz[2]) < 0.220:
        raise RuntimeError(
            f"current Z={float(current_pose.xyz[2]):.6f} m is below 0.220 m; "
            "direct joint-home recovery is not permitted"
        )

    current_expected_error_deg = float(
        np.max(np.degrees(np.abs(current_q - expected_q)))
    )
    if current_expected_error_deg > 2.0:
        raise RuntimeError(
            f"current joints are {current_expected_error_deg:.3f} deg from "
            f"pick_{point_index + 1} observation posture; limit is 2.000 deg"
        )

    delta_deg = np.degrees(target_q - current_q)
    max_delta_deg = float(np.max(np.abs(delta_deg)))
    if max_delta_deg > 30.0:
        raise RuntimeError(
            f"direct-home joint delta {max_delta_deg:.3f} deg exceeds 30.000 deg"
        )
    if max_delta_deg <= float(cfg.task_b.initial_pose_tolerance_deg):
        log("[preflight] robot already inside Task B initial-pose tolerance; no motion needed")
        confirm = False

    samples = np.linspace(current_q, target_q, 1001)
    z_values = np.asarray([float(controller_fk(q)[2, 3]) for q in samples])
    min_z = float(np.min(z_values))
    max_z = float(np.max(z_values))
    if min_z < 0.220:
        raise RuntimeError(
            f"direct joint-home path minimum Z={min_z:.6f} m is below 0.220 m"
        )
    if z_values[-1] + 1e-9 < z_values[0]:
        raise RuntimeError("direct joint-home path does not finish higher than it starts")

    log(f"[preflight] robot {state.message} {ptext(current_pose)}")
    log(f"[preflight] current_joints_deg={np.round(np.degrees(current_q), 3).tolist()}")
    log(
        f"[preflight] expected_pick_{point_index + 1}_deg="
        f"{np.round(expected_deg, 3).tolist()} "
        f"max_error_deg={current_expected_error_deg:.3f}"
    )
    log(f"[preflight] initial_target_deg={np.round(np.degrees(target_q), 3).tolist()}")
    log(
        f"[preflight] direct_delta_deg={np.round(delta_deg, 3).tolist()} "
        f"max_delta_deg={max_delta_deg:.3f}"
    )
    log(
        f"[preflight] direct_joint_path_z_min={min_z:.6f} m "
        f"z_max={max_z:.6f} m z_end={z_values[-1]:.6f} m"
    )
    log(
        f"[preflight] joint_velocity={float(cfg.motion.joint_speed_rad_s):.3f} rad/s "
        f"safety_limit={float(cfg.motion.joint_speed_safety_limit_rad_s):.3f} rad/s"
    )

    pub = motion._publisher
    waiter = getattr(pub, "wait_for_matching_subscriptions", None)
    if callable(waiter):
        if not waiter(5.0):
            raise RuntimeError("RobotMove bridge subscriber not found")
    elif int(pub.get_subscription_count()) < 1:
        raise RuntimeError("RobotMove bridge subscriber not found")
    log("[preflight] /RobotMove bridge subscriber confirmed")

    if not confirm:
        log("[preflight-pass] no arm motion was commanded; set TRACK1_CONFIRM=1 after approval")
    else:
        gripper.close_backend()
        gripper = None
        log("[motion] sending one direct six-axis return to Task B initial pose; no retry")
        motion.move_joints(target_q, name="task_b_recovery_direct_joint_home")

        seq = motion.get_feedback_sequence()
        final = motion.wait_for_feedback_state(3.0, newer_than=seq)
        if not final.ready or final.pose is None or final.joints is None:
            raise RuntimeError(f"final state failed: {final.message}")
        final_q = np.asarray(final.joints, dtype=float)
        residual_deg = float(np.max(np.degrees(np.abs(final_q - target_q))))
        log(
            f"[final] {ptext(final.pose)} "
            f"joints_deg={np.round(np.degrees(final_q), 3).tolist()} "
            f"residual_deg={residual_deg:.3f}"
        )
        if residual_deg > float(cfg.task_b.initial_pose_tolerance_deg):
            raise RuntimeError(
                f"initial pose residual {residual_deg:.3f} deg exceeds "
                f"{float(cfg.task_b.initial_pose_tolerance_deg):.3f} deg"
            )
        log("[success] direct-joint return to Task B initial pose verified")
finally:
    if gripper is not None:
        try:
            gripper.close_backend()
        except Exception:
            pass
    if motion is not None:
        try:
            motion.close()
        except Exception:
            pass
    try:
        executor.shutdown()
    except Exception:
        pass
    spin_thread.join(timeout=2.0)
    if spin_thread.is_alive():
        print("[cleanup] ROS executor thread did not stop within 2.0 s", flush=True)
    try:
        executor.remove_node(node)
    except Exception:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
PY

STATUS=${PIPESTATUS[0]}
printf '[recovery] exit=%s\n' "$STATUS" | tee -a "$MAIN_LOG"
if [ "$STATUS" -ne 0 ]; then
    tail -n 80 "$BRIDGE_LOG" | tee -a "$MAIN_LOG"
fi
exit "$STATUS"
