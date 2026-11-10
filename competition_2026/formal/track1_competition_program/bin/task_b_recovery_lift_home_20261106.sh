set -o pipefail
ROOT="$PWD"
STAMP="$(date '+%Y%m%d_%H%M%S')"
MAIN_LOG="$ROOT/logs/task_b_recovery_lift_home_${STAMP}.log"
BRIDGE_LOG="$ROOT/logs/codroid_bridge_recovery_lift_home_${STAMP}.log"
CONFIG="${TRACK1_CONFIG:-$ROOT/config/track1_task_b_pick1_single_low_speed_live_20261106.yaml}"
PY="${TRACK1_PYTHON:-/usr/bin/python3}"
SITE="${TRACK1_SITE_PACKAGES:-/media/sunshine/新加卷/taskA/venv/lib/python3.10/site-packages}"
RUNTIME="${TRACK1_RUNTIME_MOUNT:-/media/sunshine/ESTUN_RUNTIME}"
BRIDGE_PID=""
cleanup() {
    if [ -n "$BRIDGE_PID" ] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill -TERM "$BRIDGE_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$BRIDGE_PID" 2>/dev/null || break; sleep 0.1; done
        kill -KILL "$BRIDGE_PID" 2>/dev/null || true
        wait "$BRIDGE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM
"$ROOT/bin/ensure_runtime" || exit 1
source /opt/ros/humble/setup.bash
source "$RUNTIME/ros2_ws/install/setup.bash"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export PYTHONPATH="$ROOT/app/track1_grasp:$SITE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
printf 'recovery log: %s\nbridge log: %s\n' "$MAIN_LOG" "$BRIDGE_LOG" | tee "$MAIN_LOG"
"$PY" -m taskA_grasp.nodes.udp_bridge --ros-args \
  -p "listen_port:=${TRACK1_BRIDGE_PORT:-9006}" \
  -p "controller_host:=${TRACK1_CONTROLLER_HOST:-192.168.101.100}" \
  -p "controller_port:=${TRACK1_CONTROLLER_PORT:-9005}" >"$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!
sleep 0.6
kill -0 "$BRIDGE_PID" 2>/dev/null || { tail -n 80 "$BRIDGE_LOG" | tee -a "$MAIN_LOG"; exit 1; }
timeout 180s "$PY" - "$CONFIG" <<'PY' 2>&1 | tee -a "$MAIN_LOG"
import math, sys
import threading
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from track1_grasp.config import load_config, load_joint_pose_file
from track1_grasp.domain import Pose
from track1_grasp.gripper import build_gripper
from track1_grasp.kinematics import joints_within_limits
from track1_grasp.motion import build_motion

cfg = load_config(sys.argv[1])
if cfg.motion.backend != "robotmove" or cfg.motion.dry_run:
    raise RuntimeError("recovery requires live robotmove motion")
if cfg.gripper.backend != "ms42dc":
    raise RuntimeError(f"unexpected gripper backend: {cfg.gripper.backend}")
target_q = np.asarray(load_joint_pose_file(cfg.task_b.initial_pose_file, expected_role="task_b_initial_pose"), dtype=float)
if target_q.shape != (6,) or not np.all(np.isfinite(target_q)):
    raise RuntimeError("invalid initial joint target")
if not joints_within_limits(target_q, math.radians(float(cfg.motion.joint_limit_margin_deg))):
    raise RuntimeError("initial pose violates joint limits")

def log(s): print(s, flush=True)
def ptext(p): return f"xyz=({p.xyz[0]:.6f}, {p.xyz[1]:.6f}, {p.xyz[2]:.6f}) rpy=({p.rpy[0]:.6f}, {p.rpy[1]:.6f}, {p.rpy[2]:.6f})"

rclpy.init()
node = Node("task_b_recovery_lift_home")
motion = gripper = executor = spin_thread = None
executor = SingleThreadedExecutor()
executor.add_node(node)
spin_thread = threading.Thread(
    target=executor.spin,
    name="task-b-recovery-ros",
    daemon=True,
)
spin_thread.start()
try:
    motion = build_motion(cfg.motion, node=node, log=log)
    gripper = build_gripper(cfg.gripper, log=log)
    gs = gripper.read_status()
    if gs is None:
        raise RuntimeError("gripper status read failed")
    log(f"[preflight] gripper arrived={gs.arrived} angle={gs.angle_deg:.1f} deg speed={gs.speed_rad_s:.2f} rad/s")
    if (
        not gs.arrived
        or gs.angle_deg > 100.0
        or abs(float(gs.speed_rad_s)) > 0.2
    ):
        raise RuntimeError("gripper not confirmed open and stationary; refusing arm motion")
    gripper.close_backend(); gripper = None

    st = motion.wait_for_state(5.0)
    if not st.ready or st.pose is None or st.joints is None:
        raise RuntimeError(f"robot preflight failed: {st.message}")
    q0 = np.asarray(st.joints, dtype=float)
    p0 = st.pose
    log(f"[preflight] robot {st.message} {ptext(p0)}")
    log(f"[preflight] joints_deg={np.round(np.degrees(q0),3).tolist()}")
    log(f"[preflight] initial_target_deg={np.round(np.degrees(target_q),3).tolist()} max_delta_deg={float(np.max(np.abs(np.degrees(target_q-q0)))):.3f}")

    pub = motion._publisher
    waiter = getattr(pub, "wait_for_matching_subscriptions", None)
    if callable(waiter):
        if not waiter(5.0): raise RuntimeError("RobotMove bridge subscriber not found")
    elif int(pub.get_subscription_count()) < 1:
        raise RuntimeError("RobotMove bridge subscriber not found")
    log("[preflight] /RobotMove bridge subscriber confirmed")

    z0 = float(p0.xyz[2]); lift_z = max(z0, 0.300)
    if lift_z > z0 + 1e-6:
        lift = Pose((float(p0.xyz[0]), float(p0.xyz[1]), lift_z), p0.rpy)
        log(f"[plan] vertical lift z {z0:.6f} -> {lift_z:.6f} m distance={lift_z-z0:.6f} m; XY/RPY unchanged")
        route = motion.precheck_linear_targets((lift,), step_m=float(cfg.motion.cartesian_step_m))
        seg = route.segments[0]
        if not math.isfinite(float(seg.min_jacobian_singular_value)) or float(seg.min_jacobian_singular_value) < float(cfg.motion.cartesian_min_jacobian_singular_value):
            raise RuntimeError("vertical lift failed Jacobian singularity gate")
        log(f"[preflight] lift route PASS waypoints={max(0,len(seg.waypoints)-1)} max_joint_step={seg.max_joint_step_deg:.3f} deg max_joint_accel={seg.max_joint_acceleration_deg:.3f} deg min_sigma={seg.min_jacobian_singular_value:.6f}")
        seq = motion.get_feedback_sequence(); fresh = motion.wait_for_feedback_state(3.0, newer_than=seq)
        if not fresh.ready or fresh.pose is None:
            raise RuntimeError(f"fresh post-IK state failed: {fresh.message}")
        drift = float(np.linalg.norm(np.asarray(fresh.pose.xyz)-np.asarray(p0.xyz)))
        if drift > 0.002: raise RuntimeError(f"robot drifted {drift*1000:.3f} mm during precheck")
        log(f"[preflight:post-ik] robot remained within {drift*1000:.3f} mm")
        motion.execute_linear_route(route, names=("task_b_recovery_vertical_lift",), speed_mps=float(cfg.motion.speed_mps))
    else:
        log("[plan] robot already at or above safe Z=0.300 m")

    seq = motion.get_feedback_sequence(); lifted = motion.wait_for_feedback_state(3.0, newer_than=seq)
    if not lifted.ready or lifted.pose is None or lifted.joints is None:
        raise RuntimeError(f"post-lift state failed: {lifted.message}")
    zerr = abs(float(lifted.pose.xyz[2])-lift_z)
    log(f"[post-lift] {ptext(lifted.pose)} z_error={zerr*1000:.3f} mm")
    if zerr > 0.003: raise RuntimeError("vertical lift endpoint verification failed")

    seq = motion.get_feedback_sequence(); latest = motion.wait_for_feedback_state(3.0, newer_than=seq)
    if not latest.ready or latest.joints is None:
        raise RuntimeError(f"robot not ready before initial return: {latest.message}")
    log("[motion] moving six axes to recorded Task B initial pose")
    motion.move_joints(target_q, name="task_b_recovery_return_initial")
    seq = motion.get_feedback_sequence(); final = motion.wait_for_feedback_state(3.0, newer_than=seq)
    if not final.ready or final.pose is None or final.joints is None:
        raise RuntimeError(f"final state failed: {final.message}")
    fq = np.asarray(final.joints, dtype=float)
    residual = float(np.max(np.abs(np.degrees(fq-target_q))))
    log(f"[final] {ptext(final.pose)} joints_deg={np.round(np.degrees(fq),3).tolist()} residual_deg={residual:.3f}")
    if residual > 0.5: raise RuntimeError(f"initial pose residual {residual:.3f} deg exceeds 0.500 deg")
    log("[success] vertical lift and Task B initial-pose return verified")
finally:
    if gripper is not None:
        try: gripper.close_backend()
        except Exception: pass
    if motion is not None:
        try: motion.close()
        except Exception: pass
    if executor is not None:
        try: executor.shutdown()
        except Exception: pass
    if spin_thread is not None:
        spin_thread.join(timeout=2.0)
        if spin_thread.is_alive():
            print("[cleanup] ROS executor thread did not stop within 2.0 s", flush=True)
    if executor is not None:
        try: executor.remove_node(node)
        except Exception: pass
    node.destroy_node()
    if rclpy.ok(): rclpy.shutdown()
PY
STATUS=${PIPESTATUS[0]}
printf '[recovery] exit=%s\n' "$STATUS" | tee -a "$MAIN_LOG"
if [ "$STATUS" -ne 0 ]; then tail -n 80 "$BRIDGE_LOG" | tee -a "$MAIN_LOG"; fi
exit "$STATUS"
