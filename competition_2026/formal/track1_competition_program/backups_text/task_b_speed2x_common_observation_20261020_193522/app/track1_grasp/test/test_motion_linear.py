import math
import threading

import numpy as np
import pytest

from track1_grasp.config import MotionConfig
from track1_grasp.domain import Pose, RobotState
from track1_grasp.kinematics import (
    controller_fk,
    matrix_to_rpy,
    rotation_angle_error,
    solve_pose_ik,
    solve_pose_ik_multistart,
)
from track1_grasp.motion import (
    LinearSegmentPlan,
    ROBOT_CONTROLLER_MAX_TRAJECTORY_BYTES,
    ROBOT_CONTROLLER_MAX_TRAJECTORY_POINTS,
    RobotMoveMotionBackend,
    UDP_MAX_DATAGRAM_BYTES,
    linear_pose_waypoints,
)


def test_linear_pose_waypoints_keep_every_step_at_or_below_one_centimetre():
    start = Pose((0.1, 0.2, 0.3), (math.pi, 0.0, 0.0))
    target = Pose((0.3, 0.2, 0.3), (math.pi, 0.0, 0.0))

    waypoints = linear_pose_waypoints(start, target, step_m=0.01)

    assert waypoints[0] == start
    assert waypoints[-1].xyz == target.xyz
    assert len(waypoints) == 21
    assert all(
        math.dist(left.xyz, right.xyz) <= 0.01 + 1e-9
        for left, right in zip(waypoints, waypoints[1:])
    )
    assert all(
        abs(left.xyz[1] - 0.2) < 1e-12 and abs(left.xyz[2] - 0.3) < 1e-12
        for left in waypoints
    )


def test_linear_waypoint_count_is_not_inflated_by_float_roundoff():
    start = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    target = Pose((0.010, 0.0, 0.0), (0.0, 0.0, 0.0))

    waypoints = linear_pose_waypoints(start, target, step_m=0.0025)

    assert len(waypoints) == 5


def test_linear_pose_waypoints_collapse_an_unchanged_pose():
    start = Pose((0.1, 0.2, 0.3), (math.pi, 0.0, 0.0))

    waypoints = linear_pose_waypoints(start, start, step_m=0.001)

    assert waypoints == (start,)


def test_pose_ik_handles_small_move_from_live_singular_home_pose():
    joints = np.radians([90.0, 0.0, 0.0, 0.0, 90.0, 0.0])
    transform = controller_fk(joints)
    target_rotation = transform[:3, :3]
    target_position = transform[:3, 3].copy()
    target_position[1] += 0.010

    solved = solve_pose_ik(joints, target_position, target_rotation)
    solved_transform = controller_fk(solved)

    assert np.linalg.norm(solved_transform[:3, 3] - target_position) < 4e-4
    assert rotation_angle_error(solved_transform[:3, :3], target_rotation) < 4e-4
    assert np.max(np.abs(np.degrees(solved - joints))) < 2.0


def test_pose_ik_multistart_stops_after_first_step_bounded_solution(monkeypatch):
    joints = np.radians([94.191, 71.955, -16.289, -34.463, 88.164, 0.248])
    transform = controller_fk(joints)
    calls = []

    def solved(_q, _position, _rotation, *, damping, **_kwargs):
        calls.append(float(damping))
        return joints.copy()

    monkeypatch.setattr("track1_grasp.kinematics.solve_pose_ik", solved)

    result = solve_pose_ik_multistart(
        joints,
        transform[:3, 3],
        transform[:3, :3],
        damping_candidates=(0.01, 0.02, 0.03),
        accept_delta_rad=math.radians(0.5),
    )

    assert np.allclose(result, joints)
    assert calls == [0.01]


def _robot_backend(monkeypatch, state, *, dry_run, **config_overrides):
    values = {
        "backend": "robotmove",
        "dry_run": dry_run,
        "workspace_min": (-0.65, -0.65, 0.02),
        "workspace_max": (0.65, 0.65, 0.80),
        "workspace_margin": 0.01,
    }
    values.update(config_overrides)
    config = MotionConfig(**values)
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = config
    backend._log = lambda message: None
    monkeypatch.setattr(backend, "wait_for_state", lambda timeout: state)
    monkeypatch.setattr(backend, "get_state", lambda: state)
    return backend


def test_robot_linear_precheck_builds_smooth_axis_bounded_joint_path(monkeypatch):
    joints = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    transform = controller_fk(joints)
    start_pose = Pose(
        tuple(float(v) for v in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    target = Pose(
        (start_pose.xyz[0] + 0.03, start_pose.xyz[1], start_pose.xyz[2]),
        start_pose.rpy,
    )
    state = RobotState(
        ready=True,
        pose=start_pose,
        joints=tuple(float(v) for v in joints),
        message="AutoReady",
    )
    backend = _robot_backend(monkeypatch, state, dry_run=True)

    route = backend.precheck_linear_targets((target,), step_m=0.01)

    assert len(route.segments) == 1
    segment = route.segments[0]
    assert abs(segment.distance_m - 0.03) < 1e-9
    assert len(segment.joint_waypoints) == len(segment.waypoints)
    assert len(segment.joint_waypoints) > 2
    assert segment.planning_step_m <= backend.config.cartesian_step_m
    assert segment.max_joint_step_deg <= backend.config.cartesian_joint_step_deg
    assert (
        segment.max_joint_acceleration_deg
        <= backend.config.cartesian_joint_acceleration_limit_deg
    )
    assert segment.joint_metrics is not None
    assert len(segment.joint_metrics.max_step_deg) == 6
    assert len(segment.joint_metrics.max_acceleration_deg) == 6
    assert len(segment.joint_metrics.travel_deg) == 6
    assert max(segment.joint_metrics.travel_deg) > (
        10.0 * min(segment.joint_metrics.travel_deg)
    )
    assert backend._axis_metric_list(segment, "max_step_deg").count(",") == 5
    deltas_deg = np.degrees(
        np.diff(np.asarray(segment.joint_waypoints, dtype=float), axis=0)
    )
    assert float(np.max(np.abs(deltas_deg))) <= (
        backend.config.cartesian_joint_step_deg + 1e-9
    )
    if len(deltas_deg) > 1:
        acceleration = np.diff(deltas_deg, axis=0)
        assert float(np.max(np.abs(acceleration))) <= (
            backend.config.cartesian_joint_acceleration_limit_deg + 1e-9
        )


def test_robot_unchanged_prechecked_segment_is_a_noop(monkeypatch):
    joints = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    transform = controller_fk(joints)
    pose = Pose(
        tuple(float(v) for v in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    state = RobotState(
        ready=True,
        pose=pose,
        joints=tuple(float(v) for v in joints),
        message="AutoReady",
    )
    backend = _robot_backend(monkeypatch, state, dry_run=False)
    route = backend.precheck_linear_targets((pose,), step_m=0.001)
    segment = route.segments[0]
    assert segment.distance_m == 0.0
    assert segment.waypoints == (pose,)
    assert segment.joint_waypoints == (tuple(float(v) for v in joints),)

    sent = []
    monkeypatch.setattr(
        backend,
        "_send_joint_waypoint",
        lambda *args, **kwargs: sent.append((args, kwargs)),
    )

    backend.execute_prechecked_segment(segment, name="transfer", speed_mps=0.01)

    assert sent == []


def test_robot_prechecked_segment_publishes_one_complete_trajectory(monkeypatch):
    start_q = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    end_q = start_q + np.radians([0.2, -0.2, 0.3, -0.3, 0.2, -0.2])
    start_transform = controller_fk(start_q)
    end_transform = controller_fk(end_q)
    start_pose = Pose(
        tuple(float(value) for value in start_transform[:3, 3]),
        matrix_to_rpy(start_transform[:3, :3]),
    )
    target = Pose(
        tuple(float(value) for value in end_transform[:3, 3]),
        matrix_to_rpy(end_transform[:3, :3]),
    )
    joint_waypoints = (
        tuple(float(value) for value in start_q),
        tuple(float(value) for value in (start_q + end_q) / 2.0),
        tuple(float(value) for value in end_q),
    )
    segment = LinearSegmentPlan(
        target=target,
        waypoints=(start_pose, start_pose, target),
        joint_waypoints=joint_waypoints,
    )
    start_state = RobotState(
        ready=True,
        pose=start_pose,
        joints=joint_waypoints[0],
        message="AutoReady",
    )
    end_state = RobotState(
        ready=True,
        pose=target,
        joints=joint_waypoints[-1],
        message="AutoReady",
    )
    backend = _robot_backend(monkeypatch, start_state, dry_run=False)
    current = {"state": start_state}
    monkeypatch.setattr(backend, "wait_for_state", lambda timeout: current["state"])
    sent = []

    def send_complete_trajectory(targets, *, name, timeout_s, reject_timeout_s):
        sent.append(
            (
                name,
                np.asarray(targets, dtype=float),
                float(timeout_s),
                float(reject_timeout_s),
            )
        )
        current["state"] = end_state
        return np.asarray(targets, dtype=float)[-1]

    monkeypatch.setattr(backend, "_send_joint_trajectory", send_complete_trajectory)

    backend.execute_prechecked_segment(segment, name="grasp", speed_mps=0.06)

    assert len(sent) == 1
    name, targets, timeout_s, _reject_timeout_s = sent[0]
    assert name == "grasp"
    assert targets == pytest.approx(np.asarray(joint_waypoints[1:]))
    assert timeout_s >= 11.0


def test_robot_move_joints_publishes_all_interpolation_points_once(monkeypatch):
    start_q = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    target_q = start_q.copy()
    target_q[0] += math.radians(1.2)
    transform = controller_fk(start_q)
    start_pose = Pose(
        tuple(float(value) for value in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    state = RobotState(
        ready=True,
        pose=start_pose,
        joints=tuple(float(value) for value in start_q),
        message="AutoReady",
    )
    backend = _robot_backend(
        monkeypatch,
        state,
        dry_run=False,
        joint_max_step_deg=0.5,
    )
    sent = []

    def send_complete_trajectory(targets, *, name, timeout_s, reject_timeout_s):
        sent.append((name, np.asarray(targets, dtype=float), timeout_s))
        return np.asarray(targets, dtype=float)[-1]

    monkeypatch.setattr(backend, "_send_joint_trajectory", send_complete_trajectory)

    result = backend.move_joints(target_q, name="initial-return")

    assert np.allclose(result, target_q)
    assert len(sent) == 1
    name, targets, timeout_s = sent[0]
    assert name == "initial-return"
    assert len(targets) == 3
    assert targets[-1] == pytest.approx(target_q)
    assert timeout_s >= 11.5


def test_robot_precheck_refines_until_every_axis_step_is_bounded(monkeypatch):
    joints = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    transform = controller_fk(joints)
    start_pose = Pose(
        tuple(float(v) for v in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    target = Pose(
        (start_pose.xyz[0] + 0.01, start_pose.xyz[1], start_pose.xyz[2]),
        start_pose.rpy,
    )
    state = RobotState(
        ready=True,
        pose=start_pose,
        joints=tuple(float(v) for v in joints),
        message="AutoReady",
    )
    backend = _robot_backend(monkeypatch, state, dry_run=True)
    endpoint = np.asarray(joints, dtype=float).copy()
    endpoint[2] += math.radians(9.0)

    def synthetic_continuous_solver(q_initial, waypoint, *, label):
        fraction = (waypoint.xyz[0] - start_pose.xyz[0]) / 0.01
        candidate = np.asarray(joints, dtype=float) + fraction * (
            endpoint - np.asarray(joints, dtype=float)
        )
        return candidate, 0.1

    monkeypatch.setattr(
        backend,
        "_solve_checked_waypoint",
        synthetic_continuous_solver,
    )

    route = backend.precheck_linear_targets((target,), step_m=0.01)

    segment = route.segments[0]
    assert len(segment.waypoints) > 2
    assert segment.waypoints[-1] == target
    assert np.allclose(segment.joint_waypoints[-1], endpoint)
    assert segment.joint_metrics is not None
    assert max(segment.joint_metrics.max_step_deg) <= (
        backend.config.cartesian_joint_step_deg + 1e-9
    )
    assert max(segment.joint_metrics.max_acceleration_deg) <= (
        backend.config.cartesian_joint_acceleration_limit_deg + 1e-9
    )


def test_recorded_banana_approach_keeps_base_axis_quiet(monkeypatch):
    """Preserve the coordinated-axis result from the recorded 60 mm approach.

    The base Joint1 should make only the small yaw correction required by the
    target.  Joint2/Joint3/Joint4 carry the main posture change, while every
    axis remains on a continuous, finely subdivided path before any command is
    published.
    """

    start_joints = np.radians(
        [83.23, 81.13, -55.21, -62.62, 86.20, -0.74]
    )
    target_joints = np.radians(
        [81.929, 95.839, -73.431, -66.264, 86.180, -2.049]
    )
    start_transform = controller_fk(start_joints)
    target_transform = controller_fk(target_joints)
    start_pose = Pose(
        tuple(float(v) for v in start_transform[:3, 3]),
        matrix_to_rpy(start_transform[:3, :3]),
    )
    target = Pose(
        tuple(float(v) for v in target_transform[:3, 3]),
        matrix_to_rpy(target_transform[:3, :3]),
    )
    state = RobotState(
        ready=True,
        pose=start_pose,
        joints=tuple(float(v) for v in start_joints),
        message="AutoReady",
    )
    backend = _robot_backend(monkeypatch, state, dry_run=True)

    route = backend.precheck_linear_targets((target,), step_m=0.002)

    segment = route.segments[0]
    metrics = segment.joint_metrics
    assert metrics is not None
    assert 0.059 < segment.distance_m < 0.061
    assert segment.max_joint_step_deg <= (
        backend.config.cartesian_joint_step_deg + 1e-9
    )
    assert segment.max_joint_acceleration_deg <= (
        backend.config.cartesian_joint_acceleration_limit_deg + 1e-9
    )

    # The base yaw is nearly stationary; the shoulder/elbow/wrist do the work.
    assert metrics.travel_deg[0] < 2.0
    assert metrics.max_step_deg[0] < 0.05
    assert metrics.travel_deg[1] > 10.0
    assert metrics.travel_deg[2] > 10.0
    assert metrics.travel_deg[3] > 2.0
    assert metrics.travel_deg[1] * metrics.travel_deg[2] > 0.0

    deltas_deg = np.degrees(
        np.diff(np.asarray(segment.joint_waypoints, dtype=float), axis=0)
    )
    assert np.sign(np.sum(deltas_deg[:, 1])) != np.sign(
        np.sum(deltas_deg[:, 2])
    )


def test_robot_move_to_never_falls_back_to_one_endpoint_joint_move(monkeypatch):
    joints = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    transform = controller_fk(joints)
    start_pose = Pose(
        tuple(float(v) for v in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    target = Pose(
        (start_pose.xyz[0] + 0.01, start_pose.xyz[1], start_pose.xyz[2]),
        start_pose.rpy,
    )
    state = RobotState(
        ready=True,
        pose=start_pose,
        joints=tuple(float(v) for v in joints),
        message="AutoReady",
    )
    backend = _robot_backend(monkeypatch, state, dry_run=True)
    route = object()
    calls = []

    def precheck(targets, *, step_m):
        calls.append(("precheck", tuple(targets), step_m))
        return route

    def execute(planned_route, *, names, speed_mps):
        calls.append(("execute", planned_route, tuple(names), speed_mps))

    monkeypatch.setattr(backend, "precheck_linear_targets", precheck)
    monkeypatch.setattr(backend, "execute_linear_route", execute)
    monkeypatch.setattr(
        backend,
        "move_joints",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("move_to used a direct endpoint joint move")
        ),
    )

    backend.move_to(target, name="route-only-test", speed_mps=0.01)

    assert calls == [
        (
            "precheck",
            (target,),
            backend.config.cartesian_step_m,
        ),
        ("execute", route, ("route-only-test",), 0.01),
    ]


def test_robot_linear_dry_run_publishes_no_joint_targets(monkeypatch):
    joints = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    transform = controller_fk(joints)
    start_pose = Pose(
        tuple(float(v) for v in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    target = Pose(
        (start_pose.xyz[0] + 0.03, start_pose.xyz[1], start_pose.xyz[2]),
        start_pose.rpy,
    )
    state = RobotState(
        ready=True,
        pose=start_pose,
        joints=tuple(float(v) for v in joints),
        message="AutoReady",
    )
    backend = _robot_backend(monkeypatch, state, dry_run=True)
    route = backend.precheck_linear_targets((target,), step_m=0.01)
    monkeypatch.setattr(
        backend,
        "move_joints",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("published")),
    )

    backend.execute_linear_route(
        route,
        names=("dry-run-test",),
        speed_mps=0.01,
    )


def test_robot_precheck_rejects_large_ik_branch_jump_before_publishing(monkeypatch):
    joints = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    transform = controller_fk(joints)
    start_pose = Pose(
        tuple(float(v) for v in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    target = Pose(
        (start_pose.xyz[0] + 0.001, start_pose.xyz[1], start_pose.xyz[2]),
        start_pose.rpy,
    )
    state = RobotState(
        ready=True,
        pose=start_pose,
        joints=tuple(float(v) for v in joints),
        message="AutoReady",
    )
    backend = _robot_backend(
        monkeypatch,
        state,
        dry_run=False,
        cartesian_min_step_m=0.002,
    )
    published = []

    def jump_to_target(q_initial, waypoint, *, label):
        jumped = np.asarray(q_initial, dtype=float).copy()
        jumped[2] += math.radians(6.0)
        return jumped, 0.1

    monkeypatch.setattr(backend, "_solve_checked_waypoint", jump_to_target)
    monkeypatch.setattr(
        backend,
        "_send_joint_waypoint",
        lambda *args, **kwargs: published.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="possible IK branch jump"):
        backend.precheck_linear_targets((target,), step_m=0.002)

    assert published == []


def test_robot_precheck_rejects_near_singular_start_before_publishing(monkeypatch):
    joints = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    transform = controller_fk(joints)
    start_pose = Pose(
        tuple(float(v) for v in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    target = Pose(
        (start_pose.xyz[0] + 0.01, start_pose.xyz[1], start_pose.xyz[2]),
        start_pose.rpy,
    )
    state = RobotState(
        ready=True,
        pose=start_pose,
        joints=tuple(float(v) for v in joints),
        message="AutoReady",
    )
    backend = _robot_backend(
        monkeypatch,
        state,
        dry_run=False,
        cartesian_min_jacobian_singular_value=1.0e9,
    )
    published = []
    monkeypatch.setattr(
        backend,
        "_send_joint_waypoint",
        lambda *args, **kwargs: published.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="near-singular posture"):
        backend.precheck_linear_targets((target,), step_m=0.002)

    assert published == []


def test_robot_workspace_violation_fails_precheck_before_publishing(monkeypatch):
    joints = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    transform = controller_fk(joints)
    start_xyz = transform[:3, 3]
    start_pose = Pose(
        tuple(float(v) for v in start_xyz),
        matrix_to_rpy(transform[:3, :3]),
    )
    target = Pose(
        (start_pose.xyz[0], start_pose.xyz[1], start_xyz[2] - 0.05),
        start_pose.rpy,
    )
    state = RobotState(
        ready=True,
        pose=start_pose,
        joints=tuple(float(v) for v in joints),
        message="AutoReady",
    )
    backend = _robot_backend(
        monkeypatch,
        state,
        dry_run=False,
        workspace_min=(
            -0.65,
            -0.65,
            float(start_xyz[2]) - 0.001,
        ),
        workspace_margin=0.0,
    )
    published = []
    monkeypatch.setattr(
        backend,
        "_send_joint_waypoint",
        lambda *args, **kwargs: published.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="outside the configured workspace"):
        backend.precheck_linear_targets((target,), step_m=0.002)

    assert published == []

class _FakeTrajectoryPoint:
    def __init__(self):
        self.positions = []
        self.velocities = []


class _FakeJointTrajectory:
    def __init__(self):
        self.joint_names = []
        self.points = []


class _FakePublisher:
    def __init__(self, backend):
        self.backend = backend
        self.messages = []

    def publish(self, message):
        self.messages.append(message)
        self.backend._state = "AutoReady"
        self.backend._stamp += 1.0


class _FakeMonotonicClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, float(seconds))


def _trajectory_test_backend(
    monkeypatch,
    state_for_elapsed,
    *,
    point_count=94,
    poll_period_s=0.5,
):
    targets = np.zeros((int(point_count), 6), dtype=float)
    targets[:, 0] = np.radians(np.linspace(0.02, 1.0, int(point_count)))
    targets[:, 2] = np.radians(np.linspace(-0.02, -0.5, int(point_count)))
    start = np.zeros(6, dtype=float)
    start_state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        joints=tuple(float(v) for v in start),
        message="AutoReady",
    )
    target_state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        joints=tuple(float(v) for v in targets[-1]),
        message="AutoReady",
    )
    config = MotionConfig(
        backend="robotmove",
        dry_run=False,
        joint_tolerance_deg=0.35,
        joint_start_tolerance_deg=0.05,
        poll_period_s=float(poll_period_s),
    )
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = config
    backend._log = lambda message: None
    backend._trajectory_type = _FakeJointTrajectory
    backend._point_type = _FakeTrajectoryPoint
    backend._lock = threading.Lock()
    backend._state = ""
    backend._stamp = 0.0
    publisher = _FakePublisher(backend)
    backend._publisher = publisher
    clock = _FakeMonotonicClock()
    monkeypatch.setattr(backend, "wait_for_state", lambda timeout: start_state)

    def get_state():
        state = state_for_elapsed(clock.now, start_state, target_state)
        backend._state = state.message
        backend._stamp += 1.0
        return state

    monkeypatch.setattr(backend, "get_state", get_state)
    monkeypatch.setattr("track1_grasp.motion.time.monotonic", clock.monotonic)
    monkeypatch.setattr("track1_grasp.motion.time.sleep", clock.sleep)
    return backend, targets, start_state, target_state, clock


def test_controller_safe_chunk_waits_past_short_reject_window_during_planning(
    monkeypatch,
):
    """A 40-point command can remain AutoReady while the controller plans."""

    def state_for_elapsed(elapsed, start_state, target_state):
        if elapsed < 5.1:
            return start_state
        if elapsed < 5.6:
            return RobotState(
                ready=True,
                pose=start_state.pose,
                joints=start_state.joints,
                message="AutoRunning",
            )
        return target_state

    backend, targets, _start, _target, clock = _trajectory_test_backend(
        monkeypatch,
        state_for_elapsed,
        point_count=ROBOT_CONTROLLER_MAX_TRAJECTORY_POINTS,
    )
    reject_timeout_s = backend._reject_confirmation_timeout_s(len(targets))

    assert reject_timeout_s > backend.config.joint_reject_timeout_s
    result = backend._send_joint_trajectory(
        targets,
        name="controller_safe_chunk",
        timeout_s=backend._trajectory_timeout_s(len(targets)),
        reject_timeout_s=reject_timeout_s,
    )

    assert np.allclose(result, targets[-1])
    assert clock.now >= 5.5


def _chunked_trajectory_backend(monkeypatch, *, point_count=94):
    targets = np.zeros((int(point_count), 6), dtype=float)
    targets[:, 0] = np.radians(np.linspace(0.02, 1.0, int(point_count)))
    targets[:, 2] = np.radians(np.linspace(-0.02, -0.5, int(point_count)))
    start_state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        joints=(0.0,) * 6,
        message="AutoReady",
    )
    config = MotionConfig(
        backend="robotmove",
        dry_run=False,
        joint_tolerance_deg=0.35,
        joint_start_tolerance_deg=0.05,
        joint_limit_margin_deg=2.0,
        joint_max_step_deg=1.0,
        cartesian_joint_step_deg=0.5,
        poll_period_s=0.0,
    )
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = config
    backend._log = lambda message: None
    backend._lock = threading.Lock()
    backend._stamp = 0.0
    current = {"state": start_state}
    monkeypatch.setattr(
        backend,
        "wait_for_state",
        lambda timeout: current["state"],
    )
    monkeypatch.setattr(backend, "get_state", lambda: current["state"])
    return backend, targets, current


def test_94_point_trajectory_splits_into_controller_safe_chunks(monkeypatch):
    backend, targets, _current = _chunked_trajectory_backend(monkeypatch)

    chunks = backend._split_joint_trajectory(
        targets,
        name="task_search_position_03",
    )

    assert [len(chunk) for chunk in chunks] == [40, 40, 14]
    assert np.array_equal(np.concatenate(chunks, axis=0), targets)
    assert all(
        len(chunk) <= ROBOT_CONTROLLER_MAX_TRAJECTORY_POINTS
        for chunk in chunks
    )
    assert all(
        backend._joint_trajectory_payload_size(chunk)
        <= ROBOT_CONTROLLER_MAX_TRAJECTORY_BYTES
        for chunk in chunks
    )


def test_chunked_trajectory_publishes_chunks_in_order(monkeypatch):
    backend, targets, current = _chunked_trajectory_backend(monkeypatch)
    sent = []

    def complete_chunk(chunk_targets, *, name, timeout_s, reject_timeout_s):
        chunk = np.asarray(chunk_targets, dtype=float).copy()
        sent.append((name, chunk))
        current["state"] = RobotState(
            ready=True,
            pose=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            joints=tuple(float(value) for value in chunk[-1]),
            message="AutoReady",
        )
        return chunk[-1]

    monkeypatch.setattr(
        backend,
        "_send_joint_trajectory_once",
        complete_chunk,
    )

    result = backend._send_joint_trajectory(
        targets,
        name="task_search_position_03",
        timeout_s=60.0,
        reject_timeout_s=60.0,
    )

    assert np.allclose(result, targets[-1])
    assert [name for name, _chunk in sent] == [
        "task_search_position_03 chunk 1/3",
        "task_search_position_03 chunk 2/3",
        "task_search_position_03 chunk 3/3",
    ]
    assert np.array_equal(
        np.concatenate([chunk for _name, chunk in sent], axis=0),
        targets,
    )
    assert [len(chunk) for _name, chunk in sent] == [40, 40, 14]


def test_chunked_trajectory_stops_after_first_chunk_is_ignored(monkeypatch):
    backend, targets, _current = _chunked_trajectory_backend(monkeypatch)
    sent = []

    def ignored_chunk(chunk_targets, *, name, timeout_s, reject_timeout_s):
        sent.append(np.asarray(chunk_targets, dtype=float).copy())
        raise RuntimeError(f"controller ignored joint move {name!r}")

    monkeypatch.setattr(
        backend,
        "_send_joint_trajectory_once",
        ignored_chunk,
    )

    with pytest.raises(RuntimeError, match="controller ignored joint move"):
        backend._send_joint_trajectory(
            targets,
            name="task_search_position_03",
            timeout_s=60.0,
            reject_timeout_s=60.0,
        )

    assert len(sent) == 1
    assert np.array_equal(sent[0], targets[:40])


def test_chunked_trajectory_refuses_next_chunk_after_stale_feedback(monkeypatch):
    backend, targets, current = _chunked_trajectory_backend(monkeypatch)
    chunks = backend._split_joint_trajectory(
        targets,
        name="task_search_position_03",
    )
    sent = []

    def complete_with_stale_boundary(
        chunk_targets,
        *,
        name,
        timeout_s,
        reject_timeout_s,
    ):
        chunk = np.asarray(chunk_targets, dtype=float).copy()
        sent.append(chunk)
        stale_joints = chunks[1][0].copy()
        stale_joints[1] += math.radians(2.0)
        current["state"] = RobotState(
            ready=True,
            pose=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            joints=tuple(float(value) for value in stale_joints),
            message="AutoReady",
        )
        return chunk[-1]

    monkeypatch.setattr(
        backend,
        "_send_joint_trajectory_once",
        complete_with_stale_boundary,
    )

    with pytest.raises(RuntimeError, match="feedback is .* from its first target"):
        backend._send_joint_trajectory(
            targets,
            name="task_search_position_03",
            timeout_s=60.0,
            reject_timeout_s=60.0,
        )

    assert len(sent) == 1


def test_chunked_trajectory_stops_on_fault_between_chunks(monkeypatch):
    backend, targets, current = _chunked_trajectory_backend(monkeypatch)
    sent = []

    def complete_then_fault(chunk_targets, *, name, timeout_s, reject_timeout_s):
        chunk = np.asarray(chunk_targets, dtype=float).copy()
        sent.append(chunk)
        current["state"] = RobotState(
            ready=False,
            pose=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            fault=True,
            joints=tuple(float(value) for value in chunk[-1]),
            message="Fault",
        )
        return chunk[-1]

    monkeypatch.setattr(
        backend,
        "_send_joint_trajectory_once",
        complete_then_fault,
    )

    with pytest.raises(RuntimeError, match="controller fault between"):
        backend._send_joint_trajectory(
            targets,
            name="task_search_position_03",
            timeout_s=60.0,
            reject_timeout_s=60.0,
        )

    assert len(sent) == 1


def test_single_controller_safe_chunk_still_stops_on_controller_fault(
    monkeypatch,
):
    def state_for_elapsed(elapsed, start_state, target_state):
        if elapsed < 1.0:
            return start_state
        return RobotState(
            ready=False,
            pose=start_state.pose,
            fault=True,
            joints=start_state.joints,
            message="Fault",
        )

    backend, targets, _start, _target, clock = _trajectory_test_backend(
        monkeypatch,
        state_for_elapsed,
        point_count=4,
    )

    with pytest.raises(RuntimeError, match="controller fault"):
        backend._send_joint_trajectory(
            targets,
            name="single_chunk_fault",
            timeout_s=60.0,
            reject_timeout_s=backend._reject_confirmation_timeout_s(len(targets)),
        )

    assert clock.now == pytest.approx(1.0)


def test_single_controller_safe_chunk_whole_move_timeout_remains_effective(
    monkeypatch,
):
    def state_for_elapsed(elapsed, start_state, target_state):
        if elapsed < 0.5:
            return start_state
        return RobotState(
            ready=True,
            pose=start_state.pose,
            joints=start_state.joints,
            message="AutoRunning",
        )

    backend, targets, _start, _target, _clock = _trajectory_test_backend(
        monkeypatch,
        state_for_elapsed,
        point_count=4,
    )

    with pytest.raises(RuntimeError, match="timed out after 2.0s"):
        backend._send_joint_trajectory(
            targets,
            name="single_chunk_timeout",
            timeout_s=2.0,
            reject_timeout_s=backend._reject_confirmation_timeout_s(len(targets)),
        )


def test_point2_pregrasp_at_observation_anchor_skips_sub_tolerance_command(
    monkeypatch,
):
    """Point 2 observation feedback can make pregrasp a near-zero command.

    The live 2026-10-20 run reached the taught point-2 observation pose, then
    published a nominal ``pregrasp_p001`` target whose residual was only a few
    thousandths of a degree.  The controller stayed AutoReady and the backend
    incorrectly classified that no-op as an ignored non-zero command.  A target
    inside ``joint_start_tolerance_deg`` must be accepted without publishing.
    """

    observation_deg = (
        91.405846375,
        74.366790849,
        -18.643502588,
        -35.900559036,
        87.641336807,
        -19.600974643,
    )
    actual_q = np.radians(np.asarray(observation_deg, dtype=float))
    target_q = actual_q.copy()
    target_q[0] += math.radians(0.010)
    transform = controller_fk(actual_q)
    state = RobotState(
        ready=True,
        pose=Pose(
            tuple(float(value) for value in transform[:3, 3]),
            matrix_to_rpy(transform[:3, :3]),
        ),
        joints=tuple(float(value) for value in actual_q),
        message="AutoReady",
    )
    config = MotionConfig(
        backend="robotmove",
        dry_run=False,
        joint_start_tolerance_deg=0.05,
    )
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = config
    backend._log = lambda message: None
    monkeypatch.setattr(backend, "wait_for_state", lambda timeout: state)

    class RejectPublisher:
        def publish(self, message):
            raise AssertionError("sub-tolerance pregrasp command was published")

    backend._publisher = RejectPublisher()

    result = backend._send_joint_waypoint(
        target_q,
        name="pregrasp_p001",
        timeout_s=1.0,
        reject_timeout_s=0.1,
    )

    assert np.array_equal(result, actual_q)


def test_robot_joint_trajectory_appends_all_ros2_trajectory_points(monkeypatch):
    start = np.zeros(6, dtype=float)
    midpoint = np.radians(np.arange(6, dtype=float)) * 0.5
    target = np.radians(np.arange(6, dtype=float))
    config = MotionConfig(
        backend="robotmove",
        dry_run=False,
        joint_tolerance_deg=0.35,
        joint_start_tolerance_deg=0.05,
        poll_period_s=0.0,
    )
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = config
    backend._log = lambda message: None
    backend._trajectory_type = _FakeJointTrajectory
    backend._point_type = _FakeTrajectoryPoint
    backend._lock = threading.Lock()
    backend._state = ""
    backend._stamp = 0.0
    publisher = _FakePublisher(backend)
    backend._publisher = publisher
    start_state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        joints=tuple(float(v) for v in start),
        message="AutoReady",
    )
    target_state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        joints=tuple(float(v) for v in target),
        message="AutoReady",
    )
    monkeypatch.setattr(backend, "wait_for_state", lambda timeout: start_state)
    monkeypatch.setattr(backend, "get_state", lambda: target_state)
    monkeypatch.setattr("track1_grasp.motion.time.sleep", lambda seconds: None)

    result = backend._send_joint_trajectory(
        (midpoint, target),
        name="message-api-test",
        timeout_s=1.0,
        reject_timeout_s=0.1,
    )

    assert np.allclose(result, target)
    assert len(publisher.messages) == 1
    message = publisher.messages[0]
    assert message.joint_names == [
        "Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"
    ]
    assert len(message.points) == 2
    assert message.points[0].positions == midpoint.tolist()
    assert message.points[1].positions == target.tolist()


def test_robot_continuous_trajectory_udp_payload_stays_below_datagram_limit():
    points = np.linspace(-3.1, 3.1, 147 * 6, dtype=float).reshape(147, 6)

    payload_bytes = RobotMoveMotionBackend._joint_trajectory_payload_size(points)

    assert payload_bytes < UDP_MAX_DATAGRAM_BYTES


def test_robot_joint_move_rejects_floor_violation_before_publishing(monkeypatch):
    start = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    start_transform = controller_fk(start)
    start_xyz = start_transform[:3, 3]
    config = MotionConfig(
        backend="robotmove",
        dry_run=False,
        workspace_min=(-0.65, -0.65, float(start_xyz[2]) - 0.001),
        workspace_max=(0.65, 0.65, 0.80),
        workspace_margin=0.0,
        joint_max_step_deg=0.5,
    )
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = config
    backend._log = lambda message: None
    state = RobotState(
        ready=True,
        pose=Pose(tuple(float(v) for v in start_xyz), (math.pi, 0.0, 0.0)),
        joints=tuple(float(v) for v in start),
        message="AutoReady",
    )
    monkeypatch.setattr(backend, "wait_for_state", lambda timeout: state)
    monkeypatch.setattr(
        backend,
        "_send_joint_waypoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("published a floor-violating waypoint")
        ),
    )
    target_xyz = start_xyz.copy()
    target_xyz[2] -= 0.03
    target = solve_pose_ik(start, target_xyz, start_transform[:3, :3])

    with pytest.raises(RuntimeError, match="would leave the configured workspace"):
        backend.move_joints(target, name="floor-guard-test")
