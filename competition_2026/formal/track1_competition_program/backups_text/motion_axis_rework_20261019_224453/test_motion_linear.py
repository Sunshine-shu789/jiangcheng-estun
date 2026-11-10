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
)
from track1_grasp.motion import (
    RobotMoveMotionBackend,
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


def _robot_backend(monkeypatch, state, *, dry_run):
    config = MotionConfig(
        backend="robotmove",
        dry_run=dry_run,
        workspace_min=(-0.65, -0.65, 0.02),
        workspace_max=(0.65, 0.65, 0.80),
        workspace_margin=0.01,
    )
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = config
    backend._log = lambda message: None
    monkeypatch.setattr(backend, "wait_for_state", lambda timeout: state)
    monkeypatch.setattr(backend, "get_state", lambda: state)
    return backend


def test_robot_linear_precheck_builds_joint_waypoints_for_the_whole_path(monkeypatch):
    joints = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    transform = controller_fk(joints)
    start_pose = Pose(
        tuple(float(v) for v in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    target = Pose(
        (start_pose.xyz[0] + 0.15, start_pose.xyz[1], start_pose.xyz[2]),
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
    assert abs(segment.distance_m - 0.15) < 1e-9
    assert len(segment.joint_waypoints) == len(segment.waypoints)
    assert len(segment.joint_waypoints) == 16


def test_robot_linear_dry_run_publishes_no_joint_targets(monkeypatch):
    joints = np.radians([0.0, -20.0, 80.0, 0.0, 60.0, 0.0])
    transform = controller_fk(joints)
    start_pose = Pose(
        tuple(float(v) for v in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    target = Pose(
        (start_pose.xyz[0] + 0.15, start_pose.xyz[1], start_pose.xyz[2]),
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


def test_robot_joint_waypoint_appends_ros2_trajectory_point(monkeypatch):
    start = np.zeros(6, dtype=float)
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

    result = backend._send_joint_waypoint(
        target,
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
    assert len(message.points) == 1
    assert message.points[0].positions == target.tolist()


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
