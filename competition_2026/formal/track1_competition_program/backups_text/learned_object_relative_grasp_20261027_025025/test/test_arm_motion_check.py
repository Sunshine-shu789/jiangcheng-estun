import numpy as np
import pytest
from dataclasses import dataclass
from types import SimpleNamespace

from track1_grasp.domain import Pose, RobotState
from track1_grasp.nodes.arm_motion_check_node import (
    bounded_radial_up_round_trip,
    bounded_round_trip,
    choose_direction,
    motion_config_for_execution_mode,
    route_joint_max_step_deg,
    route_joint_travel_deg,
)


def ready_state(xyz=(0.1, 0.2, 0.4)):
    return RobotState(
        ready=True,
        pose=Pose(xyz, (3.14159265, 0.0, 0.0)),
        message="controller state=AutoReady",
        joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def test_auto_direction_uses_the_side_with_more_workspace():
    assert choose_direction(np.array([0.1, 0.0, 0.0]), 0, -0.2, 0.5, "auto") == 1
    assert choose_direction(np.array([0.4, 0.0, 0.0]), 0, -0.2, 0.5, "auto") == -1
    assert choose_direction(np.array([0.4, 0.0, 0.0]), 0, -0.2, 0.5, "+") == 1


def test_round_trip_preserves_orientation_and_returns_to_start():
    state = ready_state()
    target, start = bounded_round_trip(
        state,
        axis=0,
        direction=1,
        distance_m=0.01,
        low=(-0.65, -0.65, 0.02),
        high=(0.65, 0.65, 0.80),
    )
    assert target.xyz == pytest.approx((0.11, 0.2, 0.4))
    assert start.xyz == state.pose.xyz
    assert target.rpy == start.rpy == state.pose.rpy


def test_round_trip_rejects_target_outside_workspace():
    with pytest.raises(ValueError, match="outside"):
        bounded_round_trip(
            ready_state(xyz=(0.09, 0.2, 0.4)),
            axis=0,
            direction=1,
            distance_m=0.02,
            low=(-0.65, -0.65, 0.02),
            high=(0.10, 0.65, 0.80),
        )


def test_radial_up_round_trip_points_toward_base_and_preserves_orientation():
    state = ready_state(xyz=(0.1, 0.2, 0.4))
    target, start = bounded_radial_up_round_trip(
        state,
        inward_m=0.002,
        up_m=0.005,
        low=(-0.65, -0.65, 0.02),
        high=(0.65, 0.65, 0.80),
    )

    delta = np.asarray(target.xyz) - np.asarray(start.xyz)
    start_xy = np.asarray(start.xyz[:2])
    assert float(np.dot(delta[:2], start_xy)) < 0.0
    assert float(np.linalg.norm(delta[:2])) == pytest.approx(0.002)
    assert float(delta[2]) == pytest.approx(0.005)
    assert target.rpy == start.rpy == state.pose.rpy


def test_radial_up_round_trip_rejects_downward_or_outside_workspace():
    with pytest.raises(ValueError, match="inward and upward"):
        bounded_radial_up_round_trip(
            ready_state(),
            inward_m=0.002,
            up_m=-0.001,
            low=(-0.65, -0.65, 0.02),
            high=(0.65, 0.65, 0.80),
        )

    with pytest.raises(ValueError, match="outside"):
        bounded_radial_up_round_trip(
            ready_state(xyz=(0.1, 0.2, 0.49)),
            inward_m=0.002,
            up_m=0.02,
            low=(-0.65, -0.65, 0.02),
            high=(0.65, 0.65, 0.50),
        )


def test_route_joint_limits_sum_travel_and_take_worst_step():
    first = SimpleNamespace(
        joint_metrics=SimpleNamespace(
            travel_deg=(0.1, 1.0, 2.0, 3.0, 4.0, 5.0),
            max_step_deg=(0.02, 0.2, 0.3, 0.4, 0.5, 0.6),
        )
    )
    second = SimpleNamespace(
        joint_metrics=SimpleNamespace(
            travel_deg=(0.2, 1.0, 2.0, 3.0, 4.0, 5.0),
            max_step_deg=(0.01, 0.2, 0.3, 0.4, 0.5, 0.6),
        )
    )
    route = SimpleNamespace(segments=(first, second))

    assert route_joint_travel_deg(route, 0) == pytest.approx(0.3)
    assert route_joint_max_step_deg(route, 0) == pytest.approx(0.02)

def test_dry_run_mode_overrides_live_config_to_non_actuating():
    @dataclass(frozen=True)
    class MotionConfig:
        dry_run: bool
        speed_mps: float

    live = MotionConfig(dry_run=False, speed_mps=0.01)

    dry = motion_config_for_execution_mode(live, execute=False)
    execute = motion_config_for_execution_mode(live, execute=True)

    assert dry.dry_run is True
    assert execute.dry_run is False
    assert live.dry_run is False
