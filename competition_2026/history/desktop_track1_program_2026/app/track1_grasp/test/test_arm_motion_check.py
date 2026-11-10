import numpy as np
import pytest

from track1_grasp.domain import Pose, RobotState
from track1_grasp.nodes.arm_motion_check_node import (
    bounded_round_trip,
    choose_direction,
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
