from types import SimpleNamespace

import pytest

from track1_grasp.tools.recover_point4 import (
    HELD_CLOSED_ANGLE_MIN_DEG,
    _validate_held_closed_status,
    _validate_open_status,
)


def gripper_status(*, arrived=True, angle=2159.0, speed=0.0):
    return SimpleNamespace(
        device_id=1,
        arrived=arrived,
        angle_deg=angle,
        speed_rad_s=speed,
    )


def test_held_closed_status_accepts_the_recorded_point4_state():
    _validate_held_closed_status(
        gripper_status(arrived=True, angle=2159.0, speed=0.0),
        label="test",
    )


@pytest.mark.parametrize(
    "status,match",
    [
        (None, "读取失败"),
        (gripper_status(arrived=False), "未报告到位"),
        (gripper_status(angle=HELD_CLOSED_ANGLE_MIN_DEG - 1.0), "不符合"),
        (gripper_status(speed=0.5), "仍在运动"),
    ],
)
def test_held_closed_status_rejects_unsafe_feedback(status, match):
    with pytest.raises(RuntimeError, match=match):
        _validate_held_closed_status(status, label="test")


def test_open_status_requires_arrival_angle_change_and_stopped_motion():
    _validate_open_status(gripper_status(angle=3000.0), pre_open_angle_deg=2159.0)

    with pytest.raises(RuntimeError, match="未报告到位"):
        _validate_open_status(
            gripper_status(arrived=False, angle=3000.0),
            pre_open_angle_deg=2159.0,
        )
    with pytest.raises(RuntimeError, match="角度明显变化"):
        _validate_open_status(
            gripper_status(angle=2161.0),
            pre_open_angle_deg=2159.0,
        )
    with pytest.raises(RuntimeError, match="速度未归零"):
        _validate_open_status(
            gripper_status(angle=3000.0, speed=0.5),
            pre_open_angle_deg=2159.0,
        )
