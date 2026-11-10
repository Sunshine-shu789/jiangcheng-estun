import numpy as np
import pytest

from track1_grasp.approach import (
    compute_servo_command,
    pose_inside_workspace,
    select_target,
)
from track1_grasp.config import MotionConfig, ServoConfig
from track1_grasp.domain import Detection, Frame, Pose, RobotState


def make_detection(**overrides):
    values = {
        "label": "6-Foam_Brick",
        "confidence": 0.9,
        "bbox": (300.0, 220.0, 340.0, 260.0),
        "depth_m": 0.30,
        "camera_xyz": (0.0, 0.0, 0.30),
        "graspability": 1.0,
        "attributes": {
            "u": 330.0,
            "v": 245.0,
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
        },
    }
    values.update(overrides)
    return Detection(**values)


def make_servo(**overrides):
    values = {
        "desired_standoff_m": 0.30,
        "min_depth_m": 0.12,
        "center_tolerance_px": 35.0,
        "depth_tolerance_m": 0.015,
        "max_steps": 10,
        "max_step_m": 0.05,
        "max_total_travel_m": 0.35,
        "speed_mps": 0.01,
        "lateral_gain": 1.0,
        "forward_gain": 1.0,
        "settle_s_after_move": 0.0,
        "convergence_frames": 1,
        "camera_to_tool": (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    }
    values.update(overrides)
    return ServoConfig(**values)


def identity_state():
    return RobotState(
        ready=True,
        pose=Pose((0.1, -0.2, 0.4), (0.0, 0.0, 0.0)),
        joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        message="AutoReady",
    )


def test_select_target_filters_class_confidence_and_depth():
    wrong_class = make_detection(label="other", confidence=0.99)
    no_depth = make_detection(confidence=0.99, depth_m=None, camera_xyz=None)
    low_confidence = make_detection(confidence=0.20)
    selected_candidate = make_detection(confidence=0.70)
    frame = Frame(
        frame_id="test",
        detections=(wrong_class, no_depth, low_confidence, selected_candidate),
    )

    selected = select_target(
        frame,
        target_class="6-Foam_Brick",
        min_confidence=0.35,
    )

    assert selected is selected_candidate


def test_select_target_returns_none_without_usable_detection():
    frame = Frame(
        frame_id="test",
        detections=(make_detection(depth_m=None, camera_xyz=None),),
    )
    assert select_target(frame, target_class="6-Foam_Brick") is None


def test_servo_command_signs_follow_image_axes():
    command = compute_servo_command(
        identity_state(), make_detection(), make_servo(center_tolerance_px=5.0)
    )

    assert command.delta_base[0] > 0.0
    assert command.delta_base[1] > 0.0
    assert command.delta_base[2] == pytest.approx(0.0)
    assert command.pixel_error == pytest.approx((10.0, 5.0))
    assert not command.converged


def test_servo_command_moves_forward_when_too_far_from_target():
    detection = make_detection(attributes={
        "u": 320.0,
        "v": 240.0,
        "fx": 500.0,
        "fy": 500.0,
        "cx": 320.0,
        "cy": 240.0,
    }, depth_m=0.34)

    command = compute_servo_command(identity_state(), detection, make_servo())

    assert command.delta_base[2] == pytest.approx(0.04)


def test_servo_command_is_bounded_by_max_step():
    detection = make_detection(attributes={
        "u": 820.0,
        "v": 740.0,
        "fx": 500.0,
        "fy": 500.0,
        "cx": 320.0,
        "cy": 240.0,
    })

    command = compute_servo_command(
        identity_state(), detection, make_servo(max_step_m=0.005)
    )

    assert np.linalg.norm(command.delta_base) == pytest.approx(0.005)


def test_servo_command_rejects_target_inside_depth_safety_zone():
    detection = make_detection(depth_m=0.11)

    with pytest.raises(RuntimeError, match="below safety minimum"):
        compute_servo_command(identity_state(), detection, make_servo())


def test_servo_command_rotates_tool_motion_into_base_frame():
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.3), (0.0, 0.0, np.pi / 2.0)),
        joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        message="AutoReady",
    )
    detection = make_detection(attributes={
        "u": 330.0,
        "v": 240.0,
        "fx": 500.0,
        "fy": 500.0,
        "cx": 320.0,
        "cy": 240.0,
    })

    command = compute_servo_command(state, detection, make_servo())

    assert command.delta_base[0] == pytest.approx(0.0, abs=1e-9)
    assert command.delta_base[1] > 0.0


def test_workspace_check_applies_margin():
    motion = MotionConfig(
        workspace_min=(-0.5, -0.5, 0.0),
        workspace_max=(0.5, 0.5, 1.0),
        workspace_margin=0.05,
    )

    inside, low, high = pose_inside_workspace(
        Pose((0.0, 0.0, 0.3), (0.0, 0.0, 0.0)), motion
    )
    outside, _, _ = pose_inside_workspace(
        Pose((0.46, 0.0, 0.3), (0.0, 0.0, 0.0)), motion
    )

    assert inside
    assert not outside
    assert low.tolist() == pytest.approx([-0.45, -0.45, 0.05])
    assert high.tolist() == pytest.approx([0.45, 0.45, 0.95])
