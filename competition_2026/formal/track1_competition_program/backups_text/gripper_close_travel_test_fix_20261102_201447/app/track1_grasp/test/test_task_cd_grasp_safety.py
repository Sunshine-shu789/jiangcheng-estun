from dataclasses import replace
from math import pi, radians
from pathlib import Path

import numpy as np
import pytest

from track1_grasp.config import GraspSafetyConfig, load_config
from track1_grasp.domain import Detection, MotionPlan, PickStatus, Pose, RobotState
from track1_grasp.factory import build_pipeline
from track1_grasp.planner import PosePlanner, euler_to_matrix


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"
FORMAL_ROOT = Path(__file__).resolve().parents[3]
FORMAL_CD_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_cd_40class_v11_official.yaml"
)
GRASP_RPY = (pi, 0.0, 0.0)
TIP_OFFSET_M = (0.0, 0.0, 0.165)


def cd_config() -> object:
    base = load_config(CONFIG)
    safety = replace(
        base.motion.grasp_safety,
        enabled=True,
        require_flange_tip_offset=True,
        require_tool_down=True,
    )
    return replace(
        base,
        pose=replace(
            base.pose,
            robot_pose_frame="flange",
            gripper_tip_offset_flange_m=TIP_OFFSET_M,
        ),
        motion=replace(
            base.motion,
            orientation="fixed",
            fixed_rpy=GRASP_RPY,
            grasp_rpy=GRASP_RPY,
            place_pose=(0.25, 0.25, 0.08, *GRASP_RPY),
            place_pose_frame="jaw_tip",
            grasp_safety=safety,
            workspace_min=(-0.65, -0.65, 0.0),
            workspace_max=(0.65, 0.65, 0.80),
            workspace_margin=0.0,
        ),
    )


def detection() -> Detection:
    return Detection(
        label="object",
        confidence=0.95,
        bbox=(100.0, 100.0, 200.0, 200.0),
        depth_m=0.40,
        base_xyz=(0.10, 0.00, 0.02),
    )


def robot_state(rpy: tuple[float, float, float] = GRASP_RPY) -> RobotState:
    return RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), rpy),
        joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        message="mock-ready",
    )


def make_plan(
    *,
    grasp_xyz: tuple[float, float, float] = (0.10, 0.00, 0.185),
    pregrasp_xyz: tuple[float, float, float] = (0.10, 0.00, 0.265),
    release_xyz: tuple[float, float, float] = (0.25, 0.25, 0.08),
    rpy: tuple[float, float, float] = GRASP_RPY,
    release_rpy: tuple[float, float, float] = GRASP_RPY,
) -> MotionPlan:
    grasp = Pose(grasp_xyz, rpy)
    high = Pose(pregrasp_xyz, rpy)
    release = Pose(release_xyz, release_rpy)
    return MotionPlan(
        target=detection(),
        grasp_pose=grasp,
        pregrasp_pose=high,
        lift_pose=high,
        transfer_pose=Pose(
            (release_xyz[0], release_xyz[1], release_xyz[2] + 0.08),
            release_rpy,
        ),
        release_pose=release,
        retreat_pose=Pose(
            (release_xyz[0], release_xyz[1], release_xyz[2] + 0.08),
            release_rpy,
        ),
        place_name="test-place",
    )


def planner_with_safety(**safety_overrides) -> PosePlanner:
    base = cd_config()
    safety = replace(base.motion.grasp_safety, **safety_overrides)
    config = replace(
        base,
        motion=replace(base.motion, grasp_safety=safety),
    )
    return PosePlanner(config)


def test_visual_jaw_tip_is_converted_to_flange_pose_once():
    planner = PosePlanner(cd_config())
    plan = planner.plan(
        detection(),
        robot_state(),
        grasp_xyz_override_m=(0.10, 0.02, 0.02),
        grasp_target_is_jaw_tip=True,
    )

    assert plan.grasp_pose.xyz == pytest.approx((0.10, 0.02, 0.185))
    assert plan.grasp_pose.rpy == pytest.approx(GRASP_RPY)

    report = planner.validate_grasp_safety_plan(plan)
    assert report.grasp_flange_xyz_m == pytest.approx((0.10, 0.02, 0.185))
    assert report.grasp_tip_xyz_m == pytest.approx((0.10, 0.02, 0.02))
    assert report.release_flange_xyz_m == pytest.approx((0.25, 0.25, 0.245))
    assert report.release_tip_xyz_m == pytest.approx((0.25, 0.25, 0.08))
    reconstructed_tip = np.asarray(plan.grasp_pose.xyz) + (
        euler_to_matrix(plan.grasp_pose.rpy) @ np.asarray(TIP_OFFSET_M)
    )
    assert reconstructed_tip == pytest.approx((0.10, 0.02, 0.02))


def test_explicit_jaw_tip_z_overrides_visual_surface_and_converts_once():
    planner = PosePlanner(cd_config())
    plan = planner.plan(
        detection(),
        robot_state(),
        grasp_target_is_jaw_tip=True,
        grasp_rpy=GRASP_RPY,
        jaw_tip_z_m=0.000103,
    )

    assert plan.grasp_pose.xyz == pytest.approx((0.10, 0.0, 0.165103))
    report = planner.validate_grasp_safety_plan(plan)
    assert report.grasp_tip_xyz_m == pytest.approx((0.10, 0.0, 0.000103))


def test_jaw_tip_release_pose_is_converted_to_flange_once():
    planner = PosePlanner(cd_config())

    # Keep this test focused on release-frame conversion.  Supply the same
    # jaw-tip interpretation that the C/D pipeline requests explicitly.
    plan = planner.plan(
        detection(),
        robot_state(),
        grasp_xyz_override_m=(0.10, 0.0, 0.02),
        grasp_target_is_jaw_tip=True,
    )

    assert plan.release_pose.xyz == pytest.approx((0.25, 0.25, 0.245))
    assert plan.transfer_pose.xyz == pytest.approx((0.25, 0.25, 0.325))
    assert plan.retreat_pose.xyz == pytest.approx((0.25, 0.25, 0.325))
    release_tip = np.asarray(plan.release_pose.xyz) + (
        euler_to_matrix(plan.release_pose.rpy) @ np.asarray(TIP_OFFSET_M)
    )
    assert release_tip == pytest.approx((0.25, 0.25, 0.08))

    report = planner.validate_grasp_safety_plan(plan)
    assert report.place_pose_frame == "jaw_tip"
    assert report.release_flange_xyz_m == pytest.approx((0.25, 0.25, 0.245))
    assert report.release_tip_xyz_m == pytest.approx((0.25, 0.25, 0.08))
    assert report.minimum_route_tip_z_m == pytest.approx(0.02)


def test_flange_release_pose_keeps_explicit_legacy_semantics():
    base = cd_config()
    config = replace(
        base,
        motion=replace(base.motion, place_pose_frame="flange"),
    )

    plan = PosePlanner(config).plan(detection(), robot_state())

    assert plan.release_pose.xyz == pytest.approx((0.25, 0.25, 0.08))
    report = PosePlanner(config).validate_grasp_safety_plan(plan)
    assert report.release_flange_xyz_m == pytest.approx((0.25, 0.25, 0.08))
    assert report.release_tip_xyz_m == pytest.approx((0.25, 0.25, -0.085))


def test_cd_jaw_tip_conversion_uses_explicit_grasp_rpy_not_current_orientation():
    base = cd_config()
    config = replace(
        base,
        motion=replace(
            base.motion,
            orientation="current",
            fixed_rpy=(0.31, -0.22, 0.17),
            grasp_rpy=GRASP_RPY,
        ),
    )
    planner = PosePlanner(config)

    plan = planner.plan(
        detection(),
        robot_state(rpy=(0.41, -0.32, 0.23)),
        grasp_xyz_override_m=(0.10, 0.02, 0.02),
        grasp_target_is_jaw_tip=True,
    )

    assert plan.grasp_pose.rpy == pytest.approx(GRASP_RPY)
    report = planner.validate_grasp_safety_plan(plan)
    assert report.grasp_tip_xyz_m == pytest.approx((0.10, 0.02, 0.02))


@pytest.mark.parametrize(
    ("safety_field", "limit", "message"),
    (
        ("minimum_grasp_flange_z_m", 0.19, "grasp flange Z"),
        ("maximum_grasp_tip_z_m", 0.01, "grasp jaw-tip Z"),
    ),
)
def test_grasp_height_limits_reject_before_report(
    safety_field: str,
    limit: float,
    message: str,
):
    planner = planner_with_safety(**{safety_field: limit})

    with pytest.raises(RuntimeError, match=message):
        planner.validate_grasp_safety_plan(make_plan())


def test_release_route_jaw_tip_through_table_is_rejected():
    plan = make_plan(release_xyz=(0.25, 0.25, 0.08))
    release_tip = np.asarray(plan.release_pose.xyz) + (
        euler_to_matrix(plan.release_pose.rpy) @ np.asarray(TIP_OFFSET_M)
    )
    assert release_tip[2] == pytest.approx(-0.085)

    planner = planner_with_safety(minimum_route_tip_z_m=0.0)
    with pytest.raises(RuntimeError, match="minimum route jaw-tip Z"):
        planner.validate_grasp_safety_plan(plan)


def test_horizontal_reach_limit_applies_to_every_waypoint():
    planner = planner_with_safety(maximum_horizontal_reach_m=0.15)
    plan = make_plan(
        grasp_xyz=(0.20, 0.00, 0.185),
        pregrasp_xyz=(0.20, 0.00, 0.265),
    )

    with pytest.raises(RuntimeError, match="horizontal reach"):
        planner.validate_grasp_safety_plan(plan)


def test_tool_must_point_down():
    base = cd_config()
    safety = replace(base.motion.grasp_safety, require_tool_down=True)
    config = replace(
        base,
        motion=replace(
            base.motion,
            grasp_rpy=(0.0, 0.0, 0.0),
            grasp_safety=safety,
        ),
    )

    with pytest.raises(RuntimeError, match="tool Z is not downward"):
        PosePlanner(config).validate_grasp_safety_plan(
            make_plan(rpy=(0.0, 0.0, 0.0), release_rpy=(0.0, 0.0, 0.0))
        )


def test_tool_tilt_limit_rejects_tilted_grasp():
    tilted_rpy = (pi, radians(10.0), 0.0)
    base = cd_config()
    safety = replace(
        base.motion.grasp_safety,
        require_tool_down=True,
        maximum_tool_tilt_deg=2.0,
        maximum_place_tool_tilt_deg=18.0,
    )
    config = replace(
        base,
        motion=replace(
            base.motion,
            grasp_rpy=tilted_rpy,
            grasp_safety=safety,
        ),
    )

    with pytest.raises(RuntimeError, match="maximum tool tilt"):
        PosePlanner(config).validate_grasp_safety_plan(
            make_plan(rpy=tilted_rpy, release_rpy=tilted_rpy)
        )


def test_place_tool_tilt_has_a_separate_bound():
    release_rpy = (pi, radians(10.0), 0.0)
    planner = planner_with_safety(
        require_tool_down=True,
        maximum_tool_tilt_deg=2.0,
        maximum_place_tool_tilt_deg=11.0,
    )

    report = planner.validate_grasp_safety_plan(
        make_plan(release_rpy=release_rpy)
    )

    assert report.maximum_grasp_tool_tilt_deg == pytest.approx(0.0)
    assert report.maximum_place_tool_tilt_deg == pytest.approx(10.0)
    assert report.maximum_tool_tilt_deg == pytest.approx(10.0)


def test_place_tool_tilt_limit_rejects_excess_tilt():
    release_rpy = (pi, radians(12.0), 0.0)
    planner = planner_with_safety(
        require_tool_down=True,
        maximum_tool_tilt_deg=2.0,
        maximum_place_tool_tilt_deg=11.0,
    )

    with pytest.raises(RuntimeError, match="place maximum tool tilt"):
        planner.validate_grasp_safety_plan(make_plan(release_rpy=release_rpy))


def test_place_tool_tilt_limit_still_defaults_to_global_bound():
    release_rpy = (pi, radians(10.0), 0.0)
    planner = planner_with_safety(
        require_tool_down=True,
        maximum_tool_tilt_deg=2.0,
        maximum_place_tool_tilt_deg=None,
    )

    with pytest.raises(RuntimeError, match="place maximum tool tilt"):
        planner.validate_grasp_safety_plan(make_plan(release_rpy=release_rpy))


@pytest.mark.parametrize(
    ("safety_field", "limit", "message"),
    (
        ("minimum_approach_height_m", 0.09, "below"),
        ("maximum_approach_height_m", 0.07, "above"),
    ),
)
def test_pregrasp_approach_height_bounds(
    safety_field: str,
    limit: float,
    message: str,
):
    planner = planner_with_safety(**{safety_field: limit})

    with pytest.raises(RuntimeError, match=f"approach height.*{message}"):
        planner.validate_grasp_safety_plan(make_plan())


def test_pregrasp_xy_alignment_limit():
    planner = planner_with_safety(maximum_approach_xy_error_m=0.001)
    plan = make_plan(pregrasp_xyz=(0.102, 0.00, 0.265))

    with pytest.raises(RuntimeError, match="XY alignment error"):
        planner.validate_grasp_safety_plan(plan)


LIVE_LIMITS = {
    "minimum_grasp_flange_z_m": 0.15,
    "maximum_grasp_flange_z_m": 0.30,
    "minimum_grasp_tip_z_m": 0.0,
    "maximum_grasp_tip_z_m": 0.16,
    "minimum_route_flange_z_m": 0.02,
    "maximum_route_flange_z_m": 0.50,
    "minimum_route_tip_z_m": 0.0,
    "maximum_route_tip_z_m": 0.40,
    "maximum_horizontal_reach_m": 0.55,
    "maximum_tool_tilt_deg": 2.0,
    "maximum_place_tool_tilt_deg": 18.0,
    "minimum_approach_height_m": 0.05,
    "maximum_approach_height_m": 0.12,
    "maximum_approach_xy_error_m": 0.001,
}


def complete_live_safety(**overrides) -> GraspSafetyConfig:
    values = {
        "enabled": True,
        "require_flange_tip_offset": True,
        "require_tool_down": True,
        **LIVE_LIMITS,
        **overrides,
    }
    return GraspSafetyConfig(**values)


def live_config(safety: GraspSafetyConfig, *, gripper_backend: str = "mock"):
    base = cd_config()
    return replace(
        base,
        motion=replace(
            base.motion,
            backend="robotmove",
            dry_run=False,
            grasp_safety=safety,
        ),
        pose=replace(
            base.pose,
            extrinsics_file=str(
                ROOT / "config" / "camera_to_flange.d405_calibrated_20261025.yaml"
            ),
        ),
        gripper=replace(
            base.gripper,
            backend=gripper_backend,
            serial_port="/dev/null",
            require_feedback=True,
            hold_min_angle_deg=1.0,
        ),
    )


REQUIRED_LIVE_LIMITS = tuple(
    name for name in LIVE_LIMITS
    if name not in {"maximum_grasp_flange_z_m", "maximum_grasp_tip_z_m"}
)


def test_live_cd_validation_rejects_each_missing_required_safety_limit():
    for field_name in REQUIRED_LIVE_LIMITS:
        invalid_safety = complete_live_safety(**{field_name: None})
        with pytest.raises(ValueError, match=field_name):
            live_config(invalid_safety).validate()


def test_live_cd_validation_allows_unset_maximum_grasp_limits():
    safety = complete_live_safety(
        maximum_grasp_flange_z_m=None,
        maximum_grasp_tip_z_m=None,
    )

    live_config(safety, gripper_backend="ms42dc").validate()


def test_live_cd_validation_rejects_mock_gripper():
    with pytest.raises(ValueError, match="cannot use a mock gripper"):
        live_config(complete_live_safety(), gripper_backend="mock").validate()


def test_cd_safety_requires_explicit_release_pose_frame():
    base = cd_config()
    config = replace(
        base,
        motion=replace(base.motion, place_pose_frame=None),
    )

    with pytest.raises(ValueError, match="explicit.*place_pose_frame"):
        config.validate()


def test_jaw_tip_release_pose_requires_flange_tip_offset():
    base = cd_config()
    config = replace(
        base,
        pose=replace(base.pose, gripper_tip_offset_flange_m=None),
    )

    with pytest.raises(ValueError, match="jaw_tip requires.*gripper_tip"):
        config.validate()


def test_invalid_release_pose_frame_is_rejected():
    base = cd_config()
    config = replace(
        base,
        motion=replace(base.motion, place_pose_frame="tcp"),
    )

    with pytest.raises(ValueError, match="place_pose_frame"):
        config.validate()


def test_formal_cd_config_converts_release_tip_to_nonnegative_flange_route():
    config = load_config(FORMAL_CD_CONFIG)
    planner = PosePlanner(config)

    # The formal profile derives grasp XYZ from camera_xyz.  This test is about
    # the place/release frame, so bypass only that unrelated transform with an
    # explicit jaw-tip target while retaining the formal safety envelope.
    plan = planner.plan(
        detection(),
        robot_state(),
        grasp_xyz_override_m=(0.10, 0.0, 0.02),
        grasp_target_is_jaw_tip=True,
    )
    report = planner.validate_grasp_safety_plan(plan)

    assert plan.release_pose.xyz == pytest.approx(
        (0.233276436105, 0.304828196977, 0.404269299444)
    )
    assert report.release_tip_xyz_m == pytest.approx(
        (0.270377682201, 0.335654065085, 0.246477463808)
    )
    assert report.maximum_grasp_tool_tilt_deg == pytest.approx(0.0)
    assert report.maximum_place_tool_tilt_deg == pytest.approx(
        16.99815791059307
    )
    assert report.minimum_route_tip_z_m == pytest.approx(0.02)


def test_cd_safety_rejection_keeps_route_precheck_arm_and_gripper_idle():
    base = cd_config()
    safety = replace(
        base.motion.grasp_safety,
        # Deterministically above every possible route tip in this fixture.
        minimum_route_tip_z_m=10.0,
    )
    config = replace(
        base,
        motion=replace(base.motion, grasp_safety=safety),
    )
    pipeline = build_pipeline(config)
    prechecked_routes = []
    pipeline.motion.precheck_linear_targets = (
        lambda targets, *, step_m: prechecked_routes.append(tuple(targets))
    )

    result = pipeline.run_once()

    assert result.status is PickStatus.REJECTED
    assert "minimum route jaw-tip Z" in result.message
    assert prechecked_routes == []
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []
