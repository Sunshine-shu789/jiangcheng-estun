from dataclasses import replace
from math import pi, radians
from pathlib import Path

import numpy as np
import pytest
import yaml

from track1_grasp.config import GraspSafetyConfig, load_config
from track1_grasp.domain import Detection, MotionPlan, PickStatus, Pose, RobotState
from track1_grasp.factory import build_pipeline
from track1_grasp.motion import RobotMoveMotionBackend
from track1_grasp.planner import PosePlanner, euler_to_matrix


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"
FORMAL_ROOT = Path(__file__).resolve().parents[3]
FORMAL_CD_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_cd_40class_v11_official.yaml"
)
FORMAL_CD_CONFIGS = (
    FORMAL_ROOT / "config" / "track1_task_c_v11_official_20261102.yaml",
    FORMAL_ROOT / "config" / "track1_task_c_v11_official_mustard_single_20261102.yaml",
    FORMAL_ROOT / "config" / "track1_task_d_v11_official_20261102.yaml",
    FORMAL_ROOT / "config" / "track1_task_cd_40class_v11_official.yaml",
)
GRASP_RPY = (pi, 0.0, 0.0)
TIP_OFFSET_M = (0.0, 0.0, 0.165)
VERTICAL_WAYPOINTS = (
    "pregrasp",
    "grasp",
    "lift",
    "transfer",
    "release",
    "retreat",
)
TOOL_AXIS_WAYPOINTS = (
    "pregrasp",
    "approach",
    "grasp",
    "lift",
    "transfer",
    "release",
    "retreat",
)
LIVE_RELEASE_XYZ = (0.25, 0.25, 0.245)


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


def make_tilted_tool_axis_plan(
    tilt_deg: float,
    *,
    jaw_tip_z: float = 0.000103,
    all_waypoints_tilted: bool = False,
) -> MotionPlan:
    """Build a tool-axis plan whose grasp tip remains at a fixed safe height."""

    rpy = (pi, radians(tilt_deg), 0.0)
    rotation = euler_to_matrix(rpy)
    jaw_tip_xyz = np.asarray((0.10, 0.0, jaw_tip_z), dtype=float)
    grasp_xyz = jaw_tip_xyz - rotation @ np.asarray(TIP_OFFSET_M)
    approach_xyz = grasp_xyz - 0.08 * rotation[:, 2]
    return replace(
        make_plan(
            grasp_xyz=tuple(float(value) for value in grasp_xyz),
            pregrasp_xyz=tuple(float(value) for value in approach_xyz),
            release_xyz=LIVE_RELEASE_XYZ,
            rpy=rpy,
            release_rpy=rpy if all_waypoints_tilted else GRASP_RPY,
        ),
        approach_pose=Pose(
            tuple(float(value) for value in approach_xyz),
            rpy,
        ),
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
    report = planner.validate_grasp_safety_plan(
        plan,
        expected_grasp_rpy=plan.grasp_pose.rpy,
    )
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


@pytest.mark.parametrize("tilt_deg", (15.0, 30.0))
def test_bounded_tilt_only_route_uses_tip_floor(tilt_deg):
    plan = make_tilted_tool_axis_plan(
        tilt_deg,
        all_waypoints_tilted=True,
    )
    planner = planner_with_safety(
        **{
            **LIVE_LIMITS,
            "minimum_grasp_flange_z_m": 0.164621,
            "minimum_route_flange_z_m": 0.164621,
            "minimum_grasp_tip_z_m": 0.000103,
            "minimum_route_tip_z_m": 0.000103,
            "maximum_tool_tilt_deg": 30.0,
            "maximum_place_tool_tilt_deg": 30.0,
        }
    )

    report = planner.validate_grasp_safety_plan(
        plan,
        expected_grasp_rpy=plan.grasp_pose.rpy,
    )

    assert report.grasp_flange_xyz_m[2] < 0.164621
    assert report.grasp_tip_xyz_m[2] == pytest.approx(0.000103)
    assert report.maximum_grasp_tool_tilt_deg == pytest.approx(tilt_deg)


def test_bounded_tilt_still_rejects_grasp_tip_below_verified_floor():
    plan = make_tilted_tool_axis_plan(15.0, jaw_tip_z=0.0)
    planner = planner_with_safety(
        **{
            **LIVE_LIMITS,
            "minimum_grasp_flange_z_m": 0.164621,
            "minimum_route_flange_z_m": 0.164621,
            "minimum_grasp_tip_z_m": 0.000103,
            "minimum_route_tip_z_m": 0.000103,
            "maximum_tool_tilt_deg": 30.0,
        }
    )

    with pytest.raises(RuntimeError, match="grasp jaw-tip Z"):
        planner.validate_grasp_safety_plan(
            plan,
            expected_grasp_rpy=plan.grasp_pose.rpy,
        )


@pytest.mark.parametrize(
    ("raw_tip_xyz", "error_match"),
    (
        (None, "no jaw-tip point"),
        (np.asarray((0.10, 0.0, np.nan)), "finite jaw-tip point"),
        (np.asarray((0.10, 0.0, np.inf)), "finite jaw-tip point"),
    ),
    ids=("missing", "nan", "inf"),
)
def test_grasp_safety_rejects_missing_or_nonfinite_jaw_tip_measurements(
    monkeypatch,
    raw_tip_xyz,
    error_match,
):
    planner = planner_with_safety()
    monkeypatch.setattr(
        PosePlanner,
        "_jaw_tip_xyz_for_flange_pose",
        staticmethod(lambda _pose, _offset: raw_tip_xyz),
    )

    with pytest.raises(RuntimeError, match=error_match):
        planner.validate_grasp_safety_plan(make_plan())


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


def test_vertical_plan_uses_exact_six_waypoint_sequence():
    plan = make_plan()

    assert plan.approach_pose is None
    assert plan.validated_waypoint_names() == VERTICAL_WAYPOINTS


def test_tool_axis_plan_uses_exact_seven_waypoint_sequence():
    plan = replace(
        make_plan(release_xyz=LIVE_RELEASE_XYZ),
        approach_pose=Pose((0.10, 0.0, 0.265), GRASP_RPY),
    )

    assert plan.validated_waypoint_names() == TOOL_AXIS_WAYPOINTS


def test_vertical_planner_candidate_keeps_six_waypoint_plan():
    planner = PosePlanner(cd_config())
    plan = planner.plan(
        detection(),
        robot_state(),
        grasp_target_is_jaw_tip=True,
        grasp_rpy=GRASP_RPY,
        jaw_tip_z_m=0.02,
    )

    assert plan.approach_pose is None
    assert plan.validated_waypoint_names() == VERTICAL_WAYPOINTS


def test_tool_axis_approach_is_collinear_and_matches_grasp_orientation():
    plan = replace(
        make_plan(release_xyz=LIVE_RELEASE_XYZ),
        approach_pose=Pose((0.10, 0.0, 0.265), GRASP_RPY),
    )
    tool_z = euler_to_matrix(plan.grasp_pose.rpy)[:, 2]
    descent = np.asarray(plan.grasp_pose.xyz) - np.asarray(
        plan.approach_pose.xyz
    )

    assert plan.approach_pose.rpy == pytest.approx(plan.grasp_pose.rpy)
    assert descent == pytest.approx(0.08 * tool_z)
    assert float(np.linalg.norm(descent)) == pytest.approx(0.08)


def test_tool_axis_approach_report_exposes_kind_and_height():
    plan = replace(
        make_plan(release_xyz=LIVE_RELEASE_XYZ),
        approach_pose=Pose((0.10, 0.0, 0.265), GRASP_RPY),
    )

    report = planner_with_safety(**LIVE_LIMITS).validate_grasp_safety_plan(plan)

    assert report.approach_kind == "tool_axis"
    assert report.approach_height_m == pytest.approx(0.08)
    assert report.approach_xy_error_m == pytest.approx(0.0, abs=1e-12)
    assert report.tool_axis_approach_height_m == pytest.approx(0.08)


def test_tool_axis_approach_rejects_lateral_residual():
    plan = replace(
        make_plan(release_xyz=LIVE_RELEASE_XYZ),
        approach_pose=Pose((0.101, 0.0, 0.265), GRASP_RPY),
    )

    with pytest.raises(RuntimeError, match="lateral residual"):
        planner_with_safety(**LIVE_LIMITS).validate_grasp_safety_plan(plan)


def test_tool_axis_approach_rejects_orientation_mismatch():
    plan = replace(
        make_plan(release_xyz=LIVE_RELEASE_XYZ),
        approach_pose=Pose((0.10, 0.0, 0.265), (pi, 0.0, 0.01)),
    )

    with pytest.raises(RuntimeError, match="orientation must match"):
        planner_with_safety(**LIVE_LIMITS).validate_grasp_safety_plan(plan)


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
            require_close_travel_evidence=True,
            hold_min_angle_deg=0.0,
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


def test_live_cd_validation_rejects_fixed_empty_angle_with_width_evidence():
    config = live_config(complete_live_safety(), gripper_backend="ms42dc")
    config = replace(
        config,
        gripper=replace(config.gripper, hold_min_angle_deg=1978.0),
    )

    with pytest.raises(ValueError, match="hold_min_angle_deg=0.*object sizes vary"):
        config.validate()


def test_live_cd_validation_requires_width_aware_ms42dc_close():
    config = live_config(complete_live_safety(), gripper_backend="ms42dc")
    config = replace(
        config,
        gripper=replace(config.gripper, require_close_travel_evidence=False),
    )

    with pytest.raises(ValueError, match="require_close_travel_evidence=true"):
        config.validate()


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


@pytest.mark.parametrize(
    "config_path",
    FORMAL_CD_CONFIGS,
    ids=lambda path: path.name,
)
def test_formal_cd_configs_use_unified_speed_and_safety_envelope(
    config_path: Path,
):
    config = load_config(config_path)

    # Goal: D405 optical center to storage-box bottom is about 0.35 m.
    # At the verified lowest grasp pose that range is 0.179600 m.  With the
    # near-vertical camera axis, a 0.1706 m flange-Z lift reaches about 0.35 m.
    assert config.motion.lift_height == pytest.approx(0.1706)
    # C/D share one code path and one formal throughput cap: 0.18 m/s at
    # 2.4 rad/s.  Servo speed remains independently capped below.
    expected_speed_mps = 0.18
    expected_joint_speed_rad_s = 2.4
    assert config.motion.speed_mps == pytest.approx(expected_speed_mps)
    # Visual-servo speed remains independently capped at 0.01 m/s.
    assert config.servo.speed_mps == pytest.approx(0.01)
    assert config.motion.workspace_min == pytest.approx(
        (-0.80, -0.80, 0.02)
    )
    assert config.motion.workspace_max == pytest.approx(
        (0.80, 0.80, 0.80)
    )
    assert config.motion.workspace_margin == pytest.approx(0.01)
    assert config.motion.joint_speed_rad_s == pytest.approx(
        expected_joint_speed_rad_s
    )
    assert config.motion.joint_speed_safety_limit_rad_s == pytest.approx(
        expected_joint_speed_rad_s
    )
    assert config.motion.grasp_safety.maximum_tool_tilt_deg == pytest.approx(
        30.0
    )
    assert (
        config.motion.grasp_safety.maximum_horizontal_reach_m
        == pytest.approx(0.82)
    )
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    assert short_edge.tilt_angles_deg == pytest.approx(
        (15.0, -15.0, 30.0, -30.0)
    )
    assert short_edge.tool_axis_approach_height_m == pytest.approx(0.08)

    planner = PosePlanner(config)
    plan = planner.plan(
        detection(),
        robot_state(),
        grasp_xyz_override_m=(0.10, 0.0, short_edge.jaw_tip_z_m),
        grasp_target_is_jaw_tip=True,
    )
    report = planner.validate_grasp_safety_plan(plan)
    safety = config.motion.grasp_safety

    assert plan.lift_pose.xyz[:2] == pytest.approx(plan.grasp_pose.xyz[:2])
    assert plan.lift_pose.xyz[2] - plan.grasp_pose.xyz[2] == pytest.approx(
        0.1706
    )
    assert report.grasp_tip_xyz_m == pytest.approx(
        (0.10, 0.0, short_edge.jaw_tip_z_m)
    )
    assert report.maximum_route_flange_z_m is not None
    assert report.maximum_route_tip_z_m is not None
    assert safety.maximum_route_flange_z_m is not None
    assert safety.maximum_route_tip_z_m is not None
    assert (
        report.maximum_route_flange_z_m
        <= safety.maximum_route_flange_z_m
    )
    assert report.maximum_route_tip_z_m <= safety.maximum_route_tip_z_m


def test_formal_cd_relocation_offsets_retain_required_scan_coverage():
    # Keep the four directions, four corners, and elevated recheck while using
    # the site-validated +Y margins measured on 2026-11-03.
    expected = (
        (-0.12, 0.0, 0.0),
        (0.12, 0.0, 0.0),
        (0.0, -0.12, 0.0),
        (0.0, 0.11, 0.0),
        (-0.085, -0.085, 0.0),
        (0.085, -0.085, 0.0),
        (-0.085, 0.075, 0.0),
        (0.085, 0.075, 0.0),
        (0.0, 0.0, 0.08),
    )
    for config_path in FORMAL_CD_CONFIGS:
        config = load_config(config_path)
        assert config.task_cd.relocation_scan.offsets_xyz_m == expected


def test_formal_c_relocation_scan_full_route_passes_captured_site_posture(
    monkeypatch,
):
    config = load_config(FORMAL_CD_CONFIGS[0])
    observation_path = (
        FORMAL_CD_CONFIGS[0].parent / config.task_cd.observation_pose_file
    )
    observation = yaml.safe_load(
        observation_path.read_text(encoding='utf-8')
    )
    anchor = Pose(
        tuple(float(value) for value in observation['flange_xyz_m']),
        tuple(float(value) for value in observation['flange_rpy_rad']),
    )
    state = RobotState(
        ready=True,
        pose=anchor,
        joints=tuple(
            float(value) for value in observation['joint_positions_rad']
        ),
        message='offline captured-site feedback; no ROS publisher',
        controller_state='AutoReady',
        fault_flag=0,
    )
    backend = RobotMoveMotionBackend.__new__(RobotMoveMotionBackend)
    backend.config = config.motion
    backend._log = lambda _message: None
    monkeypatch.setattr(backend, 'wait_for_state', lambda _timeout: state)

    poses = tuple(
        Pose(
            tuple(
                anchor.xyz[index] + float(offset[index])
                for index in range(3)
            ),
            anchor.rpy,
        )
        for offset in config.task_cd.relocation_scan.offsets_xyz_m
    )
    targets = tuple(
        pose
        for scan_pose in poses
        for pose in (scan_pose, anchor)
    )
    route = backend.precheck_linear_targets(
        targets,
        step_m=config.motion.cartesian_step_m,
    )

    assert len(route.segments) == 2 * len(poses)
    minimum_sigma = min(
        float(segment.min_jacobian_singular_value)
        for segment in route.segments
    )
    # The controller remains fail-closed at 0.010. This regression additionally
    # requires at least 20% headroom across the complete outbound/return route.
    assert minimum_sigma >= 0.012
