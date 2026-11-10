from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np
import pytest

from track1_grasp.config import (
    TaskCDClearConfig,
    TaskCDShortEdgeGraspConfig,
    apply_mode,
    load_config,
)
from track1_grasp.domain import Detection, Frame, PickStatus, Pose, RobotState
from track1_grasp.factory import build_pipeline
from track1_grasp.planner import PosePlanner, euler_to_matrix
from track1_grasp.vision import build_vision


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = Path(__file__).resolve().parents[3]
MOCK_CONFIG = ROOT / "config" / "track1_mock.yaml"
FORMAL_CD_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_cd_40class_v11_official.yaml"
)
EXTRINSICS = (
    FORMAL_ROOT / "config" / "camera_to_flange.d405_calibrated_20261025.yaml"
)
GRASP_RPY = (pi, 0.0, 0.0)
TIP_OFFSET_M = (0.0, 0.0, 0.165)
PLACE_POSE = (0.25, 0.25, 0.08, *GRASP_RPY)


class ScriptedVision:
    def __init__(self, frames):
        self.frames = tuple(frames)
        self.index = 0

    def capture(self):
        if self.index >= len(self.frames):
            raise StopIteration("scripted vision exhausted")
        frame = self.frames[self.index]
        self.index += 1
        return frame


def write_observation_pose(path: Path) -> Path:
    path.write_text(
        "role: 'task_cd_observation'\n"
        "joint_positions_rad: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n",
        encoding="utf-8",
    )
    return path


def short_edge_config(observation_pose: Path):
    base = apply_mode(load_config(MOCK_CONFIG), "C")
    safety = replace(
        base.motion.grasp_safety,
        enabled=True,
        require_flange_tip_offset=True,
        require_tool_down=True,
    )
    config = replace(
        base,
        task=replace(base.task, max_picks=1),
        task_cd=TaskCDClearConfig(
            enabled=True,
            observation_pose_file=str(observation_pose),
            observation_pose_tolerance_deg=0.5,
            return_to_observation_after_each_pick=True,
            place_pose_verified=True,
            exclude_completed_tracks=True,
            short_edge_grasp=TaskCDShortEdgeGraspConfig(
                enabled=True,
                jaw_tip_z_m=0.000103,
            ),
        ),
        vision=replace(
            base.vision,
            require_depth=True,
            stabilize=True,
            stabilizer_max_per_class=0,
        ),
        pose=replace(
            base.pose,
            camera_frame="flange",
            extrinsics_file=str(EXTRINSICS),
            robot_pose_frame="flange",
            gripper_tip_offset_flange_m=TIP_OFFSET_M,
        ),
        motion=replace(
            base.motion,
            orientation="fixed",
            fixed_rpy=GRASP_RPY,
            grasp_rpy=GRASP_RPY,
            place_pose=PLACE_POSE,
            place_pose_frame="jaw_tip",
            grasp_safety=safety,
            workspace_min=(-0.65, -0.65, 0.0),
            workspace_max=(0.65, 0.65, 0.80),
            workspace_margin=0.0,
        ),
        search=replace(
            base.search,
            enabled=False,
            observe_frames=1,
            confirm_hits=1,
        ),
    )
    config.validate()
    return config


def meat_can_detection(**overrides) -> Detection:
    values = {
        "label": "10-meat_can",
        "confidence": 0.95,
        "bbox": (100.0, 100.0, 200.0, 180.0),
        "depth_m": 0.40,
        "base_xyz": (0.10, 0.0, 0.02),
        "object_center_camera_xyz": (0.0, 0.0, 0.40),
        "major_axis_camera": (1.0, 0.0, 0.0),
        "minor_axis_camera": (0.0, 1.0, 0.0),
        "major_axis_length_m": 0.0788,
        "minor_axis_length_m": 0.0434,
        "pose_confidence": 0.620,
        "graspability": 0.95,
        "instance_id": "meat_can_1",
    }
    values.update(overrides)
    return Detection(**values)


def frame(frame_id: str, *detections: Detection) -> Frame:
    return Frame(frame_id=frame_id, detections=tuple(detections))


def pipeline_with_frames(config, frames):
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(frames)
    # Isolate the heading math from the live observation pose and calibration.
    pipeline.planner.base_camera_transform = lambda _state: np.eye(4)
    return pipeline


def test_meat_can_major_axis_x_derives_short_edge_tool_y_and_dynamic_rpy(tmp_path):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    pipeline = build_pipeline(config)
    pipeline.planner.base_camera_transform = lambda _state: np.eye(4)

    rpy, diagnostics = pipeline._task_cd_short_edge_grasp_rpy(
        meat_can_detection(),
        pipeline.motion.get_state(),
    )

    assert rpy == pytest.approx((pi, 0.0, pi / 2.0))
    tool_y_base = euler_to_matrix(rpy)[:, 1]
    assert tool_y_base == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert diagnostics["alignment"] == pytest.approx(1.0)
    assert diagnostics["axis_ratio"] == pytest.approx(0.0788 / 0.0434)
    assert diagnostics["grasp_rpy"] == pytest.approx(rpy)


def test_meat_can_plan_uses_pca_heading_and_exact_six_segment_clear_cycle(
    tmp_path,
):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    pipeline = pipeline_with_frames(
        config,
        [
            frame("meat_can_visible", meat_can_detection()),
            frame("empty_after_clear"),
        ],
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
    ]
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    grasp_cycle = [pose.rpy for _name, pose in pipeline.motion.moves]
    assert grasp_cycle[:3] == pytest.approx(
        [(pi, 0.0, pi / 2.0)] * 3
    )
    assert grasp_cycle[:3] != pytest.approx([GRASP_RPY] * 3)
    grasp_names = [name for name, _pose in pipeline.motion.moves]
    grasp_pose = pipeline.motion.moves[grasp_names.index("grasp")][1]
    # The command is flange-frame: low jaw tip Z + the calibrated 0.165 m tool
    # offset.  This proves the visual top-surface Z is not the insertion target.
    assert grasp_pose.xyz[2] == pytest.approx(0.165103)
    assert any(
        "short-edge insertion: fixed jaw-tip Z=0.000103000 m" in event.message
        for result in report.results
        for event in result.events
    )
    assert pipeline.gripper.events == ["open", "close", "open"]
    assert any(
        "short-edge PCA heading" in event.message
        and "abs(tool_y dot major_axis)=1.000000000" in event.message
        for result in report.results
        for event in result.events
    )


def test_equivalent_wrist_yaw_retried_after_primary_route_precheck_failure(
    tmp_path,
):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    pipeline = pipeline_with_frames(
        config,
        [
            frame("meat_can_visible", meat_can_detection()),
            frame("empty_after_clear"),
        ],
    )
    original_precheck = pipeline.motion.precheck_linear_targets
    primary_rpy = (pi, 0.0, pi / 2.0)
    alternate_rpy = (pi, 0.0, -pi / 2.0)
    attempted_rpy = []

    def reject_primary_only(targets, *, step_m):
        first_rpy = tuple(float(value) for value in targets[0].rpy)
        attempted_rpy.append(first_rpy)
        if np.allclose(first_rpy, primary_rpy, atol=1e-12):
            raise RuntimeError("synthetic primary wrist IK singularity")
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = reject_primary_only

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [
        PickStatus.SUCCESS,
        PickStatus.NO_TARGET,
    ]
    assert attempted_rpy == pytest.approx([primary_rpy, alternate_rpy])
    assert [name for name, _pose in pipeline.motion.moves] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    assert [pose.rpy for _name, pose in pipeline.motion.moves[:3]] == (
        pytest.approx([alternate_rpy] * 3)
    )
    messages = [event.message for result in report.results for event in result.events]
    assert any(
        "trying equivalent short-edge wrist yaw" in message
        for message in messages
    )
    assert any(
        "short_edge_variant=pca-equivalent-yaw-plus-pi" in message
        for message in messages
    )


def test_disabled_short_edge_keeps_legacy_fixed_orientation_variant(tmp_path):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    config = replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=TaskCDShortEdgeGraspConfig(enabled=False),
        ),
    )
    config.validate()
    pipeline = build_pipeline(config)

    variants = pipeline._task_cd_short_edge_grasp_rpy_variants(None, {})

    assert variants == (("legacy", None, {}),)


@pytest.mark.parametrize(
    "overrides",
    (
        {"major_axis_camera": None},
        {"pose_confidence": 0.49},
        {"major_axis_length_m": 0.0800, "minor_axis_length_m": 0.0762},
    ),
    ids=("missing-pca", "low-confidence", "low-axis-ratio"),
)
def test_invalid_short_edge_pca_fails_closed_before_motion_or_gripper(
    tmp_path,
    overrides,
):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    pipeline = pipeline_with_frames(
        config,
        [frame("invalid_pca", meat_can_detection(**overrides))],
    )
    route_prechecks = []
    pipeline.motion.precheck_linear_targets = (
        lambda targets, *, step_m: route_prechecks.append(tuple(targets))
    )

    report = pipeline.run(max_picks=1)

    assert [result.status for result in report.results] == [PickStatus.NO_TARGET]
    assert route_prechecks == []
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]


def test_formal_cd_config_enables_only_validated_pca_short_edge_path():
    config = load_config(FORMAL_CD_CONFIG)
    short_edge = config.task_cd.short_edge_grasp

    assert short_edge is not None
    assert short_edge.enabled is True
    assert short_edge.min_pose_confidence == pytest.approx(0.50)
    assert short_edge.require_validated_extrinsics is True
    assert short_edge.jaw_tip_z_m == pytest.approx(
        config.motion.grasp_safety.minimum_grasp_tip_z_m
    )
    assert short_edge.pca.min_valid_points == 200
    assert short_edge.pca.min_axis_ratio == pytest.approx(1.02)
    assert short_edge.pca.support_plane_required is False
    assert config.adaptive_inverse.enabled is False
    assert config.pose.camera_frame == "flange"
    assert config.task_cd.enabled is True
    assert config.motion.cartesian_joint_step_deg == pytest.approx(0.60)
    assert (
        config.motion.cartesian_joint_step_deg
        < config.motion.joint_max_step_deg
    )


def test_real_yolo_vision_factory_receives_short_edge_pca_config(monkeypatch):
    from track1_grasp import vision as vision_module

    config = load_config(FORMAL_CD_CONFIG)
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    captured = {}

    class FakeRealSenseYoloVision:
        def __init__(
            self,
            vision_config,
            log=None,
            adaptive_inverse=None,
            object_pose_config=None,
        ):
            captured["vision_config"] = vision_config
            captured["adaptive_inverse"] = adaptive_inverse
            captured["object_pose_config"] = object_pose_config

    monkeypatch.setattr(
        vision_module,
        "RealSenseYoloVision",
        FakeRealSenseYoloVision,
    )
    source = build_vision(
        replace(config.vision, continuous=False),
        object_pose_config=short_edge.pca,
    )

    assert isinstance(source, FakeRealSenseYoloVision)
    assert captured["adaptive_inverse"] is None
    assert captured["object_pose_config"] is short_edge.pca


def test_dynamic_short_edge_rpy_passes_safety_and_legacy_call_stays_strict():
    config = load_config(FORMAL_CD_CONFIG)
    planner = PosePlanner(config)
    dynamic_rpy = (pi, 0.0, 1.17)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), GRASP_RPY),
        joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        message="mock-ready",
    )
    plan = planner.plan(
        meat_can_detection(base_xyz=(0.10, 0.0, 0.02)),
        state,
        grasp_xyz_override_m=(0.10, 0.0, 0.02),
        grasp_target_is_jaw_tip=True,
        grasp_rpy=dynamic_rpy,
    )

    report = planner.validate_grasp_safety_plan(
        plan,
        expected_grasp_rpy=dynamic_rpy,
    )
    assert plan.grasp_pose.rpy == pytest.approx(dynamic_rpy)
    assert euler_to_matrix(plan.grasp_pose.rpy)[:, 1][2] == pytest.approx(0.0)
    assert report.maximum_grasp_tool_tilt_deg == pytest.approx(0.0)

    with pytest.raises(
        RuntimeError,
        match="planned grasp orientation differs from motion\\.grasp_rpy",
    ):
        planner.validate_grasp_safety_plan(plan)


def test_short_edge_fixed_jaw_tip_z_cannot_go_below_verified_minimum():
    config = load_config(FORMAL_CD_CONFIG)
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    minimum = config.motion.grasp_safety.minimum_grasp_tip_z_m
    assert minimum is not None

    invalid = replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=replace(
                short_edge,
                jaw_tip_z_m=minimum - 0.001,
            ),
        ),
    )
    with pytest.raises(
        ValueError,
        match="jaw_tip_z_m .* is below the verified minimum",
    ):
        invalid.validate()
