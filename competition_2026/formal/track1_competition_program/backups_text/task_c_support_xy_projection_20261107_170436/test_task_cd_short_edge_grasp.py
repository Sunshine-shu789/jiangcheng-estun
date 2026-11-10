from dataclasses import replace
from math import atan2, nan, pi
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from track1_grasp.config import (
    GripperSelfFilterConfig,
    TaskCDClearConfig,
    TaskCDEmptyConfirmationConfig,
    TaskCDRelocationScanConfig,
    TaskCDShortEdgeGraspConfig,
    apply_mode,
    load_config,
)
from track1_grasp.collision import (
    GEOMETRY_SOURCE_OFFLINE_MOCK_SYNTHETIC,
    BoxClearanceConfig,
    check_box_clearance,
)
from track1_grasp.detection_stabilizer import DetectionStabilizer
from track1_grasp.domain import Detection, Frame, PickStatus, Pose, RobotState
from track1_grasp.factory import build_pipeline
from track1_grasp.grasp_geometry import (
    GraspGeometryCandidate,
    GraspGeometryEvidence,
    RimGeometry,
    SidewallGeometry,
    rotation_from_tool_axis_and_closure,
    select_grasp_geometry,
)
from track1_grasp.kinematics import matrix_to_rpy
from track1_grasp.planner import PosePlanner, euler_to_matrix
import track1_grasp.pipeline as pipeline_module
import track1_grasp.vision as vision_module
from track1_grasp.vision import RealSenseYoloVision, build_vision


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


def short_edge_config(
    observation_pose: Path,
    closure_axis_tool_xy=(1.0, 0.0),
    grasp_point_modes=("object_center",),
    handle_grasp_labels=(),
    handle_offset_fraction=0.30,
):
    base = apply_mode(load_config(MOCK_CONFIG), "C")
    # Keep this fixture on Task C's fail-closed known-class policy while
    # covering the formal V11 labels used by the short-edge scenarios.
    base = replace(
        base,
        target=replace(
            base.target,
            known_classes=("9-Mug", "10-meat_can", "13-Bowl"),
        ),
    )
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
            gripper_self_filter=GripperSelfFilterConfig(enabled=True),
            relocation_scan=TaskCDRelocationScanConfig(
                enabled=True,
                offsets_xyz_m=((0.02, 0.0, 0.0),),
                settle_s=0.0,
                complete_box_before_planning=False,
            ),
            empty_confirmation=TaskCDEmptyConfirmationConfig(
                enabled=True,
                require_complete_full_scan=True,
                confirmation_rounds=2,
                require_all_views_successful=True,
                reject_if_any_item_present=True,
            ),
            short_edge_grasp=TaskCDShortEdgeGraspConfig(
                enabled=True,
                jaw_tip_z_m=0.000103,
                closure_axis_tool_xy=closure_axis_tool_xy,
                grasp_point_modes=grasp_point_modes,
                handle_grasp_labels=handle_grasp_labels,
                handle_offset_fraction=handle_offset_fraction,
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


def box_edge_clearance_config(
    config,
    *,
    band_m: float = 0.08,
    wall_clearance_m: float = 0.015,
    wall_top_z_m: float | None = 0.12,
):
    """Add an explicit offline box around the origin for edge-rule tests."""

    config = replace(
        config,
        task_cd=replace(
            config.task_cd,
            box_clearance=BoxClearanceConfig(
                required=True,
                geometry_validated=True,
                geometry_source=GEOMETRY_SOURCE_OFFLINE_MOCK_SYNTHETIC,
                center_xy_m=(0.0, 0.0),
                yaw_rad=0.0,
                interior_size_m=(1.0, 1.0),
                wall_clearance_m=wall_clearance_m,
                edge_align_band_m=band_m,
                edge_parallel_required=True,
                wall_top_z_m=wall_top_z_m,
                gripper_footprint_flange_xy_m=(
                    (-0.031507, -0.058971),
                    (0.031507, -0.058971),
                    (0.031507, 0.058971),
                    (-0.031507, 0.058971),
                ),
            ),
        ),
    )
    config.validate()
    return config


def geometry_config(
    observation_pose: Path,
    *,
    allow_rim: bool,
    allow_vertical_center: bool = False,
    allow_horizontal_sidewall: bool = False,
    grasp_z_offset: float = 0.0,
):
    """Enable the opt-in geometry selector on the short-edge test profile."""

    config = short_edge_config(observation_pose)
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    short_edge = replace(
        short_edge,
        jaw_tip_z_m=None,
        height_adaptive_enabled=True,
        geometry_policy=replace(
            short_edge.geometry_policy,
            enabled=True,
            allow_rim=allow_rim,
            allow_vertical_center=allow_vertical_center,
            allow_horizontal_sidewall=allow_horizontal_sidewall,
        ),
        pca=replace(short_edge.pca, defer_axis_ratio_rejection=True),
    )
    config = replace(
        config,
        task_cd=replace(config.task_cd, short_edge_grasp=short_edge),
        motion=replace(config.motion, grasp_z_offset=grasp_z_offset),
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


def empty_full_scan_frames(prefix: str):
    """Return two complete fixed+relocation empty-scan rounds."""

    return [
        frame(f"{prefix}_round_{round_index}_{view}")
        for round_index in (1, 2)
        for view in ("fixed", "relocation")
    ]


def geometry_detection(
    evidence: GraspGeometryEvidence,
    *,
    label: str,
    instance_id: str,
) -> Detection:
    """Build a geometry-only observation without any legacy PCA fields."""

    return Detection(
        label=label,
        confidence=0.95,
        bbox=(100.0, 100.0, 200.0, 200.0),
        depth_m=0.40,
        base_xyz=(0.10, 0.0, 0.02),
        geometry_evidence=evidence,
        graspability=0.95,
        instance_id=instance_id,
    )


def box_geometry_evidence() -> GraspGeometryEvidence:
    return GraspGeometryEvidence(
        object_center_camera_xyz=(0.0, 0.0, 0.38),
        support_plane_valid=True,
        support_plane_quality=0.98,
        support_plane_center_camera_xyz=(0.0, 0.0, 0.40),
        support_plane_normal_camera=(0.0, 0.0, 1.0),
        up_camera=(0.0, 0.0, -1.0),
        visible_height_m=0.04,
        footprint_center_camera_xyz=(0.0, 0.0, 0.40),
        footprint_major_length_m=0.14,
        footprint_minor_length_m=0.05,
        footprint_axis_ratio=2.8,
        footprint_major_axis_camera=(1.0, 0.0, 0.0),
        footprint_minor_axis_camera=(0.0, 1.0, 0.0),
        top_center_camera_xyz=(0.0, 0.0, 0.36),
        top_height_m=0.04,
        short_edge_confidence=0.95,
        rim=None,
        sidewall=None,
        valid_points=5000,
    )


def rim_geometry_evidence() -> GraspGeometryEvidence:
    rim_z_m = 0.32568
    return GraspGeometryEvidence(
        object_center_camera_xyz=(0.0, 0.0, 0.36),
        support_plane_valid=True,
        support_plane_quality=0.98,
        support_plane_center_camera_xyz=(0.0, 0.0, 0.40),
        support_plane_normal_camera=(0.0, 0.0, 1.0),
        up_camera=(0.0, 0.0, -1.0),
        visible_height_m=0.08,
        footprint_center_camera_xyz=(0.0, 0.0, 0.40),
        footprint_major_length_m=0.06,
        footprint_minor_length_m=0.06,
        footprint_axis_ratio=1.0,
        footprint_major_axis_camera=(1.0, 0.0, 0.0),
        footprint_minor_axis_camera=(0.0, 1.0, 0.0),
        top_center_camera_xyz=(0.0, 0.0, rim_z_m),
        top_height_m=0.40 - rim_z_m,
        short_edge_confidence=0.20,
        rim=RimGeometry(
            center_camera_xyz=(0.0, 0.0, rim_z_m),
            center_xy_m=(0.0, 0.0),
            radius_m=0.03,
            depression_m=0.01432,
            circularity=1.0,
            coverage=1.0,
            fit_error_m=1e-6,
            inlier_ratio=1.0,
            valid_points=720,
            confidence=0.98,
            # Offline fixture only: the production path must measure this
            # wall closure span from the same RGB-D observation.
            rim_closure_width_m=0.006,
        ),
        sidewall=None,
        valid_points=5000,
    )


def closed_cylinder_geometry_evidence() -> GraspGeometryEvidence:
    return GraspGeometryEvidence(
        object_center_camera_xyz=(0.0, 0.0, 0.36),
        support_plane_valid=True,
        support_plane_quality=0.98,
        support_plane_center_camera_xyz=(0.0, 0.0, 0.40),
        support_plane_normal_camera=(0.0, 0.0, 1.0),
        up_camera=(0.0, 0.0, -1.0),
        visible_height_m=0.08,
        footprint_center_camera_xyz=(0.0, 0.0, 0.40),
        footprint_major_length_m=0.06,
        footprint_minor_length_m=0.06,
        footprint_axis_ratio=1.0,
        footprint_major_axis_camera=(1.0, 0.0, 0.0),
        footprint_minor_axis_camera=(0.0, 1.0, 0.0),
        top_center_camera_xyz=(0.0, 0.0, 0.32),
        top_height_m=0.08,
        short_edge_confidence=0.10,
        rim=None,
        sidewall=SidewallGeometry(
            center_camera_xyz=(0.0, 0.0, 0.36),
            center_xy_m=(0.0, 0.0),
            radius_m=0.03,
            height_m=0.08,
            outward_direction_camera=(1.0, 0.0, 0.0),
            circularity=1.0,
            coverage=1.0,
            fit_error_m=1e-6,
            inlier_ratio=1.0,
            valid_points=1000,
            confidence=0.90,
        ),
        valid_points=5000,
    )


def geometry_pipeline_with_frames(config, frames):
    pipeline = pipeline_with_frames(config, frames)

    # The camera optical +Z axis points towards the table. This deterministic
    # transform maps it to robot base -Z and places the support plane at Z=0 so
    # the geometry pipeline exercises the same downward-tool safety envelope.
    base_camera = np.eye(4)
    base_camera[:3, :3] = np.diag((1.0, -1.0, -1.0))
    base_camera[:3, 3] = (0.10, 0.0, 0.40)
    pipeline.planner.base_camera_transform = lambda _state: base_camera
    return pipeline


def live_style_geometry_scene(config):
    """Build one camera-like RGB-D observation without opening a camera.

    This mirrors the data passed by ``RealSenseYoloVision.capture`` to
    ``_make_detection``: a detector item, a depth image, camera intrinsics and
    the current RGB frame.  The point cloud has a measured support plane and a
    short elongated object, so the real PCA path must produce geometry
    evidence rather than a hand-built test fixture.
    """

    height, width = 240, 320
    v, u = np.mgrid[0:height, 0:width]
    intrinsics = SimpleNamespace(fx=600.0, fy=600.0, cx=160.0, cy=120.0)
    ray_u = (u - intrinsics.cx) / intrinsics.fx
    ray_v = (v - intrinsics.cy) / intrinsics.fy
    support_z = 1.0 / (2.5 - 0.03 * ray_u + 0.02 * ray_v)
    depth = support_z.copy()
    object_mask = (
        ((u - 160.0) / 52.0) ** 2 + ((v - 120.0) / 14.0) ** 2
    ) <= 1.0
    depth[object_mask] = support_z[object_mask] - 0.050

    bbox = (105.0, 96.0, 215.0, 144.0)
    item = SimpleNamespace(
        class_name="10-meat_can",
        confidence=0.95,
        bbox=bbox,
        u=160.0,
        v=120.0,
        depth_m=0.40,
        cam_xyz=(0.0, 0.0, 0.40),
    )
    return depth, intrinsics, item


def make_live_style_detection(config, depth, intrinsics, item, monkeypatch):
    """Call the real detector-to-domain adapter without starting a camera."""

    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    monkeypatch.setattr(
        vision_module,
        "target_support_guard_mask",
        lambda _color, _label: None,
    )
    adapter = RealSenseYoloVision(
        config.vision,
        object_pose_config=short_edge.pca,
    )
    return adapter._make_detection(
        item,
        depth,
        intrinsics,
        np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8),
    )


def non_scan_moves(pipeline):
    """Return commanded Cartesian moves excluding relocation-scan travel."""

    scan_prefixes = (
        "task_cd_relocation_scan_",
        "task_cd_relocation_return_",
    )
    return [
        (name, pose)
        for name, pose in pipeline.motion.moves
        if not name.startswith(scan_prefixes)
    ]


def pipeline_with_frames(config, frames):
    pipeline = build_pipeline(config)
    pipeline.vision = ScriptedVision(frames)
    # Isolate the heading math from the live observation pose and calibration.
    pipeline.planner.base_camera_transform = lambda _state: np.eye(4)
    return pipeline


@pytest.mark.parametrize(
    ("closure_axis_tool_xy", "expected_yaw"),
    (
        ((0.0, 1.0), pi),
        ((1.0, 0.0), pi / 2.0),
    ),
    ids=("cad-nominal-tool-y", "field-installed-tool-x"),
)
def test_meat_can_major_axis_x_derives_configured_minor_axis_rpy(
    tmp_path,
    closure_axis_tool_xy,
    expected_yaw,
):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        closure_axis_tool_xy=closure_axis_tool_xy,
    )
    pipeline = build_pipeline(config)
    pipeline.planner.base_camera_transform = lambda _state: np.eye(4)

    rpy, diagnostics = pipeline._task_cd_short_edge_grasp_rpy(
        meat_can_detection(),
        pipeline.motion.get_state(),
    )

    assert rpy == pytest.approx((pi, 0.0, expected_yaw))
    closure_axis_base = euler_to_matrix(rpy)[:, :2] @ np.asarray(
        closure_axis_tool_xy,
        dtype=float,
    )
    assert closure_axis_base == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert diagnostics["closure_axis_tool_xy"] == pytest.approx(
        closure_axis_tool_xy
    )
    assert diagnostics["closure_axis_base"] == pytest.approx(
        closure_axis_base,
        abs=1e-12,
    )
    assert diagnostics["alignment"] == pytest.approx(1.0)
    assert diagnostics["major_axis_base_xy"] == pytest.approx((1.0, 0.0))
    assert diagnostics["minor_axis_base_xy"] == pytest.approx((0.0, 1.0))
    assert diagnostics["horizontal_axis_unit_xy"] == pytest.approx((1.0, 0.0))
    assert diagnostics["minor_axis_unit_xy"] == pytest.approx((0.0, 1.0))
    assert diagnostics["axis_ratio"] == pytest.approx(0.0788 / 0.0434)
    assert diagnostics["grasp_rpy"] == pytest.approx(rpy)


def test_meat_can_plan_closes_across_pca_minor_axis_and_exact_six_segment_cycle(
    tmp_path,
):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    target = meat_can_detection(
        major_axis_camera=(0.8, 0.6, 0.0),
        minor_axis_camera=(-0.6, 0.8, 0.0),
    )
    expected_yaw = atan2(0.8, -0.6)
    pipeline = pipeline_with_frames(
        config,
        [
            frame("meat_can_visible", target),
            *empty_full_scan_frames("empty_after_clear"),
        ],
    )

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    grasp_cycle = [pose.rpy for _name, pose in pipeline.motion.moves]
    assert grasp_cycle[:3] == pytest.approx(
        [(pi, 0.0, expected_yaw)] * 3
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
        and "closure_axis_tool_xy=(1.000000000, 0.000000000)" in event.message
        and "minor_axis_base_xy=(-0.600000, 0.800000)" in event.message
        and "abs(configured_closure_axis dot minor_axis)=1.000000000"
        in event.message
        for result in report.results
        for event in result.events
    )


def test_live_4_frenchs_mustard_geometry_closes_perpendicular_to_major_axis(
    tmp_path,
):
    """Regression for the 2026-11-02 mustard attempt that closed along its length."""

    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    major = np.asarray((0.422413, -0.905578, 0.0), dtype=float)
    minor = np.asarray((0.905578, 0.422413, 0.0), dtype=float)
    major /= np.linalg.norm(major)
    minor /= np.linalg.norm(minor)
    target = meat_can_detection(
        label="4-Frenchs_Mustard",
        major_axis_camera=tuple(major),
        minor_axis_camera=tuple(minor),
        pose_confidence=0.583622,
    )
    pipeline = build_pipeline(config)
    pipeline.planner.base_camera_transform = lambda _state: np.eye(4)

    rpy, diagnostics = pipeline._task_cd_short_edge_grasp_rpy(
        target,
        pipeline.motion.get_state(),
    )

    closure_axis = np.asarray(diagnostics["closure_axis_base"], dtype=float)
    assert rpy == pytest.approx((pi, 0.0, atan2(minor[1], minor[0])))
    assert abs(float(np.dot(closure_axis, major))) < 1e-9
    assert abs(float(np.dot(closure_axis, minor))) == pytest.approx(1.0)
    assert diagnostics["alignment"] == pytest.approx(1.0)
    assert diagnostics["major_axis_base_xy"] == pytest.approx(major[:2])
    assert diagnostics["minor_axis_base_xy"] == pytest.approx(minor[:2])


def test_short_edge_uses_point_cloud_centre_instead_of_detection_anchor(
    tmp_path,
):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    target = meat_can_detection(
        base_xyz=(0.40, 0.30, 0.02),
        object_center_camera_xyz=(0.012, -0.007, 0.40),
    )
    pipeline = pipeline_with_frames(
        config,
        [
            frame("centre_visible", target),
            *empty_full_scan_frames("empty_after_clear"),
        ],
    )
    base_camera = np.eye(4)
    base_camera[:3, 3] = (0.20, -0.10, 0.0)
    pipeline.planner.base_camera_transform = lambda _state: base_camera

    report = pipeline.run(acceptance_max_successes=1)

    expected_xy = (0.212, -0.107)
    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    np.testing.assert_allclose(
        [pose.xyz[:2] for _name, pose in pipeline.motion.moves[:3]],
        [expected_xy] * 3,
        atol=1e-12,
    )
    assert pipeline.motion.moves[0][1].xyz[:2] != pytest.approx((0.40, 0.30))
    messages = [event.message for result in report.results for event in result.events]
    assert any(
        "mode=object_center" in message
        and "source=point-cloud geometric centre" in message
        and "xy_correction_m=" in message
        for message in messages
    )
    assert any(
        "grasp_point_mode=object_center" in message for message in messages
    )


def test_handle_fallback_runs_only_after_centre_route_rejection(tmp_path):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        grasp_point_modes=("object_center", "handle"),
        handle_grasp_labels=("10-meat_can",),
        handle_offset_fraction=0.30,
    )
    pipeline = pipeline_with_frames(
        config,
        [
            frame("handle_visible", meat_can_detection()),
            *empty_full_scan_frames("empty_after_clear"),
        ],
    )
    original_precheck = pipeline.motion.precheck_linear_targets
    attempted_xy = []
    expected_offset = 0.30 * 0.0788

    def reject_centre_route(targets, *, step_m):
        if len(targets) != 6:
            return original_precheck(targets, step_m=step_m)
        grasp_xy = tuple(float(value) for value in targets[0].xyz[:2])
        attempted_xy.append(grasp_xy)
        if np.linalg.norm(np.asarray(grasp_xy)) <= 1e-9:
            raise RuntimeError("synthetic centre route rejection")
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = reject_centre_route

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    np.testing.assert_allclose(
        attempted_xy[:2],
        [(0.0, 0.0), (0.0, 0.0)],
        atol=1e-12,
    )
    assert attempted_xy[2] == pytest.approx((expected_offset, 0.0))
    np.testing.assert_allclose(
        [pose.xyz[:2] for _name, pose in pipeline.motion.moves[:3]],
        [(expected_offset, 0.0)] * 3,
        atol=1e-12,
    )
    messages = [event.message for result in report.results for event in result.events]
    assert any(
        "endpoint/handle fallback" in message
        and "not a semantic detected handle point" in message
        and "major_axis_offset_m=0.023640000" in message
        for message in messages
    )
    assert any(
        "grasp_point_mode=handle_positive_major_axis" in message
        for message in messages
    )


def test_handle_fallback_is_disabled_for_unlisted_label(tmp_path):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        grasp_point_modes=("object_center", "handle"),
        handle_grasp_labels=("10-meat_can",),
    )
    unlisted = meat_can_detection(label="13-Bowl", instance_id="bowl_1")
    pipeline = pipeline_with_frames(
        config,
        [
            frame("unlisted_visible", unlisted),
            frame("fixed_view_unlisted_target_excluded", unlisted),
            frame("relocation_view_unlisted_target_excluded", unlisted),
            frame("fixed_retry_unlisted_target_excluded", unlisted),
        ],
    )
    attempted_xy = []
    original_precheck = pipeline.motion.precheck_linear_targets

    def reject_every_route(targets, *, step_m):
        if len(targets) == 6:
            attempted_xy.append(
                tuple(float(value) for value in targets[0].xyz[:2])
            )
            raise RuntimeError("synthetic centre rejection")
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = reject_every_route

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.REJECTED]
    assert report.successes == 0
    assert report.results[0].target.instance_id == "bowl_1"
    np.testing.assert_allclose(
        attempted_xy[:2],
        [(0.0, 0.0), (0.0, 0.0)],
        atol=1e-12,
    )
    assert len(attempted_xy) == 2
    assert non_scan_moves(pipeline) == []
    assert pipeline.gripper.events == []
    assert not any(
        "endpoint/handle fallback" in event.message
        for result in report.results
        for event in result.events
    )


@pytest.mark.parametrize("task_mode", ("C", "D"))
def test_pca_pregrasp_endpoint_rejects_before_full_route_precheck(
    tmp_path,
    task_mode,
):
    """A near-singular high endpoint must not trigger route refinement."""

    config = apply_mode(
        short_edge_config(
            write_observation_pose(tmp_path / "observation.yaml")
        ),
        task_mode,
    )
    target = meat_can_detection()
    pipeline = pipeline_with_frames(
        config,
        [
            frame("near_singular_pca_visible", target),
            frame("fixed_view_pca_target_excluded", target),
            frame("relocation_view_pca_target_excluded", target),
            frame("fixed_retry_pca_target_excluded", target),
        ],
    )
    original_precheck = pipeline.motion.precheck_linear_targets
    endpoint_checks = []
    grasp_route_prechecks = []

    def reject_pregrasp_endpoint(pose, *, label):
        endpoint_checks.append((pose, label))
        return config.motion.cartesian_min_jacobian_singular_value / 2.0

    def record_route_precheck(targets, *, step_m):
        if len(targets) == 6:
            grasp_route_prechecks.append(tuple(targets))
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_pose_condition = reject_pregrasp_endpoint
    pipeline.motion.precheck_linear_targets = record_route_precheck

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.REJECTED]
    assert report.results[0].target is not None
    assert endpoint_checks
    assert grasp_route_prechecks == []
    assert non_scan_moves(pipeline) == []
    assert pipeline.gripper.events == []
    messages = [event.message for result in report.results for event in result.events]
    assert any(
        "pregrasp endpoint minimum Jacobian sigma" in message
        and "is below required" in message
        and "rejected before motion" in message
        for message in messages
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
            *empty_full_scan_frames("empty_after_clear"),
        ],
    )
    original_precheck = pipeline.motion.precheck_linear_targets
    primary_rpy = (pi, 0.0, pi / 2.0)
    alternate_rpy = (pi, 0.0, -pi / 2.0)
    attempted_rpy = []

    def reject_primary_only(targets, *, step_m):
        if len(targets) != 6:
            return original_precheck(targets, step_m=step_m)
        first_rpy = tuple(float(value) for value in targets[0].rpy)
        attempted_rpy.append(first_rpy)
        if np.allclose(first_rpy, primary_rpy, atol=1e-12):
            raise RuntimeError("synthetic primary wrist IK singularity")
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = reject_primary_only

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert attempted_rpy == pytest.approx([primary_rpy, alternate_rpy])
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
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


def test_tilted_route_retried_after_both_vertical_route_prechecks_fail(tmp_path):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    short_edge = replace(
        short_edge,
        tilt_angles_deg=(15.0, -15.0, 30.0, -30.0),
        tool_axis_approach_height_m=0.08,
    )
    config = replace(
        config,
        task_cd=replace(config.task_cd, short_edge_grasp=short_edge),
    )
    config.validate()
    pipeline = pipeline_with_frames(
        config,
        [
            frame("meat_can_visible", meat_can_detection()),
            *empty_full_scan_frames("empty_after_clear"),
        ],
    )
    original_precheck = pipeline.motion.precheck_linear_targets
    attempted_lengths = []

    def reject_vertical_routes(targets, *, step_m):
        if len(targets) in (6, 7):
            attempted_lengths.append(len(targets))
        if len(targets) == 6:
            raise RuntimeError("synthetic vertical route rejection")
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = reject_vertical_routes

    report = pipeline.run(acceptance_max_successes=1)

    assert attempted_lengths == [6, 6, 7]
    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
        "pregrasp",
        "approach",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    messages = [event.message for result in report.results for event in result.events]
    assert any(
        "short_edge_variant=pca-tilt-plus-15-primary" in message
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


def test_pca_tilt_variants_keep_deterministic_bounded_order(tmp_path):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    short_edge = replace(
        short_edge,
        tilt_angles_deg=(15.0, -15.0, 30.0, -30.0),
        tool_axis_approach_height_m=0.08,
    )
    config = replace(
        config,
        task_cd=replace(config.task_cd, short_edge_grasp=short_edge),
    )
    config.validate()
    pipeline = pipeline_with_frames(config, [])

    grasp_rpy, diagnostics = pipeline._task_cd_short_edge_grasp_rpy(
        meat_can_detection(),
        pipeline.motion.get_state(),
    )
    variants = pipeline._task_cd_short_edge_grasp_rpy_variants(
        grasp_rpy,
        diagnostics,
    )

    assert [name for name, _rpy, _diagnostics in variants] == [
        "pca-primary",
        "pca-equivalent-yaw-plus-pi",
        "pca-tilt-plus-15-primary",
        "pca-tilt-plus-15-wrist-equivalent",
        "pca-tilt-minus-15-primary",
        "pca-tilt-minus-15-wrist-equivalent",
        "pca-tilt-plus-30-primary",
        "pca-tilt-plus-30-wrist-equivalent",
        "pca-tilt-minus-30-primary",
        "pca-tilt-minus-30-wrist-equivalent",
    ]
    for name, _rpy, candidate_diagnostics in variants[2:]:
        assert name.startswith("pca-tilt-")
        assert candidate_diagnostics["tool_axis_approach"] is True
        assert candidate_diagnostics["tool_axis_approach_height_m"] == pytest.approx(
            0.08
        )


def test_pca_tilt_preserves_closure_when_xy_axis_projections_are_not_orthogonal(
    tmp_path,
):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    short_edge = replace(
        short_edge,
        tilt_angles_deg=(15.0, -15.0, 30.0, -30.0),
        tool_axis_approach_height_m=0.08,
    )
    config = replace(
        config,
        task_cd=replace(config.task_cd, short_edge_grasp=short_edge),
    )
    config.validate()
    pipeline = build_pipeline(config)
    minor_direction = np.asarray((0.8, 0.6), dtype=float)
    primary = (pi, 0.0, atan2(minor_direction[1], minor_direction[0]))
    diagnostics = {
        "horizontal_axis_unit_xy": (1.0, 0.0),
        "minor_axis_unit_xy": tuple(minor_direction),
        "closure_axis_tool_xy": (1.0, 0.0),
    }

    variants = pipeline._task_cd_short_edge_grasp_rpy_variants(
        primary,
        diagnostics,
    )

    assert len(variants) == 10
    for _name, _rpy, candidate_diagnostics in variants[2:]:
        closure_axis = np.asarray(
            candidate_diagnostics["closure_axis_base"],
            dtype=float,
        )
        tool_axis = np.asarray(candidate_diagnostics["tool_axis_base"], dtype=float)
        tilt_rad = np.radians(candidate_diagnostics["tilt_deg"])
        assert closure_axis[2] == pytest.approx(0.0, abs=1e-12)
        assert abs(float(np.dot(closure_axis[:2], minor_direction))) == pytest.approx(
            1.0,
            abs=1e-12,
        )
        assert float(np.dot(tool_axis, closure_axis)) == pytest.approx(
            0.0,
            abs=1e-12,
        )
        assert float(np.hypot(tool_axis[0], tool_axis[1])) == pytest.approx(
            abs(np.sin(tilt_rad)),
            abs=1e-12,
        )


def test_geometry_tilt_variants_keep_deterministic_bounded_order(tmp_path):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    short_edge = replace(
        short_edge,
        tilt_angles_deg=(15.0, -15.0, 30.0, -30.0),
        tool_axis_approach_height_m=0.08,
    )
    config = replace(
        config,
        task_cd=replace(config.task_cd, short_edge_grasp=short_edge),
    )
    config.validate()
    pipeline = build_pipeline(config)
    candidate = GraspGeometryCandidate(
        mode="vertical_short_edge",
        target_camera_xyz=(0.0, 0.0, 0.36),
        tool_axis_camera=(0.0, 0.0, 1.0),
        closure_axis_camera=(1.0, 0.0, 0.0),
        confidence=0.95,
        evidence="synthetic geometry variant ordering fixture",
    )
    base_camera = np.eye(4)
    base_camera[:3, :3] = np.diag((1.0, -1.0, -1.0))

    variants = pipeline._task_cd_geometry_orientation_variants(
        candidate,
        base_camera,
    )

    assert [name for name, _rpy, _diagnostics in variants] == [
        "geometry-primary",
        "geometry-jaw-flip",
        "geometry-tilt-plus-15-primary",
        "geometry-tilt-plus-15-jaw-flip",
        "geometry-tilt-minus-15-primary",
        "geometry-tilt-minus-15-jaw-flip",
        "geometry-tilt-plus-30-primary",
        "geometry-tilt-plus-30-jaw-flip",
        "geometry-tilt-minus-30-primary",
        "geometry-tilt-minus-30-jaw-flip",
    ]
    for name, _rpy, candidate_diagnostics in variants[2:]:
        assert name.startswith("geometry-tilt-")
        assert candidate_diagnostics["tool_axis_approach"] is True
        assert candidate_diagnostics["tool_axis_approach_height_m"] == pytest.approx(
            0.08
        )


@pytest.mark.parametrize(
    (
        "target_xy",
        "expected_wall",
        "expected_parallel",
        "measured_already_parallel",
    ),
    (
        ((-0.47, 0.0), "left_inner_wall", (0.0, 1.0), False),
        ((0.47, 0.0), "right_inner_wall", (0.0, 1.0), False),
        ((0.0, -0.47), "near_inner_wall", (1.0, 0.0), True),
        ((0.0, 0.47), "far_inner_wall", (1.0, 0.0), True),
    ),
)
def test_box_edge_forces_closure_axis_parallel_to_nearest_wall(
    tmp_path,
    target_xy,
    expected_wall,
    expected_parallel,
    measured_already_parallel,
):
    config = box_edge_clearance_config(
        short_edge_config(write_observation_pose(tmp_path / "observation.yaml"))
    )
    pipeline = build_pipeline(config)
    candidate = GraspGeometryCandidate(
        mode="vertical_short_edge",
        target_camera_xyz=(target_xy[0], target_xy[1], 0.36),
        tool_axis_camera=(0.0, 0.0, 1.0),
        # The object geometry is deliberately perpendicular to three of the
        # four walls.  The edge rule must override it, not merely prefer it.
        closure_axis_camera=(1.0, 0.0, 0.0),
        confidence=0.95,
        evidence="synthetic box-edge closure-axis regression fixture",
    )

    variants = pipeline._task_cd_geometry_orientation_variants(
        candidate,
        np.eye(4),
    )

    assert variants
    for _name, _rpy, diagnostics in variants:
        assert diagnostics["box_edge_parallel_applied"] is True
        assert diagnostics["box_edge_wall"] == expected_wall
        assert diagnostics["closure_axis_source"] == "box_edge_parallel"
        closure_xy = np.asarray(
            diagnostics["closure_axis_base"],
            dtype=float,
        )[:2]
        closure_xy /= np.linalg.norm(closure_xy)
        assert abs(float(np.dot(closure_xy, expected_parallel))) == pytest.approx(
            1.0,
            abs=1e-9,
        )
        measured_xy = np.asarray(
            diagnostics["measured_closure_axis_base"],
            dtype=float,
        )[:2]
        measured_alignment = abs(
            float(np.dot(measured_xy, expected_parallel))
        )
        if measured_already_parallel:
            assert measured_alignment == pytest.approx(1.0, abs=1e-9)
        else:
            assert measured_alignment < 0.999999


@pytest.mark.parametrize(
    ("target_xy", "expected_wall", "wall_parallel"),
    (
        ((-0.425, 0.0), "left_inner_wall", (0.0, 1.0)),
        ((0.425, 0.0), "right_inner_wall", (0.0, 1.0)),
        ((0.0, -0.425), "near_inner_wall", (1.0, 0.0)),
        ((0.0, 0.425), "far_inner_wall", (1.0, 0.0)),
    ),
)
def test_box_edge_rotation_keeps_both_finger_tips_parallel_and_inside(
    tmp_path,
    target_xy,
    expected_wall,
    wall_parallel,
):
    config = box_edge_clearance_config(
        short_edge_config(write_observation_pose(tmp_path / "observation.yaml"))
    )
    pipeline = build_pipeline(config)
    candidate = GraspGeometryCandidate(
        mode="vertical_short_edge",
        target_camera_xyz=(target_xy[0], target_xy[1], 0.36),
        tool_axis_camera=(0.0, 0.0, 1.0),
        closure_axis_camera=(1.0, 0.0, 0.0),
        confidence=0.95,
        evidence="synthetic two-finger box-edge parallelism fixture",
    )

    variants = pipeline._task_cd_geometry_orientation_variants(
        candidate,
        np.eye(4),
    )
    assert [name for name, _rpy, _diagnostics in variants] == [
        "geometry-primary",
        "geometry-jaw-flip",
    ]

    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    closure_tool = np.asarray(short_edge.closure_axis_tool_xy, dtype=float)
    closure_tool /= np.linalg.norm(closure_tool)
    footprint = np.asarray(
        config.task_cd.box_clearance.gripper_footprint_flange_xy_m,
        dtype=float,
    )
    finger_half_span_m = float(np.max(np.abs(footprint @ closure_tool)))
    assert finger_half_span_m > 0.0

    target = np.asarray(target_xy, dtype=float)
    parallel = np.asarray(wall_parallel, dtype=float)
    tip_offset = np.asarray(
        config.pose.gripper_tip_offset_flange_m,
        dtype=float,
    )
    for _name, _rpy, diagnostics in variants:
        rotation = np.asarray(diagnostics["rotation_base_tool"], dtype=float)
        closure_xy = (rotation[:, :2] @ closure_tool)[:2]
        closure_xy /= np.linalg.norm(closure_xy)
        finger_tips = np.asarray(
            (
                target + finger_half_span_m * closure_xy,
                target - finger_half_span_m * closure_xy,
            )
        )

        # Both physical finger tips must lie on the same line parallel to the
        # nearest wall.  Equal signed wall distances prove that neither finger
        # crosses to the outside while the other remains inside.
        wall_normal = np.asarray((-parallel[1], parallel[0]), dtype=float)
        assert float(np.dot(finger_tips[0], wall_normal)) == pytest.approx(
            float(np.dot(finger_tips[1], wall_normal)),
            abs=1e-9,
        )
        assert abs(
            float(np.dot(finger_tips[0] - finger_tips[1], parallel))
        ) == pytest.approx(2.0 * finger_half_span_m, abs=1e-9)
        if expected_wall == "left_inner_wall":
            finger_wall_clearance = finger_tips[:, 0] + 0.5
        elif expected_wall == "right_inner_wall":
            finger_wall_clearance = 0.5 - finger_tips[:, 0]
        elif expected_wall == "near_inner_wall":
            finger_wall_clearance = finger_tips[:, 1] + 0.5
        else:
            assert expected_wall == "far_inner_wall"
            finger_wall_clearance = 0.5 - finger_tips[:, 1]
        assert np.all(
            finger_wall_clearance
            >= config.task_cd.box_clearance.wall_clearance_m
        )
        assert finger_wall_clearance[0] == pytest.approx(
            finger_wall_clearance[1],
            abs=1e-9,
        )

        flange_xy = target - (rotation @ tip_offset)[:2]
        clearance = check_box_clearance(
            config.task_cd.box_clearance,
            flange_xy_m=flange_xy,
            rotation=rotation,
        )
        assert clearance.allowed is True, clearance.reason


def test_geometry_invalid_close_width_is_checked_once_before_orientation_planning(
    tmp_path,
    monkeypatch,
):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    config = replace(
        config,
        gripper=replace(
            config.gripper,
            require_close_travel_evidence=True,
        ),
    )
    config.validate()
    pipeline = build_pipeline(config)
    invalid_width_m = 0.082202
    attempts = tuple(
        SimpleNamespace(
            geometry_mode="vertical_short_edge",
            point_mode="object_center",
            variant_name=variant_name,
            close_width_m=invalid_width_m,
        )
        for variant_name in (
            "geometry-primary",
            "geometry-jaw-flip",
            "geometry-tilt-plus-15-primary",
            "geometry-tilt-plus-15-jaw-flip",
            "geometry-tilt-minus-15-primary",
        )
    )
    pipeline._task_cd_geometry_grasp_attempts = (
        lambda _target, _state, _emit: attempts
    )

    validation_calls = []

    def reject_width(_config, width_m):
        validation_calls.append(float(width_m))
        raise RuntimeError("synthetic CAD opening rejection")

    monkeypatch.setattr(
        pipeline_module,
        "validate_close_width",
        reject_width,
    )
    planner_calls = []
    pipeline.planner.plan = lambda *args, **kwargs: planner_calls.append(
        (args, kwargs)
    )
    events = []
    emit = lambda phase, message: events.append(
        SimpleNamespace(phase=phase, message=message)
    )
    state = pipeline.motion.get_state()

    result = pipeline._precheck_task_cd_geometry_candidate(
        meat_can_detection(),
        state,
        state,
        emit,
    )

    assert result is None
    assert validation_calls == [pytest.approx(invalid_width_m)]
    assert planner_calls == []
    assert len(events) >= 1
    assert "failed the CAD opening gate" in events[0].message
    assert pipeline.motion.moves == []
    assert pipeline.motion.joint_moves == []
    assert pipeline.gripper.events == []


def test_geometry_invalid_width_skips_only_that_attempt_and_tries_safe_width(
    tmp_path,
    monkeypatch,
):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    config = replace(
        config,
        gripper=replace(
            config.gripper,
            require_close_travel_evidence=True,
        ),
    )
    config.validate()
    pipeline = build_pipeline(config)
    invalid_width_m = 0.082202
    safe_width_m = 0.040000

    def attempt(width_m, variant_name):
        return SimpleNamespace(
            geometry_mode="vertical_short_edge",
            point_mode="object_center",
            point_xyz_m=(0.10, 0.0, 0.04),
            point_info={},
            variant_name=variant_name,
            grasp_rpy=GRASP_RPY,
            diagnostics={
                "tool_axis_base": (0.0, 0.0, -1.0),
                "closure_axis_base": (0.0, 1.0, 0.0),
            },
            jaw_tip_z_m=0.04,
            preserve_grasp_target_z=True,
            tool_axis_approach=True,
            tool_axis_approach_height_m=0.10,
            close_width_m=width_m,
        )

    attempts = (
        attempt(invalid_width_m, "geometry-primary"),
        attempt(safe_width_m, "geometry-jaw-flip"),
    )
    pipeline._task_cd_geometry_grasp_attempts = (
        lambda _target, _state, _emit: attempts
    )
    validation_calls = []

    def validate_width(_config, width_m):
        validation_calls.append(float(width_m))
        if float(width_m) == pytest.approx(invalid_width_m):
            raise RuntimeError("synthetic CAD opening rejection")

    monkeypatch.setattr(
        pipeline_module,
        "validate_close_width",
        validate_width,
    )
    planner_calls = []

    def reach_orientation_planning(*args, **kwargs):
        planner_calls.append((args, kwargs))
        raise RuntimeError("synthetic orientation planning reached")

    pipeline.planner.plan = reach_orientation_planning
    events = []
    emit = lambda phase, message: events.append(
        SimpleNamespace(phase=phase, message=message)
    )
    state = pipeline.motion.get_state()

    result = pipeline._precheck_task_cd_geometry_candidate(
        meat_can_detection(),
        state,
        state,
        emit,
    )

    assert result is None
    assert validation_calls == [
        pytest.approx(invalid_width_m),
        pytest.approx(safe_width_m),
    ]
    assert len(planner_calls) == 1
    assert any(
        "skipping geometry-primary" in event.message
        and "failed the CAD opening gate" in event.message
        for event in events
    )
    assert any(
        "variant=geometry-jaw-flip" in event.message
        and "trying geometry grasp" in event.message
        for event in events
    )
    assert any(
        "synthetic orientation planning reached" in event.message
        for event in events
    )
    assert pipeline.motion.moves == []
    assert pipeline.motion.joint_moves == []
    assert pipeline.gripper.events == []


def test_box_edge_parallel_pose_that_still_breaks_wall_clearance_is_rejected(
    tmp_path,
):
    config = box_edge_clearance_config(
        short_edge_config(write_observation_pose(tmp_path / "observation.yaml"))
    )
    pipeline = build_pipeline(config)
    rotation = rotation_from_tool_axis_and_closure(
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0),
    )
    low_pose = Pose(
        (-0.47, 0.0, 0.05),
        matrix_to_rpy(rotation),
    )

    with pytest.raises(RuntimeError, match="wall clearance"):
        pipeline._validate_task_cd_box_clearance_pose(
            low_pose,
            label="edge-parallel low grasp",
        )

    # The same XY is safe while the complete tool remains above the wall top;
    # this preserves high observation/transit poses without weakening descent.
    high_pose = Pose(
        (-0.47, 0.0, 0.40),
        matrix_to_rpy(rotation),
    )
    pipeline._validate_task_cd_box_clearance_pose(
        high_pose,
        label="edge-parallel high transit",
    )


def test_box_edge_parallel_footprint_failure_reports_no_motion_allowed(
    tmp_path,
):
    config = box_edge_clearance_config(
        short_edge_config(write_observation_pose(tmp_path / "observation.yaml"))
    )
    rotation = rotation_from_tool_axis_and_closure(
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0),
    )
    result = check_box_clearance(
        config.task_cd.box_clearance,
        flange_xy_m=(-0.47, 0.0),
        rotation=rotation,
        flange_z_m=0.05,
        tool_below_flange_m=0.165,
    )
    assert result.allowed is False
    assert "wall clearance" in result.reason


@pytest.mark.parametrize("mode", ("vertical_short_edge", "vertical_rim"))
def test_geometry_top_grasp_flips_measured_tool_axis_to_the_down_hemisphere(
    tmp_path,
    mode,
):
    config = short_edge_config(
        write_observation_pose(tmp_path / "observation.yaml")
    )
    pipeline = build_pipeline(config)
    candidate = GraspGeometryCandidate(
        mode=mode,
        target_camera_xyz=(0.0, 0.0, 0.36),
        # Identity camera-to-base extrinsics make this the exact field failure:
        # the raw measured normal points upward in the robot base frame.
        tool_axis_camera=(0.0, 0.0, 1.0),
        closure_axis_camera=(1.0, 0.0, 0.0),
        confidence=0.95,
        evidence="identity-extrinsics downward tool-Z regression fixture",
    )

    variants = pipeline._task_cd_geometry_orientation_variants(
        candidate,
        np.eye(4),
    )

    assert variants
    for _name, _rpy, diagnostics in variants:
        tool_axis_base = np.asarray(
            diagnostics["tool_axis_base"],
            dtype=float,
        )
        assert float(tool_axis_base[2]) < 0.0


@pytest.mark.parametrize(
    "overrides",
    (
        {"major_axis_camera": None},
        {"minor_axis_camera": None},
        {"object_center_camera_xyz": None},
        {"pose_confidence": 0.49},
        {"major_axis_length_m": 0.0800, "minor_axis_length_m": 0.0762},
    ),
    ids=(
        "missing-major-axis",
        "missing-minor-axis",
        "missing-centre",
        "low-confidence",
        "low-axis-ratio",
    ),
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
        [
            frame("invalid_pca", meat_can_detection(**overrides)),
            frame("empty_after_invalid_pca_relocation_view"),
            *empty_full_scan_frames("empty_after_invalid_pca"),
        ],
    )
    original_precheck = pipeline.motion.precheck_linear_targets
    grasp_route_prechecks = []

    def record_grasp_route(targets, *, step_m):
        if len(targets) == 6:
            grasp_route_prechecks.append(tuple(targets))
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = record_grasp_route

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [
        PickStatus.REJECTED,
    ]
    assert grasp_route_prechecks == []
    assert non_scan_moves(pipeline) == []
    assert pipeline.gripper.events == []
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]


def test_formal_cd_config_enables_only_validated_pca_short_edge_path():
    config = load_config(FORMAL_CD_CONFIG)
    short_edge = config.task_cd.short_edge_grasp

    assert short_edge is not None
    assert short_edge.enabled is True
    assert short_edge.closure_axis_tool_xy == pytest.approx((1.0, 0.0))
    assert short_edge.grasp_point_modes == ("object_center", "handle")
    assert "4-Frenchs_Mustard" in short_edge.handle_grasp_labels
    assert short_edge.handle_offset_fraction == pytest.approx(0.30)
    assert short_edge.min_pose_confidence == pytest.approx(0.50)
    assert short_edge.require_validated_extrinsics is True
    assert short_edge.jaw_tip_z_m is None
    assert short_edge.height_adaptive_enabled is True
    assert short_edge.height_target_insertion_fraction == pytest.approx(0.50)
    assert short_edge.height_min_insertion_fraction == pytest.approx(0.30)
    assert short_edge.height_max_insertion_fraction == pytest.approx(0.75)
    assert short_edge.height_min_support_clearance_m == pytest.approx(0.003)
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
    assert config.motion.cartesian_min_step_m == pytest.approx(0.00025)
    assert config.motion.cartesian_max_waypoints == 600
    assert config.motion.cartesian_max_total_waypoints == 3200


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


def test_short_edge_closure_axis_must_be_finite_and_nonzero():
    config = load_config(FORMAL_CD_CONFIG)
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None

    invalid = replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=replace(
                short_edge,
                closure_axis_tool_xy=(0.0, 0.0),
            ),
        ),
    )
    with pytest.raises(
        ValueError,
        match="closure_axis_tool_xy must be finite and non-zero",
    ):
        invalid.validate()


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


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"grasp_point_modes": ()}, "grasp_point_modes must not be empty"),
        (
            {"grasp_point_modes": ("handle",)},
            "grasp_point_modes must start with object_center",
        ),
        (
            {"grasp_point_modes": ("object_center", "object_center")},
            "grasp_point_modes must not contain duplicates",
        ),
        (
            {"grasp_point_modes": ("object_center", "wrist")},
            "grasp_point_modes contains unsupported",
        ),
        (
            {"handle_grasp_labels": ()},
            "handle_grasp_labels must not be empty",
        ),
        (
            {
                "handle_grasp_labels": (
                    "4-Frenchs_Mustard",
                    "4-Frenchs_Mustard",
                )
            },
            "handle_grasp_labels must not contain duplicates",
        ),
        (
            {"handle_offset_fraction": 0.0},
            "handle_offset_fraction must be finite",
        ),
        (
            {"handle_offset_fraction": 0.5},
            "handle_offset_fraction must be finite",
        ),
        (
            {"handle_offset_fraction": nan},
            "handle_offset_fraction must be finite",
        ),
    ),
)
def test_short_edge_grasp_point_configuration_fails_closed(overrides, message):
    config = load_config(FORMAL_CD_CONFIG)
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    invalid = replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=replace(short_edge, **overrides),
        ),
    )

    with pytest.raises(ValueError, match=message):
        invalid.validate()


def test_short_edge_grasp_point_lists_reject_non_string_entries():
    with pytest.raises(
        ValueError,
        match="grasp_point_modes must contain only strings",
    ):
        TaskCDShortEdgeGraspConfig.from_dict(
            {"grasp_point_modes": ["object_center", 1]}
        )


def test_geometry_policy_short_edge_uses_adaptive_height_and_same_frame_evidence(
    tmp_path,
):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
        grasp_z_offset=0.020,
    )
    target = geometry_detection(
        box_geometry_evidence(),
        label="10-meat_can",
        instance_id="geometry_short_edge_1",
    )
    pipeline = geometry_pipeline_with_frames(
        config,
        [
            frame("geometry_short_edge_visible", target),
            *empty_full_scan_frames("empty_after_geometry_short_edge"),
        ],
    )

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    assert short_edge.jaw_tip_z_m is None
    assert short_edge.height_adaptive_enabled is True
    # Fixture: support Z=0.00 m, measured object top Z=0.04 m.  A 50% target
    # insertion gives jaw-tip Z=0.02 m, not the legacy 0.000103 m.
    expected_jaw_tip_z_m = 0.020
    grasp_pose = dict(non_scan_moves(pipeline))["grasp"]
    assert grasp_pose.xyz[2] == pytest.approx(
        expected_jaw_tip_z_m + TIP_OFFSET_M[2]
    )
    assert grasp_pose.xyz[2] != pytest.approx(0.000103 + TIP_OFFSET_M[2])
    messages = [event.message for result in report.results for event in result.events]
    assert any(
        "adaptive height grasp:" in message
        and "visible_height=0.040000" in message
        and "top_z=0.040000" in message
        and "selected_jaw_tip_z=0.020000" in message
        for message in messages
    )
    assert any(
        "geometry grasp point: mode=vertical_short_edge_center" in message
        and "source=measured_short_edge_footprint_center" in message
        and "geometry_mode=vertical_short_edge" in message
        for message in messages
    )
    assert any(
        "passed the complete read-only geometry precheck" in message
        and "geometry_mode=vertical_short_edge" in message
        for message in messages
    )
    assert pipeline.gripper.events == ["open", "close", "open"]


def test_geometry_policy_vertical_center_uses_measured_width_and_height_z(
    tmp_path,
):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
        allow_vertical_center=True,
        grasp_z_offset=0.020,
    )
    near_square = replace(
        box_geometry_evidence(),
        footprint_major_length_m=0.06,
        footprint_minor_length_m=0.055,
        footprint_axis_ratio=1.0,
        short_edge_confidence=0.20,
    )
    target = geometry_detection(
        near_square,
        label="10-meat_can",
        instance_id="geometry_vertical_center_1",
    )
    pipeline = geometry_pipeline_with_frames(
        config,
        [
            frame("geometry_vertical_center_visible", target),
            *empty_full_scan_frames("empty_after_geometry_vertical_center"),
        ],
    )

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    assert short_edge.height_adaptive_enabled is True
    grasp_pose = dict(non_scan_moves(pipeline))["grasp"]
    assert grasp_pose.xyz[2] == pytest.approx(0.020 + TIP_OFFSET_M[2])
    messages = [event.message for result in report.results for event in result.events]
    assert any(
        "adaptive height grasp:" in message
        and "selected_jaw_tip_z=0.020000" in message
        for message in messages
    )
    assert any(
        "geometry grasp point: mode=vertical_center" in message
        and "source=measured_vertical_center_footprint" in message
        and "geometry_mode=vertical_center" in message
        for message in messages
    )
    assert any(
        "passed the complete read-only geometry precheck" in message
        and "geometry_mode=vertical_center" in message
        for message in messages
    )
    assert pipeline.gripper.events == ["open", "close", "open"]


def test_adaptive_height_target_tracks_different_object_heights(tmp_path):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
    )
    tall_evidence = replace(
        box_geometry_evidence(),
        object_center_camera_xyz=(0.0, 0.0, 0.36),
        visible_height_m=0.08,
        top_center_camera_xyz=(0.0, 0.0, 0.32),
        top_height_m=0.08,
    )
    target = geometry_detection(
        tall_evidence,
        label="10-meat_can",
        instance_id="geometry_adaptive_tall_1",
    )
    pipeline = geometry_pipeline_with_frames(
        config,
        [
            frame("geometry_adaptive_tall_visible", target),
            *empty_full_scan_frames("empty_after_geometry_adaptive_tall"),
        ],
    )

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    grasp_pose = dict(non_scan_moves(pipeline))["grasp"]
    # Support Z=0.00 m, top Z=0.08 m, target insertion fraction 50%.
    assert grasp_pose.xyz[2] == pytest.approx(0.040 + TIP_OFFSET_M[2])
    messages = [event.message for result in report.results for event in result.events]
    assert any(
        "visible_height=0.080000" in message
        and "top_z=0.080000" in message
        and "selected_jaw_tip_z=0.040000" in message
        for message in messages
    )


def test_adaptive_height_rejects_inconsistent_top_surface_before_motion(tmp_path):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
    )
    inconsistent = replace(
        box_geometry_evidence(),
        # Height says the visible top is 0.040 m above support, while the
        # independently measured top point is 0.080 m above it.
        top_center_camera_xyz=(0.0, 0.0, 0.32),
    )
    target = geometry_detection(
        inconsistent,
        label="10-meat_can",
        instance_id="geometry_adaptive_inconsistent_1",
    )
    pipeline = geometry_pipeline_with_frames(
        config,
        [
            frame("geometry_adaptive_inconsistent_visible", target),
            *empty_full_scan_frames("empty_after_adaptive_inconsistent"),
        ],
    )

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.REJECTED]
    assert non_scan_moves(pipeline) == []
    assert pipeline.gripper.events == []
    messages = [event.message for result in report.results for event in result.events]
    assert any(
        "adaptive height top-surface evidence is inconsistent" in message
        for message in messages
    )


def test_unusable_reason_reports_support_plane_quality(tmp_path):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
        allow_vertical_center=True,
    )
    target = geometry_detection(
        replace(box_geometry_evidence(), support_plane_quality=0.20),
        label="10-meat_can",
        instance_id="low_support_quality",
    )
    pipeline = build_pipeline(config)

    assert pipeline._unusable_reason(target) == (
        "support_plane_quality_below_policy"
    )


@pytest.mark.parametrize("failure", ("invalid", "quality"))
def test_support_plane_failure_uses_complete_pca_legacy_fallback(
    tmp_path,
    failure,
):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
        allow_vertical_center=True,
    )
    # This regression covers the legacy PCA fallback itself.  Height-adaptive
    # grasping intentionally disables that fallback, so opt into the legacy
    # fixed-Z profile here while a separate test below covers fail-closed
    # behavior when adaptive height evidence is missing.
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    config = replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=replace(
                short_edge,
                height_adaptive_enabled=False,
                jaw_tip_z_m=0.000103,
            ),
        ),
    )
    evidence = replace(box_geometry_evidence(), support_plane_quality=0.20)
    if failure == "invalid":
        evidence = replace(
            evidence,
            support_plane_valid=False,
            support_plane_quality=0.0,
        )
    target = replace(
        meat_can_detection(instance_id=f"support_{failure}_pca"),
        geometry_evidence=evidence,
    )
    pipeline = build_pipeline(config)
    events = []

    assert pipeline._task_cd_legacy_short_edge_fallback_allowed(target)
    assert pipeline._unusable_reason(target) is None
    state = pipeline.motion.get_state()
    result = pipeline._precheck_task_cd_candidate(
        target,
        state,
        state,
        lambda phase, message: events.append((phase, message)),
    )

    assert result is not None
    plan, route, waypoints = result
    assert plan.grasp_geometry_mode == "legacy_short_edge"
    assert route.segments
    assert waypoints
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []
    assert any(
        "support_plane_fallback=legacy_short_edge" in message
        for _phase, message in events
    )


def test_adaptive_height_support_plane_failure_does_not_use_legacy_fallback(
    tmp_path,
):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
        allow_vertical_center=True,
    )
    target = replace(
        meat_can_detection(instance_id="adaptive_missing_support"),
        geometry_evidence=replace(
            box_geometry_evidence(),
            support_plane_valid=False,
            support_plane_quality=0.0,
        ),
    )
    pipeline = build_pipeline(config)
    events = []

    assert not pipeline._task_cd_legacy_short_edge_fallback_allowed(target)
    assert pipeline._unusable_reason(target) is not None
    state = pipeline.motion.get_state()
    result = pipeline._precheck_task_cd_candidate(
        target,
        state,
        state,
        lambda phase, message: events.append((phase, message)),
    )

    assert result is None
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []
    assert not any(
        "support_plane_fallback=legacy_short_edge" in message
        for _phase, message in events
    )


def test_support_plane_fallback_is_disabled_when_plane_is_required(tmp_path):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
        allow_vertical_center=True,
    )
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    config = replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=replace(
                short_edge,
                pca=replace(short_edge.pca, support_plane_required=True),
            ),
        ),
    )
    target = replace(
        meat_can_detection(instance_id="required_support_plane"),
        geometry_evidence=replace(
            box_geometry_evidence(),
            support_plane_valid=False,
            support_plane_quality=0.0,
        ),
    )
    pipeline = build_pipeline(config)

    assert not pipeline._task_cd_legacy_short_edge_fallback_allowed(target)
    assert pipeline._unusable_reason(target) is not None


def test_support_plane_fallback_rejects_incomplete_pca_before_motion(tmp_path):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
        allow_vertical_center=True,
    )
    target = replace(
        meat_can_detection(instance_id="support_incomplete_pca"),
        geometry_evidence=replace(
            box_geometry_evidence(),
            support_plane_valid=False,
            support_plane_quality=0.0,
        ),
        minor_axis_length_m=None,
    )
    pipeline = build_pipeline(config)
    events = []

    assert not pipeline._task_cd_legacy_short_edge_fallback_allowed(target)
    assert pipeline._unusable_reason(target) is not None
    state = pipeline.motion.get_state()
    result = pipeline._precheck_task_cd_candidate(
        target,
        state,
        state,
        lambda phase, message: events.append((phase, message)),
    )

    assert result is None
    assert pipeline.motion.moves == []
    assert pipeline.gripper.events == []
    assert any("support plane" in message.lower() for _phase, message in events)


def test_non_support_geometry_rejection_does_not_use_pca_fallback(tmp_path):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
    )
    evidence = replace(
        box_geometry_evidence(),
        footprint_axis_ratio=None,
        footprint_major_length_m=None,
    )
    target = replace(
        meat_can_detection(instance_id="non_support_geometry_rejection"),
        geometry_evidence=evidence,
    )
    pipeline = build_pipeline(config)

    assert not pipeline._task_cd_legacy_short_edge_fallback_allowed(target)
    assert pipeline._unusable_reason(target) is not None


def test_task_cd_candidate_order_prefers_measured_geometry_over_yolo_score(
    tmp_path,
):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
    )
    high_evidence = replace(
        box_geometry_evidence(),
        support_plane_quality=0.70,
        short_edge_confidence=0.80,
        object_center_camera_xyz=(0.0, 0.0, 0.38),
        footprint_center_camera_xyz=(0.0, 0.0, 0.40),
        top_center_camera_xyz=(0.0, 0.0, 0.36),
    )
    strong_evidence = replace(
        box_geometry_evidence(),
        object_center_camera_xyz=(0.12, 0.0, 0.38),
        footprint_center_camera_xyz=(0.12, 0.0, 0.40),
        top_center_camera_xyz=(0.12, 0.0, 0.36),
    )
    high_yolo_confidence = replace(
        geometry_detection(
            high_evidence,
            label="10-meat_can",
            instance_id="high_yolo_confidence",
        ),
        confidence=0.99,
        bbox=(100.0, 100.0, 200.0, 180.0),
        base_xyz=(0.10, 0.0, 0.02),
    )
    strong_measured_geometry = replace(
        geometry_detection(
            strong_evidence,
            label="10-meat_can",
            instance_id="strong_measured_geometry",
        ),
        confidence=0.40,
        bbox=(250.0, 100.0, 350.0, 180.0),
        base_xyz=(0.22, 0.0, 0.02),
    )
    pipeline = geometry_pipeline_with_frames(
        config,
        [frame("geometry_quality_order", high_yolo_confidence, strong_measured_geometry)],
    )

    observation = pipeline._observe_task_cd_candidates(lambda *_args: None)

    assert [
        candidate.detection.instance_id for candidate in observation.candidates
    ] == [
        "strong_measured_geometry",
        "high_yolo_confidence",
    ]


@pytest.mark.parametrize("task_mode", ("C", "D"))
def test_geometry_pregrasp_endpoint_rejects_before_full_route_precheck(
    tmp_path,
    task_mode,
):
    """The geometry path must share the same fast endpoint rejection."""

    config = apply_mode(
        geometry_config(
            write_observation_pose(tmp_path / "observation.yaml"),
            allow_rim=True,
        ),
        task_mode,
    )
    target = geometry_detection(
        rim_geometry_evidence(),
        label="9-Mug",
        instance_id=f"near_singular_geometry_{task_mode.lower()}",
    )
    pipeline = geometry_pipeline_with_frames(
        config,
        [
            frame("near_singular_geometry_visible", target),
            frame("fixed_view_geometry_target_excluded", target),
            frame("relocation_view_geometry_target_excluded", target),
            frame("fixed_retry_geometry_target_excluded", target),
        ],
    )
    original_precheck = pipeline.motion.precheck_linear_targets
    endpoint_checks = []
    grasp_route_prechecks = []

    def reject_pregrasp_endpoint(pose, *, label):
        endpoint_checks.append((pose, label))
        return config.motion.cartesian_min_jacobian_singular_value / 2.0

    def record_route_precheck(targets, *, step_m):
        if len(targets) in (6, 7):
            grasp_route_prechecks.append(tuple(targets))
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_pose_condition = reject_pregrasp_endpoint
    pipeline.motion.precheck_linear_targets = record_route_precheck

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.REJECTED]
    assert report.results[0].target is not None
    assert endpoint_checks
    assert grasp_route_prechecks == []
    assert non_scan_moves(pipeline) == []
    assert pipeline.gripper.events == []
    messages = [event.message for result in report.results for event in result.events]
    assert any(
        "pregrasp endpoint minimum Jacobian sigma" in message
        and "is below required" in message
        and "geometry point=vertical_rim" in message
        for message in messages
    )


def test_geometry_handle_fallback_offsets_along_major_not_closure_axis(
    tmp_path,
):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
    )
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    short_edge = replace(
        short_edge,
        grasp_point_modes=("object_center", "handle"),
        handle_grasp_labels=("4-Frenchs_Mustard",),
        handle_offset_fraction=0.30,
    )
    config = replace(
        config,
        task_cd=replace(config.task_cd, short_edge_grasp=short_edge),
    )
    config.validate()
    target = geometry_detection(
        box_geometry_evidence(),
        label="4-Frenchs_Mustard",
        instance_id="geometry_handle_1",
    )
    decision = select_grasp_geometry(
        target.geometry_evidence,
        short_edge.geometry_policy,
        label=target.label,
    )
    assert [candidate.mode for candidate in decision.candidates] == [
        "vertical_short_edge"
    ]
    candidate = decision.candidates[0]
    assert candidate.closure_axis_camera == pytest.approx((0.0, 1.0, 0.0))
    pipeline = build_pipeline(config)
    base_camera = np.eye(4)
    base_camera[:3, :3] = np.diag((1.0, -1.0, -1.0))
    base_camera[:3, 3] = (0.10, 0.0, 0.40)

    points = pipeline._task_cd_geometry_grasp_points(
        target,
        candidate,
        base_camera,
    )

    assert [name for name, _xyz, _info in points] == [
        "vertical_short_edge_center",
        "vertical_short_edge_handle_positive_major_axis",
        "vertical_short_edge_handle_negative_major_axis",
    ]
    offset_m = 0.30 * 0.14
    assert points[1][1] == pytest.approx((0.10 + offset_m, 0.0, 0.0))
    assert points[2][1] == pytest.approx((0.10 - offset_m, 0.0, 0.0))
    assert points[1][2]["major_axis_unit_xy"] == pytest.approx((1.0, 0.0))


def test_geometry_policy_round_rim_ignores_legacy_pca_and_preserves_measured_z(
    tmp_path,
):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=True,
        grasp_z_offset=0.030,
    )
    target = geometry_detection(
        rim_geometry_evidence(),
        label="9-Mug",
        instance_id="geometry_rim_1",
    )
    pipeline = geometry_pipeline_with_frames(
        config,
        [
            frame("geometry_rim_visible", target),
            *empty_full_scan_frames("empty_after_geometry_rim"),
        ],
    )

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.SUCCESS]
    assert [name for name, _pose in non_scan_moves(pipeline)] == [
        "pregrasp",
        "grasp",
        "lift",
        "transfer",
        "release",
        "retreat",
    ]
    measured_jaw_tip_z_m = 0.40 - 0.32568
    grasp_pose = dict(non_scan_moves(pipeline))["grasp"]
    assert grasp_pose.xyz[2] == pytest.approx(
        measured_jaw_tip_z_m + TIP_OFFSET_M[2]
    )
    assert grasp_pose.xyz[2] != pytest.approx(
        measured_jaw_tip_z_m + TIP_OFFSET_M[2] + 0.030
    )
    messages = [event.message for result in report.results for event in result.events]
    assert any(
        "geometry grasp point: mode=vertical_rim" in message
        and "source=measured_open_rim" in message
        and "geometry_mode=vertical_rim" in message
        for message in messages
    )
    assert any(
        "passed the complete read-only geometry precheck" in message
        and "geometry_mode=vertical_rim" in message
        for message in messages
    )
    assert pipeline.gripper.events == ["open", "close", "open"]


def test_geometry_policy_closed_cylinder_sidewall_rejects_before_any_command(
    tmp_path,
):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=True,
        allow_horizontal_sidewall=True,
    )
    target = geometry_detection(
        closed_cylinder_geometry_evidence(),
        label="10-meat_can",
        instance_id="geometry_sidewall_1",
    )
    pipeline = geometry_pipeline_with_frames(
        config,
        [
            frame("geometry_sidewall_visible", target),
            frame("fixed_view_sidewall_target_excluded", target),
            frame("relocation_view_sidewall_target_excluded", target),
            frame("fixed_retry_sidewall_target_excluded", target),
        ],
    )
    route_precheck_lengths = []
    original_precheck = pipeline.motion.precheck_linear_targets

    def record_scan_prechecks(targets, *, step_m):
        route_precheck_lengths.append(len(targets))
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = record_scan_prechecks

    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [PickStatus.REJECTED]
    assert report.successes == 0
    assert report.results[0].target.instance_id == "geometry_sidewall_1"
    # The first planning pass rejects the sidewall before any grasp route.
    # Only the bounded relocation route needed to check for another object is
    # allowed on the next observation; no grasp Cartesian motion or gripper
    # command may be issued.
    assert all(length < 6 for length in route_precheck_lengths)
    assert non_scan_moves(pipeline) == []
    assert pipeline.gripper.events == []
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]
    assert any(
        "horizontal_sidewall candidate retained as diagnostic only" in event.message
        and "rejected before motion" in event.message
        for result in report.results
        for event in result.events
    )


def test_geometry_policy_without_same_frame_evidence_is_not_usable(tmp_path):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=True,
    )
    target = meat_can_detection(instance_id="legacy_pca_without_geometry")
    pipeline = geometry_pipeline_with_frames(
        config,
        [
            frame("legacy_pca_without_geometry_visible", target),
            frame("empty_after_legacy_pca_relocation_view"),
            *empty_full_scan_frames("empty_after_legacy_pca"),
        ],
    )
    original_precheck = pipeline.motion.precheck_linear_targets
    grasp_route_prechecks = []

    def record_grasp_route(targets, *, step_m):
        if len(targets) == 6:
            grasp_route_prechecks.append(tuple(targets))
        return original_precheck(targets, step_m=step_m)

    pipeline.motion.precheck_linear_targets = record_grasp_route

    assert pipeline._is_usable(target) is False
    report = pipeline.run(acceptance_max_successes=1)

    assert [result.status for result in report.results] == [
        PickStatus.REJECTED,
    ]
    assert grasp_route_prechecks == []
    assert non_scan_moves(pipeline) == []
    assert pipeline.gripper.events == []
    assert [name for name, _joints in pipeline.motion.joint_moves] == [
        "task_cd_observation_entry"
    ]


def test_live_style_geometry_evidence_survives_stabilizer_and_is_usable(
    tmp_path,
    monkeypatch,
):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
    )
    config = replace(
        config,
        vision=replace(config.vision, stabilizer_confirm_hits=1),
    )
    config.validate()
    depth, intrinsics, item = live_style_geometry_scene(config)

    raw = make_live_style_detection(
        config,
        depth,
        intrinsics,
        item,
        monkeypatch,
    )

    assert raw.geometry_evidence is not None
    assert raw.attributes["short_edge_geometry_evidence_valid"] is True
    assert raw.attributes["short_edge_footprint_axis_ratio"] > 1.08
    stabilizer = DetectionStabilizer(config.vision)
    stable = stabilizer.update((raw,), depth.shape[1], depth.shape[0])

    assert len(stable) == 1
    assert stable[0].attributes["held"] is False
    assert stable[0].geometry_evidence is raw.geometry_evidence
    pipeline = geometry_pipeline_with_frames(config, [])
    assert pipeline._is_usable(stable[0]) is True


def test_live_style_pose_failure_clears_old_geometry_evidence_and_fails_closed(
    tmp_path,
    monkeypatch,
):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
    )
    config = replace(
        config,
        vision=replace(config.vision, stabilizer_confirm_hits=1),
    )
    config.validate()
    depth, intrinsics, item = live_style_geometry_scene(config)
    fresh = make_live_style_detection(
        config,
        depth,
        intrinsics,
        item,
        monkeypatch,
    )
    assert fresh.geometry_evidence is not None

    invalid_depth = np.full_like(depth, np.nan)
    rejected = make_live_style_detection(
        config,
        invalid_depth,
        intrinsics,
        item,
        monkeypatch,
    )
    assert rejected.geometry_evidence is None
    assert "short_edge_pose_error" in rejected.attributes

    stabilizer = DetectionStabilizer(config.vision)
    first = stabilizer.update((fresh,), depth.shape[1], depth.shape[0])
    current = stabilizer.update(
        (rejected,),
        depth.shape[1],
        depth.shape[0],
    )

    assert first[0].geometry_evidence is not None
    assert len(current) == 1
    assert current[0].attributes["held"] is False
    assert current[0].geometry_evidence is None
    pipeline = geometry_pipeline_with_frames(config, [])
    assert pipeline._is_usable(current[0]) is False


def test_planner_preserve_measured_jaw_tip_z_requires_override_and_no_fixed_z(
    tmp_path,
):
    config = geometry_config(
        write_observation_pose(tmp_path / "observation.yaml"),
        allow_rim=False,
        grasp_z_offset=0.030,
    )
    planner = PosePlanner(config)
    state = RobotState(
        ready=True,
        pose=Pose((0.0, 0.0, 0.30), GRASP_RPY),
        joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        message="mock-ready",
    )
    target = meat_can_detection()
    measured_jaw_tip = (0.10, 0.0, 0.07432)

    with pytest.raises(
        ValueError,
        match="preserve_grasp_target_z requires grasp_target_is_jaw_tip=True",
    ):
        planner.plan(
            target,
            state,
            grasp_xyz_override_m=measured_jaw_tip,
            preserve_grasp_target_z=True,
            grasp_rpy=GRASP_RPY,
        )

    with pytest.raises(
        ValueError,
        match="preserve_grasp_target_z requires grasp_xyz_override_m",
    ):
        planner.plan(
            target,
            state,
            grasp_target_is_jaw_tip=True,
            preserve_grasp_target_z=True,
            grasp_rpy=GRASP_RPY,
        )

    with pytest.raises(
        ValueError,
        match="preserve_grasp_target_z and jaw_tip_z_m are mutually exclusive",
    ):
        planner.plan(
            target,
            state,
            grasp_xyz_override_m=measured_jaw_tip,
            grasp_target_is_jaw_tip=True,
            preserve_grasp_target_z=True,
            jaw_tip_z_m=0.000103,
            grasp_rpy=GRASP_RPY,
        )

    plan = planner.plan(
        target,
        state,
        grasp_xyz_override_m=measured_jaw_tip,
        grasp_target_is_jaw_tip=True,
        preserve_grasp_target_z=True,
        grasp_rpy=GRASP_RPY,
    )
    assert plan.grasp_pose.xyz[2] == pytest.approx(
        measured_jaw_tip[2] + TIP_OFFSET_M[2]
    )
    assert plan.grasp_pose.xyz[2] != pytest.approx(
        measured_jaw_tip[2] + TIP_OFFSET_M[2] + 0.030
    )
