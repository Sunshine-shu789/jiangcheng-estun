from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from track1_grasp.config import (
    GraspGeometryPolicyConfig,
    apply_mode,
    load_config,
    load_config_for_read_only_vision,
)
from track1_grasp.domain import CameraIntrinsics, Detection, Frame, TaskMode
from track1_grasp.grasp_geometry import GraspGeometryEvidence, RimGeometry
from track1_grasp.tools import diagnose_task_c_geometry as diagnosis


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = Path(__file__).resolve().parents[3]
FORMAL_C_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_c_v11_official_20261102.yaml"
)
LAUNCHER = FORMAL_ROOT / "bin" / "diagnose_task_c_geometry"
INTRINSICS = CameraIntrinsics(
    width=640,
    height=480,
    fx=600.0,
    fy=600.0,
    cx=320.0,
    cy=240.0,
)


def _task_c_config():
    config = load_config(FORMAL_C_CONFIG)
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    # This diagnostic tool intentionally exercises the legacy short-edge path.
    # The official C/D profile now enables the same-frame geometry policy, so
    # isolate the legacy path here; the policy path has separate acceptance and
    # negative tests below.
    return replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=replace(
                short_edge,
                geometry_policy=GraspGeometryPolicyConfig(enabled=False),
            ),
        ),
    )


def _detection(
    track_id: str,
    *,
    confidence: float = 0.90,
    bbox: tuple[float, float, float, float] = (250.0, 180.0, 390.0, 300.0),
) -> Detection:
    return Detection(
        label="10-meat_can",
        confidence=confidence,
        bbox=bbox,
        depth_m=0.40,
        camera_xyz=(0.010, -0.010, 0.40),
        object_center_camera_xyz=(0.0, 0.0, 0.40),
        major_axis_camera=(1.0, 0.0, 0.0),
        minor_axis_camera=(0.0, 1.0, 0.0),
        major_axis_length_m=0.080,
        minor_axis_length_m=0.045,
        pose_confidence=0.80,
        graspability=0.92,
        instance_id=track_id,
        attributes={
            "track_id": track_id,
            "stable_confidence": confidence,
            "raw_confidence": max(0.0, confidence - 0.05),
            "hits": 5,
            "missed": 0,
            "held": False,
        },
    )


def _frame(index: int, detections: tuple[Detection, ...]) -> Frame:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 1] = 30
    depth = np.full((480, 640), 0.40, dtype=np.float32)
    depth[:3, :] = 0.0
    depth[:, :3] = np.nan
    return Frame(
        frame_id=f"fake-{index}",
        detections=detections,
        stamp_s=float(index),
        image=image,
        camera_intrinsics=INTRINSICS,
        depth_m=depth,
    )


def _rim_evidence() -> GraspGeometryEvidence:
    return GraspGeometryEvidence(
        object_center_camera_xyz=(0.0, 0.0, 0.40),
        support_plane_valid=True,
        support_plane_quality=0.95,
        support_plane_center_camera_xyz=(0.0, 0.0, 0.43),
        support_plane_normal_camera=(0.0, 0.0, 1.0),
        up_camera=(0.0, 0.0, -1.0),
        visible_height_m=0.055,
        footprint_center_camera_xyz=(0.0, 0.0, 0.38),
        footprint_major_length_m=0.070,
        footprint_minor_length_m=0.070,
        footprint_axis_ratio=1.0,
        footprint_major_axis_camera=(1.0, 0.0, 0.0),
        footprint_minor_axis_camera=(0.0, 1.0, 0.0),
        top_center_camera_xyz=(0.0, 0.0, 0.38),
        top_height_m=0.055,
        short_edge_confidence=0.30,
        rim=RimGeometry(
            center_camera_xyz=(0.0, 0.0, 0.38),
            center_xy_m=(0.0, 0.0),
            radius_m=0.030,
            depression_m=0.012,
            circularity=0.96,
            coverage=0.82,
            fit_error_m=0.0012,
            inlier_ratio=0.94,
            valid_points=520,
            confidence=0.88,
            rim_closure_width_m=0.006,
            rim_closure_valid_sectors=14,
            rim_closure_mad_m=0.0002,
        ),
        sidewall=None,
        valid_points=1200,
    )


def _rim_detection(*, closure_width_m: float | None = 0.006) -> Detection:
    evidence = _rim_evidence()
    assert evidence.rim is not None
    measured = closure_width_m is not None
    evidence = replace(
        evidence,
        rim=replace(
            evidence.rim,
            rim_closure_width_m=closure_width_m,
            rim_closure_valid_sectors=14 if measured else None,
            rim_closure_mad_m=0.0002 if measured else None,
        ),
    )
    return replace(
        _detection("track-open-container"),
        geometry_evidence=evidence,
    )


def _read_only_official_c_config():
    return load_config_for_read_only_vision(
        FORMAL_C_CONFIG,
        task_mode=TaskMode.C,
    )


class _FakeSource:
    def __init__(self, frames):
        self.frames = tuple(frames)
        self.index = 0
        self.closed = False

    def capture(self):
        if self.index >= len(self.frames):
            raise StopIteration("scripted vision exhausted")
        frame = self.frames[self.index]
        self.index += 1
        return frame

    def close(self):
        self.closed = True


class _FailingSource:
    def __init__(self):
        self.closed = False

    def capture(self):
        raise RuntimeError("synthetic camera failure")

    def close(self):
        self.closed = True


def _fake_builder(source, calls=None):
    def build(vision_config, *, log=None, **kwargs):
        assert vision_config.backend == "realsense_yolo"
        assert vision_config.show is False
        assert vision_config.save_annotated is False
        assert log is not None
        assert kwargs.get("object_pose_config") is not None
        if calls is not None:
            calls.append((vision_config, kwargs))
        return source

    return build


def test_capture_aggregates_track_reids_and_same_frame_duplicates(tmp_path):
    source = _FakeSource(
        [
            _frame(
                index,
                (
                    _detection(f"track-{100 + index}", confidence=0.90),
                    _detection(f"duplicate-{index}", confidence=0.70),
                ),
            )
            for index in range(1, 6)
        ]
    )
    output_dir = tmp_path / "diagnosis"
    build_calls = []

    payload = diagnosis.run_diagnosis(
        _task_c_config(),
        output_dir=output_dir,
        frame_count=5,
        require_known=True,
        require_stable_geometry=True,
        min_cluster_frames=5,
        config_path=FORMAL_C_CONFIG,
        build_vision_fn=_fake_builder(source, build_calls),
    )

    assert payload["status"] == "passed"
    assert payload["scope"] == "vision_only_no_robot_info_no_motion_no_gripper"
    assert payload["safety"] == {
        "robot_info_required": False,
        "robot_bridge_started": False,
        "pick_pipeline_constructed": False,
        "pose_planner_constructed": False,
        "motion_backend_constructed": False,
        "gripper_backend_constructed": False,
        "motion_commands_published": False,
        "gripper_commands_sent": False,
        "execution_authorized": False,
        "motion_backend": "mock",
        "motion_dry_run": True,
        "gripper_backend": "mock",
    }
    assert payload["config"]["task_mode"] == "C"
    assert payload["config"]["show_window"] is False
    assert payload["config"]["short_edge_grasp"]["enabled"] is True
    assert payload["config"]["short_edge_grasp"]["geometry_policy"]["enabled"] is False
    assert payload["aggregate"] == {
        **payload["aggregate"],
        "requested_frames": 5,
        "captured_frames": 5,
        "raw_observations": 10,
        "duplicate_observations_removed": 5,
        "known_observations_deduplicated": 5,
        "geometry_accepted_observations_deduplicated": 5,
        "physical_clusters": 1,
        "stable_clusters": 1,
        "stable_geometry_clusters": 1,
        "unique_labels": ["10-meat_can"],
    }
    cluster = payload["clusters"][0]
    assert cluster["status"] == "stable_geometry"
    assert cluster["frame_hits"] == 5
    assert cluster["observation_count"] == 5
    assert cluster["track_ids"] == [
        "track-101",
        "track-102",
        "track-103",
        "track-104",
        "track-105",
    ]
    assert cluster["track_id_count"] == 5
    assert cluster["geometry_accepted_fraction"] == pytest.approx(1.0)
    assert cluster["estimated_width_median_m"] == pytest.approx(0.045)
    assert cluster["xy_correction_horizontal_norm_median_m"] > 0.0
    assert len(build_calls) == 1
    assert build_calls[0][1]["object_pose_config"] == _task_c_config().task_cd.short_edge_grasp.pca
    assert source.closed is True

    summary_path = output_dir / "summary.json"
    assert summary_path.is_file()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "passed"
    for frame in payload["frames"]:
        assert frame["raw_detection_count"] == 2
        assert frame["deduplicated_detection_count"] == 1
        assert frame["duplicate_count"] == 1
        assert frame["geometry_accepted_count"] == 2
        for key in ("rgb", "depth_color", "depth_m_npz", "detections_overlay"):
            assert (output_dir / frame["images"][key]).is_file(), key
    assert payload["representative_frame"] is not None
    for filename in payload["representative_frame"]["images"].values():
        assert (output_dir / filename).is_file()
    npz = np.load(output_dir / payload["frames"][0]["images"]["depth_m_npz"])
    assert npz["depth_m"].shape == (480, 640)


def test_require_known_and_stable_geometry_fail_but_write_report(tmp_path):
    source = _FakeSource([_frame(1, ()), _frame(2, ())])
    output_dir = tmp_path / "requirements-failed"

    payload = diagnosis.run_diagnosis(
        _task_c_config(),
        output_dir=output_dir,
        frame_count=2,
        require_known=True,
        require_stable_geometry=True,
        min_cluster_frames=3,
        build_vision_fn=_fake_builder(source),
    )

    assert payload["status"] == "failed"
    assert payload["aggregate"]["known_observations_deduplicated"] == 0
    assert payload["aggregate"]["stable_geometry_clusters"] == 0
    assert any("--require-known" in item for item in payload["diagnosis"]["failures"])
    assert any(
        "--require-stable-geometry" in item
        for item in payload["diagnosis"]["failures"]
    )
    assert any("No known-class positive" in note for note in payload["notes"])
    assert (output_dir / "summary.json").is_file()


def test_camera_failure_still_writes_failed_summary(tmp_path):
    source = _FailingSource()
    output_dir = tmp_path / "camera-failure"

    payload = diagnosis.run_diagnosis(
        _task_c_config(),
        output_dir=output_dir,
        frame_count=2,
        build_vision_fn=_fake_builder(source),
    )

    assert payload["status"] == "failed"
    assert payload["aggregate"]["captured_frames"] == 0
    assert "synthetic camera failure" in payload["capture"]["capture_error"]
    assert source.closed is True
    assert (output_dir / "summary.json").is_file()


def test_task_mode_gate_rejects_non_c_without_building_vision(tmp_path):
    config = apply_mode(_task_c_config(), TaskMode.D)
    output_dir = tmp_path / "wrong-mode"
    called = False

    def should_not_build(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("vision must not be constructed for wrong task")

    with pytest.raises(diagnosis.DiagnosisError, match="task.mode=C"):
        diagnosis.run_diagnosis(
            config,
            output_dir=output_dir,
            frame_count=1,
            build_vision_fn=should_not_build,
        )
    assert called is False
    assert not output_dir.exists()


@pytest.mark.raw_formal_cd_config
def test_geometry_policy_accepts_four_measured_radial_rim_candidates(tmp_path):
    config = _read_only_official_c_config()
    source = _FakeSource([_frame(1, (_rim_detection(),))])
    output_dir = tmp_path / "geometry-policy-accepted"

    payload = diagnosis.run_diagnosis(
        config,
        output_dir=output_dir,
        frame_count=1,
        require_geometry_candidate=True,
        require_rim_evidence=True,
        require_rim_closure_width=True,
        require_rim_candidate=True,
        build_vision_fn=_fake_builder(source),
    )

    assert payload["status"] == "passed"
    assert payload["diagnosis"]["failures"] == []
    aggregate = payload["aggregate"]
    assert aggregate["geometry_policy_enabled"] is True
    assert aggregate["geometry_policy_evaluated_observations"] == 1
    assert aggregate["geometry_policy_accepted_observations"] == 1
    assert aggregate["geometry_policy_rejected_observations"] == 0
    assert aggregate["geometry_policy_candidate_mode_observations"] == {
        "vertical_rim": 4
    }
    assert aggregate["rim_evidence_observations_deduplicated"] == 1
    assert aggregate["rim_closure_width_observations_deduplicated"] == 1
    assert aggregate["rim_closure_width_median_m"] == pytest.approx(0.006)
    assert aggregate["rim_closure_valid_sector_counts"] == {"14": 1}
    assert aggregate["vertical_rim_candidate_total"] == 4
    assert aggregate["vertical_rim_candidate_count_distribution"] == {"4": 1}
    assert aggregate["complete_vertical_rim_candidate_observations"] == 1
    assert aggregate["vertical_rim_candidate_close_width_min_m"] == pytest.approx(
        0.006
    )
    assert aggregate["vertical_rim_candidate_close_width_median_m"] == pytest.approx(
        0.006
    )
    assert aggregate["vertical_rim_candidate_close_width_max_m"] == pytest.approx(
        0.006
    )
    assert aggregate["vertical_rim_radial_alignment_min"] == pytest.approx(1.0)
    assert aggregate["vertical_rim_radial_alignment_median"] == pytest.approx(1.0)

    record = payload["frames"][0]["detections"][0]
    assert record["effective_geometry_accepted"] is True
    assert record["effective_geometry_reason"] == "accepted"
    policy = record["geometry_policy"]
    assert policy["accepted"] is True
    assert policy["reason_category"] == "accepted"
    rim_candidates = [
        candidate
        for candidate in policy["candidates"]
        if candidate["mode"] == "vertical_rim"
    ]
    assert len(rim_candidates) == 4
    assert all(
        candidate["close_width_m"] == pytest.approx(0.006)
        for candidate in rim_candidates
    )
    assert all(
        candidate["close_width_source"] == "same_frame_rim_wall_thickness"
        for candidate in rim_candidates
    )
    assert all(
        candidate["rim_radial_alignment"] == pytest.approx(1.0)
        for candidate in rim_candidates
    )
    assert source.closed is True


@pytest.mark.raw_formal_cd_config
def test_rim_candidates_never_fabricate_missing_closure_width(tmp_path):
    config = _read_only_official_c_config()
    source = _FakeSource(
        [_frame(1, (_rim_detection(closure_width_m=None),))]
    )
    output_dir = tmp_path / "geometry-policy-no-rim-width"

    payload = diagnosis.run_diagnosis(
        config,
        output_dir=output_dir,
        frame_count=1,
        require_geometry_candidate=True,
        require_rim_evidence=True,
        require_rim_closure_width=True,
        require_rim_candidate=True,
        build_vision_fn=_fake_builder(source),
    )

    assert payload["status"] == "failed"
    failures = payload["diagnosis"]["failures"]
    assert any("--require-rim-closure-width" in item for item in failures)
    assert any("--require-rim-candidate" in item for item in failures)
    assert not any("--require-rim-evidence" in item for item in failures)
    aggregate = payload["aggregate"]
    assert aggregate["geometry_policy_accepted_observations"] == 1
    assert aggregate["rim_evidence_observations_deduplicated"] == 1
    assert aggregate["rim_closure_width_observations_deduplicated"] == 0
    assert aggregate["vertical_rim_candidate_total"] == 4
    assert aggregate["vertical_rim_candidate_count_distribution"] == {"4": 1}
    assert aggregate["complete_vertical_rim_candidate_observations"] == 0
    assert aggregate["vertical_rim_candidate_close_width_median_m"] is None

    record = payload["frames"][0]["detections"][0]
    assert record["effective_geometry_accepted"] is True
    rim_candidates = [
        candidate
        for candidate in record["geometry_policy"]["candidates"]
        if candidate["mode"] == "vertical_rim"
    ]
    assert len(rim_candidates) == 4
    assert all(candidate["close_width_m"] is None for candidate in rim_candidates)
    assert all(candidate["close_width_source"] is None for candidate in rim_candidates)
    assert source.closed is True


@pytest.mark.raw_formal_cd_config
def test_read_only_formal_c_config_preserves_geometry_policy_and_forces_mock():
    config = _read_only_official_c_config()

    assert config.task.mode is TaskMode.C
    assert config.motion.backend == "mock"
    assert config.motion.dry_run is True
    assert config.gripper.backend == "mock"
    assert config.task_cd.box_clearance.required is False
    assert config.task_cd.short_edge_grasp is not None
    assert config.task_cd.short_edge_grasp.geometry_policy.enabled is True


@pytest.mark.raw_formal_cd_config
def test_main_loads_formal_c_config_through_read_only_loader(tmp_path, monkeypatch):
    captured = {}

    def fake_run(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {
            "status": "passed",
            "aggregate": {
                "captured_frames": 1,
                "known_observations_deduplicated": 0,
                "geometry_accepted_observations_deduplicated": 0,
                "geometry_policy_accepted_observations": 0,
                "rim_evidence_observations_deduplicated": 0,
                "rim_closure_width_observations_deduplicated": 0,
                "vertical_rim_candidate_total": 0,
                "complete_vertical_rim_candidate_observations": 0,
                "physical_clusters": 0,
                "stable_clusters": 0,
                "stable_geometry_clusters": 0,
                "mean_frame_depth_in_range_ratio": None,
            },
            "diagnosis": {"failures": []},
        }

    monkeypatch.setattr(diagnosis, "run_diagnosis", fake_run)
    result = diagnosis.main(
        [
            "--config",
            str(FORMAL_C_CONFIG),
            "--frames",
            "1",
            "--output-dir",
            str(tmp_path / "main-loading"),
        ]
    )

    assert result == 0
    config = captured["config"]
    assert config.task.mode is TaskMode.C
    assert config.motion.backend == "mock"
    assert config.motion.dry_run is True
    assert config.gripper.backend == "mock"
    assert config.task_cd.box_clearance.required is False
    assert config.task_cd.short_edge_grasp is not None
    assert config.task_cd.short_edge_grasp.geometry_policy.enabled is True
    assert captured["kwargs"]["config_path"] == FORMAL_C_CONFIG.resolve()


def test_short_edge_heading_uses_live_mustard_minor_axis_regression():
    config = _task_c_config()
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    major = np.asarray((0.422413, -0.905578, 0.0), dtype=float)
    minor = np.asarray((0.905578, 0.422413, 0.0), dtype=float)
    major /= np.linalg.norm(major)
    minor /= np.linalg.norm(minor)
    detection = replace(
        _detection("track-mustard"),
        label="4-Frenchs_Mustard",
        major_axis_camera=tuple(major),
        minor_axis_camera=tuple(minor),
        pose_confidence=0.583622,
    )

    heading = diagnosis._short_edge_heading(
        detection,
        np.eye(4),
        short_edge,
    )

    closure = np.asarray(heading["closure_axis_base"], dtype=float)
    assert heading["yaw_rad"] == pytest.approx(
        np.arctan2(float(minor[1]), float(minor[0]))
    )
    assert abs(float(np.dot(closure, major))) < 1e-9
    assert abs(float(np.dot(closure, minor))) == pytest.approx(1.0)
    assert heading["major_axis_base_xy_unit"] == pytest.approx(major[:2])
    assert heading["minor_axis_base_xy_unit"] == pytest.approx(minor[:2])
    assert heading["alignment"] == pytest.approx(1.0)
    assert heading["alignment_error_deg"] == pytest.approx(0.0, abs=1e-5)


def test_launcher_is_executable_and_has_no_task_or_motion_entrypoint():
    assert LAUNCHER.is_file()
    assert os.access(LAUNCHER, os.X_OK)
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "track1_grasp.tools.diagnose_task_c_geometry" in source
    assert "track1_task_runner" not in source
    assert "run_task_c" not in source
    assert "RobotMove" not in source
    assert "bin/run_task_" not in source
    assert "connect" not in source.lower()
