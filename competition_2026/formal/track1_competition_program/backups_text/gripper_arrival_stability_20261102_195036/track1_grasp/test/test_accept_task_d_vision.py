from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from track1_grasp.config import GraspGeometryPolicyConfig, apply_mode, load_config
from track1_grasp.domain import CameraIntrinsics, Detection, Frame, TaskMode
from track1_grasp.grasp_geometry import GraspGeometryEvidence
from track1_grasp.tools import accept_task_d_vision as acceptance


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = Path(__file__).resolve().parents[3]
FORMAL_CD_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_cd_40class_v11_official.yaml"
)
INTRINSICS = CameraIntrinsics(
    width=320,
    height=240,
    fx=300.0,
    fy=300.0,
    cx=160.0,
    cy=120.0,
)

GEOMETRY_EVIDENCE = GraspGeometryEvidence(
    object_center_camera_xyz=(-0.04, 0.01, 0.37),
    support_plane_valid=False,
    support_plane_quality=0.0,
    support_plane_center_camera_xyz=None,
    support_plane_normal_camera=None,
    up_camera=None,
    visible_height_m=None,
    footprint_center_camera_xyz=None,
    footprint_major_length_m=None,
    footprint_minor_length_m=None,
    footprint_axis_ratio=None,
    footprint_major_axis_camera=None,
    footprint_minor_axis_camera=None,
    top_center_camera_xyz=None,
    top_height_m=None,
    short_edge_confidence=0.0,
    rim=None,
    sidewall=None,
    valid_points=100,
)

ACCEPTABLE_GEOMETRY_EVIDENCE = replace(
    GEOMETRY_EVIDENCE,
    support_plane_valid=True,
    support_plane_quality=0.98,
    support_plane_center_camera_xyz=(-0.04, 0.01, 0.42),
    support_plane_normal_camera=(0.0, 0.0, 1.0),
    up_camera=(0.0, 0.0, -1.0),
    visible_height_m=0.05,
    footprint_center_camera_xyz=(-0.04, 0.01, 0.42),
    footprint_major_length_m=0.09,
    footprint_minor_length_m=0.05,
    footprint_axis_ratio=1.8,
    footprint_major_axis_camera=(1.0, 0.0, 0.0),
    footprint_minor_axis_camera=(0.0, 1.0, 0.0),
    top_center_camera_xyz=(-0.04, 0.01, 0.37),
    top_height_m=0.05,
    short_edge_confidence=0.95,
    valid_points=1000,
)


def _task_d_config():
    # The formal C/D file is now an explicitly authorized live configuration.
    # Read-only vision acceptance still must prove that it runs with no actuator
    # backend, so give these tests an explicit non-actuating override instead of
    # weakening the acceptance tool's safety gate.
    config = apply_mode(load_config(FORMAL_CD_CONFIG), TaskMode.D)
    return replace(
        config,
        motion=replace(config.motion, dry_run=True),
        gripper=replace(config.gripper, backend="mock"),
    )


def _task_d_geometry_policy_config():
    config = _task_d_config()
    short_edge = config.task_cd.short_edge_grasp
    assert short_edge is not None
    return replace(
        config,
        task_cd=replace(
            config.task_cd,
            short_edge_grasp=replace(
                short_edge,
                geometry_policy=GraspGeometryPolicyConfig(
                    enabled=True,
                    allow_rim=True,
                    allow_horizontal_sidewall=False,
                ),
                pca=replace(short_edge.pca, defer_axis_ratio_rejection=True),
            ),
        ),
    )


def _frame(
    index: int,
    *,
    include_known: bool = True,
    include_unknown: bool = True,
    include_geometry: bool = True,
    geometry_evidence: GraspGeometryEvidence = GEOMETRY_EVIDENCE,
) -> Frame:
    depth = np.full((240, 320), 0.40, dtype=np.float32)
    depth[:4, :] = 0.0
    depth[:, :4] = np.nan
    detections = []
    if include_known:
        detections.append(
            Detection(
                label="9-Mug",
                confidence=0.88,
                bbox=(20.0, 40.0, 100.0, 150.0),
                depth_m=0.40,
                camera_xyz=(0.01, 0.02, 0.40),
                instance_id="track-2",
                attributes={
                    "track_id": 2,
                    "stable_confidence": 0.88,
                    "raw_confidence": 0.82,
                    "hits": 4,
                    "missed": 0,
                    "held": False,
                },
            )
        )
    if include_unknown:
        detections.append(
            Detection(
                label="unknown",
                confidence=0.80,
                bbox=(170.0, 70.0, 260.0, 180.0),
                depth_m=0.37,
                camera_xyz=(-0.04, 0.01, 0.37),
                object_center_camera_xyz=(-0.04, 0.01, 0.37),
                major_axis_camera=(1.0, 0.0, 0.0),
                minor_axis_camera=(0.0, 1.0, 0.0),
                major_axis_length_m=0.09,
                minor_axis_length_m=0.05,
                pose_confidence=0.75,
                geometry_evidence=geometry_evidence if include_geometry else None,
                graspability=0.72,
                instance_id="track-9",
                attributes={
                    "track_id": 9,
                    "stable_confidence": 0.80,
                    "raw_confidence": 0.76,
                    "hits": 3,
                    "missed": 0,
                    "held": False,
                    "proposal_source": "depth_support_plane",
                    "opaque_foreground_only": True,
                    "transparent_objects_supported": False,
                },
            )
        )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :, 1] = np.arange(320, dtype=np.uint8)[None, :]
    image[:, :, 2] = np.arange(240, dtype=np.uint8)[:, None]
    return Frame(
        frame_id=f"fake-{index:03d}",
        detections=tuple(detections),
        stamp_s=1000.0 + index,
        image=image,
        camera_intrinsics=INTRINSICS,
        depth_m=depth,
    )


class _FakeSource:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.closed = False

    def capture(self):
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


def _fake_builder(source):
    def build(vision_config, *, log=None, **kwargs):
        assert vision_config.backend == "realsense_yolo"
        assert vision_config.unknown_proposer.enabled is True
        assert vision_config.show is False
        assert log is not None
        assert kwargs.get("object_pose_config") is not None
        return source

    return build


def test_acceptance_captures_rgbd_detections_and_audit_artifacts(tmp_path):
    source = _FakeSource([_frame(1), _frame(2), _frame(3)])
    output_dir = tmp_path / "acceptance"

    payload = acceptance.run_acceptance(
        _task_d_config(),
        output_dir=output_dir,
        frame_count=3,
        require_known=True,
        require_unknown=True,
        min_valid_depth_ratio=0.95,
        config_path=FORMAL_CD_CONFIG,
        build_vision_fn=_fake_builder(source),
    )

    assert payload["status"] == "passed"
    assert payload["scope"] == "vision_only_no_motion_no_gripper"
    assert payload["safety"] == {
        "motion_dry_run": True,
        "gripper_backend": "mock",
        "robot_info_required": False,
        "robot_bridge_started": False,
        "motion_backend_constructed": False,
        "gripper_backend_constructed": False,
        "motion_commands_published": False,
        "gripper_commands_sent": False,
    }
    assert payload["config"]["task_mode"] == "D"
    assert payload["config"]["unknown_proposer_enabled"] is True
    assert payload["config"]["short_edge_grasp_enabled"] is True
    assert payload["config"]["geometry_policy_enabled"] is False
    assert payload["config"]["pca_defer_axis_ratio_rejection"] is False
    assert payload["acceptance"]["captured_frames"] == 3
    assert payload["aggregate"]["known_observations"] == 3
    assert payload["aggregate"]["unknown_observations"] == 3
    assert payload["aggregate"]["unique_known_labels"] == ["9-Mug"]
    assert payload["aggregate"]["unique_unknown_track_ids"] == ["track-9"]
    assert payload["aggregate"]["geometry_evidence_observations"] == 3
    assert payload["aggregate"]["unknown_geometry_evidence_observations"] == 3
    assert payload["aggregate"]["rim_evidence_observations"] == 0
    assert payload["aggregate"]["sidewall_evidence_observations"] == 0
    assert payload["aggregate"]["minimum_frame_depth_valid_ratio"] > 0.95
    assert payload["frames"][0]["unknown_track_ids"] == ["track-9"]
    assert payload["frames"][0]["detections"][1]["geometry_evidence"] == {
        "object_center_camera_xyz": [-0.04, 0.01, 0.37],
        "support_plane_valid": False,
        "support_plane_quality": 0.0,
        "support_plane_center_camera_xyz": None,
        "support_plane_normal_camera": None,
        "up_camera": None,
        "visible_height_m": None,
        "footprint_center_camera_xyz": None,
        "footprint_major_length_m": None,
        "footprint_minor_length_m": None,
        "footprint_axis_ratio": None,
        "footprint_major_axis_camera": None,
        "footprint_minor_axis_camera": None,
        "top_center_camera_xyz": None,
        "top_height_m": None,
        "short_edge_confidence": 0.0,
        "rim": None,
        "sidewall": None,
        "valid_points": 100,
        "method_version": "geometry-policy-v1",
    }
    assert payload["frames"][0]["depth"]["central_roi_in_range_ratio"] == 1.0
    assert source.closed is True

    summary_path = output_dir / "summary.json"
    assert summary_path.is_file()
    on_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    assert on_disk["status"] == "passed"

    for frame in payload["frames"]:
        for key in ("rgb", "depth_color", "depth_m_npz", "detection_overlay"):
            assert (output_dir / frame["images"][key]).is_file(), key
    representative = payload["representative_frame"]["images"]
    for filename in representative.values():
        assert (output_dir / filename).is_file()
    depth_npz = np.load(
        output_dir / payload["frames"][0]["images"]["depth_m_npz"]
    )
    assert depth_npz["depth_m"].shape == (240, 320)


def test_require_unknown_fails_closed_but_writes_report(tmp_path):
    source = _FakeSource(
        [
            _frame(1, include_unknown=False),
            _frame(2, include_unknown=False),
        ]
    )
    output_dir = tmp_path / "no-unknown"

    payload = acceptance.run_acceptance(
        _task_d_config(),
        output_dir=output_dir,
        frame_count=2,
        require_unknown=True,
        build_vision_fn=_fake_builder(source),
    )

    assert payload["status"] == "failed"
    assert payload["aggregate"]["unknown_observations"] == 0
    assert any("--require-unknown" in item for item in payload["acceptance"]["failures"])
    assert (output_dir / "summary.json").is_file()
    assert any("No unknown positive sample" in note for note in payload["notes"])


def test_require_geometry_fails_closed_but_writes_report(tmp_path):
    source = _FakeSource(
        [
            _frame(1, include_geometry=False),
            _frame(2, include_geometry=False),
        ]
    )
    output_dir = tmp_path / "no-geometry"

    payload = acceptance.run_acceptance(
        _task_d_config(),
        output_dir=output_dir,
        frame_count=2,
        require_geometry=True,
        build_vision_fn=_fake_builder(source),
    )

    assert payload["status"] == "failed"
    assert payload["aggregate"]["geometry_evidence_observations"] == 0
    assert any("--require-geometry" in item for item in payload["acceptance"]["failures"])
    assert any("No same-frame geometry evidence" in note for note in payload["notes"])
    assert (output_dir / "summary.json").is_file()


def test_geometry_policy_rejects_missing_support_plane_and_fails_closed(tmp_path):
    config = _task_d_geometry_policy_config()
    source = _FakeSource([_frame(1), _frame(2)])
    output_dir = tmp_path / "geometry-policy-rejected"

    payload = acceptance.run_acceptance(
        config,
        output_dir=output_dir,
        frame_count=2,
        require_geometry=True,
        require_geometry_candidate=True,
        build_vision_fn=_fake_builder(source),
    )

    policy = config.task_cd.short_edge_grasp.geometry_policy
    assert payload["status"] == "failed"
    assert payload["config"]["geometry_policy_enabled"] is True
    assert payload["config"]["geometry_policy"] == json.loads(json.dumps(asdict(policy)))
    assert payload["aggregate"]["geometry_evidence_observations"] == 2
    assert payload["aggregate"]["support_plane_valid_observations"] == 0
    assert payload["aggregate"]["support_plane_quality_mean"] == 0.0
    assert payload["aggregate"]["support_plane_quality_min"] == 0.0
    assert payload["aggregate"]["support_plane_quality_max"] == 0.0
    assert payload["aggregate"]["geometry_policy_evaluated_observations"] == 4
    assert payload["aggregate"]["geometry_policy_accepted_observations"] == 0
    assert payload["aggregate"]["geometry_policy_rejected_observations"] == 4
    assert payload["aggregate"]["geometry_policy_candidate_mode_observations"] == {}
    assert payload["aggregate"]["geometry_policy_rejection_categories"] == {
        "geometry_evidence_missing": 2,
        "support_plane_invalid": 2,
    }
    assert payload["aggregate"]["geometry_policy_accepted_track_ids_by_mode"] == {}
    for frame in payload["frames"]:
        unknown = next(
            detection
            for detection in frame["detections"]
            if detection["track_id"] == "track-9"
        )
        assert unknown["geometry_policy"] == {
            "accepted": False,
            "reason": "a reliable support plane is required",
            "reason_category": "support_plane_invalid",
            "candidates": [],
        }
    assert any(
        "--require-geometry-candidate" in item
        for item in payload["acceptance"]["failures"]
    )
    assert any(
        "raw evidence is not candidate acceptance" in note
        for note in payload["notes"]
    )
    assert (output_dir / "summary.json").is_file()


def test_geometry_policy_accepts_measured_short_edge_and_aggregates_track(tmp_path):
    config = _task_d_geometry_policy_config()
    source = _FakeSource(
        [
            _frame(1, geometry_evidence=ACCEPTABLE_GEOMETRY_EVIDENCE),
            _frame(2, geometry_evidence=ACCEPTABLE_GEOMETRY_EVIDENCE),
        ]
    )
    output_dir = tmp_path / "geometry-policy-accepted"

    payload = acceptance.run_acceptance(
        config,
        output_dir=output_dir,
        frame_count=2,
        require_geometry=True,
        require_geometry_candidate=True,
        build_vision_fn=_fake_builder(source),
    )

    policy = config.task_cd.short_edge_grasp.geometry_policy
    assert payload["status"] == "passed"
    assert payload["config"]["geometry_policy"] == json.loads(json.dumps(asdict(policy)))
    assert payload["aggregate"]["support_plane_valid_observations"] == 2
    assert payload["aggregate"]["support_plane_quality_mean"] == pytest.approx(0.98)
    assert payload["aggregate"]["support_plane_quality_min"] == pytest.approx(0.98)
    assert payload["aggregate"]["support_plane_quality_max"] == pytest.approx(0.98)
    assert payload["aggregate"]["geometry_policy_evaluated_observations"] == 4
    assert payload["aggregate"]["geometry_policy_accepted_observations"] == 2
    assert payload["aggregate"]["geometry_policy_rejected_observations"] == 2
    assert payload["aggregate"]["geometry_policy_candidate_mode_observations"] == {
        "vertical_short_edge": 2
    }
    assert payload["aggregate"]["geometry_policy_rejection_categories"] == {
        "geometry_evidence_missing": 2
    }
    assert payload["aggregate"]["geometry_policy_accepted_track_ids_by_mode"] == {
        "vertical_short_edge": ["track-9"]
    }
    for frame in payload["frames"]:
        unknown = next(
            detection
            for detection in frame["detections"]
            if detection["track_id"] == "track-9"
        )
        decision = unknown["geometry_policy"]
        assert decision["accepted"] is True
        assert decision["reason"] == ""
        assert decision["reason_category"] == "accepted"
        assert [candidate["mode"] for candidate in decision["candidates"]] == [
            "vertical_short_edge"
        ]
        assert decision["candidates"][0]["tool_axis_camera"] == [0.0, 0.0, 1.0]
        # Parallel jaws close across the short edges, along the PCA minor
        # axis; the PCA major axis remains the object's long direction.
        assert decision["candidates"][0]["closure_axis_camera"] == [0.0, 1.0, 0.0]
    assert payload["acceptance"]["failures"] == []
    assert (output_dir / "summary.json").is_file()


def test_camera_failure_still_writes_failed_summary(tmp_path):
    source = _FailingSource()
    output_dir = tmp_path / "camera-failure"

    payload = acceptance.run_acceptance(
        _task_d_config(),
        output_dir=output_dir,
        frame_count=2,
        build_vision_fn=_fake_builder(source),
    )

    assert payload["status"] == "failed"
    assert payload["acceptance"]["captured_frames"] == 0
    assert "synthetic camera failure" in payload["capture"]["capture_error"]
    assert source.closed is True
    assert (output_dir / "summary.json").is_file()


def test_read_only_safety_rejects_real_motion_configuration(tmp_path):
    config = _task_d_config()
    unsafe = replace(
        config,
        motion=replace(config.motion, dry_run=False),
    )

    with pytest.raises(acceptance.AcceptanceError, match="motion.dry_run"):
        acceptance.run_acceptance(
            unsafe,
            output_dir=tmp_path / "unsafe-motion",
            frame_count=1,
        )
    assert not (tmp_path / "unsafe-motion").exists()


def test_read_only_safety_rejects_non_mock_gripper(tmp_path):
    config = _task_d_config()
    unsafe = replace(
        config,
        gripper=replace(config.gripper, backend="ms42dc"),
    )

    with pytest.raises(acceptance.AcceptanceError, match="gripper.backend"):
        acceptance.run_acceptance(
            unsafe,
            output_dir=tmp_path / "unsafe-gripper",
            frame_count=1,
        )
    assert not (tmp_path / "unsafe-gripper").exists()


def test_task_mode_gate_rejects_non_d_without_building_vision(tmp_path):
    config = apply_mode(load_config(FORMAL_CD_CONFIG), TaskMode.C)

    with pytest.raises(acceptance.AcceptanceError, match="task.mode=D"):
        acceptance.run_acceptance(
            config,
            output_dir=tmp_path / "wrong-mode",
            frame_count=1,
        )
    assert not (tmp_path / "wrong-mode").exists()
