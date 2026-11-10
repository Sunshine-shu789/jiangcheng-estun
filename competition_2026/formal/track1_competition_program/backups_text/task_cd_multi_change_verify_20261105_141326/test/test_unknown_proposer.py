from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from track1_grasp.config import (
    CornerSmallObjectFilterConfig,
    UnknownObjectProposerConfig,
    VisionConfig,
    apply_mode,
    load_config,
)
from track1_grasp.domain import CameraIntrinsics, Detection, TaskMode
from track1_grasp.unknown_proposer import UnknownObjectProposer
from track1_grasp import vision as vision_module
from track1_grasp.vision import RealSenseYoloVision


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = Path(__file__).resolve().parents[3]
FORMAL_CD_CONFIG = (
    FORMAL_ROOT / "config" / "track1_task_cd_40class_v11_official.yaml"
)

WIDTH = 640
HEIGHT = 480
INTRINSICS = CameraIntrinsics(
    width=WIDTH,
    height=HEIGHT,
    fx=600.0,
    fy=600.0,
    cx=320.0,
    cy=240.0,
)


def proposer_config(**overrides) -> UnknownObjectProposerConfig:
    values = {
        "enabled": True,
        "plane_ransac_iterations": 48,
        "min_area_px": 400,
        "min_depth_points": 120,
    }
    values.update(overrides)
    return UnknownObjectProposerConfig(**values)


def scene(*, include_noise: bool = False):
    depth = np.full((HEIGHT, WIDTH), 0.50, dtype=float)
    color = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    depth[180:300, 120:240] = 0.42
    depth[170:290, 380:500] = 0.42
    if include_noise:
        depth[30:42, 30:42] = 0.42
    return color, depth


def test_unknown_proposer_disabled_returns_no_candidates():
    proposer = UnknownObjectProposer(proposer_config(enabled=False))
    assert proposer.propose(None, None, None) == ()


def test_unknown_proposer_returns_two_separate_geometric_instances():
    color, depth = scene()
    proposals = UnknownObjectProposer(proposer_config()).propose(
        color, depth, INTRINSICS
    )

    assert len(proposals) == 2
    assert all(proposal.label == "unknown" for proposal in proposals)
    assert proposals[0].bbox[0] > proposals[1].bbox[2] or (
        proposals[1].bbox[0] > proposals[0].bbox[2]
    )
    for proposal in proposals:
        assert proposal.depth_m is not None and proposal.depth_m > 0.0
        assert proposal.camera_xyz is not None
        assert proposal.object_center_camera_xyz is not None
        assert proposal.major_axis_camera is not None
        assert proposal.minor_axis_camera is not None
        assert proposal.major_axis_length_m is not None
        assert proposal.minor_axis_length_m is not None
        assert proposal.pose_confidence is not None
        assert proposal.pose_confidence >= 0.55
        assert proposal.graspability > 0.0
        assert proposal.attributes["proposal_source"] == "depth_support_plane"
        assert proposal.attributes["unknown_proposal"] is True
        assert proposal.attributes["opaque_foreground_only"] is True
        assert proposal.attributes["transparent_objects_supported"] is False


def test_unknown_proposer_filters_small_depth_noise():
    color, depth = scene(include_noise=True)
    proposals = UnknownObjectProposer(proposer_config()).propose(
        color, depth, INTRINSICS
    )

    assert len(proposals) == 2
    assert all(
        not (proposal.bbox[0] <= 30.0 and proposal.bbox[1] <= 30.0)
        for proposal in proposals
    )


def test_unknown_proposer_excludes_foreground_already_known_to_v11():
    color, depth = scene()
    known = Detection(
        label="9-Mug",
        confidence=0.90,
        bbox=(110.0, 170.0, 250.0, 310.0),
    )
    proposals = UnknownObjectProposer(proposer_config()).propose(
        color,
        depth,
        INTRINSICS,
        known_detections=(known,),
    )

    assert len(proposals) == 1
    assert proposals[0].bbox[0] >= 380.0


def test_unknown_proposer_fails_closed_without_valid_depth():
    color, _depth = scene()
    depth = np.full((HEIGHT, WIDTH), np.nan, dtype=float)
    proposer = UnknownObjectProposer(proposer_config())

    assert proposer.propose(color, depth, INTRINSICS) == ()


def test_unknown_proposer_rejects_low_pose_quality():
    color, depth = scene()
    proposer = UnknownObjectProposer(
        proposer_config(min_pose_confidence=0.99)
    )

    assert proposer.propose(color, depth, INTRINSICS) == ()


def test_unknown_graspability_is_only_a_conservative_geometric_proxy():
    color, depth = scene()
    proposals = UnknownObjectProposer(
        proposer_config(graspability_scale=0.5)
    ).propose(color, depth, INTRINSICS)

    assert len(proposals) == 2
    assert all(proposal.graspability < 0.65 for proposal in proposals)


def test_task_mode_gate_enables_unknown_branch_only_for_task_d():
    base = load_config(FORMAL_CD_CONFIG)

    assert base.vision.unknown_proposer.enabled is False
    assert apply_mode(base, TaskMode.C).vision.unknown_proposer.enabled is False
    assert apply_mode(base, TaskMode.D).vision.unknown_proposer.enabled is True
    assert apply_mode(base, TaskMode.E).vision.unknown_proposer.enabled is False


def test_empty_instance_tracking_label_is_rejected():
    base = load_config(ROOT / "config" / "track1_mock.yaml")
    invalid = replace(
        base,
        vision=replace(
            base.vision,
            stabilizer_instance_tracking_labels=("unknown", " "),
        ),
    )

    try:
        invalid.validate()
    except ValueError as exc:
        assert "instance_tracking_labels" in str(exc)
    else:
        raise AssertionError("blank instance-tracking label must be rejected")


class _FakeStream:
    def __init__(self, color, depth) -> None:
        self.color = color
        self.depth = depth
        self.stopped = False

    def read(self):
        return self.color, self.depth, INTRINSICS

    def stop(self) -> None:
        self.stopped = True


class _FakeDetector:
    def __init__(self, detections) -> None:
        self.detections = detections

    def detect(self, _color, _depth, _intrinsics):
        return self.detections


class _ExplodingProposer:
    def propose(self, *_args, **_kwargs):
        raise RuntimeError("synthetic proposal failure")


def _yolo_vision_with_fake_frame(
    *,
    include_known: bool,
    target_class: str = "",
    unknown_proposer=None,
):
    color, depth = scene()
    known_items = []
    if include_known:
        known_items.append(
            SimpleNamespace(
                class_name="9-Mug",
                confidence=0.90,
                bbox=(110.0, 170.0, 250.0, 310.0),
                u=180.0,
                v=240.0,
                depth_m=0.42,
                cam_xyz=(0.0, 0.0, 0.42),
            )
        )
    config = VisionConfig(
        backend="realsense_yolo",
        weights="unused-in-test",
        classes=("9-Mug",),
        target_class=target_class,
        min_confidence=0.15,
        stabilize=False,
        continuous=False,
        show=False,
        unknown_proposer=unknown_proposer or proposer_config(),
    )
    source = RealSenseYoloVision(config)
    source._started = True
    source._stream = _FakeStream(color, depth)
    source._detector = _FakeDetector(known_items)
    source._intrinsics = INTRINSICS
    return source


def test_realsense_vision_adds_unknown_proposals_without_overlapping_v11():
    source = _yolo_vision_with_fake_frame(include_known=True)
    frame = source.capture()

    assert len(frame.detections) == 2
    assert frame.detections[0].label == "9-Mug"
    assert frame.detections[1].label == "unknown"
    assert frame.detections[1].bbox[0] >= 380.0


def test_realsense_vision_filters_other_yolo_classes_when_target_is_pinned():
    color, depth = scene()
    source = _yolo_vision_with_fake_frame(
        include_known=False,
        target_class="4-Frenchs_Mustard",
        unknown_proposer=proposer_config(enabled=False),
    )
    source._detector = _FakeDetector(
        (
            SimpleNamespace(
                class_name="4-Frenchs_Mustard",
                confidence=0.90,
                bbox=(110.0, 170.0, 250.0, 310.0),
                u=180.0,
                v=240.0,
                depth_m=0.42,
                cam_xyz=(0.0, 0.0, 0.42),
            ),
            SimpleNamespace(
                class_name="10-meat_can",
                confidence=0.99,
                bbox=(380.0, 170.0, 500.0, 310.0),
                u=440.0,
                v=240.0,
                depth_m=0.42,
                cam_xyz=(0.0, 0.0, 0.42),
            ),
        )
    )

    frame = source.capture()

    assert [detection.label for detection in frame.detections] == [
        "4-Frenchs_Mustard"
    ]


def test_realsense_vision_keeps_v11_result_if_unknown_proposer_raises():
    source = _yolo_vision_with_fake_frame(include_known=True)
    source._unknown_proposer = _ExplodingProposer()
    frame = source.capture()

    assert [detection.label for detection in frame.detections] == ["9-Mug"]


def test_scene_support_uses_only_formally_eligible_detection_boxes(monkeypatch):
    source = _yolo_vision_with_fake_frame(
        include_known=False,
        unknown_proposer=proposer_config(enabled=False),
    )
    source.config = replace(
        source.config,
        classes=(),
        stabilize=True,
        stabilizer_inference_min_confidence=0.10,
    )
    high_bbox = (110.0, 170.0, 250.0, 310.0)
    low_bbox = (165.0, 363.0, 213.0, 409.0)
    source._detector = _FakeDetector(
        (
            SimpleNamespace(
                class_name="9-Mug",
                confidence=0.16,
                bbox=high_bbox,
                u=180.0,
                v=240.0,
                depth_m=0.42,
                cam_xyz=(0.0, 0.0, 0.42),
            ),
            SimpleNamespace(
                class_name="9-Mug",
                confidence=0.12,
                bbox=low_bbox,
                u=189.0,
                v=386.0,
                depth_m=0.42,
                cam_xyz=(0.0, 0.0, 0.42),
            ),
        )
    )
    source._object_pose_config = object()
    fitted_boxes = []

    def fake_scene_support(_depth, boxes, _intrinsics, _config):
        fitted_boxes.append(tuple(boxes))
        return None

    def reject_pose(*_args, **_kwargs):
        raise ValueError("synthetic pose rejection")

    monkeypatch.setattr(
        vision_module,
        "estimate_scene_support_plane",
        fake_scene_support,
    )
    monkeypatch.setattr(vision_module, "estimate_object_pose", reject_pose)

    frame = source.capture()

    assert fitted_boxes == [(high_bbox,)]
    assert [detection.bbox for detection in frame.detections] == [
        high_bbox,
        low_bbox,
    ]
    assert all(
        detection.attributes["scene_support_bboxes_xyxy_px"] == (high_bbox,)
        for detection in frame.detections
    )


def test_scene_support_excludes_configured_corner_small_detection(monkeypatch):
    source = _yolo_vision_with_fake_frame(
        include_known=False,
        unknown_proposer=proposer_config(enabled=False),
    )
    source.config = replace(
        source.config,
        classes=(),
        stabilize=True,
        corner_small_object_filter=CornerSmallObjectFilterConfig(
            enabled=True,
            mode="adaptive",
        ),
    )
    normal_bbox = (110.0, 170.0, 250.0, 310.0)
    corner_small_bbox = (165.0, 362.0, 212.0, 409.0)
    source._detector = _FakeDetector(
        (
            SimpleNamespace(
                class_name="26-Plastic_Strawberry",
                confidence=0.95,
                bbox=normal_bbox,
                u=180.0,
                v=240.0,
                depth_m=0.42,
                cam_xyz=(0.0, 0.0, 0.42),
            ),
            SimpleNamespace(
                class_name="26-Plastic_Strawberry",
                confidence=0.95,
                bbox=corner_small_bbox,
                u=188.0,
                v=386.0,
                depth_m=0.42,
                cam_xyz=(0.0, 0.0, 0.42),
            ),
        )
    )
    source._object_pose_config = object()
    fitted_boxes = []

    def fake_scene_support(_depth, boxes, _intrinsics, _config):
        fitted_boxes.append(tuple(boxes))
        return None

    def reject_pose(*_args, **_kwargs):
        raise ValueError("synthetic pose rejection")

    monkeypatch.setattr(
        vision_module,
        "estimate_scene_support_plane",
        fake_scene_support,
    )
    monkeypatch.setattr(vision_module, "estimate_object_pose", reject_pose)

    frame = source.capture()

    assert fitted_boxes == [(normal_bbox,)]
    assert [detection.bbox for detection in frame.detections] == [
        normal_bbox,
        corner_small_bbox,
    ]
    assert all(
        detection.attributes["scene_support_bboxes_xyxy_px"] == (normal_bbox,)
        for detection in frame.detections
    )
