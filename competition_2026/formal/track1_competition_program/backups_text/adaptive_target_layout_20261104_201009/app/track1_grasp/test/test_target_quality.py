from track1_grasp.config import CornerSmallObjectFilterConfig
from track1_grasp.domain import Detection
from track1_grasp.grasp_geometry import GraspGeometryEvidence
from track1_grasp.target_quality import (
    candidate_region_priority,
    detection_quality_key,
    is_corner_region_object,
    is_corner_small_object,
)


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def _filter_config(*, enabled: bool = True) -> CornerSmallObjectFilterConfig:
    return CornerSmallObjectFilterConfig(
        enabled=enabled,
        corner_zone_ratio=0.30,
        max_box_width_ratio=0.12,
        max_box_height_ratio=0.14,
    )


def _geometry_evidence(
    *,
    support_plane_valid: bool = True,
    support_plane_quality: float = 0.95,
    short_edge_confidence: float = 0.95,
) -> GraspGeometryEvidence:
    return GraspGeometryEvidence(
        object_center_camera_xyz=(0.0, 0.0, 0.40),
        support_plane_valid=support_plane_valid,
        support_plane_quality=support_plane_quality,
        support_plane_center_camera_xyz=(0.0, 0.0, 0.42),
        support_plane_normal_camera=(0.0, 0.0, 1.0),
        up_camera=(0.0, 0.0, -1.0),
        visible_height_m=0.05,
        footprint_center_camera_xyz=(0.0, 0.0, 0.42),
        footprint_major_length_m=0.14,
        footprint_minor_length_m=0.05,
        footprint_axis_ratio=2.8,
        footprint_major_axis_camera=(1.0, 0.0, 0.0),
        footprint_minor_axis_camera=(0.0, 1.0, 0.0),
        top_center_camera_xyz=(0.0, 0.0, 0.37),
        top_height_m=0.05,
        short_edge_confidence=short_edge_confidence,
        rim=None,
        sidewall=None,
        valid_points=1000,
    )


def test_corner_small_object_filter_excludes_left_bottom_and_right_top():
    config = _filter_config()

    assert is_corner_small_object(
        (5.0, 405.0, 45.0, 445.0),
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    )
    assert is_corner_small_object(
        (600.0, 5.0, 635.0, 40.0),
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    )


def test_corner_small_object_filter_keeps_normal_sized_edge_object():
    config = _filter_config()

    assert not is_corner_small_object(
        (0.0, 0.0, 180.0, 180.0),
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    )


def test_corner_small_object_filter_keeps_small_centred_object():
    config = _filter_config()

    assert not is_corner_small_object(
        (300.0, 220.0, 340.0, 260.0),
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    )


def test_candidate_region_priority_puts_middle_before_large_corner_object():
    config = _filter_config()
    middle_bbox = (250.0, 150.0, 390.0, 290.0)
    large_corner_bbox = (5.0, 405.0, 145.0, 445.0)

    assert candidate_region_priority(
        middle_bbox,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    ) == 1
    assert is_corner_region_object(
        large_corner_bbox,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    )
    assert not is_corner_small_object(
        large_corner_bbox,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    )
    assert candidate_region_priority(
        large_corner_bbox,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    ) == 0


def test_corner_small_object_filter_matches_live_strawberry_bbox():
    # Measured live V11 detection: centre (0.2945, 0.8031), size
    # (0.0734, 0.0979) relative to the 640x480 colour frame.
    assert is_corner_small_object(
        (165.0, 362.0, 212.0, 409.0),
        FRAME_WIDTH,
        FRAME_HEIGHT,
        _filter_config(),
    )


def test_quality_key_prefers_measured_geometry_over_raw_yolo_confidence():
    strong_geometry = Detection(
        label="10-meat_can",
        confidence=0.40,
        bbox=(250.0, 190.0, 390.0, 290.0),
        pose_confidence=0.60,
        geometry_evidence=_geometry_evidence(),
        graspability=0.95,
    )
    high_yolo_confidence = Detection(
        label="10-meat_can",
        confidence=0.99,
        bbox=(250.0, 190.0, 390.0, 290.0),
        pose_confidence=0.99,
        geometry_evidence=None,
        graspability=0.95,
    )

    assert detection_quality_key(
        strong_geometry,
        FRAME_WIDTH,
        FRAME_HEIGHT,
    ) > detection_quality_key(
        high_yolo_confidence,
        FRAME_WIDTH,
        FRAME_HEIGHT,
    )


def test_disabled_corner_filter_keeps_corner_small_object():
    assert not is_corner_small_object(
        (5.0, 405.0, 45.0, 445.0),
        FRAME_WIDTH,
        FRAME_HEIGHT,
        _filter_config(enabled=False),
    )
