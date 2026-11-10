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
MIDDLE_BBOX = (250.0, 150.0, 390.0, 290.0)
LARGE_PERIPHERAL_BBOX = (5.0, 405.0, 145.0, 445.0)
SMALL_PERIPHERAL_BBOX = (165.0, 362.0, 212.0, 409.0)
CENTRAL_SMALL_BBOX = (285.0, 250.0, 325.0, 290.0)


def _filter_config(
    *,
    enabled: bool = True,
    mode: str = "adaptive",
) -> CornerSmallObjectFilterConfig:
    return CornerSmallObjectFilterConfig(enabled=enabled, mode=mode)


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


def test_adaptive_filter_excludes_live_small_corner_strawberry():
    scene = (MIDDLE_BBOX, SMALL_PERIPHERAL_BBOX)

    assert is_corner_small_object(
        SMALL_PERIPHERAL_BBOX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        _filter_config(),
        scene_bboxes=scene,
    )


def test_adaptive_filter_still_catches_singleton_corner_small_object():
    assert is_corner_small_object(
        SMALL_PERIPHERAL_BBOX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        _filter_config(),
    )


def test_adaptive_filter_keeps_large_peripheral_object():
    scene = (MIDDLE_BBOX, LARGE_PERIPHERAL_BBOX)

    assert is_corner_region_object(
        LARGE_PERIPHERAL_BBOX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        _filter_config(),
    )
    assert not is_corner_small_object(
        LARGE_PERIPHERAL_BBOX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        _filter_config(),
        scene_bboxes=scene,
    )


def test_adaptive_filter_keeps_small_centred_object():
    scene = (MIDDLE_BBOX, CENTRAL_SMALL_BBOX)

    assert not is_corner_small_object(
        CENTRAL_SMALL_BBOX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        _filter_config(),
        scene_bboxes=scene,
    )


def test_adaptive_priority_is_continuous_and_centre_first():
    scene = (MIDDLE_BBOX, LARGE_PERIPHERAL_BBOX)

    middle_priority = candidate_region_priority(
        MIDDLE_BBOX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        _filter_config(),
        scene_bboxes=scene,
    )
    peripheral_priority = candidate_region_priority(
        LARGE_PERIPHERAL_BBOX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        _filter_config(),
        scene_bboxes=scene,
    )

    assert middle_priority == 1.0
    assert 0.0 <= peripheral_priority < middle_priority


def test_old_030_boundary_no_longer_changes_small_corner_decision():
    # These two bboxes put the live strawberry centre on opposite sides of the
    # old x=0.30 cutoff while remaining at the same y position.  Adaptive mode
    # must not flip merely because the target crosses that historical boundary.
    left_of_old_boundary = (158.0, 362.0, 205.0, 409.0)   # centre x = 0.2836
    right_of_old_boundary = (211.0, 362.0, 258.0, 409.0)  # centre x = 0.3664
    config = _filter_config()

    assert is_corner_small_object(
        left_of_old_boundary,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    )
    assert is_corner_small_object(
        right_of_old_boundary,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    )


def test_adaptive_priority_tracks_shifted_target_distribution():
    original_scene = (MIDDLE_BBOX, LARGE_PERIPHERAL_BBOX)
    shifted_scene = tuple(
        (x0 + 45.0, y0 + 10.0, x1 + 45.0, y1 + 10.0)
        for x0, y0, x1, y1 in original_scene
    )
    config = _filter_config()

    original = [
        candidate_region_priority(
            bbox,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            config,
            scene_bboxes=original_scene,
        )
        for bbox in original_scene
    ]
    shifted = [
        candidate_region_priority(
            bbox,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            config,
            scene_bboxes=shifted_scene,
        )
        for bbox in shifted_scene
    ]

    assert original[0] > original[1]
    assert shifted[0] > shifted[1]
    assert original == shifted


def test_mixed_scene_excludes_only_peripheral_small_clutter():
    scene = (MIDDLE_BBOX, LARGE_PERIPHERAL_BBOX, SMALL_PERIPHERAL_BBOX)
    config = _filter_config()

    excluded = {
        bbox: is_corner_small_object(
            bbox,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            config,
            scene_bboxes=scene,
        )
        for bbox in scene
    }

    assert excluded == {
        MIDDLE_BBOX: False,
        LARGE_PERIPHERAL_BBOX: False,
        SMALL_PERIPHERAL_BBOX: True,
    }


def test_legacy_mode_remains_parse_compatible_when_explicitly_selected():
    config = CornerSmallObjectFilterConfig(
        enabled=True,
        corner_zone_ratio=0.30,
        max_box_width_ratio=0.12,
        max_box_height_ratio=0.14,
        mode="legacy",
    )

    assert is_corner_small_object(
        (5.0, 405.0, 45.0, 445.0),
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    )
    assert candidate_region_priority(
        (5.0, 405.0, 45.0, 445.0),
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    ) == 0.0
    assert candidate_region_priority(
        MIDDLE_BBOX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        config,
    ) == 1.0


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


def test_disabled_adaptive_filter_keeps_peripheral_small_object():
    assert not is_corner_small_object(
        SMALL_PERIPHERAL_BBOX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        _filter_config(enabled=False),
    )
