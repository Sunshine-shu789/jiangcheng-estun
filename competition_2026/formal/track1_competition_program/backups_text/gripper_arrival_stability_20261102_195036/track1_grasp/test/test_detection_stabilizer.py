from track1_grasp.config import VisionConfig
from track1_grasp.detection_stabilizer import DetectionStabilizer
from track1_grasp.domain import Detection


def stabilizer_config(**overrides) -> VisionConfig:
    values = {
        "backend": "mock",
        "show": False,
        "continuous": False,
        "min_confidence": 0.35,
        "stabilizer_inference_min_confidence": 0.10,
        "stabilizer_confirm_hits": 3,
        "stabilizer_max_misses": 3,
        "stabilizer_confidence_alpha": 0.25,
        "stabilizer_box_alpha": 0.55,
        "stabilizer_depth_alpha": 0.25,
        "stabilizer_class_switch_hits": 3,
    }
    values.update(overrides)
    return VisionConfig(**values)


def detection(
    label: str = "6-Foam_Brick",
    confidence: float = 0.90,
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 300.0, 240.0),
    depth_m: float = 0.30,
) -> Detection:
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        depth_m=depth_m,
        camera_xyz=(0.01, -0.02, depth_m),
        attributes={
            "u": (bbox[0] + bbox[2]) * 0.5,
            "v": (bbox[1] + bbox[3]) * 0.5,
        },
    )


def test_new_track_is_hidden_until_confirm_hits():
    tracker = DetectionStabilizer(stabilizer_config())

    assert tracker.update((detection(),), 640, 480) == ()
    assert tracker.update((detection(),), 640, 480) == ()
    stable = tracker.update((detection(),), 640, 480)

    assert len(stable) == 1
    assert stable[0].label == "6-Foam_Brick"
    assert stable[0].instance_id == "track-1"
    assert stable[0].attributes["hits"] == 3
    assert stable[0].attributes["held"] is False


def test_same_class_duplicate_is_suppressed():
    tracker = DetectionStabilizer(
        stabilizer_config(stabilizer_confirm_hits=1)
    )
    stable = tracker.update(
        (
            detection(confidence=0.92),
            detection(confidence=0.70, bbox=(104.0, 104.0, 296.0, 236.0)),
        ),
        640,
        480,
    )

    assert len(stable) == 1
    assert stable[0].attributes["raw_confidence"] == 0.92


def test_cross_class_overlap_keeps_only_the_stronger_box():
    tracker = DetectionStabilizer(
        stabilizer_config(stabilizer_confirm_hits=1)
    )
    stable = tracker.update(
        (
            detection("6-Foam_Brick", confidence=0.90),
            detection("10-meat_can", confidence=0.50),
        ),
        640,
        480,
    )

    assert len(stable) == 1
    assert stable[0].label == "6-Foam_Brick"


def test_class_switch_requires_consecutive_challenger_hits():
    tracker = DetectionStabilizer(
        stabilizer_config(
            stabilizer_confirm_hits=1,
            stabilizer_class_switch_hits=3,
        )
    )

    first = tracker.update((detection("6-Foam_Brick"),), 640, 480)
    challenger_one = tracker.update((detection("10-meat_can"),), 640, 480)
    challenger_two = tracker.update((detection("10-meat_can"),), 640, 480)
    switched = tracker.update((detection("10-meat_can"),), 640, 480)

    assert first[0].label == "6-Foam_Brick"
    assert challenger_one[0].label == "6-Foam_Brick"
    assert challenger_two[0].label == "6-Foam_Brick"
    assert switched[0].label == "10-meat_can"


def test_confirmed_track_is_briefly_held_then_expires():
    tracker = DetectionStabilizer(
        stabilizer_config(
            stabilizer_confirm_hits=1,
            stabilizer_max_misses=3,
        )
    )
    tracker.update((detection(confidence=0.90),), 640, 480)

    held = [tracker.update((), 640, 480) for _ in range(3)]
    expired = tracker.update((), 640, 480)

    assert all(len(frame) == 1 for frame in held)
    assert all(frame[0].attributes["held"] is True for frame in held)
    assert held[-1][0].attributes["missed"] == 3
    assert expired == ()


def test_box_confidence_and_depth_are_smoothed():
    tracker = DetectionStabilizer(
        stabilizer_config(stabilizer_confirm_hits=1)
    )
    first = tracker.update((detection(confidence=0.90, depth_m=0.30),), 640, 480)
    second = tracker.update(
        (
            detection(
                confidence=0.80,
                bbox=(102.0, 100.0, 302.0, 240.0),
                depth_m=0.32,
            ),
        ),
        640,
        480,
    )

    assert second[0].bbox[0] > first[0].bbox[0]
    assert second[0].bbox[0] < 102.0
    assert 0.80 < second[0].confidence < 0.90
    assert 0.30 < second[0].depth_m < 0.32


def test_preview_raw_confidence_tracks_latest_frame_not_smoothed_value():
    tracker = DetectionStabilizer(
        stabilizer_config(stabilizer_confirm_hits=1)
    )
    first = tracker.update((detection(confidence=0.90),), 640, 480)
    second = tracker.update(
        (
            detection(
                confidence=0.60,
                bbox=(102.0, 100.0, 302.0, 240.0),
            ),
        ),
        640,
        480,
    )

    assert first[0].attributes["raw_confidence"] == 0.90
    assert second[0].attributes["raw_confidence"] == 0.60
    assert second[0].confidence > 0.60


def test_low_confidence_edge_false_positive_is_ignored():
    tracker = DetectionStabilizer(
        stabilizer_config(
            stabilizer_confirm_hits=1,
            stabilizer_edge_suppress_confidence=0.55,
        )
    )
    edge_box = detection(
        confidence=0.40,
        bbox=(2.0, 2.0, 80.0, 70.0),
    )

    assert tracker.update((edge_box,), 640, 480) == ()


def test_separate_unknown_instances_are_not_collapsed_by_max_per_class():
    tracker = DetectionStabilizer(
        stabilizer_config(
            stabilizer_confirm_hits=1,
            stabilizer_max_per_class=1,
            stabilizer_instance_tracking_labels=("unknown",),
        )
    )
    first = detection(
        "unknown",
        confidence=0.90,
        bbox=(80.0, 100.0, 180.0, 220.0),
    )
    second = detection(
        "unknown",
        confidence=0.88,
        bbox=(420.0, 120.0, 540.0, 250.0),
    )

    stable = tracker.update((first, second), 640, 480)

    assert len(stable) == 2
    assert {item.instance_id for item in stable} == {"track-1", "track-2"}


def test_unknown_track_ids_survive_short_occlusion_without_swapping():
    tracker = DetectionStabilizer(
        stabilizer_config(
            stabilizer_confirm_hits=1,
            stabilizer_max_misses=3,
            stabilizer_max_per_class=1,
            stabilizer_instance_tracking_labels=("unknown",),
        )
    )
    first = detection(
        "unknown",
        confidence=0.90,
        bbox=(80.0, 100.0, 180.0, 220.0),
    )
    second = detection(
        "unknown",
        confidence=0.88,
        bbox=(420.0, 120.0, 540.0, 250.0),
    )
    initial = tracker.update((first, second), 640, 480)
    initial_ids = {
        item.bbox[0]: item.instance_id for item in initial
    }

    held = tracker.update((first,), 640, 480)
    recovered = tracker.update((first, second), 640, 480)

    assert len(held) == 2
    recovered_ids = {
        round(item.bbox[0] / 100.0): item.instance_id for item in recovered
    }
    assert recovered_ids[1] == initial_ids[80.0]
    assert recovered_ids[4] == initial_ids[420.0]
    assert all(item.instance_id is not None for item in recovered)
