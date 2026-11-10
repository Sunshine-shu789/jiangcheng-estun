from dataclasses import replace
from pathlib import Path

from track1_grasp.config import apply_mode, load_config
from track1_grasp.vision import LegacyHttpVision, build_vision


ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG = ROOT / "config" / "track1_task_a_legacy.yaml"


def test_task_a_legacy_config_is_pinned_to_old_http_vision():
    config = apply_mode(load_config(LEGACY_CONFIG), "A")

    assert config.vision.backend == "legacy_http"
    assert config.vision.legacy_status_url == "http://127.0.0.1:9100/status.json"
    assert config.vision.target_class == "6-Foam_Brick"
    assert config.vision.weights == ""
    assert config.vision.continuous is True
    assert config.vision.show is False


def test_task_a_legacy_config_uses_fast_three_axis_patrol():
    config = apply_mode(load_config(LEGACY_CONFIG), "A")
    patrol = config.task_a.patrol

    assert patrol.enabled is True
    assert patrol.position_count == 10
    assert patrol.axes == ("x", "y", "z")
    assert patrol.stride_m == 0.17
    assert patrol.linear_step_m == 0.02
    assert patrol.settle_s == 0.5
    assert config.motion.speed_mps == 0.02
    assert config.motion.joint_max_step_deg == 2.0


def test_legacy_backend_builds_the_http_adapter_without_yolo():
    config = apply_mode(load_config(LEGACY_CONFIG), "A")
    source = build_vision(replace(config.vision, continuous=False))

    try:
        assert isinstance(source, LegacyHttpVision)
    finally:
        source.close()


class SequencedLegacyVision(LegacyHttpVision):
    def __init__(self, statuses):
        config = replace(
            apply_mode(load_config(LEGACY_CONFIG), "A").vision,
            continuous=False,
            min_confidence=0.35,
            legacy_poll_s=0.001,
            first_frame_timeout_s=0.2,
            capture_timeout_s=0.2,
        )
        super().__init__(config)
        self.statuses = iter(statuses)
        self.fetch_count = 0

    def _fetch_status(self):
        self.fetch_count += 1
        return next(self.statuses)


def test_legacy_adapter_filters_class_prefers_stable_confidence_and_deduplicates():
    status = {
        "ready": True,
        "frameSequence": 7,
        "detections": [
            {
                "class": "6-Foam_Brick",
                "confidence": 0.42,
                "stableConfidence": 0.91,
                "bbox": [60, 70, 180, 230],
            },
            {
                "class": "1-Cheez-it",
                "confidence": 0.99,
                "stableConfidence": 0.99,
                "bbox": [1, 2, 3, 4],
            },
            {
                "class": "6-Foam_Brick",
                "confidence": 0.20,
                "stableConfidence": 0.20,
                "bbox": [5, 6, 7, 8],
            },
        ],
    }
    vision = SequencedLegacyVision(
        [status, status, {**status, "frameSequence": 8}]
    )

    first = vision.capture()
    second = vision.capture()

    assert first.frame_id == "legacy-00000007"
    assert len(first.detections) == 1
    assert first.detections[0].label == "6-Foam_Brick"
    assert first.detections[0].confidence == 0.91
    assert first.detections[0].bbox == (60.0, 70.0, 180.0, 230.0)
    assert first.detections[0].attributes["source"] == "taskA-legacy"
    assert second.frame_id == "legacy-00000008"
    assert vision.fetch_count == 3
