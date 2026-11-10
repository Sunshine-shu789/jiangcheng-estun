from pathlib import Path

from track1_grasp.config import apply_mode, load_config
from track1_grasp.domain import BoxAnnotation, Frame, VisionReport
from track1_grasp.factory import build_pipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "track1_mock.yaml"


def build_task_a():
    config = apply_mode(load_config(CONFIG), "A")
    return config, build_pipeline(config)


def test_task_a_scores_ten_positions_by_iou():
    config, pipeline = build_task_a()
    report = pipeline.run()
    assert isinstance(report, VisionReport)
    assert report.frames == 10
    assert report.total_annotations == 10
    assert report.iou_threshold == 0.75
    # the scripted scene contains two deliberately weak positions
    assert report.passed == 8
    assert abs(report.pass_rate - 0.8) < 1e-9


def test_task_a_uses_the_dedicated_scenario_not_the_pick_scene():
    config, pipeline = build_task_a()
    assert Path(pipeline.vision.config.scenario_file).name == "mock_task_a_scene.yaml"


def test_task_a_does_not_touch_motion_or_gripper():
    _config, pipeline = build_task_a()
    assert not hasattr(pipeline, "motion")
    assert not hasattr(pipeline, "gripper")
    pipeline.run()
    pipeline.close()


def test_task_a_flat_frames_without_annotations_are_still_processed():
    config, pipeline = build_task_a()
    pipeline._ground_truth = {}
    box = BoxAnnotation("mug", (0, 0, 100, 100))
    pipeline.vision._frames = [
        Frame(
            frame_id="only",
            detections=[
                # reuse a detection-shaped object through the annotation helper
                box.as_detection(base_xyz=(0.1, 0.1, 0.05)),
            ],
            annotations=(box,),
        )
    ]
    report = pipeline.run()
    assert report.frames == 1
    assert report.passed == 1
