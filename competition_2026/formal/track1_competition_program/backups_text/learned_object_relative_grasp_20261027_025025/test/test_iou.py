from track1_grasp.annotator import iou, match_frame, score_frame
from track1_grasp.domain import BoxAnnotation, Detection, Frame


def detection(label, bbox, confidence=0.9):
    return Detection(label=label, confidence=confidence, bbox=bbox)


def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_of_half_overlap():
    # 10x10 boxes sharing a 5x10 strip -> 50 / 150
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_label_is_required_but_generic_labels_are_forgiven():
    strict = match_frame(
        [detection("bowl", (0, 0, 10, 10))],
        [BoxAnnotation("mug", (0, 0, 10, 10))],
        threshold=0.75,
    )
    assert strict[0].detection is None
    assert not strict[0].passed

    generic = match_frame(
        [detection("object", (0, 0, 10, 10))],
        [BoxAnnotation("mug", (0, 0, 10, 10))],
        threshold=0.75,
    )
    assert generic[0].detection is not None
    assert generic[0].passed


def test_score_frame_counts_misses_and_spurious_boxes():
    frame = Frame(
        frame_id="f",
        detections=[detection("mug", (0, 0, 100, 100)), detection("bowl", (300, 300, 340, 340))],
        annotations=[
            BoxAnnotation("mug", (0, 0, 100, 100)),
            BoxAnnotation("book", (500, 500, 540, 540)),
        ],
    )
    matches, report = score_frame(frame, threshold=0.75)
    assert report.total_annotations == 2
    assert report.passed == 1
    assert report.matched == 1
    assert report.pass_rate == 0.5
    # mug matched, book missed, bowl spurious
    assert len(matches) == 3


def test_score_frame_reports_low_iou_as_failure():
    frame = Frame(
        frame_id="f",
        detections=[detection("mug", (0, 0, 100, 100))],
        annotations=[BoxAnnotation("mug", (30, 30, 130, 130))],
    )
    _matches, report = score_frame(frame, threshold=0.75)
    assert report.passed == 0
    assert 0.0 < report.mean_iou < 0.75
