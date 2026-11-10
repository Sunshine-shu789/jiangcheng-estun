from dataclasses import dataclass

import numpy as np
import pytest

from track1_grasp.vision import (
    pad_rgbd_for_inference,
    rotate_180_intrinsics,
    rotate_180_rgbd,
    unpad_detection,
)


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class DetectionItem:
    bbox: tuple[int, int, int, int]
    u: float
    v: float
    confidence: float


def test_rotate_180_intrinsics_moves_principal_point():
    intrinsics = Intrinsics(
        width=640,
        height=480,
        fx=400.0,
        fy=401.0,
        cx=300.25,
        cy=220.75,
    )

    rotated = rotate_180_intrinsics(intrinsics)

    assert rotated.width == 640
    assert rotated.height == 480
    assert rotated.fx == 400.0
    assert rotated.fy == 401.0
    assert rotated.cx == pytest.approx(338.75)
    assert rotated.cy == pytest.approx(258.25)
    # The camera object must not be mutated in place.
    assert intrinsics.cx == 300.25
    assert intrinsics.cy == 220.75


def test_rotate_180_rgbd_keeps_colour_and_depth_aligned():
    color = np.arange(6, dtype=np.uint8).reshape(2, 3, 1)
    depth = np.arange(6, dtype=np.float32).reshape(2, 3) / 100.0
    intrinsics = Intrinsics(
        width=3,
        height=2,
        fx=2.0,
        fy=2.0,
        cx=1.0,
        cy=0.5,
    )

    rotated_color, rotated_depth, rotated_intrinsics = rotate_180_rgbd(
        color, depth, intrinsics
    )

    expected = np.asarray([[5, 4, 3], [2, 1, 0]])
    assert np.array_equal(rotated_color[..., 0], expected)
    assert np.array_equal(rotated_depth, expected.astype(np.float32) / 100.0)
    assert rotated_intrinsics.cx == pytest.approx(1.0)
    assert rotated_intrinsics.cy == pytest.approx(0.5)
    assert rotated_color.flags.c_contiguous
    assert rotated_depth.flags.c_contiguous


def test_rotate_180_rgbd_rejects_misaligned_frames():
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((479, 640), dtype=np.float32)
    intrinsics = Intrinsics(
        width=640,
        height=480,
        fx=400.0,
        fy=400.0,
        cx=320.0,
        cy=240.0,
    )

    with pytest.raises(RuntimeError, match="shape mismatch"):
        rotate_180_rgbd(color, depth, intrinsics)


def test_inference_padding_keeps_rgbd_and_intrinsics_aligned():
    color = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.arange(6, dtype=np.float32).reshape(2, 3) / 100.0
    intrinsics = Intrinsics(
        width=3,
        height=2,
        fx=2.0,
        fy=3.0,
        cx=1.0,
        cy=0.5,
    )

    padded_color, padded_depth, padded_intrinsics = pad_rgbd_for_inference(
        color,
        depth,
        intrinsics,
        2,
    )

    assert padded_color.shape == (6, 7, 3)
    assert padded_depth.shape == (6, 7)
    assert np.all(padded_color[:2] == 114)
    assert np.all(padded_color[:, :2] == 114)
    assert np.array_equal(padded_color[2:4, 2:5], color)
    assert np.all(padded_depth[:2] == 0.0)
    assert np.all(padded_depth[:, :2] == 0.0)
    assert np.array_equal(padded_depth[2:4, 2:5], depth)
    assert padded_intrinsics.width == 7
    assert padded_intrinsics.height == 6
    assert padded_intrinsics.fx == 2.0
    assert padded_intrinsics.fy == 3.0
    assert padded_intrinsics.cx == 3.0
    assert padded_intrinsics.cy == 2.5


def test_unpad_detection_maps_box_and_sample_pixel_to_original_frame():
    item = DetectionItem(
        bbox=(42, 52, 152, 132),
        u=102.0,
        v=92.0,
        confidence=0.91,
    )

    mapped = unpad_detection(item, 20, 640, 480)

    assert mapped.bbox == (22, 32, 132, 112)
    assert mapped.u == 82.0
    assert mapped.v == 72.0
    assert mapped.confidence == 0.91


def test_unpad_detection_drops_items_entirely_inside_neutral_border():
    item = DetectionItem(
        bbox=(2, 3, 12, 13),
        u=7.0,
        v=8.0,
        confidence=0.91,
    )

    assert unpad_detection(item, 20, 640, 480) is None
