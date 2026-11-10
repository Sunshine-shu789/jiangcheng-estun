from dataclasses import dataclass

import numpy as np
import pytest

from track1_grasp.vision import rotate_180_intrinsics, rotate_180_rgbd


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


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
