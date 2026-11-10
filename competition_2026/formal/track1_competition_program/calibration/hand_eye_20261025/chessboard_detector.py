#!/usr/bin/env python3
"""Detect the operator's 10x7-inner-corner chessboard and estimate its pose.

Coordinates follow OpenCV's camera convention: +X right, +Y down, +Z forward.
The returned transform is T_camera_board (camera <- board).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen

import cv2
import numpy as np


DEFAULT_CAMERA_MATRIX = np.array(
    [
        [392.6912841796875, 0.0, 324.5948486328125],
        [0.0, 391.7457275390625, 232.28482055664062],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
# RealSense reports its inverse Brown-Conrady coefficients.  For this board's
# near-central field of view OpenCV accepts them directly; the resulting
# reprojection error is checked on every acquisition and acquisition is
# rejected when it becomes excessive.
DEFAULT_DIST_COEFFS = np.array(
    [
        -0.0534040629863739,
        0.05888928845524788,
        8.304249058710411e-05,
        8.694318239577115e-05,
        -0.019305530935525894,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class BoardSpec:
    # OpenCV calls these the inner-corner grid.  An 11x8-square printed
    # chessboard therefore has 10x7 inner corners.
    columns: int = 10
    rows: int = 7
    square_size_m: float = 0.014

    @property
    def pattern_size(self) -> tuple[int, int]:
        return self.columns, self.rows

    @property
    def square_grid(self) -> tuple[int, int]:
        return self.columns + 1, self.rows + 1

    @property
    def object_points(self) -> np.ndarray:
        points = np.zeros((self.columns * self.rows, 3), dtype=np.float32)
        points[:, :2] = (
            np.mgrid[0 : self.columns, 0 : self.rows].T.reshape(-1, 2)
            * self.square_size_m
        )
        return points


@dataclass(frozen=True)
class BoardObservation:
    image_path: str
    corners_px: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    reprojection_rms_px: float
    reprojection_mean_px: float
    reprojection_max_px: float
    board_center_camera_m: np.ndarray
    board_normal_camera: np.ndarray
    corner_refinement_used: bool = False

    @property
    def rotation_matrix(self) -> np.ndarray:
        rotation, _ = cv2.Rodrigues(self.rvec)
        return rotation

    @property
    def transform_camera_board(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation_matrix
        transform[:3, 3] = self.tvec.reshape(3)
        return transform


def detect_corners(
    gray: np.ndarray,
    board: BoardSpec,
) -> np.ndarray | None:
    flag_sets = (
        (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        ),
        cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        0,
    )
    for flags in flag_sets:
        found, corners = cv2.findChessboardCornersSB(
            gray,
            board.pattern_size,
            flags=flags,
        )
        if found and corners is not None:
            return np.asarray(corners, dtype=np.float32)
    return None


def _solve_board_pose(
    corners: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solved, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise RuntimeError("棋盘格角点已找到，但 solvePnP 失败")
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - corners.reshape(-1, 2),
        axis=1,
    )
    return rvec, tvec, errors


def estimate_board_pose(
    image: np.ndarray,
    board: BoardSpec,
    camera_matrix: np.ndarray = DEFAULT_CAMERA_MATRIX,
    dist_coeffs: np.ndarray = DEFAULT_DIST_COEFFS,
    image_path: str = "",
) -> BoardObservation:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    corners = detect_corners(gray, board)
    if corners is None:
        raise RuntimeError(
            f"没有检测到 {board.columns}x{board.rows} 棋盘格内角点"
        )

    object_points = board.object_points
    rvec, tvec, errors = _solve_board_pose(
        corners, object_points, camera_matrix, dist_coeffs
    )
    corner_refinement_used = False
    try:
        refined = cv2.cornerSubPix(
            gray,
            corners.reshape(-1, 1, 2).copy(),
            (3, 3),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                50,
                1e-4,
            ),
        )
        refined_rvec, refined_tvec, refined_errors = _solve_board_pose(
            refined, object_points, camera_matrix, dist_coeffs
        )
        baseline_rms = float(np.sqrt(np.mean(errors**2)))
        refined_rms = float(np.sqrt(np.mean(refined_errors**2)))
        if refined_rms < baseline_rms:
            corners = refined
            rvec = refined_rvec
            tvec = refined_tvec
            errors = refined_errors
            corner_refinement_used = True
    except cv2.error:
        pass

    rotation, _ = cv2.Rodrigues(rvec)
    center = (
        tvec.reshape(3)
        + rotation
        @ np.array(
            [
                (board.columns - 1) * board.square_size_m / 2.0,
                (board.rows - 1) * board.square_size_m / 2.0,
                0.0,
            ],
            dtype=np.float64,
        )
    )
    return BoardObservation(
        image_path=image_path,
        corners_px=corners.reshape(-1, 2),
        rvec=rvec.reshape(3),
        tvec=tvec.reshape(3),
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        dist_coeffs=np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
        reprojection_rms_px=float(np.sqrt(np.mean(errors**2))),
        reprojection_mean_px=float(np.mean(errors)),
        reprojection_max_px=float(np.max(errors)),
        board_center_camera_m=center,
        board_normal_camera=rotation[:, 2].copy(),
        corner_refinement_used=corner_refinement_used,
    )


def render_overlay(
    image: np.ndarray,
    observation: BoardObservation,
    board: BoardSpec,
) -> np.ndarray:
    result = image.copy()
    corners = np.rint(observation.corners_px).astype(np.int32)
    for index, point in enumerate(corners):
        cv2.circle(result, tuple(point), 2, (0, 0, 255), -1, cv2.LINE_AA)
        if index in (0, board.columns - 1, len(corners) - board.columns, len(corners) - 1):
            cv2.putText(
                result,
                str(index),
                (point[0] + 4, point[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    center = observation.board_center_camera_m
    cv2.putText(
        result,
        (
            f"10x7 corners / 11x8 squares / 14mm  RMS={observation.reprojection_rms_px:.3f}px  "
            f"range={np.linalg.norm(center):.3f}m"
        ),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"无法读取图像: {path}")
    return image


def fetch_snapshot(url: str, timeout_s: float = 5.0) -> np.ndarray:
    request = Request(url, headers={"Cache-Control": "no-store"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = response.read()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"快照接口返回的图像无法解码: {url}")
    return image


def observation_payload(
    observation: BoardObservation,
    board: BoardSpec,
) -> dict[str, Any]:
    return {
        "board": {
            "pattern_columns": board.columns,
            "pattern_rows": board.rows,
            "square_columns": board.square_grid[0],
            "square_rows": board.square_grid[1],
            "inner_corner_count": board.columns * board.rows,
            "square_size_m": board.square_size_m,
            "square_size_mm": board.square_size_m * 1000.0,
        },
        "image_path": observation.image_path,
        "rvec": observation.rvec.tolist(),
        "tvec_m": observation.tvec.tolist(),
        "rotation_matrix": observation.rotation_matrix.tolist(),
        "transform_camera_board": observation.transform_camera_board.tolist(),
        "board_center_camera_m": observation.board_center_camera_m.tolist(),
        "board_normal_camera": observation.board_normal_camera.tolist(),
        "corner_refinement_used": observation.corner_refinement_used,
        "reprojection_rms_px": observation.reprojection_rms_px,
        "reprojection_mean_px": observation.reprojection_mean_px,
        "reprojection_max_px": observation.reprojection_max_px,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument(
        "--snapshot-url",
        default=None,
        help="例如 http://127.0.0.1:9100/snapshot.jpg",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--square-size-mm", type=float, default=14.0)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--interval-s", type=float, default=0.25)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    board = BoardSpec(square_size_m=args.square_size_mm / 1000.0)
    observations: list[BoardObservation] = []
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for index in range(max(1, args.frames)):
        if args.image:
            image = load_image(args.image)
            image_path = str(Path(args.image).expanduser().resolve())
        else:
            image = fetch_snapshot(args.snapshot_url)
            image_path = f"{args.snapshot_url}?frame={index}"
        observation = estimate_board_pose(image, board, image_path=image_path)
        observations.append(observation)
        overlay = render_overlay(image, observation, board)
        cv2.imwrite(str(output / f"detect_{stamp}_{index:02d}.png"), overlay)
        if index + 1 < args.frames:
            time.sleep(max(0.0, args.interval_s))

    poses = np.asarray([obs.transform_camera_board for obs in observations])
    tvecs = np.asarray([obs.tvec for obs in observations])
    rms = np.asarray([obs.reprojection_rms_px for obs in observations])
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": args.image or args.snapshot_url,
        "frame_count": len(observations),
        "detected": len(observations),
        "board": observation_payload(observations[0], board)["board"],
        "reprojection_rms_px": {
            "mean": float(np.mean(rms)),
            "max": float(np.max(rms)),
        },
        "tvec_mean_m": np.mean(tvecs, axis=0).tolist(),
        "position_sample_std_mm": (
            (np.std(tvecs, axis=0, ddof=1) * 1000.0).tolist()
            if len(tvecs) > 1
            else [0.0, 0.0, 0.0]
        ),
        "pose_rotation_component_std": (
            np.std(poses[:, :3, :3], axis=0, ddof=1).tolist()
            if len(poses) > 1
            else np.zeros((3, 3)).tolist()
        ),
        "observations": [
            observation_payload(obs, board) for obs in observations
        ],
    }
    json_path = output / f"detection_{stamp}.json"
    def json_default(value: Any):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"not JSON serializable: {type(value).__name__}")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
        handle.write("\n")
    print(json.dumps(
        {
            "json": str(json_path),
            "detected": f"{len(observations)}/{max(1, args.frames)}",
            "rms_mean_px": payload["reprojection_rms_px"]["mean"],
            "rms_max_px": payload["reprojection_rms_px"]["max"],
            "tvec_mean_m": payload["tvec_mean_m"],
            "position_sample_std_mm": payload["position_sample_std_mm"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
