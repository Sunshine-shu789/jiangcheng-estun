"""Fail-closed box pose fitting and dynamic five-point generation.

The box centre and yaw are never configured in the dynamic five-point mode.
They are fitted from four ordered RGB-D corners in one captured frame after
the corners have been transformed through the calibrated eye-in-hand chain.
This module intentionally has no ROS or hardware dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, degrees, isfinite
from typing import Sequence

import numpy as np

from .collision import BoxClearanceConfig
from .config import BoxRelativeFivePointConfig
from .domain import (
    BOX_CORNER_NAMES,
    BOX_POINT_NAMES,
    BoxPoint,
    BoxPointLayout,
    BoxPoseEstimate,
    CameraIntrinsics,
    Detection,
    Frame,
)
from .kinematics import validate_rotation_matrix


class BoxPoseError(ValueError):
    """A box-pose observation failed a safety or geometry gate."""


@dataclass(frozen=True)
class BoxPointMatch:
    """Nearest fixed box point for the current item centre."""

    point: BoxPoint
    point_index: int
    item_local_xy_m: tuple[float, float]
    distance_m: float
    second_distance_m: float
    margin_m: float


_BOX_POINT_SIGNS = {
    "corner_nw": (-1.0, 1.0),
    "corner_ne": (1.0, 1.0),
    "corner_se": (1.0, -1.0),
    "corner_sw": (-1.0, -1.0),
    "center": (0.0, 0.0),
}


def _finite_vector(value: Sequence[float], length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,):
        raise BoxPoseError(f"{name} must contain exactly {length} values")
    if not np.all(np.isfinite(vector)):
        raise BoxPoseError(f"{name} contains non-finite values")
    return vector


def _as_transform(matrix: Sequence[Sequence[float]], name: str) -> np.ndarray:
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise BoxPoseError(f"{name} must be a finite 4x4 matrix")
    validate_rotation_matrix(transform[:3, :3], name=f"{name} rotation")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise BoxPoseError(f"{name} must be a homogeneous transform")
    return transform


def _sample_corner_depth_std(
    depth_m: object,
    uv_px: Sequence[float],
    *,
    radius_px: int = 2,
) -> float:
    """Return robust local depth spread around one pixel corner."""

    if depth_m is None:
        raise BoxPoseError("RGB-D box fitting requires frame depth_m")
    uv = _finite_vector(uv_px, 2, "box corner UV")
    values = np.asarray(depth_m, dtype=float)
    if values.ndim == 0:
        scalar = float(values)
        if not isfinite(scalar) or scalar <= 0.0:
            raise BoxPoseError("frame depth_m scalar must be positive")
        return 0.0
    if values.ndim != 2:
        raise BoxPoseError("frame depth_m must be a scalar or a 2-D image")
    row = int(round(float(uv[1])))
    column = int(round(float(uv[0])))
    if row < 0 or column < 0 or row >= values.shape[0] or column >= values.shape[1]:
        raise BoxPoseError("box corner UV lies outside frame depth_m")
    radius = max(0, int(radius_px))
    patch = values[
        max(0, row - radius) : min(values.shape[0], row + radius + 1),
        max(0, column - radius) : min(values.shape[1], column + radius + 1),
    ]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size < 3:
        raise BoxPoseError(
            "box corner depth patch has fewer than three positive samples"
        )
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    # Reject mixed foreground/background patches instead of accepting a
    # corner whose depth is visually ambiguous.
    return max(float(np.std(valid)), 1.4826 * mad)


def _corner_camera_points(
    frame: Frame,
    *,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, float]:
    raw_xyz = tuple(frame.box_corner_camera_xyz_m)
    raw_uv = tuple(frame.box_corner_uv_px)
    if len(raw_xyz) != 4 and len(raw_uv) != 4:
        raise BoxPoseError(
            "box fitting requires four ordered camera XYZ corners or four UV corners"
        )
    if raw_uv and len(raw_uv) != 4:
        raise BoxPoseError("box_corner_uv_px must contain exactly four ordered corners")

    corner_std: list[float] = []
    if raw_xyz:
        if len(raw_xyz) != 4:
            raise BoxPoseError(
                "box_corner_camera_xyz_m must contain exactly four ordered corners"
            )
        camera_points = np.asarray(
            [
                _finite_vector(point, 3, f"box corner {index} camera XYZ")
                for index, point in enumerate(raw_xyz)
            ],
            dtype=float,
        )
        if raw_uv:
            corner_std = [
                _sample_corner_depth_std(frame.depth_m, uv)
                for uv in raw_uv
            ]
    else:
        if frame.depth_m is None:
            raise BoxPoseError("box fitting from UV requires frame depth_m")
        depth_values = np.asarray(frame.depth_m, dtype=float)
        camera_points_list: list[np.ndarray] = []
        for index, uv in enumerate(raw_uv):
            uv_vector = _finite_vector(uv, 2, f"box corner {index} UV")
            row = int(round(float(uv_vector[1])))
            column = int(round(float(uv_vector[0])))
            if (
                row < 0
                or column < 0
                or (
                    depth_values.ndim == 2
                    and (row >= depth_values.shape[0] or column >= depth_values.shape[1])
                )
            ):
                raise BoxPoseError(f"box corner {index} UV lies outside depth image")
            if depth_values.ndim == 2:
                radius = 2
                patch = depth_values[
                    max(0, row - radius) : min(depth_values.shape[0], row + radius + 1),
                    max(0, column - radius) : min(depth_values.shape[1], column + radius + 1),
                ]
                valid = patch[np.isfinite(patch) & (patch > 0.0)]
                if valid.size < 3:
                    raise BoxPoseError(
                        f"box corner {index} depth patch has too few valid samples"
                    )
                depth = float(np.median(valid))
                std = max(
                    float(np.std(valid)),
                    1.4826 * float(np.median(np.abs(valid - depth))),
                )
            else:
                depth = float(depth_values)
                std = 0.0
            if not isfinite(depth) or depth <= 0.0:
                raise BoxPoseError(f"box corner {index} depth must be positive")
            x_camera = (float(uv_vector[0]) - float(intrinsics.cx)) * depth / float(
                intrinsics.fx
            )
            y_camera = (float(uv_vector[1]) - float(intrinsics.cy)) * depth / float(
                intrinsics.fy
            )
            camera_points_list.append(np.array((x_camera, y_camera, depth)))
            corner_std.append(std)
        camera_points = np.asarray(camera_points_list, dtype=float)

    if camera_points.shape != (4, 3):
        raise BoxPoseError("box corners must form a 4x3 camera-point array")
    if not np.all(np.isfinite(camera_points)):
        raise BoxPoseError("box corner camera points contain non-finite values")
    if not raw_uv:
        raise BoxPoseError(
            "box fitting requires ordered UV corners for reprojection quality"
        )
    return camera_points, max(corner_std, default=0.0)


def _project_camera_points(
    camera_points: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    z = camera_points[:, 2]
    if np.any(z <= 1e-9):
        raise BoxPoseError("box corner has non-positive camera depth")
    return np.column_stack(
        (
            float(intrinsics.fx) * camera_points[:, 0] / z + float(intrinsics.cx),
            float(intrinsics.fy) * camera_points[:, 1] / z + float(intrinsics.cy),
        )
    )


def _validate_ordered_corner_geometry(
    base_points: np.ndarray,
    *,
    expected_size_m: Sequence[float],
    config: BoxRelativeFivePointConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    expected = _finite_vector(expected_size_m, 2, "expected interior size")
    if np.any(expected <= 0.0):
        raise BoxPoseError("expected interior size must be positive")

    width_edges = (
        base_points[1] - base_points[0],
        base_points[2] - base_points[3],
    )
    depth_edges = (
        base_points[0] - base_points[3],
        base_points[1] - base_points[2],
    )
    x_axis = np.mean(width_edges, axis=0)
    y_axis = np.mean(depth_edges, axis=0)
    x_norm = float(np.linalg.norm(x_axis))
    y_norm = float(np.linalg.norm(y_axis))
    if x_norm <= 1e-9 or y_norm <= 1e-9:
        raise BoxPoseError("box corners do not define two usable edges")
    x_axis = x_axis / x_norm
    y_axis = y_axis / y_norm
    orthogonality_error_deg = abs(90.0 - degrees(float(np.arccos(
        np.clip(float(np.dot(x_axis, y_axis)), -1.0, 1.0)
    ))))
    if orthogonality_error_deg > float(config.max_corner_orthogonality_error_deg):
        raise BoxPoseError(
            "box corner axes are not orthogonal: "
            f"{orthogonality_error_deg:.3f} deg exceeds "
            f"{config.max_corner_orthogonality_error_deg:.3f} deg"
        )

    y_axis = y_axis - x_axis * float(np.dot(x_axis, y_axis))
    y_axis_norm = float(np.linalg.norm(y_axis))
    if y_axis_norm <= 1e-9:
        raise BoxPoseError("box corner Y axis collapsed during orthonormalisation")
    y_axis = y_axis / y_axis_norm
    normal = np.cross(x_axis, y_axis)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        raise BoxPoseError("box corner normal is degenerate")
    normal = normal / normal_norm
    if normal[2] < 0.0:
        normal = -normal
        y_axis = -y_axis

    width_m = float(np.mean([np.linalg.norm(edge) for edge in width_edges]))
    depth_m = float(np.mean([np.linalg.norm(edge) for edge in depth_edges]))
    width_error_m = abs(width_m - float(expected[0]))
    depth_error_m = abs(depth_m - float(expected[1]))
    if width_error_m > float(config.max_corner_edge_error_m):
        raise BoxPoseError(
            f"box width {width_m:.4f} m differs from expected "
            f"{float(expected[0]):.4f} m by {width_error_m:.4f} m"
        )
    if depth_error_m > float(config.max_corner_edge_error_m):
        raise BoxPoseError(
            f"box depth {depth_m:.4f} m differs from expected "
            f"{float(expected[1]):.4f} m by {depth_error_m:.4f} m"
        )
    return (
        x_axis,
        y_axis,
        normal,
        base_points,
        width_m,
        depth_m,
        width_error_m,
        depth_error_m,
    )


def fit_box_pose_from_frame(
    frame: Frame,
    *,
    base_camera_transform: Sequence[Sequence[float]],
    expected_interior_size_m: Sequence[float],
    config: BoxRelativeFivePointConfig,
) -> BoxPoseEstimate:
    """Fit centre and yaw from the four ordered corners in one exact frame."""

    if frame.camera_intrinsics is None:
        raise BoxPoseError("RGB-D box fitting requires camera_intrinsics")
    intrinsics = frame.camera_intrinsics
    base_camera = _as_transform(base_camera_transform, "base_camera_transform")
    camera_points, corner_depth_std_m = _corner_camera_points(
        frame,
        intrinsics=intrinsics,
    )
    if corner_depth_std_m > float(config.max_corner_depth_std_m):
        raise BoxPoseError(
            f"box corner depth std {corner_depth_std_m:.4f} m exceeds "
            f"{config.max_corner_depth_std_m:.4f} m"
        )

    homogeneous = np.column_stack((camera_points, np.ones(4)))
    base_points = (base_camera @ homogeneous.T).T[:, :3]
    if np.any(base_points[:, 2] <= 0.0):
        raise BoxPoseError("box corners transform below the robot base XY plane")
    centroid = np.mean(base_points, axis=0)
    centered = base_points - centroid
    try:
        _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise BoxPoseError(f"box corner plane fit failed: {exc}") from exc
    if singular_values.shape != (3,) or singular_values[1] <= 1e-9:
        raise BoxPoseError("box corner plane fit is degenerate")
    plane_normal = vh[-1].copy()
    if plane_normal[2] < 0.0:
        plane_normal = -plane_normal
    plane_rms_m = float(
        np.sqrt(np.mean(np.square(centered @ plane_normal)))
    )
    if plane_rms_m > float(config.max_corner_plane_rms_m):
        raise BoxPoseError(
            f"box corner plane RMS {plane_rms_m:.4f} m exceeds "
            f"{config.max_corner_plane_rms_m:.4f} m"
        )

    (
        x_axis,
        y_axis,
        fitted_normal,
        _points,
        width_m,
        depth_m,
        width_error_m,
        depth_error_m,
    ) = _validate_ordered_corner_geometry(
        base_points,
        expected_size_m=expected_interior_size_m,
        config=config,
    )
    normal_z = float(fitted_normal[2])
    if normal_z < float(config.min_plane_normal_z):
        raise BoxPoseError(
            f"box plane normal Z {normal_z:.4f} is below "
            f"{config.min_plane_normal_z:.4f}"
        )

    projected_uv = _project_camera_points(camera_points, intrinsics)
    observed_uv = np.asarray(frame.box_corner_uv_px, dtype=float)
    if observed_uv.shape != (4, 2) or not np.all(np.isfinite(observed_uv)):
        raise BoxPoseError("box_corner_uv_px must be a finite 4x2 array")
    reprojection_rms_px = float(
        np.sqrt(np.mean(np.sum(np.square(projected_uv - observed_uv), axis=1)))
    )
    if reprojection_rms_px > float(config.max_corner_reprojection_rms_px):
        raise BoxPoseError(
            f"box corner reprojection RMS {reprojection_rms_px:.3f} px exceeds "
            f"{config.max_corner_reprojection_rms_px:.3f} px"
        )

    ratios = (
        plane_rms_m / float(config.max_corner_plane_rms_m),
        width_error_m / float(config.max_corner_edge_error_m),
        depth_error_m / float(config.max_corner_edge_error_m),
        orthogonality_error_deg
        / float(config.max_corner_orthogonality_error_deg),
        reprojection_rms_px / float(config.max_corner_reprojection_rms_px),
        corner_depth_std_m / float(config.max_corner_depth_std_m),
    )
    quality_score = float(np.clip(1.0 - max(ratios), 0.0, 1.0))
    center_base = tuple(float(value) for value in centroid)
    center_homogeneous = np.linalg.inv(base_camera) @ np.array(
        (*center_base, 1.0),
        dtype=float,
    )
    center_camera = center_homogeneous[:3]
    return BoxPoseEstimate(
        frame_id=frame.frame_id,
        corner_names=BOX_CORNER_NAMES,
        center_camera_xyz_m=tuple(float(value) for value in center_camera),
        center_base_xyz_m=center_base,
        yaw_rad=atan2(float(x_axis[1]), float(x_axis[0])),
        x_axis_base=tuple(float(value) for value in x_axis),
        y_axis_base=tuple(float(value) for value in y_axis),
        normal_base=tuple(float(value) for value in fitted_normal),
        width_m=width_m,
        depth_m=depth_m,
        plane_rms_m=plane_rms_m,
        width_error_m=width_error_m,
        depth_error_m=depth_error_m,
        orthogonality_error_deg=orthogonality_error_deg,
        reprojection_rms_px=reprojection_rms_px,
        corner_depth_std_m=corner_depth_std_m,
        normal_z=normal_z,
        quality_score=quality_score,
        corner_camera_xyz_m=tuple(
            tuple(float(value) for value in point) for point in camera_points
        ),
        corner_base_xyz_m=tuple(
            tuple(float(value) for value in point) for point in base_points
        ),
    )


def build_box_point_layout(
    interior_size_m: Sequence[float],
    *,
    gripper_max_outer_m: float,
    wall_clearance_m: float,
) -> BoxPointLayout:
    """Build the fixed five-point layout in the fitted box-local frame."""

    size = _finite_vector(interior_size_m, 2, "box interior size")
    half_extent = float(gripper_max_outer_m) * 0.5
    clearance = float(wall_clearance_m)
    if np.any(size <= 0.0):
        raise BoxPoseError("box interior size must be positive")
    if not isfinite(half_extent) or half_extent <= 0.0:
        raise BoxPoseError("gripper maximum outer size must be positive")
    if not isfinite(clearance) or clearance < 0.0:
        raise BoxPoseError("wall clearance must be finite and non-negative")
    x = float(size[0]) * 0.5 - half_extent - clearance
    y = float(size[1]) * 0.5 - half_extent - clearance
    if x <= 0.0 or y <= 0.0:
        raise BoxPoseError(
            "gripper envelope and wall clearance leave no usable box points"
        )
    points = tuple(
        BoxPoint(
            name=name,
            local_xy_m=(float(_BOX_POINT_SIGNS[name][0] * x), float(
                _BOX_POINT_SIGNS[name][1] * y
            )),
        )
        for name in BOX_POINT_NAMES
    )
    return BoxPointLayout(
        points=points,
        gripper_half_extent_m=half_extent,
        wall_clearance_m=clearance,
    )


def box_local_xy_to_base(
    local_xy_m: Sequence[float],
    pose: BoxPoseEstimate,
) -> tuple[float, float]:
    local = _finite_vector(local_xy_m, 2, "box-local point")
    cosine = float(np.cos(pose.yaw_rad))
    sine = float(np.sin(pose.yaw_rad))
    rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=float)
    base = rotation @ local + np.asarray(pose.center_base_xy_m, dtype=float)
    return (float(base[0]), float(base[1]))


def base_xy_to_box_local(
    base_xy_m: Sequence[float],
    pose: BoxPoseEstimate,
) -> tuple[float, float]:
    base = _finite_vector(base_xy_m, 2, "base XY point")
    cosine = float(np.cos(pose.yaw_rad))
    sine = float(np.sin(pose.yaw_rad))
    rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=float)
    local = rotation.T @ (
        base - np.asarray(pose.center_base_xy_m, dtype=float)
    )
    return (float(local[0]), float(local[1]))


def detection_center_base_xyz(
    detection: Detection,
    *,
    base_camera_transform: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    if detection.object_center_camera_xyz is None:
        raise BoxPoseError("item detection has no object_center_camera_xyz")
    base_camera = _as_transform(base_camera_transform, "base_camera_transform")
    point = _finite_vector(
        detection.object_center_camera_xyz,
        3,
        "item object_center_camera_xyz",
    )
    transformed = base_camera[:3, :3] @ point + base_camera[:3, 3]
    return tuple(float(value) for value in transformed)


def match_item_to_box_point(
    item_center_base_xyz_m: Sequence[float],
    *,
    pose: BoxPoseEstimate,
    layout: BoxPointLayout,
    config: BoxRelativeFivePointConfig,
) -> BoxPointMatch:
    """Map the current item centre to one unambiguous fixed box point."""

    item_base = _finite_vector(item_center_base_xyz_m, 3, "item centre")
    item_local = base_xy_to_box_local(item_base[:2], pose)
    distances = [
        float(
            np.linalg.norm(
                np.asarray(item_local, dtype=float)
                - np.asarray(point.local_xy_m, dtype=float)
            )
        )
        for point in layout.points
    ]
    order = np.argsort(np.asarray(distances, dtype=float))
    nearest_index = int(order[0])
    nearest_distance = float(distances[nearest_index])
    second_distance = float(distances[int(order[1])]) if len(order) > 1 else float("inf")
    margin = second_distance - nearest_distance
    if nearest_distance > float(config.item_match_max_distance_m):
        raise BoxPoseError(
            f"item is {nearest_distance:.4f} m from the nearest box point, "
            f"above {config.item_match_max_distance_m:.4f} m"
        )
    if margin < float(config.item_match_min_margin_m):
        raise BoxPoseError(
            f"item-to-point match is ambiguous: nearest={nearest_distance:.4f} m, "
            f"second={second_distance:.4f} m, margin={margin:.4f} m"
        )
    return BoxPointMatch(
        point=layout.points[nearest_index],
        point_index=nearest_index,
        item_local_xy_m=item_local,
        distance_m=nearest_distance,
        second_distance_m=second_distance,
        margin_m=margin,
    )


def dynamic_box_clearance(
    template: BoxClearanceConfig,
    pose: BoxPoseEstimate,
) -> BoxClearanceConfig:
    """Create the per-frame clearance config from the fitted box pose."""

    return replace(
        template,
        center_xy_m=pose.center_base_xy_m,
        yaw_rad=float(pose.yaw_rad),
    )
