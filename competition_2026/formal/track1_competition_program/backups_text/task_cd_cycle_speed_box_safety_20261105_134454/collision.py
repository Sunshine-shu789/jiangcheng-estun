"""Conservative planar box-clearance checks for adaptive grasps.

The model is intentionally two-dimensional: the gripper footprint is projected
onto the robot base XY plane for the complete candidate orientation, then
checked against the inner box rectangle and every configured reinforcement
polygon.  This is a fast pre-IK rejection gate, not a replacement for full
six-dimensional collision checking or field calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


GEOMETRY_SOURCE_UNCONFIGURED = "unconfigured"
GEOMETRY_SOURCE_FIELD_MEASURED = "field_measured"
GEOMETRY_SOURCE_OFFLINE_MOCK_SYNTHETIC = "offline_mock_synthetic_test_only"
_GEOMETRY_SOURCES = {
    GEOMETRY_SOURCE_UNCONFIGURED,
    GEOMETRY_SOURCE_FIELD_MEASURED,
    GEOMETRY_SOURCE_OFFLINE_MOCK_SYNTHETIC,
}
_GEOMETRY_EPS = 1e-9


def _pair(value: Any, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ValueError(f"{field_name} must be a 2-value vector")
    return (float(value[0]), float(value[1]))


def _points(value: Any, field_name: str) -> tuple[tuple[float, float], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of XY points")
    result: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        pair = _pair(point, f"{field_name}[{index}]")
        assert pair is not None
        result.append(pair)
    return tuple(result)


@dataclass(frozen=True)
class BoxObstacleConfig:
    """One convex obstacle polygon in box-local XY metres."""

    name: str
    vertices_m: tuple[tuple[float, float], ...]

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | Sequence[Any], *, index: int
    ) -> "BoxObstacleConfig":
        if isinstance(data, Mapping):
            name = str(data.get("name", f"obstacle_{index}")).strip()
            raw_vertices = data.get(
                "vertices_m",
                data.get("vertices_xy_m", data.get("vertices")),
            )
        else:
            name = f"obstacle_{index}"
            raw_vertices = data
        if not name:
            raise ValueError(
                f"adaptive_inverse.box_clearance.obstacles_m[{index}].name "
                "must not be empty"
            )
        return cls(
            name=name,
            vertices_m=_points(
                raw_vertices,
                f"adaptive_inverse.box_clearance.obstacles_m[{index}]."
                "vertices_m",
            ),
        )


@dataclass(frozen=True)
class BoxClearanceConfig:
    """Configured inner-box geometry and gripper flange footprint.

    ``center_xy_m`` and ``yaw_rad`` place the box-local frame in the robot base
    frame.  ``interior_size_m`` is the full inner width/depth before clearance
    shrinking.  Obstacles are convex polygons in the same box-local frame.
    ``geometry_validated`` is a claim that the geometry source is complete and
    reviewed.  Real hardware additionally requires the source to be
    ``field_measured``; synthetic values are accepted only by the offline Mock
    chain.
    """

    required: bool = False
    geometry_validated: bool = False
    geometry_source: str = GEOMETRY_SOURCE_UNCONFIGURED
    center_xy_m: tuple[float, float] | None = None
    yaw_rad: float = 0.0
    interior_size_m: tuple[float, float] | None = None
    wall_clearance_m: float = 0.0
    obstacle_clearance_m: float = 0.0
    gripper_footprint_flange_xy_m: tuple[
        tuple[float, float], ...
    ] = ()
    obstacles_m: tuple[BoxObstacleConfig, ...] = ()

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None
    ) -> "BoxClearanceConfig":
        data = dict(data or {})
        raw_obstacles = data.get("obstacles_m", data.get("obstacles", ()))
        if raw_obstacles is None:
            raw_obstacles = ()
        if (
            isinstance(raw_obstacles, (str, bytes))
            or not isinstance(raw_obstacles, Sequence)
        ):
            raise ValueError(
                "adaptive_inverse.box_clearance.obstacles_m must be a sequence"
            )
        obstacles = tuple(
            BoxObstacleConfig.from_dict(item, index=index)
            for index, item in enumerate(raw_obstacles)
        )
        return cls(
            required=bool(data.get("required", False)),
            geometry_validated=bool(data.get("geometry_validated", False)),
            geometry_source=str(
                data.get("geometry_source", GEOMETRY_SOURCE_UNCONFIGURED)
            ).strip(),
            center_xy_m=_pair(
                data.get("center_xy_m"),
                "adaptive_inverse.box_clearance.center_xy_m",
            ),
            yaw_rad=float(data.get("yaw_rad", 0.0)),
            interior_size_m=_pair(
                data.get("interior_size_m"),
                "adaptive_inverse.box_clearance.interior_size_m",
            ),
            wall_clearance_m=float(data.get("wall_clearance_m", 0.0)),
            obstacle_clearance_m=float(
                data.get("obstacle_clearance_m", 0.0)
            ),
            gripper_footprint_flange_xy_m=_points(
                data.get("gripper_footprint_flange_xy_m"),
                "adaptive_inverse.box_clearance."
                "gripper_footprint_flange_xy_m",
            ),
            obstacles_m=obstacles,
        )

    def has_configured_geometry(self) -> bool:
        return bool(
            self.center_xy_m is not None
            or self.interior_size_m is not None
            or self.gripper_footprint_flange_xy_m
            or self.obstacles_m
            or self.geometry_source != GEOMETRY_SOURCE_UNCONFIGURED
            or self.geometry_validated
        )


@dataclass(frozen=True)
class BoxClearanceResult:
    allowed: bool
    reason: str
    footprint_base_xy_m: tuple[tuple[float, float], ...] = ()
    min_wall_clearance_m: float | None = None
    min_obstacle_clearance_m: float | None = None
    limiting_obstacle: str = ""


def _as_polygon(
    points: Sequence[Sequence[float]],
    field_name: str,
) -> np.ndarray:
    polygon = np.asarray(points, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError(f"{field_name} must contain at least three XY points")
    if not np.all(np.isfinite(polygon)):
        raise ValueError(f"{field_name} must contain only finite values")
    if np.any(
        np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        <= _GEOMETRY_EPS
    ):
        raise ValueError(f"{field_name} must not contain duplicate vertices")
    area2 = float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
    )
    if abs(area2) <= _GEOMETRY_EPS:
        raise ValueError(f"{field_name} is degenerate")
    crosses = []
    count = len(polygon)
    for index in range(count):
        first = polygon[(index + 1) % count] - polygon[index]
        second = polygon[(index + 2) % count] - polygon[(index + 1) % count]
        crosses.append(float(first[0] * second[1] - first[1] * second[0]))
    crosses_array = np.asarray(crosses, dtype=float)
    positive = crosses_array > _GEOMETRY_EPS
    negative = crosses_array < -_GEOMETRY_EPS
    if not (bool(np.all(positive)) or bool(np.all(negative))):
        raise ValueError(f"{field_name} must be strictly convex and non-self-intersecting")
    for first in range(count):
        for second in range(first + 1, count):
            adjacent = second == first + 1 or (first == 0 and second == count - 1)
            if adjacent:
                continue
            if _segments_intersect(
                polygon[first],
                polygon[(first + 1) % count],
                polygon[second],
                polygon[(second + 1) % count],
            ):
                raise ValueError(f"{field_name} must be non-self-intersecting")
    return polygon


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def _on_segment(a: np.ndarray, b: np.ndarray, point: np.ndarray) -> bool:
    return bool(
        min(a[0], b[0]) - _GEOMETRY_EPS
        <= point[0]
        <= max(a[0], b[0]) + _GEOMETRY_EPS
        and min(a[1], b[1]) - _GEOMETRY_EPS
        <= point[1]
        <= max(a[1], b[1]) + _GEOMETRY_EPS
        and abs(_orientation(a, b, point)) <= _GEOMETRY_EPS
    )


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    first_a = _orientation(first_start, first_end, second_start)
    first_b = _orientation(first_start, first_end, second_end)
    second_a = _orientation(second_start, second_end, first_start)
    second_b = _orientation(second_start, second_end, first_end)
    if first_a * first_b < 0.0 and second_a * second_b < 0.0:
        return True
    return bool(
        (abs(first_a) <= _GEOMETRY_EPS and _on_segment(first_start, first_end, second_start))
        or (abs(first_b) <= _GEOMETRY_EPS and _on_segment(first_start, first_end, second_end))
        or (abs(second_a) <= _GEOMETRY_EPS and _on_segment(second_start, second_end, first_start))
        or (abs(second_b) <= _GEOMETRY_EPS and _on_segment(second_start, second_end, first_end))
    )


def _point_in_convex_polygon(
    point: np.ndarray,
    polygon: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> bool:
    signed_area = 0.0
    count = len(polygon)
    for index in range(count):
        first = polygon[index]
        second = polygon[(index + 1) % count]
        signed_area += float(first[0] * second[1] - first[1] * second[0])
    orientation = 1.0 if signed_area >= 0.0 else -1.0
    for index in range(count):
        if (
            orientation
            * _orientation(
                polygon[index],
                polygon[(index + 1) % count],
                point,
            )
            < -tolerance
        ):
            return False
    return True


def _point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= _GEOMETRY_EPS:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, segment) / length_squared)
    fraction = min(1.0, max(0.0, fraction))
    nearest = start + fraction * segment
    return float(np.linalg.norm(point - nearest))


def _polygon_boundary_distance(
    point: np.ndarray,
    polygon: np.ndarray,
) -> float:
    return min(
        _point_segment_distance(
            point,
            polygon[index],
            polygon[(index + 1) % len(polygon)],
        )
        for index in range(len(polygon))
    )


def _vertices_to_polygon_edges_distance(
    vertices: np.ndarray,
    polygon: np.ndarray,
) -> float:
    return min(
        _point_segment_distance(
            vertices[vertex_index],
            polygon[edge_index],
            polygon[(edge_index + 1) % len(polygon)],
        )
        for vertex_index in range(len(vertices))
        for edge_index in range(len(polygon))
    )


def _convex_polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    if any(
        _point_in_convex_polygon(point, second)
        for point in first
    ) or any(
        _point_in_convex_polygon(point, first)
        for point in second
    ):
        return 0.0
    if any(
        _segments_intersect(
            first[first_index],
            first[(first_index + 1) % len(first)],
            second[second_index],
            second[(second_index + 1) % len(second)],
        )
        for first_index in range(len(first))
        for second_index in range(len(second))
    ):
        return 0.0
    # The closest pair between disjoint convex polygons can have a vertex of
    # either polygon against an edge of the other.  Both directions are needed;
    # checking only first-vertices -> second-edges can seriously overestimate
    # clearance for long, thin footprints near a corner-block vertex.
    return min(
        _vertices_to_polygon_edges_distance(first, second),
        _vertices_to_polygon_edges_distance(second, first),
    )


def _transform_box_points(
    points: np.ndarray,
    center_xy_m: tuple[float, float],
    yaw_rad: float,
) -> np.ndarray:
    cosine = float(np.cos(yaw_rad))
    sine = float(np.sin(yaw_rad))
    rotation = np.array(
        ((cosine, -sine), (sine, cosine)),
        dtype=float,
    )
    return points @ rotation.T + np.asarray(center_xy_m, dtype=float)


def _rectangle(half_width_m: float, half_height_m: float) -> np.ndarray:
    return np.array(
        (
            (-half_width_m, -half_height_m),
            (half_width_m, -half_height_m),
            (half_width_m, half_height_m),
            (-half_width_m, half_height_m),
        ),
        dtype=float,
    )


def validate_box_clearance_geometry(
    config: BoxClearanceConfig,
    *,
    require_pose: bool = True,
    field_prefix: str = "adaptive_inverse.box_clearance",
) -> None:
    """Validate configured box geometry, or fail closed when it is required.

    Dynamic box-relative task modes intentionally leave ``center_xy_m`` and
    ``yaw_rad`` unset in the template because both values come from the
    current RGB-D frame.  ``require_pose=False`` validates every frame-local
    dimension and collision shape while deferring those two fields.
    """

    prefix = str(field_prefix).strip()
    if not prefix:
        raise ValueError("box-clearance field prefix must be non-empty")

    if not config.has_configured_geometry():
        if config.required:
            raise ValueError(
                f"{prefix} is required but complete box "
                "geometry and a validated gripper footprint are missing"
            )
        return
    if config.geometry_source not in _GEOMETRY_SOURCES:
        raise ValueError(
            f"{prefix}.geometry_source must be one of "
            f"{sorted(_GEOMETRY_SOURCES)}"
        )
    if config.required and not config.geometry_validated:
        raise ValueError(
            f"{prefix}.geometry_validated must be true "
            "before motion is allowed"
        )
    if config.required and config.geometry_source == GEOMETRY_SOURCE_UNCONFIGURED:
        raise ValueError(
            f"{prefix}.geometry_source must identify the "
            "measured or offline synthetic geometry"
        )
    if not config.geometry_validated:
        return
    if require_pose and config.center_xy_m is None:
        raise ValueError(f"{prefix}.center_xy_m is required")
    if config.interior_size_m is None:
        raise ValueError(f"{prefix}.interior_size_m is required")
    if not config.gripper_footprint_flange_xy_m:
        raise ValueError(
            f"{prefix}.gripper_footprint_flange_xy_m is required"
        )
    size = np.asarray(config.interior_size_m, dtype=float)
    if require_pose:
        assert config.center_xy_m is not None
        center = np.asarray(config.center_xy_m, dtype=float)
        if not np.all(np.isfinite(center)) or not isfinite(config.yaw_rad):
            raise ValueError(f"{prefix} center/yaw must be finite")
    if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ValueError(
            f"{prefix}.interior_size_m must be positive and finite"
        )
    if (
        not isfinite(config.wall_clearance_m)
        or config.wall_clearance_m < 0.0
        or not isfinite(config.obstacle_clearance_m)
        or config.obstacle_clearance_m < 0.0
    ):
        raise ValueError(
            f"{prefix} wall and obstacle clearances must be finite and >= 0"
        )
    half_width = float(size[0]) * 0.5
    half_height = float(size[1]) * 0.5
    if config.wall_clearance_m >= min(half_width, half_height):
        raise ValueError(
            f"{prefix}.wall_clearance_m leaves no usable interior"
        )
    _as_polygon(
        config.gripper_footprint_flange_xy_m,
        f"{prefix}.gripper_footprint_flange_xy_m",
    )
    for obstacle in config.obstacles_m:
        polygon = _as_polygon(
            obstacle.vertices_m,
            f"{prefix} obstacle {obstacle.name!r}",
        )
        relative = np.abs(polygon)
        if np.any(relative[:, 0] > half_width + _GEOMETRY_EPS) or np.any(
            relative[:, 1] > half_height + _GEOMETRY_EPS
        ):
            raise ValueError(
                f"{prefix} obstacle {obstacle.name!r} "
                "lies outside the box interior"
            )


def check_box_clearance(
    config: BoxClearanceConfig,
    *,
    flange_xy_m: Sequence[float],
    rotation: Sequence[Sequence[float]],
) -> BoxClearanceResult:
    """Check one flange XY/orientation against the configured inner box.

    The return value describes collisions.  Malformed or unvalidated required
    geometry raises ``ValueError`` instead of silently allowing motion.
    """

    if not config.required:
        return BoxClearanceResult(True, "box clearance is disabled")
    validate_box_clearance_geometry(config)
    flange_xy = np.asarray(flange_xy_m, dtype=float)
    if flange_xy.shape != (2,) or not np.all(np.isfinite(flange_xy)):
        raise ValueError("adaptive grasp flange XY must be a finite 2-vector")
    rotation_matrix = np.asarray(rotation, dtype=float)
    if rotation_matrix.shape != (3, 3) or not np.all(
        np.isfinite(rotation_matrix)
    ):
        raise ValueError("adaptive grasp rotation must be a finite 3x3 matrix")
    if not np.allclose(
        rotation_matrix.T @ rotation_matrix,
        np.eye(3),
        atol=1e-7,
        rtol=0.0,
    ) or float(np.linalg.det(rotation_matrix)) <= 0.0:
        raise ValueError("adaptive grasp rotation must be a proper orthonormal matrix")
    assert config.center_xy_m is not None
    assert config.interior_size_m is not None
    footprint = _as_polygon(
        config.gripper_footprint_flange_xy_m,
        "adaptive_inverse.box_clearance.gripper_footprint_flange_xy_m",
    )
    footprint_3d = np.column_stack((footprint, np.zeros(len(footprint))))
    projected = (rotation_matrix @ footprint_3d.T).T[:, :2]
    footprint_base = projected + flange_xy
    projected = _as_polygon(
        footprint_base,
        "projected gripper footprint",
    )

    half_width = float(config.interior_size_m[0]) * 0.5
    half_height = float(config.interior_size_m[1]) * 0.5
    interior = _transform_box_points(
        _rectangle(half_width, half_height),
        config.center_xy_m,
        config.yaw_rad,
    )
    inset_half_width = half_width - float(config.wall_clearance_m)
    inset_half_height = half_height - float(config.wall_clearance_m)
    inset = _transform_box_points(
        _rectangle(inset_half_width, inset_half_height),
        config.center_xy_m,
        config.yaw_rad,
    )
    min_wall_clearance = min(
        _polygon_boundary_distance(point, interior)
        for point in projected
    )
    if not all(
        _point_in_convex_polygon(point, inset)
        for point in projected
    ):
        return BoxClearanceResult(
            False,
            "gripper footprint violates the configured wall clearance "
            f"{float(config.wall_clearance_m):.6f} m",
            tuple(tuple(float(value) for value in point) for point in footprint_base),
            min_wall_clearance,
            None,
            "",
        )

    min_obstacle_clearance: float | None = None
    limiting_obstacle = ""
    for obstacle in config.obstacles_m:
        local_polygon = _as_polygon(
            obstacle.vertices_m,
            f"adaptive_inverse.box_clearance obstacle {obstacle.name!r}",
        )
        base_polygon = _transform_box_points(
            local_polygon,
            config.center_xy_m,
            config.yaw_rad,
        )
        distance = _convex_polygon_distance(projected, base_polygon)
        if (
            min_obstacle_clearance is None
            or distance < min_obstacle_clearance
        ):
            min_obstacle_clearance = distance
            limiting_obstacle = obstacle.name
        if distance <= _GEOMETRY_EPS or distance < (
            float(config.obstacle_clearance_m) - _GEOMETRY_EPS
        ):
            return BoxClearanceResult(
                False,
                "gripper footprint violates obstacle clearance "
                f"{float(config.obstacle_clearance_m):.6f} m to "
                f"{obstacle.name!r} (distance {distance:.6f} m)",
                tuple(
                    tuple(float(value) for value in point)
                    for point in footprint_base
                ),
                min_wall_clearance,
                min_obstacle_clearance,
                limiting_obstacle,
            )

    return BoxClearanceResult(
        True,
        "gripper footprint clears the configured box walls and obstacles",
        tuple(tuple(float(value) for value in point) for point in footprint_base),
        min_wall_clearance,
        min_obstacle_clearance,
        limiting_obstacle,
    )
