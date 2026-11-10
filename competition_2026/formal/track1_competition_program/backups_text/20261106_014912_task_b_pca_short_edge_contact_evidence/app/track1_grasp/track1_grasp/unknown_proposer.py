"""Class-agnostic RGB-D proposals for Task D unseen objects.

The V11 detector is a closed-set model.  It cannot manufacture an ``unknown``
observation for an object that is outside its forty trained classes.  This
module fills that gap conservatively by segmenting opaque foreground connected
components above a robustly fitted support plane.

The output is deliberately a proposal, not a grasp decision.  It supplies
enough geometry for the existing target policy and planner to reject weak
observations, while every accepted route still has to pass the normal IK,
workspace, collision and motion safety gates.  Transparent objects are not
claimed to be solved: missing depth splits or removes their foreground and the
proposal then fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import cv2
import numpy as np

from .config import UnknownObjectProposerConfig
from .domain import Detection


@dataclass(frozen=True)
class _PlaneModel:
    center: np.ndarray
    normal: np.ndarray
    inlier_ratio: float
    rms_m: float
    inliers: int
    candidates: int

    def signed_height(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.center) @ self.normal


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return np.asarray(vector, dtype=float) / norm


def _odd_kernel(value: int) -> int:
    size = max(0, int(value))
    if size <= 1:
        return 0
    return size if size % 2 == 1 else size + 1


class UnknownObjectProposer:
    """Generate depth-foreground proposals for the Task D unknown branch."""

    def __init__(self, config: UnknownObjectProposerConfig) -> None:
        self.config = config

    def propose(
        self,
        color: Any,
        depth_m: Any,
        intrinsics: Any,
        *,
        known_detections: Sequence[Detection] = (),
    ) -> tuple[Detection, ...]:
        """Return deterministic unknown-object proposals for one RGB-D frame."""

        if not self.config.enabled:
            return ()

        depth = np.asarray(depth_m, dtype=float)
        image = np.asarray(color)
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError("unknown proposer needs a non-empty 2-D depth image")
        if image.ndim != 3 or image.shape[:2] != depth.shape:
            raise ValueError("unknown proposer colour/depth shapes do not match")

        fx = float(getattr(intrinsics, "fx"))
        fy = float(getattr(intrinsics, "fy"))
        cx = float(getattr(intrinsics, "cx"))
        cy = float(getattr(intrinsics, "cy"))
        if not all(math.isfinite(value) for value in (fx, fy, cx, cy)):
            raise ValueError("unknown proposer intrinsics are not finite")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("unknown proposer focal lengths must be positive")

        valid = (
            np.isfinite(depth)
            & (depth >= float(self.config.min_depth_m))
            & (depth <= float(self.config.max_depth_m))
        )
        if int(np.count_nonzero(valid)) < int(self.config.plane_min_inliers):
            return ()

        plane = self._fit_support_plane(depth, valid, intrinsics)
        if plane is None:
            return ()

        height = self._signed_height_image(depth, valid, plane, fx, fy, cx, cy)
        foreground = (
            valid
            & (height >= float(self.config.min_object_height_m))
            & (height <= float(self.config.max_object_height_m))
        )
        foreground = self._morphology(foreground)
        if int(np.count_nonzero(foreground)) < int(self.config.min_area_px):
            return ()

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8, copy=False),
            connectivity=8,
        )
        proposals: list[Detection] = []
        for component_id in range(1, int(count)):
            proposal = self._component_detection(
                component_id=component_id,
                labels=labels,
                stats=stats,
                depth=depth,
                valid=valid,
                height=height,
                plane=plane,
                intrinsics=intrinsics,
                known_detections=known_detections,
            )
            if proposal is not None:
                proposals.append(proposal)

        proposals.sort(
            key=lambda detection: (
                float(detection.graspability),
                float(detection.confidence),
                float(detection.major_axis_length_m or 0.0)
                * float(detection.minor_axis_length_m or 0.0),
            ),
            reverse=True,
        )
        return tuple(proposals[: max(0, int(self.config.max_proposals))])

    def _fit_support_plane(
        self,
        depth: np.ndarray,
        valid: np.ndarray,
        intrinsics: Any,
    ) -> _PlaneModel | None:
        valid_v, valid_u = np.nonzero(valid)
        z = depth[valid_v, valid_u]
        x = (valid_u - float(intrinsics.cx)) / float(intrinsics.fx) * z
        y = (valid_v - float(intrinsics.cy)) / float(intrinsics.fy) * z
        points = np.column_stack((x, y, z)).astype(float, copy=False)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < int(self.config.plane_min_inliers):
            return None

        max_samples = max(3, int(self.config.plane_max_samples))
        if len(points) > max_samples:
            rng = np.random.default_rng(int(self.config.seed))
            sample_indices = rng.choice(len(points), max_samples, replace=False)
            fit_points = points[sample_indices]
        else:
            fit_points = points

        threshold = float(self.config.plane_ransac_threshold_m)
        best_mask: np.ndarray | None = None
        best_score: tuple[int, float] | None = None
        rng = np.random.default_rng(int(self.config.seed))
        iterations = max(1, int(self.config.plane_ransac_iterations))
        for iteration in range(iterations + 1):
            if iteration == 0:
                center = np.median(fit_points, axis=0)
                try:
                    _, _, vh = np.linalg.svd(
                        fit_points - center,
                        full_matrices=False,
                    )
                except np.linalg.LinAlgError:
                    continue
                normal = np.asarray(vh[-1], dtype=float)
            else:
                selected = fit_points[
                    rng.choice(len(fit_points), 3, replace=False)
                ]
                normal = np.cross(
                    selected[1] - selected[0],
                    selected[2] - selected[0],
                )
                center = np.mean(selected, axis=0)
            unit_normal = _unit(normal)
            if unit_normal is None:
                continue
            if abs(float(unit_normal[2])) < float(self.config.plane_min_normal_z):
                continue
            residual = np.abs((fit_points - center) @ unit_normal)
            mask = residual <= threshold
            inliers = int(np.count_nonzero(mask))
            if inliers < 3:
                continue
            score = (inliers, -float(np.sqrt(np.mean(residual[mask] ** 2))))
            if best_score is None or score > best_score:
                best_score = score
                best_mask = mask

        if best_mask is None:
            return None

        selected_points = fit_points[best_mask]
        normal = np.asarray((0.0, 0.0, 1.0), dtype=float)
        center = np.median(selected_points, axis=0)
        for _ in range(4):
            try:
                center = np.median(selected_points, axis=0)
                _, _, vh = np.linalg.svd(
                    selected_points - center,
                    full_matrices=False,
                )
            except np.linalg.LinAlgError:
                return None
            normal = np.asarray(vh[-1], dtype=float)
            residual = np.abs((fit_points - center) @ normal)
            next_mask = residual <= threshold
            if int(np.count_nonzero(next_mask)) < 3 or np.array_equal(
                next_mask, best_mask
            ):
                break
            best_mask = next_mask
            selected_points = fit_points[best_mask]

        unit_normal = _unit(normal)
        if unit_normal is None:
            return None
        # Point the normal towards the camera.  Objects resting on the support
        # then have a positive signed height instead of an orientation-dependent
        # negative value.
        if float(-unit_normal @ center) < 0.0:
            unit_normal = -unit_normal
        residual = np.abs((fit_points - center) @ unit_normal)
        inlier_mask = residual <= threshold
        inliers = int(np.count_nonzero(inlier_mask))
        candidates = int(len(fit_points))
        inlier_ratio = inliers / max(candidates, 1)
        if inliers < int(self.config.plane_min_inliers):
            return None
        if inlier_ratio < float(self.config.plane_min_inlier_ratio):
            return None
        rms = float(np.sqrt(np.mean(residual[inlier_mask] ** 2)))
        if not math.isfinite(rms) or rms > float(self.config.plane_max_rms_m):
            return None
        return _PlaneModel(
            center=center,
            normal=unit_normal,
            inlier_ratio=inlier_ratio,
            rms_m=rms,
            inliers=inliers,
            candidates=candidates,
        )

    @staticmethod
    def _signed_height_image(
        depth: np.ndarray,
        valid: np.ndarray,
        plane: _PlaneModel,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        height = np.full(depth.shape, np.nan, dtype=float)
        valid_v, valid_u = np.nonzero(valid)
        z = depth[valid_v, valid_u]
        x = (valid_u - cx) / fx * z
        y = (valid_v - cy) / fy * z
        points = np.column_stack((x, y, z))
        height[valid_v, valid_u] = plane.signed_height(points)
        return height

    def _morphology(self, mask: np.ndarray) -> np.ndarray:
        result = np.asarray(mask, dtype=np.uint8) * 255
        open_size = _odd_kernel(self.config.morph_open_px)
        if open_size:
            result = cv2.morphologyEx(
                result,
                cv2.MORPH_OPEN,
                np.ones((open_size, open_size), dtype=np.uint8),
            )
        close_size = _odd_kernel(self.config.morph_close_px)
        if close_size:
            result = cv2.morphologyEx(
                result,
                cv2.MORPH_CLOSE,
                np.ones((close_size, close_size), dtype=np.uint8),
            )
        return result > 0

    def _component_detection(
        self,
        *,
        component_id: int,
        labels: np.ndarray,
        stats: np.ndarray,
        depth: np.ndarray,
        valid: np.ndarray,
        height: np.ndarray,
        plane: _PlaneModel,
        intrinsics: Any,
        known_detections: Sequence[Detection],
    ) -> Detection | None:
        x0, y0, width, height_px, area = (
            int(stats[component_id, cv2.CC_STAT_LEFT]),
            int(stats[component_id, cv2.CC_STAT_TOP]),
            int(stats[component_id, cv2.CC_STAT_WIDTH]),
            int(stats[component_id, cv2.CC_STAT_HEIGHT]),
            int(stats[component_id, cv2.CC_STAT_AREA]),
        )
        image_h, image_w = depth.shape
        x1 = x0 + width
        y1 = y0 + height_px
        if area < int(self.config.min_area_px):
            return None
        if width < int(self.config.min_component_width_px):
            return None
        if height_px < int(self.config.min_component_height_px):
            return None
        if area > float(self.config.max_area_ratio) * float(depth.size):
            return None
        margin = int(self.config.edge_margin_px)
        if (
            margin > 0
            and (
                x0 <= margin
                or y0 <= margin
                or x1 >= image_w - margin
                or y1 >= image_h - margin
            )
        ):
            return None
        bbox = (float(x0), float(y0), float(x1), float(y1))
        if self._overlaps_known(bbox, known_detections):
            return None

        component = labels == component_id
        bbox_valid = valid[y0:y1, x0:x1]
        bbox_valid_ratio = float(np.count_nonzero(bbox_valid)) / max(
            int(bbox_valid.size), 1
        )
        component_depth = depth[component]
        component_height = height[component]
        finite_component = (
            np.isfinite(component_depth) & np.isfinite(component_height)
        )
        component_depth = component_depth[finite_component]
        component_height = component_height[finite_component]
        if len(component_depth) < int(self.config.min_depth_points):
            return None
        if bbox_valid_ratio < float(self.config.min_valid_ratio):
            return None

        median_height = float(np.median(component_height))
        if median_height < float(self.config.min_object_height_m):
            return None
        top_cut = float(np.percentile(component_height, 60.0))
        top_mask = component & (height >= top_cut)
        top_v, top_u = np.nonzero(top_mask)
        if len(top_v) < max(12, int(self.config.min_depth_points) // 5):
            return None

        z = depth[top_v, top_u]
        x = (top_u - float(intrinsics.cx)) / float(intrinsics.fx) * z
        y = (top_v - float(intrinsics.cy)) / float(intrinsics.fy) * z
        top_points = np.column_stack((x, y, z)).astype(float, copy=False)
        top_points = top_points[np.all(np.isfinite(top_points), axis=1)]
        if len(top_points) < max(12, int(self.config.min_depth_points) // 5):
            return None

        center_xyz = np.median(top_points, axis=0)
        median_u = float(np.median(top_u))
        median_v = float(np.median(top_v))
        anchor_index = int(
            np.argmin((top_u - median_u) ** 2 + (top_v - median_v) ** 2)
        )
        anchor_z = float(depth[top_v[anchor_index], top_u[anchor_index]])
        anchor_u = float(top_u[anchor_index])
        anchor_v = float(top_v[anchor_index])
        anchor_xyz = np.asarray(
            (
                (anchor_u - float(intrinsics.cx))
                / float(intrinsics.fx)
                * anchor_z,
                (anchor_v - float(intrinsics.cy))
                / float(intrinsics.fy)
                * anchor_z,
                anchor_z,
            ),
            dtype=float,
        )

        major_axis, minor_axis, major_length, minor_length = self._image_axes(
            top_u,
            top_v,
            top_points,
            intrinsics,
            center_xyz[2],
            width,
            height_px,
        )
        quality = self._quality(
            plane=plane,
            bbox_valid_ratio=bbox_valid_ratio,
            area=area,
            median_height=median_height,
        )
        if quality < float(self.config.min_pose_confidence):
            return None

        confidence = 0.20 + 0.80 * quality
        graspability = float(
            np.clip(
                quality * float(self.config.graspability_scale),
                0.0,
                1.0,
            )
        )
        return Detection(
            label=str(self.config.label),
            confidence=float(confidence),
            bbox=bbox,
            depth_m=max(float(anchor_z), 1e-6),
            camera_xyz=tuple(float(value) for value in anchor_xyz),
            object_center_camera_xyz=tuple(float(value) for value in center_xyz),
            major_axis_camera=tuple(float(value) for value in major_axis),
            minor_axis_camera=tuple(float(value) for value in minor_axis),
            major_axis_length_m=float(major_length),
            minor_axis_length_m=float(minor_length),
            pose_confidence=float(quality),
            graspability=graspability,
            instance_id="",
            attributes={
                "u": anchor_u,
                "v": anchor_v,
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "cx": float(intrinsics.cx),
                "cy": float(intrinsics.cy),
                "proposal_source": "depth_support_plane",
                "unknown_proposal": True,
                "opaque_foreground_only": True,
                "transparent_objects_supported": False,
                "component_id": int(component_id),
                "component_area_px": int(area),
                "component_bbox_valid_ratio": float(bbox_valid_ratio),
                "component_valid_depth_points": int(len(component_depth)),
                "component_median_height_m": median_height,
                "component_max_height_m": float(np.max(component_height)),
                "support_plane_inlier_ratio": float(plane.inlier_ratio),
                "support_plane_rms_m": float(plane.rms_m),
                "support_plane_inliers": int(plane.inliers),
                "support_plane_candidates": int(plane.candidates),
                "graspability_source": "geometric_proposal_quality_proxy",
            },
        )

    def _overlaps_known(
        self,
        bbox: tuple[float, float, float, float],
        known_detections: Sequence[Detection],
    ) -> bool:
        for known in known_detections:
            if _box_iou(bbox, known.bbox) >= float(self.config.known_overlap_iou):
                return True
            if _box_containment(bbox, known.bbox) >= float(
                self.config.known_containment
            ):
                return True
        return False

    def _image_axes(
        self,
        top_u: np.ndarray,
        top_v: np.ndarray,
        top_points: np.ndarray,
        intrinsics: Any,
        center_z: float,
        bbox_width_px: int,
        bbox_height_px: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        uv = np.column_stack((top_u.astype(float), top_v.astype(float)))
        centered = uv - np.median(uv, axis=0)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        major_px = np.asarray((1.0, 0.0), dtype=float)
        if np.all(np.isfinite(covariance)):
            try:
                _, eigenvectors = np.linalg.eigh(covariance)
                major_px = np.asarray(eigenvectors[:, -1], dtype=float)
            except np.linalg.LinAlgError:
                pass
        if abs(float(major_px[0])) <= 1e-12 and abs(float(major_px[1])) <= 1e-12:
            major_px = np.asarray((1.0, 0.0), dtype=float)
        major_px = major_px / float(np.linalg.norm(major_px))
        if major_px[0] < -1e-12 or (
            abs(float(major_px[0])) <= 1e-12 and major_px[1] < 0.0
        ):
            major_px = -major_px
        minor_px = np.asarray((-major_px[1], major_px[0]), dtype=float)

        z_mid = max(float(center_z), float(self.config.min_depth_m))
        major_camera = _unit(
            np.asarray(
                (
                    major_px[0] / float(intrinsics.fx) * z_mid,
                    major_px[1] / float(intrinsics.fy) * z_mid,
                    0.0,
                ),
                dtype=float,
            )
        )
        if major_camera is None:
            major_camera = np.asarray((1.0, 0.0, 0.0), dtype=float)
        minor_camera = _unit(np.cross(np.asarray((0.0, 0.0, 1.0)), major_camera))
        if minor_camera is None:
            minor_camera = np.asarray((0.0, 1.0, 0.0), dtype=float)

        major_values = top_points @ major_camera
        minor_values = top_points @ minor_camera
        major_length = float(
            np.percentile(major_values, 95.0) - np.percentile(major_values, 5.0)
        )
        minor_length = float(
            np.percentile(minor_values, 95.0) - np.percentile(minor_values, 5.0)
        )
        major_fallback = (
            float(bbox_width_px) * z_mid / max(float(intrinsics.fx), 1e-6)
        )
        minor_fallback = (
            float(bbox_height_px) * z_mid / max(float(intrinsics.fy), 1e-6)
        )
        if not math.isfinite(major_length) or major_length < 0.005:
            major_length = max(major_fallback, 0.005)
        if not math.isfinite(minor_length) or minor_length < 0.005:
            minor_length = max(minor_fallback, 0.005)
        if major_length < minor_length:
            major_camera, minor_camera = minor_camera, major_camera
            major_length, minor_length = minor_length, major_length
        return major_camera, minor_camera, major_length, minor_length

    def _quality(
        self,
        *,
        plane: _PlaneModel,
        bbox_valid_ratio: float,
        area: int,
        median_height: float,
    ) -> float:
        minimum_ratio = float(self.config.plane_min_inlier_ratio)
        ratio_scale = max(1e-6, 1.0 - minimum_ratio)
        plane_score = _clamp01((plane.inlier_ratio - minimum_ratio) / ratio_scale)
        depth_score = _clamp01(bbox_valid_ratio)
        area_score = _clamp01(area / max(float(self.config.min_area_px) * 4.0, 1.0))
        height_span = max(
            float(self.config.max_object_height_m)
            - float(self.config.min_object_height_m),
            1e-6,
        )
        height_score = _clamp01(
            (median_height - float(self.config.min_object_height_m))
            / min(0.03, height_span)
        )
        return float(
            0.35 * plane_score
            + 0.30 * depth_score
            + 0.20 * area_score
            + 0.15 * height_score
        )


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _box_containment(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    smaller = min(
        max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1]),
        max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1]),
    )
    return intersection / smaller if smaller > 0.0 else 0.0
