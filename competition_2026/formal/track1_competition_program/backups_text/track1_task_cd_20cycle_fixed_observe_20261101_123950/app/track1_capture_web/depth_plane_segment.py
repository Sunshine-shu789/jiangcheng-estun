#!/usr/bin/env python3
"""Segment tabletop objects from aligned RealSense RGB-D captures.

The depth image is unprojected with the aligned color intrinsics, a dominant
plane is fitted with RANSAC, and pixels closer to the camera than the plane are
reported as object candidates.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray
    offset: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.offset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--plane-threshold-m", type=float, default=0.006)
    parser.add_argument("--object-min-height-m", type=float, default=0.010)
    parser.add_argument("--object-max-height-m", type=float, default=0.40)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.50)
    parser.add_argument("--sample-step", type=int, default=4)
    parser.add_argument("--ransac-iterations", type=int, default=1200)
    parser.add_argument("--ransac-seed", type=int, default=20260926)
    parser.add_argument("--min-component-pixels", type=int, default=150)
    parser.add_argument("--morph-kernel", type=int, default=5)
    parser.add_argument(
        "--keep-border-components",
        action="store_true",
        help="Keep object components that touch the image boundary.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def adjusted_intrinsics(data: dict[str, Any], rotation: int) -> Intrinsics:
    width = int(data["width"])
    height = int(data["height"])
    fx = float(data["fx"])
    fy = float(data["fy"])
    ppx = float(data["ppx"])
    ppy = float(data["ppy"])
    if rotation == 90:
        width, height = height, width
        fx, fy = fy, fx
        ppx, ppy = ppy, height - 1 - ppx
    elif rotation == 180:
        ppx = width - 1 - ppx
        ppy = height - 1 - ppy
    elif rotation == 270:
        width, height = height, width
        fx, fy = fy, fx
        ppx, ppy = width - 1 - ppy, ppx
    elif rotation != 0:
        raise ValueError(f"unsupported rotation: {rotation}")
    return Intrinsics(width, height, fx, fy, ppx, ppy)


def unproject(
    depth_m: np.ndarray,
    valid: np.ndarray,
    intrinsics: Intrinsics,
) -> np.ndarray:
    ys, xs = np.nonzero(valid)
    z = depth_m[ys, xs]
    x = (xs.astype(np.float32) - intrinsics.ppx) * z / intrinsics.fx
    y = (ys.astype(np.float32) - intrinsics.ppy) * z / intrinsics.fy
    return np.column_stack((x, y, z)).astype(np.float32)


def fit_plane_least_squares(points: np.ndarray) -> Plane:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    offset = float(-normal @ centroid)
    if offset < 0:
        normal = -normal
        offset = -offset
    return Plane(normal, offset)


def fit_dominant_plane(
    points: np.ndarray,
    threshold_m: float,
    iterations: int,
    seed: int,
) -> tuple[Plane, np.ndarray]:
    if len(points) < 3:
        raise RuntimeError("not enough valid depth points to fit a plane")
    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_count = -1
    for _ in range(iterations):
        indices = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[indices]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            continue
        normal /= norm
        offset = float(-normal @ p0)
        distances = np.abs(points @ normal + offset)
        inlier_count = int(np.count_nonzero(distances <= threshold_m))
        if inlier_count > best_count:
            best_count = inlier_count
            best_inliers = distances <= threshold_m
    if best_inliers is None:
        raise RuntimeError("RANSAC failed to find a plane")
    for _ in range(3):
        plane = fit_plane_least_squares(points[best_inliers])
        best_inliers = (
            np.abs(points @ plane.normal + plane.offset) <= threshold_m
        )
    return plane, best_inliers


def clean_mask(
    mask: np.ndarray,
    kernel_size: int,
    min_component_pixels: int,
    keep_border_components: bool,
) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("--morph-kernel must be a positive odd number")
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    keep = np.zeros_like(cleaned)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_component_pixels:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = x == 0 or y == 0 or x + width == mask.shape[1] or y + height == mask.shape[0]
        if touches_border and not keep_border_components:
            continue
        keep[labels == label] = 255
    return keep > 0


def overlay_result(
    color: np.ndarray,
    table_mask: np.ndarray,
    object_mask: np.ndarray,
) -> np.ndarray:
    overlay = color.copy()
    table_tint = np.zeros_like(color)
    table_tint[:, :, 0] = 180
    table_selection = table_mask & ~object_mask
    overlay[table_selection] = cv2.addWeighted(
        color[table_selection],
        0.82,
        table_tint[table_selection],
        0.18,
        0,
    )
    contours, _ = cv2.findContours(
        object_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (255, 200, 20), 2)
    return overlay


def depth_visualization(depth_m: np.ndarray) -> np.ndarray:
    scaled = np.clip(depth_m * 1000.0, 0, 4000).astype(np.uint16)
    return cv2.applyColorMap(
        cv2.convertScaleAbs(scaled, alpha=0.06),
        cv2.COLORMAP_TURBO,
    )


def read_frame_rows(session: Path) -> list[dict[str, Any]]:
    manifest = session / "frames.jsonl"
    rows: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no frames found in {manifest}")
    return rows


def component_summary(
    mask: np.ndarray,
    height_map: np.ndarray,
) -> tuple[int, int, list[int], list[dict[str, Any]]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )
    areas = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, count)
    ]
    areas.sort(reverse=True)
    details: list[dict[str, Any]] = []
    for label in range(1, count):
        component_pixels = labels == label
        component_heights = height_map[component_pixels]
        valid_heights = component_heights[np.isfinite(component_heights)]
        if not valid_heights.size:
            continue
        details.append(
            {
                "area_pixels": int(stats[label, cv2.CC_STAT_AREA]),
                "bbox_xywh": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "valid_depth_fraction": float(
                    valid_heights.size / component_pixels.sum()
                ),
                "height_mean_m": float(valid_heights.mean()),
                "height_median_m": float(np.median(valid_heights)),
                "height_p95_m": float(np.percentile(valid_heights, 95)),
                "height_max_m": float(valid_heights.max()),
            }
        )
    details.sort(key=lambda item: int(item["area_pixels"]), reverse=True)
    return count - 1, sum(areas), areas, details


def process_frame(
    row: dict[str, Any],
    session_data: dict[str, Any],
    args: argparse.Namespace,
    output: Path,
) -> dict[str, Any]:
    depth_path = Path(row["depth"])
    stem = str(row["stem"])
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise RuntimeError(f"cannot read depth: {depth_path}")
    depth_m = depth_raw.astype(np.float32) * float(
        session_data["depth_scale_m_per_unit"]
    )
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= args.min_depth_m)
        & (depth_m <= args.max_depth_m)
    )
    intrinsics = adjusted_intrinsics(
        session_data["color_intrinsics"],
        int(session_data["rotation_degrees"]),
    )
    if depth_m.shape != (intrinsics.height, intrinsics.width):
        raise RuntimeError(
            f"depth shape {depth_m.shape} does not match intrinsics "
            f"{(intrinsics.height, intrinsics.width)}"
        )

    points = unproject(depth_m, valid, intrinsics)
    sample_points = points[:: args.sample_step]
    plane, sample_inliers = fit_dominant_plane(
        sample_points,
        args.plane_threshold_m,
        args.ransac_iterations,
        args.ransac_seed,
    )

    ys, xs = np.nonzero(valid)
    distances = points @ plane.normal + plane.offset
    plane_error = np.abs(distances)
    table_mask = (
        np.zeros(valid.shape, dtype=bool)
    )
    table_mask[ys, xs] = plane_error <= args.plane_threshold_m
    height_map = np.full(valid.shape, np.nan, dtype=np.float32)
    height_map[ys, xs] = distances
    raw_object_mask = np.zeros(valid.shape, dtype=bool)
    raw_object_mask[ys, xs] = (
        (distances >= args.object_min_height_m)
        & (distances <= args.object_max_height_m)
    )
    object_mask = clean_mask(
        raw_object_mask,
        args.morph_kernel,
        args.min_component_pixels,
        args.keep_border_components,
    )

    rgb_path = Path(row.get("rgb", ""))
    color = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if color is None or color.shape[:2] != depth_m.shape:
        color = depth_visualization(depth_m)
    else:
        overlay = overlay_result(color, table_mask, object_mask)

    mask_dir = output / "masks"
    overlay_dir = output / "overlays"
    depth_vis_dir = output / "depth_vis"
    for directory in (mask_dir, overlay_dir, depth_vis_dir):
        directory.mkdir(parents=True, exist_ok=True)
    mask_visual = np.zeros(valid.shape, dtype=np.uint8)
    mask_visual[table_mask] = 128
    mask_visual[object_mask] = 255
    if not cv2.imwrite(str(mask_dir / f"{stem}.png"), mask_visual):
        raise RuntimeError(f"cannot write mask for {stem}")
    if not cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), overlay):
        raise RuntimeError(f"cannot write overlay for {stem}")
    if not cv2.imwrite(
        str(depth_vis_dir / f"{stem}.jpg"),
        depth_visualization(depth_m),
    ):
        raise RuntimeError(f"cannot write depth visualization for {stem}")

    inlier_distances = plane_error[plane_error <= args.plane_threshold_m]
    (
        object_components,
        object_pixels,
        component_areas,
        component_details,
    ) = component_summary(object_mask, height_map)
    result = {
        "stem": stem,
        "valid_depth_pixels": int(valid.sum()),
        "valid_depth_fraction": float(valid.mean()),
        "ransac_sample_points": int(len(sample_points)),
        "ransac_inlier_count": int(sample_inliers.sum()),
        "ransac_inlier_ratio": float(sample_inliers.mean()),
        "plane_normal_camera_xyz": [float(value) for value in plane.normal],
        "camera_plane_distance_m": float(plane.offset),
        "plane_tilt_from_optical_axis_deg": float(
            np.degrees(np.arccos(np.clip(abs(plane.normal[2]), 0.0, 1.0)))
        ),
        "table_pixels": int(table_mask.sum()),
        "table_fraction_of_valid": float(table_mask.sum() / max(valid.sum(), 1)),
        "plane_rms_error_m": (
            float(np.sqrt(np.mean(inlier_distances**2)))
            if inlier_distances.size
            else None
        ),
        "plane_median_error_m": (
            float(np.median(inlier_distances))
            if inlier_distances.size
            else None
        ),
        "object_pixels": object_pixels,
        "object_fraction_of_valid": float(
            object_pixels / max(valid.sum(), 1)
        ),
        "object_components": object_components,
        "largest_component_pixels": component_areas[0] if component_areas else 0,
        "object_component_details": component_details,
    }
    return result


def write_summary(output: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = list(rows[0].keys()) if rows else []
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key, value in flat.items():
                if isinstance(value, (list, dict)):
                    flat[key] = json.dumps(value)
            writer.writerow(flat)
    (output / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.sample_step < 1:
        raise SystemExit("--sample-step must be positive")
    if args.ransac_iterations < 1:
        raise SystemExit("--ransac-iterations must be positive")
    if args.plane_threshold_m <= 0:
        raise SystemExit("--plane-threshold-m must be positive")
    if not 0 < args.object_min_height_m <= args.object_max_height_m:
        raise SystemExit("object height range is invalid")

    session = Path(args.session).resolve()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    session_data = load_json(session / "session.json")
    rows = [
        process_frame(row, session_data, args, output)
        for row in read_frame_rows(session)
    ]
    write_summary(output, rows)
    for row in rows:
        print(
            f"{row['stem']}: table={row['table_fraction_of_valid']:.1%}, "
            f"RMS={row['plane_rms_error_m']:.4f}m, "
            f"objects={row['object_pixels']}px"
        )
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
