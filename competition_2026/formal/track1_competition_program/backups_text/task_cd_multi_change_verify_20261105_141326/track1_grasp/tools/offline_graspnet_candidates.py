#!/usr/bin/env python3
"""Offline RGB-D -> GraspNet -> CAD jaw -> base clearance -> IK diagnostic.

This tool is intentionally read-only.  It does not import ROS, contact a robot,
open a gripper, or publish a trajectory.  It is a reproducible research
pipeline for one captured frame.

Coordinate flow (all transforms are homogeneous 4x4 matrices):

    depth pixel -> camera optical frame
    T_base_camera = controller_fk(q) @ T_flange_camera
    T_base_contact = T_base_camera @ GraspNet/contact score
    T_base_tcp     = T_base_camera @ score.tcp_transform
    T_base_flange  = T_base_tcp @ inv(T_flange_tcp)

The last line matters because the controller FK in this repository returns the
flange pose.  ``grasp_tcp`` is a fixed child of ``gripper_body_link`` in the
CAD/URDF assembly, not the flange itself.

The output is fail-closed diagnostic data.  It contains no motion authorization
and marks the third-party GraspNet checkpoint as quality-unvalidated.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage
import yaml

from track1_grasp.graspnet_inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CHECKPOINT_SHA256,
    GraspNetInferenceResult,
    provenance_payload,
    run_inference,
)
from track1_grasp.jaw_grasp_scoring import (
    JawAwareGraspScorer,
    JawGraspCandidate,
    JawGraspScore,
    load_jaw_scoring_config,
    nms_jaw_grasp_scores,
)
from track1_grasp.jaw_profile import JawProfile
from track1_grasp.jaw_opening_search import (
    JawOpeningSearchConfig,
    JawOpeningTrial,
    search_jaw_opening,
)
from track1_grasp.jaw_world_clearance import (
    GripperClearanceConfig,
    GripperWorldClearanceResult,
    evaluate_gripper_world_clearance,
)
from track1_grasp.kinematics import (
    controller_fk,
    solve_pose_ik_multistart,
)
from track1_grasp.planner import load_extrinsics
from track1_grasp.support_plane import (
    SupportPlaneFitConfig,
    fit_support_plane_base,
    select_local_support_points,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCENE_DIR = (
    PROJECT_ROOT
    / "logs"
    / "static_calibration_20261025"
    / "archive_20261025_125826"
    / "batches"
    / "static_capture_20261025"
)
DEFAULT_PROFILE = (
    PROJECT_ROOT
    / "robot_description"
    / "estun_s3_60_g2"
    / "config"
    / "fae2mr86m_ms42dc_jaw_profile.json"
)
DEFAULT_DIMENSIONS = (
    PROJECT_ROOT
    / "robot_description"
    / "estun_s3_60_g2"
    / "config"
    / "fae2mr86m_ms42dc_dimensions.json"
)
DEFAULT_SCORING_CONFIG = (
    PROJECT_ROOT
    / "app"
    / "track1_grasp"
    / "config"
    / "jaw_scoring_fae2mr86m_ms42dc_cad_offline.yaml"
)
DEFAULT_EXTRINSICS = (
    PROJECT_ROOT
    / "app"
    / "track1_grasp"
    / "config"
    / "camera_to_flange.d405_calibrated_20261025.yaml"
)
DEFAULT_ASSEMBLY = (
    PROJECT_ROOT
    / "robot_description"
    / "estun_s3_60_g2"
    / "config"
    / "gripper_assembly.fae2mr86m_ms42dc.yaml"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs"
DEFAULT_FRAME = 23
DEFAULT_POINTS = 20000
DEFAULT_SEED = 20261029
DEFAULT_ROI_PAD_PX = 40


# ---------------------------------------------------------------------------
# Basic I/O and geometry helpers


def _json_value(value: Any) -> Any:
    """Recursively convert NumPy/Path values into JSON-safe values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _as_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB image must have shape (H, W, 3)")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def segment_banana_mask(
    rgb: np.ndarray,
    *,
    hsv_low: Sequence[int] = (21, 90, 90),
    hsv_high: Sequence[int] = (57, 255, 255),
    morphology_size: int = 7,
) -> np.ndarray:
    """Segment the yellow banana using PIL HSV (H is 0..255).

    The saved frame is read with PIL, so the hue convention here is PIL's
    0..255 hue, not OpenCV's 0..179 hue.  The default ranges reproduce the
    reviewed frame-23 segmentation and keep this tool usable without cv2.
    """

    rgb_u8 = _as_rgb_u8(rgb)
    hsv = np.asarray(Image.fromarray(rgb_u8, mode="RGB").convert("HSV"), dtype=np.uint8)
    low = np.asarray(hsv_low, dtype=np.uint8).reshape(3)
    high = np.asarray(hsv_high, dtype=np.uint8).reshape(3)
    if np.any(low > high):
        raise ValueError("hsv_low must be <= hsv_high for each channel")
    mask = np.all((hsv >= low) & (hsv <= high), axis=2)
    size = int(morphology_size)
    if size < 1 or size % 2 == 0:
        raise ValueError("morphology_size must be a positive odd integer")
    structure = np.ones((size, size), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=structure)
    mask = ndimage.binary_closing(mask, structure=structure)
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    if sizes[largest] == 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == largest


def make_roi_mask(mask: np.ndarray, pad_px: int) -> np.ndarray:
    roi = np.zeros_like(mask, dtype=bool)
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return roi
    pad = int(pad_px)
    if pad < 0:
        raise ValueError("roi pad must be non-negative")
    y0 = max(0, int(rows.min()) - pad)
    y1 = min(mask.shape[0], int(rows.max()) + 1 + pad)
    x0 = max(0, int(columns.min()) - pad)
    x1 = min(mask.shape[1], int(columns.max()) + 1 + pad)
    roi[y0:y1, x0:x1] = True
    return roi


def backproject_depth(
    depth_m: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backproject a depth image into an ``N x 3`` camera-optical cloud."""

    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    values = (float(fx), float(fy), float(cx), float(cy))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("camera intrinsics must be finite")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx and fy must be positive")
    valid = np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    z = depth[rows, columns]
    x = (columns.astype(np.float64) - float(cx)) / float(fx) * z
    y = (rows.astype(np.float64) - float(cy)) / float(fy) * z
    points = np.column_stack((x, y, z)).astype(np.float32)
    return points, valid, np.column_stack((rows, columns)).astype(np.int32)


def sample_points(
    points: np.ndarray,
    count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically sample exactly ``count`` points, with replacement if needed."""

    cloud = np.asarray(points, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0:
        raise ValueError("points must be a non-empty (N, 3) array")
    count = int(count)
    if count <= 0:
        raise ValueError("point count must be positive")
    rng = np.random.default_rng(int(seed))
    if len(cloud) >= count:
        indices = rng.choice(len(cloud), size=count, replace=False)
    else:
        indices = np.concatenate(
            (
                np.arange(len(cloud), dtype=np.int64),
                rng.choice(
                    len(cloud),
                    size=count - len(cloud),
                    replace=True,
                ),
            )
        )
    return np.ascontiguousarray(cloud[indices], dtype=np.float32), np.asarray(
        indices, dtype=np.int64
    )


def load_frame_joints(metadata: Mapping[str, Any], frame_number: int) -> np.ndarray:
    frames = metadata.get("frames", ())
    if not isinstance(frames, Sequence):
        raise ValueError("metadata frames must be a sequence")
    for item in frames:
        if not isinstance(item, Mapping):
            continue
        numbers = {
            item.get("index"),
            item.get("color_frame_number"),
            item.get("depth_frame_number"),
        }
        if int(frame_number) not in {int(value) for value in numbers if value is not None}:
            continue
        robot = item.get("robot")
        if not isinstance(robot, Mapping):
            continue
        values = robot.get("joint_positions")
        if values is None:
            continue
        joints = np.asarray(values, dtype=float).reshape(-1)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("metadata joint_positions must contain six finite values")
        return joints
    raise ValueError(
        f"no six-joint robot posture found for frame {frame_number} in metadata"
    )


def parse_joints(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    try:
        joints = np.asarray([float(part) for part in values], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--joints must be a comma-separated list") from exc
    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
        raise argparse.ArgumentTypeError("--joints must contain six finite radians")
    return joints


def transform_matrix(payload: Mapping[str, Any], key: str) -> np.ndarray:
    value = payload.get(key)
    if isinstance(value, Mapping):
        value = value.get("matrix")
    if value is None:
        raise ValueError(f"missing transform matrix in {key}")
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{key} must be a finite 4x4 matrix")
    return matrix


def _homogeneous_translation(xyz: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, 3] = np.asarray(xyz, dtype=float)
    return result


def load_flange_tcp(assembly_path: Path, tcp_x_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return T_flange_body and T_flange_tcp from the offline assembly YAML."""

    with assembly_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    mounting = payload.get("mounting", {})
    if not isinstance(mounting, Mapping):
        raise ValueError("assembly mounting section must be a mapping")
    t_flange_body = transform_matrix(mounting, "flange_from_body")
    body_tcp = _homogeneous_translation((float(tcp_x_m), 0.0, 0.0))
    t_flange_tcp = t_flange_body @ body_tcp
    return t_flange_body, t_flange_tcp


def _transform_payload(transform: np.ndarray) -> list[list[float]]:
    return np.asarray(transform, dtype=float).tolist()


def score_to_record(
    *,
    candidate: JawGraspCandidate,
    raw_score: JawGraspScore,
    t_base_camera: np.ndarray,
    clearance: GripperWorldClearanceResult | None,
    final_feasible: bool,
    final_reasons: Sequence[str],
    base_contact: np.ndarray,
    base_body: np.ndarray,
    base_tcp: np.ndarray,
    base_flange: np.ndarray,
    opening_search: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source_point_index": int(candidate.source_point_index),
        "graspnet": {
            "translation_camera_m": list(candidate.translation_m),
            "rotation_camera": list(candidate.rotation),
            "width_m": float(candidate.width_m),
            "width_semantics": "reference/upper-bound opening, not final jaw contact width",
            "depth_m": float(candidate.depth_m),
            "height_m": float(candidate.height_m),
            "tolerance_m": float(candidate.tolerance_m),
            "network_score": float(candidate.network_score),
            "metadata": dict(candidate.metadata),
            "translation_semantics": "surface seed/reference, not grasp_tcp",
        },
        "score_camera": {
            "selected_opening_m": float(candidate.jaw_opening_m),
            "feasible": bool(raw_score.feasible),
            "quality": float(raw_score.quality),
            "reason_codes": list(raw_score.reason_codes),
            "measured_width_m": raw_score.measured_width_m,
            "inner_target_points": int(raw_score.inner_target_points),
            "contact_left_points": int(raw_score.contact_left_points),
            "contact_right_points": int(raw_score.contact_right_points),
            "target_penetration_points": int(raw_score.target_penetration_points),
            "obstacle_finger_points": int(raw_score.obstacle_finger_points),
            "obstacle_cavity_points": int(raw_score.obstacle_cavity_points),
            "obstacle_body_points": int(raw_score.obstacle_body_points),
            "obstacle_approach_points": int(raw_score.obstacle_approach_points),
            "obstacle_clearance_m": float(raw_score.obstacle_clearance_m),
            "opening_margin_m": float(raw_score.opening_margin_m),
            "quality_components": dict(raw_score.quality_components),
            "section": raw_score.section.to_dict(),
        },
        "transforms": {
            "T_camera_contact": _transform_payload(raw_score.contact_transform),
            "T_base_contact": _transform_payload(base_contact),
            "T_camera_body": _transform_payload(raw_score.body_transform),
            "T_base_body": _transform_payload(base_body),
            "T_camera_tcp": _transform_payload(raw_score.tcp_transform),
            "T_base_tcp": _transform_payload(base_tcp),
            "T_base_flange": _transform_payload(base_flange),
            "T_base_camera": _transform_payload(t_base_camera),
        },
        "world_clearance": None if clearance is None else clearance.to_dict(),
        "opening_search": None if opening_search is None else dict(opening_search),
        "final_feasible": bool(final_feasible),
        "final_reason_codes": list(dict.fromkeys(final_reasons)),
        "ik": None,
    }


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a finite homogeneous transform to an ``N x 3`` point cloud."""

    cloud = np.asarray(points, dtype=float)
    matrix = np.asarray(transform, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be a finite 4x4 matrix")
    return cloud @ matrix[:3, :3].T + matrix[:3, 3]


def _ik_payload(solution: Any) -> dict[str, Any]:
    return {
        "success": True,
        "joints_rad": np.asarray(solution.joints, dtype=float).tolist(),
        "joints_deg": np.rad2deg(np.asarray(solution.joints, dtype=float)).tolist(),
        "seed_joints_rad": np.asarray(solution.seed_joints, dtype=float).tolist(),
        "seed_source": str(solution.seed_source),
        "damping": float(solution.damping),
        "iterations": int(solution.iterations),
        "position_error_m": float(solution.position_error_m),
        "orientation_error_rad": float(solution.orientation_error_rad),
        "minimum_jacobian_singular_value": float(
            solution.minimum_jacobian_singular_value
        ),
        "at_joint_limit": bool(solution.at_joint_limit),
        "max_joint_delta_rad": float(solution.max_joint_delta_rad),
        "max_joint_delta_deg": float(math.degrees(solution.max_joint_delta_rad)),
        "note": "converged IK is diagnostic only; no collision or execution authorization",
    }


def _report_markdown(payload: Mapping[str, Any]) -> str:
    scene = payload["scene"]
    summary = payload["summary"]
    network = payload["network"]
    limits = payload["safety"]
    lines = [
        "# Offline GraspNet Candidate Report",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- offline_only: `{str(payload['offline_only']).lower()}`",
        f"- motion_attempted: `{str(payload['motion_attempted']).lower()}`",
        f"- quality_validated: `{str(payload['quality_validated']).lower()}`",
        f"- motion_authorized: `{str(payload['motion_authorized']).lower()}`",
        "",
        "## Input",
        "",
        f"- scene: `{scene['scene_dir']}`",
        f"- frame: `{scene['frame_number']}`",
        f"- RGB shape: `{scene['rgb_shape']}`",
        f"- depth shape: `{scene['depth_shape']}`",
        f"- depth units: `{scene['depth_units']}`",
        f"- target mask pixels: `{scene['target_mask_pixels']}`",
        f"- target valid depth points: `{scene['target_valid_points']}`",
        f"- ROI obstacle points: `{scene['obstacle_points']}`",
        f"- local support-plane candidate points: `{scene['support_plane_candidate_points']}`",
        f"- local support-plane XY radius: `{scene['support_plane_local_xy_radius_m']:.3f} m`",
        f"- GraspNet sampled points: `{scene['sampled_points']}` (seed `{scene['sample_seed']}`)",
        f"- segmentation bbox `[x0,x1,y0,y1]`: `{scene['target_bbox_xyxy']}`",
        "",
        "## Model",
        "",
        f"- checkpoint: `{network['checkpoint_path']}`",
        f"- checkpoint SHA256: `{network['checkpoint_sha256']}`",
        f"- provenance: `{network['checkpoint_provenance']}`",
        f"- device: `{network['device']}`",
        f"- objectness positive seeds: `{network['stats'].get('objectness_positive_seeds')}`",
        f"- candidates after `max_candidates`: `{network['stats'].get('selected_after_max_candidates')}`",
        f"- inference elapsed: `{network['elapsed_seconds']:.3f} s`",
        "",
        "## Support plane",
        "",
    ]
    support = payload["support_plane"]
    if support["valid"]:
        normal_text = ", ".join(
            f"{float(value):.6f}" for value in support["normal_base_m"]
        )
        origin_text = ", ".join(
            f"{float(value):.6f}" for value in support["origin_base_m"]
        )
        lines.extend(
            [
                f"- valid estimate: `true`",
                f"- normal in base frame: `[{normal_text}]`",
                f"- closest plane origin in base frame: `[{origin_text}]`",
                f"- tilt from base XY plane: `{support['tilt_deg']:.4f} deg`",
                f"- inliers: `{support['inlier_points']} / {support['candidate_points']}` (`{support['inlier_ratio']:.4f}`)",
                f"- residual median / P95 / RMS: `{support['median_abs_residual_m']:.6f}` / `{support['p95_abs_residual_m']:.6f}` / `{support['rms_m']:.6f}` m",
                f"- deterministic seed / threshold / iterations: `{support['ransac']['seed']}` / `{support['ransac']['threshold_m']:.6f} m` / `{support['ransac']['iterations']}`",
            ]
        )
    else:
        reasons = ", ".join(support["reason_codes"])
        lines.extend(
            [
                "- valid estimate: `false`",
                f"- fail-closed reasons: `{reasons}`",
                "- no fallback to a base-frame z=0 support plane was used",
            ]
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            f"- generated candidates: `{summary['candidate_count']}`",
            f"- GraspNet reference-width score feasible: `{summary['graspnet_reference_score_feasible_count']}`",
            f"- aperture-search trials: `{summary['aperture_search_trial_count']}`",
            f"- camera-jaw score feasible after aperture search: `{summary['score_feasible_count']}`",
            f"- base-frame whole-gripper clearance feasible: `{summary['world_clearance_feasible_count']}`",
            f"- final score + clearance feasible: `{summary['final_feasible_count']}`",
            f"- NMS survivors: `{summary['nms_survivor_count']}`",
            f"- IK evaluated: `{summary['ik_evaluated_count']}`",
            f"- IK success: `{summary['ik_success_count']}`",
            f"- final reason histogram: `{summary['final_reason_histogram']}`",
            "",
            "## Final candidates",
            "",
        ]
    )
    final_records = [item for item in payload["candidates"] if item["final_feasible"]]
    if not final_records:
        lines.append(
            "**0 final feasible candidates.** "
            "No threshold was relaxed and no motion is authorized."
        )
    else:
        lines.extend(
            [
                "| candidate | quality | GraspNet ref width (mm) | selected opening (mm) | TCP base (m) | grip min clearance (m) | TCP z margin (m) | IK |",
                "|---|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for item in final_records:
            world = item["world_clearance"] or {}
            ik = item.get("ik")
            ik_text = "not run"
            if ik is not None:
                ik_text = "success" if ik.get("success") else "failed"
            tcp = (world.get("tcp_base_m") or [None, None, None])
            tcp_text = ", ".join(
                "n/a" if value is None else f"{float(value):.6f}" for value in tcp
            )
            lines.append(
                "| {id} | {quality:.5f} | {ref:.3f} | {selected:.3f} | {tcp} | {clear:.6f} | {margin:.6f} | {ik} |".format(
                    id=item["candidate_id"],
                    quality=float(item["score_camera"]["quality"]),
                    ref=1000.0 * float(item["graspnet"]["width_m"]),
                    selected=1000.0
                    * float(item["score_camera"]["selected_opening_m"]),
                    tcp=tcp_text,
                    clear=float(world.get("support_minimum_clearance_m", math.nan)),
                    margin=float(world.get("tcp_workspace_margin_m", math.nan)),
                    ik=ik_text,
                )
            )
    lines.extend(
        [
            "",
            "## Safety boundary",
            "",
            f"- contact points minimum: `{limits['min_contact_points_per_side']}` per side",
            f"- visible width required: `{str(limits['require_measured_width']).lower()}`",
            f"- width conflict threshold: `{limits['width_tolerance_m']:.3f} m`",
            f"- whole-gripper support clearance: `{limits['minimum_clearance_m']:.3f} m`",
            f"- TCP workspace minimum z: `{limits['workspace_min_z_m']:.3f} m`",
            "- no physical robot, camera, controller, or gripper was connected by this tool",
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in payload["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline RGB-D GraspNet candidate diagnostic; never commands hardware."
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--dimensions", type=Path, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--scoring-config", type=Path, default=DEFAULT_SCORING_CONFIG)
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    parser.add_argument("--assembly", type=Path, default=DEFAULT_ASSEMBLY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--joints", type=str, default=None, help="six joint radians, comma separated")
    parser.add_argument("--num-points", type=int, default=DEFAULT_POINTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--roi-pad-px", type=int, default=DEFAULT_ROI_PAD_PX)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--ik-limit", type=int, default=16)
    parser.add_argument("--skip-ik", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.max_candidates <= 0:
        raise SystemExit("--max-candidates must be positive")
    if args.num_points <= 0:
        raise SystemExit("--num-points must be positive")
    if args.ik_limit < 0:
        raise SystemExit("--ik-limit must be non-negative")
    if args.depth_scale <= 0.0 or not math.isfinite(args.depth_scale):
        raise SystemExit("--depth-scale must be positive and finite")

    scene_dir = args.scene_dir.resolve()
    marker_dir = args.output_dir
    if marker_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        marker_dir = DEFAULT_OUTPUT_ROOT / f"offline_graspnet_{stamp}"
    output_dir = marker_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    clouds_dir = output_dir / "point_clouds"
    masks_dir.mkdir(exist_ok=True)
    clouds_dir.mkdir(exist_ok=True)

    frame_name = f"{int(args.frame):02d}"
    color_path = scene_dir / f"color_{frame_name}.png"
    depth_path = scene_dir / f"depth_{frame_name}.npy"
    metadata_path = scene_dir / "metadata.json"
    for path in (color_path, depth_path, metadata_path):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    rgb = _as_rgb_u8(np.asarray(Image.open(color_path).convert("RGB")))
    depth = np.asarray(np.load(depth_path), dtype=np.float64)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if depth.shape != rgb.shape[:2]:
        raise SystemExit(
            f"depth shape {depth.shape} does not match RGB shape {rgb.shape[:2]}"
        )

    explicit_joints = parse_joints(args.joints)
    joints = explicit_joints if explicit_joints is not None else load_frame_joints(metadata, args.frame)
    if joints.shape != (6,):
        raise SystemExit("frame posture must contain six joints")

    intrinsics = metadata.get("rotated_intrinsics") or metadata.get("raw_intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise SystemExit("metadata has no intrinsics")
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    if intrinsics.get("width") is not None and int(intrinsics["width"]) != rgb.shape[1]:
        raise SystemExit("intrinsics width does not match RGB")
    if intrinsics.get("height") is not None and int(intrinsics["height"]) != rgb.shape[0]:
        raise SystemExit("intrinsics height does not match RGB")

    banana_mask = segment_banana_mask(rgb)
    if not np.any(banana_mask):
        raise SystemExit("banana segmentation produced an empty mask")
    roi_mask = make_roi_mask(banana_mask, args.roi_pad_px)
    depth_m = depth * float(args.depth_scale)
    points, valid, _pixel_indices = backproject_depth(
        depth_m,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    # ``points`` contains only valid-depth pixels, so select mask values at
    # those same rows/columns before indexing the compressed cloud.
    target_points = points[banana_mask[valid]]
    obstacle_points = points[roi_mask[valid] & ~banana_mask[valid]]
    if len(target_points) == 0:
        raise SystemExit("no valid target depth points in banana mask")
    if len(obstacle_points) == 0:
        raise SystemExit("no valid obstacle depth points in ROI")

    # GraspNet sees a deterministic 20k cloud sampled from the ROI.  Scoring
    # still uses the full target and obstacle point sets above.
    roi_points = points[roi_mask[valid]]
    sampled_cloud, sampled_indices = sample_points(
        roi_points,
        args.num_points,
        seed=args.seed,
    )

    with args.scoring_config.open("r", encoding="utf-8") as handle:
        scoring_payload = yaml.safe_load(handle) or {}
    if not isinstance(scoring_payload, Mapping):
        raise SystemExit("scoring config must contain a mapping")
    profile_payload = scoring_payload.get("profile", {})
    if not isinstance(profile_payload, Mapping):
        raise SystemExit("scoring config profile section must be a mapping")
    tcp_x_m = float(profile_payload.get("tcp_x_m", 0.050))
    scoring_config = load_jaw_scoring_config(args.scoring_config)
    opening_search_config = JawOpeningSearchConfig.from_mapping(scoring_payload)
    clearance_config = GripperClearanceConfig.from_mapping(scoring_payload)
    support_plane_config = SupportPlaneFitConfig.from_mapping(scoring_payload)
    profile = JawProfile.from_files(
        args.profile,
        args.dimensions,
        tcp_x_m=tcp_x_m,
    )
    scorer = JawAwareGraspScorer(profile, scoring_config)
    t_flange_body, t_flange_tcp = load_flange_tcp(args.assembly, tcp_x_m)

    t_base_flange = controller_fk(joints)
    t_base_camera = t_base_flange @ load_extrinsics(args.extrinsics)

    target_points_base = transform_points(target_points, t_base_camera)
    obstacle_points_base = transform_points(obstacle_points, t_base_camera)
    support_plane_candidates, support_plane_center_xy = select_local_support_points(
        obstacle_points_base,
        target_points_base,
        local_xy_radius_m=support_plane_config.local_xy_radius_m,
    )
    support_plane = fit_support_plane_base(
        support_plane_candidates,
        support_plane_config,
    )
    support_plane_payload = support_plane.to_dict()
    support_plane_payload["source"] = "estimated_from_current_frame_obstacle_cloud"
    support_plane_payload["selection"] = {
        "target_center_xy_base_m": list(support_plane_center_xy),
        "local_xy_radius_m": float(support_plane_config.local_xy_radius_m),
        "roi_obstacle_points": int(len(obstacle_points_base)),
        "selected_candidate_points": int(len(support_plane_candidates)),
    }
    effective_clearance_config: GripperClearanceConfig | None = None
    if support_plane.valid:
        assert support_plane.origin_base_m is not None
        assert support_plane.normal_base_m is not None
        effective_clearance_config = replace(
            clearance_config,
            support_origin_base_m=support_plane.origin_base_m,
            support_normal_base=support_plane.normal_base_m,
        )
        target_support_clearance = (
            target_points_base - np.asarray(support_plane.origin_base_m, dtype=float)
        ) @ np.asarray(support_plane.normal_base_m, dtype=float)
        support_plane_payload["target_support_clearance_m"] = {
            "minimum": float(np.min(target_support_clearance)),
            "p01": float(np.percentile(target_support_clearance, 1.0)),
            "p05": float(np.percentile(target_support_clearance, 5.0)),
            "median": float(np.median(target_support_clearance)),
            "p95": float(np.percentile(target_support_clearance, 95.0)),
            "maximum": float(np.max(target_support_clearance)),
        }

    inference = run_inference(
        sampled_cloud,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_candidates=int(args.max_candidates),
    )

    records: list[dict[str, Any]] = []
    feasible_scores: list[JawGraspScore] = []
    reference_score_feasible_count = 0
    score_feasible_count = 0
    world_feasible_count = 0
    aperture_search_trial_count = 0
    scoring_errors: list[dict[str, Any]] = []
    for index in range(len(inference)):
        candidate = JawGraspCandidate.from_graspnet_result(inference, index)
        try:
            opening_search = search_jaw_opening(
                profile,
                scorer,
                candidate,
                target_points,
                obstacle_points,
                config=opening_search_config,
            )
        except Exception as exc:  # retain all candidates for diagnosis
            scoring_errors.append(
                {
                    "index": index,
                    "candidate_id": candidate.candidate_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        aperture_search_trial_count += len(opening_search.trials)
        if opening_search.reference_score.feasible:
            reference_score_feasible_count += 1

        # Prefer every aperture that passed the complete local CAD scorer,
        # ordered by quality.  The first one that also passes base-frame
        # support clearance wins.  If none pass locally, retain the best
        # diagnostic aperture (or the original reference score) fail-closed.
        if opening_search.feasible_trials:
            trial_candidates = list(opening_search.feasible_trials)
        else:
            diagnostic = opening_search.selected_trial
            if diagnostic is not None:
                trial_candidates = [diagnostic]
            else:
                reference_trial = JawOpeningTrial(
                    opening_m=candidate.jaw_opening_m,
                    candidate=candidate,
                    score=opening_search.reference_score,
                )
                trial_candidates = [reference_trial]

        has_local_feasible = any(trial.score.feasible for trial in trial_candidates)
        if has_local_feasible:
            score_feasible_count += 1

        evaluated: list[
            tuple[
                JawOpeningTrial,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                GripperWorldClearanceResult | None,
            ]
        ] = []
        for trial in trial_candidates:
            base_contact = t_base_camera @ trial.score.contact_transform
            base_body = t_base_camera @ trial.score.body_transform
            base_tcp = t_base_camera @ trial.score.tcp_transform
            base_flange = base_tcp @ np.linalg.inv(t_flange_tcp)
            clearance: GripperWorldClearanceResult | None = None
            if effective_clearance_config is not None:
                clearance = evaluate_gripper_world_clearance(
                    profile,
                    base_contact,
                    width_m=trial.candidate.jaw_opening_m,
                    depth_m=trial.candidate.depth_m,
                    config=effective_clearance_config,
                )
            evaluated.append(
                (trial, base_contact, base_body, base_tcp, base_flange, clearance)
            )
            if clearance is None or clearance.feasible:
                break

        chosen_trial, base_contact, base_body, base_tcp, base_flange, clearance = evaluated[0]
        raw_score = chosen_trial.score
        selected_candidate = chosen_trial.candidate
        has_world_feasible = any(
            item[0].score.feasible and item[5] is not None and item[5].feasible
            for item in evaluated
        )
        if has_world_feasible:
            world_feasible_count += 1
        clearance_reasons = () if clearance is None else clearance.reason_codes
        support_reasons = () if support_plane.valid else ("support_plane_invalid",)
        opening_reasons = (
            ()
            if raw_score.feasible or opening_search.status == "disabled"
            else opening_search.reason_codes
        )
        final_feasible = bool(
            has_local_feasible
            and support_plane.valid
            and has_world_feasible
        )
        final_reasons = list(
            dict.fromkeys(
                (
                    *support_reasons,
                    *opening_reasons,
                    *raw_score.reason_codes,
                    *clearance_reasons,
                )
            )
        )
        if final_feasible:
            assert clearance is not None
            feasible_scores.append(
                replace(
                    raw_score,
                    feasible=True,
                    reason_codes=(),
                    contact_transform=base_contact,
                    body_transform=base_body,
                    tcp_transform=base_tcp,
                    world_clearance=clearance.to_dict(),
                )
            )
        records.append(
            score_to_record(
                candidate=selected_candidate,
                raw_score=raw_score,
                t_base_camera=t_base_camera,
                clearance=clearance,
                final_feasible=final_feasible,
                final_reasons=final_reasons,
                base_contact=base_contact,
                base_body=base_body,
                base_tcp=base_tcp,
                base_flange=base_flange,
                opening_search={
                    **opening_search.to_dict(),
                    "selected_opening_m": float(selected_candidate.jaw_opening_m),
                    "selected_quality": float(raw_score.quality),
                    "selected_after_world_clearance": bool(
                        has_local_feasible and has_world_feasible
                    ),
                    "local_feasible": bool(has_local_feasible),
                    "world_clearance_feasible": bool(has_world_feasible),
                },
            )
        )

    kept, suppressed = nms_jaw_grasp_scores(feasible_scores, scoring_config)
    kept_ids = {item.candidate.candidate_id for item in kept}
    suppressed_ids = {item.candidate_id: item.kept_candidate_id for item in suppressed}
    for record in records:
        record["nms_kept"] = record["candidate_id"] in kept_ids
        record["nms_suppressed_by"] = suppressed_ids.get(record["candidate_id"])

    ik_evaluated_count = 0
    ik_success_count = 0
    if not args.skip_ik:
        for record in records:
            if not record["final_feasible"] or not record["nms_kept"]:
                continue
            if ik_evaluated_count >= args.ik_limit:
                record["ik"] = {
                    "success": False,
                    "skipped": True,
                    "reason": "ik_limit_reached",
                }
                continue
            base_flange = np.asarray(record["transforms"]["T_base_flange"], dtype=float)
            ik_evaluated_count += 1
            try:
                solution = solve_pose_ik_multistart(
                    joints,
                    base_flange[:3, 3],
                    base_flange[:3, :3],
                    joint_limit_margin_rad=math.radians(1.0),
                    minimum_singular_value=0.0,
                    return_diagnostics=True,
                )
                record["ik"] = _ik_payload(solution)
                ik_success_count += 1
            except Exception as exc:
                record["ik"] = {
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

    final_reason_histogram = Counter(
        reason for record in records for reason in record["final_reason_codes"]
    )
    nms_survivor_count = len(kept)
    final_feasible_count = sum(1 for record in records if record["final_feasible"])
    generated_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": generated_at,
        "offline_only": True,
        "motion_attempted": False,
        "quality_validated": False,
        "motion_authorized": False,
        "hardware_connected": False,
        "scene": {
            "scene_dir": str(scene_dir),
            "frame_number": int(args.frame),
            "color_file": str(color_path),
            "depth_file": str(depth_path),
            "metadata_file": str(metadata_path),
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "depth_units": "metadata_npy_values_multiplied_by_depth_scale",
            "depth_scale": float(args.depth_scale),
            "depth_valid_pixels": int(np.count_nonzero(valid)),
            "target_mask_pixels": int(np.count_nonzero(banana_mask)),
            "target_valid_points": int(len(target_points)),
            "obstacle_points": int(len(obstacle_points)),
            "support_plane_candidate_points": int(len(support_plane_candidates)),
            "support_plane_local_xy_radius_m": float(
                support_plane_config.local_xy_radius_m
            ),
            "roi_points": int(len(roi_points)),
            "sampled_points": int(len(sampled_cloud)),
            "sample_seed": int(args.seed),
            "roi_pad_px": int(args.roi_pad_px),
            "target_bbox_xyxy": [
                int(np.nonzero(banana_mask)[1].min()),
                int(np.nonzero(banana_mask)[1].max() + 1),
                int(np.nonzero(banana_mask)[0].min()),
                int(np.nonzero(banana_mask)[0].max() + 1),
            ],
            "joints_rad": np.asarray(joints, dtype=float).tolist(),
            "intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
            },
        },
        "network": dict(inference.metadata),
        "network_provenance_file": provenance_payload(),
        "coordinate_transforms": {
            "T_base_flange_at_capture": _transform_payload(t_base_flange),
            "T_flange_camera": _transform_payload(load_extrinsics(args.extrinsics)),
            "T_base_camera": _transform_payload(t_base_camera),
            "T_flange_body": _transform_payload(t_flange_body),
            "T_flange_tcp": _transform_payload(t_flange_tcp),
            "T_base_flange_target_rule": "T_base_tcp @ inverse(T_flange_tcp)",
        },
        "support_plane": support_plane_payload,
        "safety": {
            "min_contact_points_per_side": scoring_config.min_contact_points_per_side,
            "min_inner_points": scoring_config.min_inner_points,
            "require_measured_width": scoring_config.require_measured_width,
            "width_tolerance_m": scoring_config.width_tolerance_m,
            "graspnet_width_semantics": "reference/upper-bound opening",
            "aperture_search": opening_search_config.to_dict(),
            "minimum_clearance_m": clearance_config.minimum_clearance_m,
            "workspace_min_z_m": clearance_config.workspace_min_z_m,
            "workspace_min_z_rule": (
                "absolute_base_frame_tcp_z_independent_of_support_plane"
            ),
            "support_plane_valid": bool(support_plane.valid),
            "support_plane_source": (
                "estimated_from_current_frame_obstacle_cloud"
            ),
            "support_plane_config": support_plane_config.to_dict(),
            "effective_support_origin_base_m": (
                None
                if effective_clearance_config is None
                else list(effective_clearance_config.support_origin_base_m)
            ),
            "effective_support_normal_base_m": (
                None
                if effective_clearance_config is None
                else list(effective_clearance_config.support_normal_base)
            ),
            "scoring_config": str(args.scoring_config),
            "profile": str(args.profile),
            "dimensions": str(args.dimensions),
            "assembly": str(args.assembly),
            "extrinsics": str(args.extrinsics),
        },
        "summary": {
            "candidate_count": int(len(inference)),
            "scoring_error_count": int(len(scoring_errors)),
            "support_plane_valid": bool(support_plane.valid),
            "support_plane_inlier_points": int(support_plane.inlier_points),
            "graspnet_reference_score_feasible_count": int(
                reference_score_feasible_count
            ),
            "aperture_search_trial_count": int(aperture_search_trial_count),
            "score_feasible_count": int(score_feasible_count),
            "world_clearance_feasible_count": int(world_feasible_count),
            "final_feasible_count": int(final_feasible_count),
            "nms_survivor_count": int(nms_survivor_count),
            "ik_evaluated_count": int(ik_evaluated_count),
            "ik_success_count": int(ik_success_count),
            "final_reason_histogram": dict(final_reason_histogram),
        },
        "scoring_errors": scoring_errors,
        "nms": {
            "kept_candidate_ids": sorted(kept_ids),
            "suppressed": [
                {
                    "candidate_id": item.candidate_id,
                    "kept_candidate_id": item.kept_candidate_id,
                }
                for item in suppressed
            ],
        },
        "candidates": records,
        "caveats": [
            "GraspNet checkpoint is a third-party mirror, not the official checkpoint; its license and real-world quality are not validated.",
            "The CAD jaw installation clocking has not been measured on the physical arm; canonical 0-degree clocking is used for offline development only.",
            "Shim stack and finger zero position have not been measured.",
            "grasp_tcp_x=0.050 m is CAD/geometry derived, not a measured tool-center calibration.",
            "The support plane is estimated from one RGB-D frame and its inlier diagnostics are reported; it has not yet been cross-validated against a robot-base/table calibration or inclinometer.",
            "Support-plane estimation fails closed when the fit is unreliable; the old base-frame z=0 assumption is not used as a fallback.",
            "Whole-gripper clearance is a sampled CAD-envelope diagnostic, not a full mesh/MoveIt/whole-arm collision check.",
            "Transparent-box and real-gripper segmentation still need scene-specific validation.",
            "The banana mask is a fixed HSV segmentation for this captured frame, not a general perception result.",
            "IK convergence is not an executable-motion authorization; joint limits, singularities, full collision, hand-eye timing, and site validation remain mandatory.",
            "No camera, robot, controller, or gripper was connected; no motion or gripper command was attempted.",
        ],
    }

    _write_json(output_dir / "offline_candidates.json", payload)
    _write_json(output_dir / "support_plane.json", support_plane_payload)
    (output_dir / "report.md").write_text(
        _report_markdown(payload),
        encoding="utf-8",
    )
    np.save(masks_dir / "banana_mask.npy", banana_mask)
    Image.fromarray((banana_mask.astype(np.uint8) * 255)).save(
        masks_dir / "banana_mask.png"
    )
    Image.fromarray((roi_mask.astype(np.uint8) * 255)).save(
        masks_dir / "roi_mask.png"
    )
    np.savez_compressed(
        clouds_dir / "camera_clouds.npz",
        target_points=target_points,
        obstacle_points=obstacle_points,
        roi_points=roi_points,
        sampled_points=sampled_cloud,
        sampled_indices=sampled_indices,
        target_points_base=target_points_base,
        obstacle_points_base=obstacle_points_base,
        support_plane_candidate_points_base=support_plane_candidates,
        support_plane_inlier_mask=support_plane.inlier_mask,
        T_base_camera=t_base_camera,
        joints_rad=joints,
    )
    _write_json(
        output_dir / "scene_summary.json",
        {
            "scene": payload["scene"],
            "coordinate_transforms": payload["coordinate_transforms"],
            "support_plane": payload["support_plane"],
            "safety": payload["safety"],
            "summary": payload["summary"],
        },
    )

    print(json.dumps(_json_value({
        "offline_only": True,
        "motion_attempted": False,
        "quality_validated": False,
        "motion_authorized": False,
        "output_dir": str(output_dir),
        "support_plane_valid": bool(support_plane.valid),
        "summary": payload["summary"],
    }), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
