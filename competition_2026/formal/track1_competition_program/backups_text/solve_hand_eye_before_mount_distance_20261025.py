#!/usr/bin/env python3
"""Solve and validate the eye-in-hand transform from captured chessboard poses.

Convention returned by this tool:
    T_flange_camera maps camera-frame points into the robot flange frame.

The primary estimate uses OpenCV TSAI and is then optionally refined by robust
non-linear least squares against the fixed-board constraint.  Quality is
reported with fixed-board residuals and leave-one-pose-out validation.  The
script never connects to the robot or camera and never commands motion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml

from track1_grasp.kinematics import (
    controller_fk,
    matrix_to_rpy,
    rotation_angle_error,
    rotation_matrix_to_vector,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "calibration/hand_eye_20261025/results"
METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proper_rotation(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def transform_from_rotation_translation(
    rotation: np.ndarray, translation: Sequence[float]
) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = proper_rotation(rotation)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def average_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("at least one transform is required")
    translation = np.mean([transform[:3, 3] for transform in transforms], axis=0)
    rotation_sum = np.sum([transform[:3, :3] for transform in transforms], axis=0)
    return transform_from_rotation_translation(rotation_sum, translation)


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(proper_rotation(rotation))[0].reshape(3)


@dataclass(frozen=True)
class CalibrationPose:
    index: int
    pose_id: str
    base_flange: np.ndarray
    camera_board: np.ndarray


def load_poses(manifest_path: Path, *, allow_incomplete: bool) -> tuple[dict[str, Any], list[CalibrationPose]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not allow_incomplete and manifest.get("completed") is not True:
        raise RuntimeError("标定 manifest 尚未完成；仅预览时可使用 --allow-incomplete")
    records = manifest.get("pose_records")
    if not isinstance(records, list) or len(records) < 3:
        raise RuntimeError("manifest 至少需要 3 条姿态记录")

    poses: list[CalibrationPose] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for position, record in enumerate(records):
        try:
            index = int(record["index"])
            pose_id = str(record["id"])
            joints = np.asarray(record["actual_joints_rad"], dtype=float)
            camera_board = np.asarray(
                record["board_transform_camera_board"], dtype=float
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"第 {position} 条姿态记录字段不完整") from exc
        if pose_id in seen_ids or index in seen_indices:
            raise RuntimeError(f"姿态 id/index 重复: {pose_id}/{index}")
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"{pose_id}: actual_joints_rad 无效")
        if camera_board.shape != (4, 4) or not np.all(np.isfinite(camera_board)):
            raise RuntimeError(f"{pose_id}: camera-board 位姿无效")
        if not np.allclose(camera_board[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise RuntimeError(f"{pose_id}: camera-board 位姿末行不是 [0,0,0,1]")
        camera_board = camera_board.copy()
        camera_board[:3, :3] = proper_rotation(camera_board[:3, :3])
        seen_ids.add(pose_id)
        seen_indices.add(index)
        poses.append(
            CalibrationPose(
                index=index,
                pose_id=pose_id,
                base_flange=controller_fk(joints),
                camera_board=camera_board,
            )
        )
    return manifest, poses


def cv_calibrate(
    poses: Sequence[CalibrationPose], method_name: str
) -> tuple[np.ndarray, dict[str, Any]]:
    rotations_gripper = [pose.base_flange[:3, :3] for pose in poses]
    translations_gripper = [
        pose.base_flange[:3, 3].reshape(3, 1) for pose in poses
    ]
    rotations_target = [pose.camera_board[:3, :3] for pose in poses]
    translations_target = [
        pose.camera_board[:3, 3].reshape(3, 1) for pose in poses
    ]
    rotation, translation = cv2.calibrateHandEye(
        rotations_gripper,
        translations_gripper,
        rotations_target,
        translations_target,
        method=METHODS[method_name],
    )
    transform = transform_from_rotation_translation(
        rotation, np.asarray(translation, dtype=float).reshape(3)
    )
    return transform, {
        "method": method_name,
        "rotation_rpy_deg": np.degrees(matrix_to_rpy(transform[:3, :3])).tolist(),
        "translation_mm": (transform[:3, 3] * 1000.0).tolist(),
    }


def board_poses(
    poses: Sequence[CalibrationPose], flange_camera: np.ndarray
) -> list[np.ndarray]:
    return [
        pose.base_flange @ flange_camera @ pose.camera_board for pose in poses
    ]


def board_consistency(
    poses: Sequence[CalibrationPose], flange_camera: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    fixed_board_poses = board_poses(poses, flange_camera)
    average_board = average_transform(fixed_board_poses)
    translations_mm: list[float] = []
    rotations_deg: list[float] = []
    per_pose: list[dict[str, Any]] = []
    for pose, fixed_board in zip(poses, fixed_board_poses):
        translation_error_mm = float(
            np.linalg.norm(fixed_board[:3, 3] - average_board[:3, 3]) * 1000.0
        )
        rotation_error_deg = float(
            np.degrees(
                rotation_angle_error(fixed_board[:3, :3], average_board[:3, :3])
            )
        )
        translations_mm.append(translation_error_mm)
        rotations_deg.append(rotation_error_deg)
        per_pose.append(
            {
                "index": pose.index,
                "id": pose.pose_id,
                "translation_error_mm": translation_error_mm,
                "rotation_error_deg": rotation_error_deg,
            }
        )
    translation_array = np.asarray(translations_mm)
    rotation_array = np.asarray(rotations_deg)
    metrics = {
        "pose_count": len(poses),
        "translation_rms_mm": float(np.sqrt(np.mean(translation_array**2))),
        "translation_max_mm": float(np.max(translation_array)),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation_array**2))),
        "rotation_max_deg": float(np.max(rotation_array)),
        "average_board_transform_base_board": average_board.tolist(),
        "per_pose": per_pose,
    }
    return metrics, average_board


def refine_hand_eye(
    poses: Sequence[CalibrationPose],
    initial: np.ndarray,
    *,
    rotation_characteristic_m: float,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return initial.copy(), {"applied": False, "reason": "scipy unavailable"}

    def unpack(parameters: np.ndarray) -> np.ndarray:
        rotation = cv2.Rodrigues(parameters[3:6].reshape(3, 1))[0]
        return transform_from_rotation_translation(rotation, parameters[:3])

    def residual(parameters: np.ndarray) -> np.ndarray:
        candidate = unpack(parameters)
        fixed = board_poses(poses, candidate)
        average = average_transform(fixed)
        values: list[float] = []
        for transform in fixed:
            translation_error = (transform[:3, 3] - average[:3, 3]) * 1000.0
            relative_rotation = transform[:3, :3] @ average[:3, :3].T
            rotation_error = rotation_vector(relative_rotation)
            values.extend(translation_error.tolist())
            values.extend(
                (rotation_error * rotation_characteristic_m * 1000.0).tolist()
            )
        return np.asarray(values, dtype=float)

    initial_parameters = np.concatenate(
        [initial[:3, 3], rotation_vector(initial[:3, :3])]
    )
    lower = np.concatenate(
        [initial_parameters[:3] - 0.20, np.full(3, -math.pi + 1e-6)]
    )
    upper = np.concatenate(
        [initial_parameters[:3] + 0.20, np.full(3, math.pi - 1e-6)]
    )
    result = least_squares(
        residual,
        initial_parameters,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=max_nfev,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    refined = unpack(result.x)
    return refined, {
        "applied": bool(result.success or np.any(result.x != initial_parameters)),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "rotation_characteristic_m": rotation_characteristic_m,
    }


def estimate_pipeline(
    poses: Sequence[CalibrationPose],
    *,
    method_name: str,
    refine: bool,
    rotation_characteristic_m: float,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    initial, method_payload = cv_calibrate(poses, method_name)
    initial_metrics, _ = board_consistency(poses, initial)
    estimate = initial
    refinement: dict[str, Any] = {"applied": False}
    metrics = initial_metrics
    if refine:
        refined, refinement = refine_hand_eye(
            poses,
            initial,
            rotation_characteristic_m=rotation_characteristic_m,
            max_nfev=max_nfev,
        )
        refined_metrics, _ = board_consistency(poses, refined)
        initial_score = math.hypot(
            initial_metrics["translation_rms_mm"],
            initial_metrics["rotation_rms_deg"]
            * math.degrees(rotation_characteristic_m),
        )
        refined_score = math.hypot(
            refined_metrics["translation_rms_mm"],
            refined_metrics["rotation_rms_deg"]
            * math.degrees(rotation_characteristic_m),
        )
        refinement["initial_score"] = initial_score
        refinement["refined_score"] = refined_score
        refinement["selected"] = refined_score <= initial_score
        if refinement["selected"]:
            estimate = refined
            metrics = refined_metrics
        refinement["selected_rotation_rpy_deg"] = np.degrees(
            matrix_to_rpy(estimate[:3, :3])
        ).tolist()
        refinement["selected_translation_mm"] = (estimate[:3, 3] * 1000.0).tolist()
    return estimate, {
        "initial": method_payload,
        "initial_board_consistency": initial_metrics,
        "refinement": refinement,
        "selected_board_consistency": metrics,
    }


def leave_one_out_validation(
    poses: Sequence[CalibrationPose],
    *,
    method_name: str,
    refine: bool,
    rotation_characteristic_m: float,
    max_nfev: int,
) -> dict[str, Any]:
    if len(poses) < 4:
        return {"valid": False, "reason": "fewer than four poses"}
    rows: list[dict[str, Any]] = []
    for held_out_position in range(len(poses)):
        training = [
            pose
            for position, pose in enumerate(poses)
            if position != held_out_position
        ]
        held_out = poses[held_out_position]
        try:
            estimate, _ = estimate_pipeline(
                training,
                method_name=method_name,
                refine=refine,
                rotation_characteristic_m=rotation_characteristic_m,
                max_nfev=max_nfev,
            )
            _training_metrics, training_board = board_consistency(
                training, estimate
            )
        except (cv2.error, RuntimeError, ValueError) as exc:
            rows.append(
                {
                    "held_out_id": held_out.pose_id,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        predicted = (
            held_out.base_flange @ estimate @ held_out.camera_board
        )
        translation_error_mm = float(
            np.linalg.norm(predicted[:3, 3] - training_board[:3, 3]) * 1000.0
        )
        rotation_error_deg = float(
            np.degrees(
                rotation_angle_error(
                    predicted[:3, :3], training_board[:3, :3]
                )
            )
        )
        rows.append(
            {
                "held_out_index": held_out.index,
                "held_out_id": held_out.pose_id,
                "success": True,
                "translation_error_mm": translation_error_mm,
                "rotation_error_deg": rotation_error_deg,
            }
        )
    successful = [row for row in rows if row["success"]]
    if not successful:
        return {"valid": False, "rows": rows, "reason": "all folds failed"}
    translation = np.asarray(
        [row["translation_error_mm"] for row in successful], dtype=float
    )
    rotation = np.asarray(
        [row["rotation_error_deg"] for row in successful], dtype=float
    )
    return {
        "valid": len(successful) == len(rows),
        "fold_count": len(rows),
        "successful_fold_count": len(successful),
        "translation_rms_mm": float(np.sqrt(np.mean(translation**2))),
        "translation_max_mm": float(np.max(translation)),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation**2))),
        "rotation_max_deg": float(np.max(rotation)),
        "rows": rows,
    }


def method_comparison(
    poses: Sequence[CalibrationPose], primary: np.ndarray
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for method_name in METHODS:
        try:
            estimate, payload = cv_calibrate(poses, method_name)
            metrics, _ = board_consistency(poses, estimate)
            rows[method_name] = {
                "success": True,
                **payload,
                "relative_to_tsai_translation_mm": float(
                    np.linalg.norm(estimate[:3, 3] - primary[:3, 3]) * 1000.0
                ),
                "relative_to_tsai_rotation_deg": float(
                    np.degrees(
                        rotation_angle_error(
                            estimate[:3, :3], primary[:3, :3]
                        )
                    )
                ),
                "board_consistency": metrics,
            }
        except (cv2.error, RuntimeError) as exc:
            rows[method_name] = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return rows


def evaluate_quality(
    metrics: dict[str, Any],
    leave_one_out: dict[str, Any],
    methods: dict[str, Any],
    *,
    expected_offset_m: np.ndarray,
    offset_warning_mm: float,
) -> dict[str, Any]:
    checks = {
        "fixed_board_translation_rms_le_2mm": metrics["translation_rms_mm"] <= 2.0,
        "fixed_board_translation_max_le_4mm": metrics["translation_max_mm"] <= 4.0,
        "fixed_board_rotation_rms_le_0_5deg": metrics["rotation_rms_deg"] <= 0.5,
        "fixed_board_rotation_max_le_1deg": metrics["rotation_max_deg"] <= 1.0,
        "leave_one_out_complete": bool(leave_one_out.get("valid")),
        "leave_one_out_translation_rms_le_2mm": float(
            leave_one_out.get("translation_rms_mm", float("inf"))
        )
        <= 2.0,
        "leave_one_out_translation_max_le_5mm": float(
            leave_one_out.get("translation_max_mm", float("inf"))
        )
        <= 5.0,
        "leave_one_out_rotation_rms_le_0_5deg": float(
            leave_one_out.get("rotation_rms_deg", float("inf"))
        )
        <= 0.5,
        "leave_one_out_rotation_max_le_1deg": float(
            leave_one_out.get("rotation_max_deg", float("inf"))
        )
        <= 1.0,
    }
    offset_error = None
    offset_check = None
    if expected_offset_m is not None:
        offset_error = float(np.linalg.norm(expected_offset_m))
        offset_check = offset_error <= offset_warning_mm / 1000.0
    return {
        "checks": checks,
        "all_required_checks_pass": all(checks.values()),
        "method_tsai_available": bool(methods.get("TSAI", {}).get("success")),
        "operator_offset_within_warning": offset_check,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--method", choices=tuple(METHODS), default="TSAI")
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--rotation-characteristic-m", type=float, default=0.25)
    parser.add_argument("--max-nfev", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--expected-offset-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, -65.0, 0.0),
    )
    parser.add_argument("--offset-warning-mm", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rotation_characteristic_m <= 0.0:
        raise RuntimeError("--rotation-characteristic-m 必须为正")
    if args.max_nfev <= 0:
        raise RuntimeError("--max-nfev 必须为正")
    if args.offset_warning_mm <= 0.0:
        raise RuntimeError("--offset-warning-mm 必须为正")

    manifest_path = args.manifest.expanduser().resolve()
    manifest, poses = load_poses(
        manifest_path, allow_incomplete=args.allow_incomplete
    )
    estimate, estimation = estimate_pipeline(
        poses,
        method_name=args.method,
        refine=not args.no_refine,
        rotation_characteristic_m=args.rotation_characteristic_m,
        max_nfev=args.max_nfev,
    )
    metrics, _average_board = board_consistency(poses, estimate)
    leave_one_out = leave_one_out_validation(
        poses,
        method_name=args.method,
        refine=not args.no_refine,
        rotation_characteristic_m=args.rotation_characteristic_m,
        max_nfev=args.max_nfev,
    )
    methods = method_comparison(poses, estimate)
    expected_offset = np.asarray(args.expected_offset_mm, dtype=float) / 1000.0
    measured_offset = estimate[:3, 3]
    offset_delta_mm = (measured_offset - expected_offset) * 1000.0
    quality = evaluate_quality(
        metrics,
        leave_one_out,
        methods,
        expected_offset_m=offset_delta_mm / 1000.0,
        offset_warning_mm=args.offset_warning_mm,
    )

    stamp = datetime.now().astimezone()
    calibration_id = (
        f"hand_eye_d405_{stamp.strftime('%Y%m%d_%H%M%S')}_"
        f"{sha256_file(manifest_path)[:10]}"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / calibration_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "created_at": stamp.isoformat(timespec="seconds"),
        "calibration_id": calibration_id,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_completed": bool(manifest.get("completed")),
        "convention": {
            "name": "T_flange_camera",
            "meaning": "maps camera-frame points into the robot flange frame",
            "board_observation_convention": "T_camera_board",
        },
        "pose_count": len(poses),
        "method": args.method,
        "estimator": estimation,
        "camera_to_flange_matrix": estimate.tolist(),
        "translation_mm": (measured_offset * 1000.0).tolist(),
        "rotation_rpy_deg": np.degrees(matrix_to_rpy(estimate[:3, :3])).tolist(),
        "expected_operator_offset_mm": (
            expected_offset * 1000.0
        ).tolist(),
        "estimated_minus_expected_offset_mm": offset_delta_mm.tolist(),
        "operator_offset_error_mm": float(
            np.linalg.norm(offset_delta_mm)
        ),
        "fixed_board_consistency": metrics,
        "leave_one_out_validation": leave_one_out,
        "method_comparison": methods,
        "quality": quality,
        "validated": bool(
            quality["all_required_checks_pass"]
            and quality["method_tsai_available"]
            and quality["operator_offset_within_warning"] is True
        ),
        "ready_for_task_b_activation": False,
        "validation_note": (
            "This file is not activated automatically. Review the quality "
            "report and operator geometry check before changing Task B "
            "extrinsics metadata."
        ),
    }
    json_path = output_dir / "hand_eye_result.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist() if isinstance(value, np.ndarray) else None)
        + "\n",
        encoding="utf-8",
    )
    yaml_payload = {
        "matrix": result["camera_to_flange_matrix"],
        "translation_mm": result["translation_mm"],
        "rotation_rpy_deg": result["rotation_rpy_deg"],
        "validated": result["validated"],
        "calibration_id": calibration_id,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": result["manifest_sha256"],
        "operator_offset_error_mm": result["operator_offset_error_mm"],
        "fixed_board_translation_rms_mm": metrics["translation_rms_mm"],
        "fixed_board_rotation_rms_deg": metrics["rotation_rms_deg"],
        "leave_one_out_translation_rms_mm": leave_one_out.get(
            "translation_rms_mm"
        ),
        "leave_one_out_rotation_rms_deg": leave_one_out.get(
            "rotation_rms_deg"
        ),
    }
    (output_dir / "camera_to_flange_calibrated.yaml").write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(str(json_path), flush=True)
    print(
        json.dumps(
            {
                "validated": result["validated"],
                "translation_mm": result["translation_mm"],
                "rotation_rpy_deg": result["rotation_rpy_deg"],
                "fixed_board_translation_rms_mm": metrics["translation_rms_mm"],
                "fixed_board_rotation_rms_deg": metrics["rotation_rms_deg"],
                "leave_one_out_translation_rms_mm": leave_one_out.get(
                    "translation_rms_mm"
                ),
                "leave_one_out_rotation_rms_deg": leave_one_out.get(
                    "rotation_rms_deg"
                ),
                "operator_offset_error_mm": result["operator_offset_error_mm"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
