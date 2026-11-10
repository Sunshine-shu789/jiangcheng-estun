#!/usr/bin/env python3
"""Offline acceptance matrix for Task B arbitrary-placement grasping.

Every sample represents a fresh camera observation of the banana at a selected
tabletop XY, heading, and optional principal-axis tilt.  The sample is converted
through the calibrated eye-in-hand transform, then the *production* candidate
generator and first-safe full-route selector are exercised against the controller
FK/IK/Jacobian model from one identical measured initial joint posture.

This tool never constructs a ROS node and never publishes a robot or gripper
command.  It forces ``motion.dry_run`` in memory and injects a ready controller
state directly into the offline RobotMove backend.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from track1_grasp.config import load_config
from track1_grasp.domain import Detection, Pose, RobotState, TaskPhase
from track1_grasp.kinematics import (
    controller_fk,
    matrix_to_rpy,
    weighted_jacobian_singular_values,
)
from track1_grasp.motion import RobotMoveMotionBackend
from track1_grasp.pipeline import PickPipeline
from track1_grasp.planner import PosePlanner, transform_from_xyz_rpy, transform_point

BASE = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
DEFAULT_CONFIG = BASE / "config/track1_task_b_arbitrary_mock.yaml"
DEFAULT_OUTPUT = BASE / "logs/task_b_arbitrary_matrix.json"

# Captured read-only from the live controller immediately before the first
# boundary-point validation on 2026-10-25.  Every matrix sample starts from this
# exact posture so candidate routes are directly comparable.
DEFAULT_INITIAL_JOINTS_DEG = (
    84.471096,
    85.766795,
    -36.504083,
    -40.724928,
    89.998590,
    -3.306197,
)

# Interior points span the tested tabletop pick region.  The final point is the
# real boundary observation used by the later reachability audit.  These are test
# samples only; the production planner still derives every grasp column from a
# fresh visual observation and no coordinate is written into the task config.
DEFAULT_XY_M = (
    (-0.140, 0.400),
    (-0.060, 0.400),
    (0.040, 0.400),
    (-0.140, 0.490),
    (-0.060, 0.490),
    (0.040, 0.490),
    (-0.140, 0.580),
    (-0.060, 0.580),
    (0.040, 0.580),
    (0.051856, 0.636963),
)

DEFAULT_SUPPORT_Z_M = -0.038478
DEFAULT_YAW_DEG = (0.0, 45.0, 90.0, 135.0, 180.0)


@dataclass(frozen=True)
class MatrixSample:
    sample_id: str
    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float
    axis_tilt_deg: float


@dataclass(frozen=True)
class WorkerContext:
    config_path: str
    initial_joints_deg: tuple[float, ...]
    max_candidates: int | None = None


_WORKER_CONFIG = None
_WORKER_JOINTS = None
_WORKER_PLANNER = None
_WORKER_ROBOT_STATE = None


def _float_csv(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("expected a non-empty finite CSV list")
    return values


def _xy_csv(text: str) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        values = _float_csv(item)
        if len(values) != 2:
            raise argparse.ArgumentTypeError("each XY point must contain x,y metres")
        pairs.append((values[0], values[1]))
    if not pairs:
        raise argparse.ArgumentTypeError("at least one XY point is required")
    return tuple(pairs)


def _unique(values: Iterable[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if not any(abs(value - seen) < 1e-12 for seen in result):
            result.append(float(value))
    return tuple(result)


def build_samples(
    *,
    xy_points: Sequence[tuple[float, float]],
    support_z_m: float,
    yaw_deg: Sequence[float],
    axis_tilt_deg: Sequence[float],
    z_offsets_m: Sequence[float] = (0.0,),
) -> tuple[MatrixSample, ...]:
    samples: list[MatrixSample] = []
    for z_offset in _unique(z_offsets_m):
        for x, y in xy_points:
            for yaw in yaw_deg:
                for tilt in axis_tilt_deg:
                    sample_id = (
                        f"x{x * 1000:+.3f}_y{y * 1000:+.3f}_"
                        f"z{(support_z_m + z_offset) * 1000:+.3f}_"
                        f"yaw{yaw:+.3f}_tilt{tilt:+.3f}"
                    )
                    samples.append(
                        MatrixSample(
                            sample_id=sample_id,
                            x_m=float(x),
                            y_m=float(y),
                            z_m=float(support_z_m + z_offset),
                            yaw_deg=float(yaw),
                            axis_tilt_deg=float(tilt),
                        )
                    )
    return tuple(samples)


def _initial_state(
    initial_joints_deg: Sequence[float],
) -> tuple[np.ndarray, RobotState]:
    q_deg = np.asarray(tuple(initial_joints_deg), dtype=float)
    if q_deg.shape != (6,) or not np.all(np.isfinite(q_deg)):
        raise ValueError("initial joints must contain six finite degrees")
    q = np.radians(q_deg)
    transform = controller_fk(q)
    pose = Pose(
        tuple(float(value) for value in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    return q, RobotState(
        ready=True,
        pose=pose,
        fault=False,
        message="offline injected AutoReady",
        joints=tuple(float(value) for value in q),
        controller_state="AutoReady",
        fault_flag=0,
    )


def _backend_for_state(
    config,
    *,
    robot_state: RobotState,
    logs: list[str] | None = None,
) -> RobotMoveMotionBackend:
    """Construct only the kinematic surface of the RobotMove backend."""

    if robot_state.pose is None or robot_state.joints is None:
        raise ValueError("offline backend needs a complete robot state")
    backend = object.__new__(RobotMoveMotionBackend)
    backend.config = config.motion
    backend._log = (
        logs.append if logs is not None else (lambda _message: None)
    )
    backend._lock = threading.Lock()
    backend._state = "AutoReady"
    backend._end_positions = [
        *robot_state.pose.xyz,
        *robot_state.pose.rpy,
    ]
    backend._joint_positions = [float(value) for value in robot_state.joints]
    backend._fault_flag = 0
    backend._stamp = time.monotonic()
    return backend


def _sample_detection(
    planner: PosePlanner,
    robot_state: RobotState,
    sample: MatrixSample,
) -> Detection:
    """Express a desired base-frame support point as a camera observation."""

    if robot_state.pose is None:
        raise ValueError("sample conversion needs the initial flange pose")
    base_flange = transform_from_xyz_rpy(
        robot_state.pose.xyz,
        robot_state.pose.rpy,
    )
    base_camera = base_flange @ planner._camera_transform()
    camera_from_base = np.linalg.inv(base_camera)

    support_base = np.array(
        [sample.x_m, sample.y_m, sample.z_m],
        dtype=float,
    )
    support_camera = transform_point(camera_from_base, support_base)

    yaw = math.radians(sample.yaw_deg)
    tilt = math.radians(sample.axis_tilt_deg)
    major_axis_base = np.array(
        [
            math.cos(tilt) * math.cos(yaw),
            math.cos(tilt) * math.sin(yaw),
            math.sin(tilt),
        ],
        dtype=float,
    )
    major_axis_base /= np.linalg.norm(major_axis_base)
    minor_axis_base = np.array(
        [-math.sin(yaw), math.cos(yaw), 0.0],
        dtype=float,
    )
    camera_rotation = base_camera[:3, :3]
    major_axis_camera = camera_rotation.T @ major_axis_base
    minor_axis_camera = camera_rotation.T @ minor_axis_base

    return Detection(
        label="8-Plastic_Banana",
        confidence=0.95,
        bbox=(220.0, 180.0, 360.0, 300.0),
        depth_m=float(np.linalg.norm(support_camera)),
        camera_xyz=tuple(float(value) for value in support_camera),
        object_center_camera_xyz=tuple(
            float(value) for value in support_camera
        ),
        major_axis_camera=tuple(float(value) for value in major_axis_camera),
        minor_axis_camera=tuple(float(value) for value in minor_axis_camera),
        major_axis_length_m=0.125,
        minor_axis_length_m=0.040,
        pose_confidence=0.95,
        attributes={"held": False},
    )


def _initialize_worker(payload: WorkerContext) -> None:
    global _WORKER_CONFIG, _WORKER_JOINTS, _WORKER_PLANNER, _WORKER_ROBOT_STATE

    config = load_config(payload.config_path)
    # Hard safety gate: even when the selected YAML is a live trial config, this
    # tool only exercises pure kinematic checks and never constructs ROS.
    # There is no ROS RobotInfo callback in this offline process.  Keep the
    # injected state logically fresh while a slow candidate IK runs; this only
    # changes offline feedback age and does not relax any kinematic gate.
    config = replace(
        config,
        motion=replace(
            config.motion,
            dry_run=True,
            robot_info_timeout_s=max(3600.0, float(config.motion.robot_info_timeout_s)),
        ),
    )
    if payload.max_candidates is not None:
        config = replace(
            config,
            adaptive_inverse=replace(
                config.adaptive_inverse,
                candidates=replace(
                    config.adaptive_inverse.candidates,
                    max_count=int(payload.max_candidates),
                ),
            ),
        )
    _q, robot_state = _initial_state(payload.initial_joints_deg)
    _WORKER_CONFIG = config
    _WORKER_JOINTS = tuple(float(value) for value in payload.initial_joints_deg)
    _WORKER_PLANNER = PosePlanner(config)
    _WORKER_ROBOT_STATE = robot_state


def _route_metrics(route) -> dict[str, float | int]:
    sigmas = [
        float(segment.min_jacobian_singular_value)
        for segment in route.segments
        if math.isfinite(float(segment.min_jacobian_singular_value))
    ]
    return {
        "min_sigma": min(sigmas) if sigmas else 0.0,
        "max_joint_step_deg": max(
            (float(segment.max_joint_step_deg) for segment in route.segments),
            default=0.0,
        ),
        "max_joint_accel_deg": max(
            (
                float(segment.max_joint_acceleration_deg)
                for segment in route.segments
            ),
            default=0.0,
        ),
        "waypoints": sum(
            max(0, len(segment.waypoints) - 1) for segment in route.segments
        ),
        "segments": len(route.segments),
    }


def check_sample(sample: MatrixSample) -> dict[str, object]:
    if (
        _WORKER_CONFIG is None
        or _WORKER_JOINTS is None
        or _WORKER_PLANNER is None
        or _WORKER_ROBOT_STATE is None
    ):
        raise RuntimeError("matrix worker was not initialized")

    started = time.monotonic()
    messages: list[str] = []
    row: dict[str, object] = {
        **asdict(sample),
        "status": "FAIL",
        "error": "",
        "planning_s": None,
        "precheck_s": None,
        "total_s": None,
        "candidate_count": 0,
        "selected_yaw_offset_deg": None,
        "selected_pitch_offset_deg": None,
        "orientation_change_deg": None,
        "grasp_flange_xyz": None,
        "grasp_rpy": None,
        "commanded_tip_xyz": None,
        "tip_position_error_m": None,
    }

    try:
        detection = _sample_detection(
            _WORKER_PLANNER,
            _WORKER_ROBOT_STATE,
            sample,
        )
        backend = _backend_for_state(
            _WORKER_CONFIG,
            robot_state=_WORKER_ROBOT_STATE,
            logs=messages,
        )

        planning_started = time.monotonic()
        candidates = _WORKER_PLANNER.plan_adaptive_candidates(
            detection,
            _WORKER_ROBOT_STATE,
            place_name=sample.sample_id,
            return_to_pick=True,
        )
        planning_elapsed = time.monotonic() - planning_started
        row["candidate_count"] = len(candidates)
        row["planning_s"] = planning_elapsed

        # Reuse the production selector directly.  Only the dependencies it
        # actually reads are attached, so this exercises the same endpoint
        # prefilter, full six-segment IK route, singular-value margin,
        # branch-jump, joint-step and acceleration gates as the live pipeline.
        pipeline = object.__new__(PickPipeline)
        pipeline.config = _WORKER_CONFIG
        pipeline.motion = backend
        pipeline._log = messages.append

        def emit(_phase: TaskPhase, message: str) -> None:
            messages.append(str(message))

        precheck_started = time.monotonic()
        plan, route, failures = pipeline._precheck_adaptive_candidates(
            candidates,
            emit,
            place_name=sample.sample_id,
        )
        precheck_elapsed = time.monotonic() - precheck_started
        row["precheck_s"] = precheck_elapsed

        if plan is None or route is None:
            row["error"] = " | ".join(failures[:8])
            row["messages"] = messages[-80:]
            return row

        pipeline._validate_vertical_grasp_plan(
            plan,
            emit,
            sample.sample_id,
            require_tool_down=True,
        )
        selected = next(
            candidate for candidate in candidates if candidate.plan is plan
        )
        metrics = _route_metrics(route)
        minimum_sigma = float(metrics["min_sigma"])
        required_sigma = (
            float(_WORKER_CONFIG.motion.cartesian_min_jacobian_singular_value)
            + float(_WORKER_CONFIG.adaptive_inverse.selection.min_sigma_margin)
        )
        if minimum_sigma < required_sigma:
            raise RuntimeError(
                f"selected route min_sigma {minimum_sigma:.6f} is below "
                f"required {required_sigma:.6f}"
            )

        tip_offset = np.asarray(
            _WORKER_CONFIG.pose.gripper_tip_offset_flange_m,
            dtype=float,
        )
        if tip_offset.shape != (3,):
            raise RuntimeError("gripper tip offset must be a 3-vector")
        from track1_grasp.planner import euler_to_matrix

        commanded_tip = (
            np.asarray(plan.grasp_pose.xyz, dtype=float)
            + euler_to_matrix(plan.grasp_pose.rpy) @ tip_offset
        )
        expected_tip = np.array(
            [
                sample.x_m,
                sample.y_m,
                sample.z_m
                + float(
                    _WORKER_CONFIG.adaptive_inverse.candidates.jaw_tip_clearance_m
                ),
            ],
            dtype=float,
        )
        tip_error = float(np.linalg.norm(commanded_tip - expected_tip))
        if tip_error > 1e-6:
            raise RuntimeError(
                f"jaw-tip geometry error {tip_error * 1000:.6f} mm exceeds 0.001 mm"
            )

        start_sigma = float(
            np.min(
                weighted_jacobian_singular_values(
                    np.radians(np.asarray(_WORKER_JOINTS, dtype=float))
                )
            )
        )
        row.update(
            {
                "status": "PASS",
                "candidate_count": len(candidates),
                "selected_yaw_offset_deg": float(
                    selected.yaw_offset_deg
                ),
                "selected_pitch_offset_deg": float(
                    selected.pitch_offset_deg
                ),
                "orientation_change_deg": float(
                    math.degrees(selected.orientation_change_rad)
                ),
                "grasp_flange_xyz": [
                    float(value) for value in plan.grasp_pose.xyz
                ],
                "grasp_rpy": [
                    float(value) for value in plan.grasp_pose.rpy
                ],
                "commanded_tip_xyz": [
                    float(value) for value in commanded_tip
                ],
                "tip_position_error_m": tip_error,
                "start_sigma": start_sigma,
                **metrics,
            }
        )
        row["messages"] = messages[-80:]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["messages"] = messages[-80:]
    finally:
        row["total_s"] = time.monotonic() - started
    return row


def _summary(rows: Sequence[dict[str, object]], config_path: Path) -> dict[str, object]:
    passed = [row for row in rows if row["status"] == "PASS"]
    failed = [row for row in rows if row["status"] != "PASS"]
    return {
        "config": str(config_path),
        "sigma_floor": None,
        "samples": len(rows),
        "passed": len(passed),
        "failed": len(failed),
        "success_rate": len(passed) / len(rows) if rows else 0.0,
        "max_total_s": max(
            (float(row["total_s"] or 0.0) for row in rows),
            default=0.0,
        ),
        "max_planning_s": max(
            (float(row["planning_s"] or 0.0) for row in rows),
            default=0.0,
        ),
        "max_precheck_s": max(
            (float(row["precheck_s"] or 0.0) for row in rows),
            default=0.0,
        ),
        "failed_sample_ids": [str(row["sample_id"]) for row in failed],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--xy",
        type=_xy_csv,
        default=DEFAULT_XY_M,
        help="semicolon-separated x,y pairs in metres",
    )
    parser.add_argument(
        "--support-z-m",
        type=float,
        default=DEFAULT_SUPPORT_Z_M,
    )
    parser.add_argument(
        "--z-offset-mm",
        type=_float_csv,
        default=(0.0,),
        help="CSV support-plane vertical offsets in millimetres",
    )
    parser.add_argument("--yaw-deg", type=_float_csv, default=DEFAULT_YAW_DEG)
    parser.add_argument("--axis-tilt-deg", type=_float_csv, default=(0.0,))
    parser.add_argument(
        "--initial-joints-deg",
        type=_float_csv,
        default=DEFAULT_INITIAL_JOINTS_DEG,
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="optional in-memory max_count override; the YAML is not modified",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    if not config.adaptive_inverse.enabled:
        raise SystemExit("selected config must enable adaptive_inverse")
    if not config.adaptive_inverse.arbitrary_placement:
        raise SystemExit("selected config must enable arbitrary_placement")
    if config.pose.gripper_tip_offset_flange_m is None:
        raise SystemExit("selected config needs pose.gripper_tip_offset_flange_m")
    if not config.pose.extrinsics_file:
        raise SystemExit("selected config needs calibrated extrinsics")

    samples = build_samples(
        xy_points=args.xy,
        support_z_m=float(args.support_z_m),
        yaw_deg=args.yaw_deg,
        axis_tilt_deg=args.axis_tilt_deg,
        z_offsets_m=tuple(value / 1000.0 for value in args.z_offset_mm),
    )
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("matrix has no samples")

    payload = WorkerContext(
        config_path=str(config_path),
        initial_joints_deg=tuple(args.initial_joints_deg),
        max_candidates=args.max_candidates,
    )
    rows: list[dict[str, object]] = []
    workers = max(1, min(int(args.workers), len(samples)))
    print(
        f"Task B arbitrary matrix: {len(samples)} sample(s), "
        f"workers={workers}, config={config_path.name}",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(payload,),
    ) as pool:
        futures = {
            pool.submit(check_sample, sample): sample for sample in samples
        }
        for future in concurrent.futures.as_completed(futures):
            sample = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    **asdict(sample),
                    "status": "FAIL",
                    "error": f"worker failure: {type(exc).__name__}: {exc}",
                    "total_s": None,
                }
            rows.append(row)
            if not args.quiet or row["status"] != "PASS":
                print(
                    f"{row['status']} {sample.sample_id} "
                    f"yaw={sample.yaw_deg:.3f} tilt={sample.axis_tilt_deg:.3f} "
                    f"elapsed={float(row.get('total_s') or 0.0):.3f}s"
                    + (
                        ""
                        if row["status"] == "PASS"
                        else f" error={row.get('error', '')}"
                    ),
                    flush=True,
                )

    rows.sort(key=lambda row: str(row["sample_id"]))
    summary = _summary(rows, config_path)
    summary["sigma_floor"] = (
        float(config.motion.cartesian_min_jacobian_singular_value)
        + float(config.adaptive_inverse.selection.min_sigma_margin)
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "initial_joints_deg": list(args.initial_joints_deg),
                "support_z_m": float(args.support_z_m),
                "yaw_deg": list(args.yaw_deg),
                "axis_tilt_deg": list(args.axis_tilt_deg),
                "z_offset_mm": list(args.z_offset_mm),
                "summary": summary,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"SUMMARY pass={summary['passed']}/{summary['samples']} "
        f"failed={summary['failed']} "
        f"max_elapsed={summary['max_total_s']:.3f}s",
        flush=True,
    )
    print(f"WROTE {output}", flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
