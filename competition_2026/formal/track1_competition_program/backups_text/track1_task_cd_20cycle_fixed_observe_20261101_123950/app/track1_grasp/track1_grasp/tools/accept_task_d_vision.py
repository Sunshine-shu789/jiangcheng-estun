#!/usr/bin/env python3
"""Read-only Task D RGB-D/V11/unknown-proposal acceptance capture.

This tool deliberately has no path to robot motion or gripper control.  It
loads the formal Track 1 configuration, applies Task D's mode gates, builds
only the vision source, captures a bounded number of frames, and writes an
auditable RGB-D report.

It is not a replacement for the normal ``run_task_d`` safety gate.  The normal
task launcher still requires ``/RobotInfo`` and all motion/gripper checks.  This
entry exists so the camera, V11 model, depth stream, stabilizer and Task D
unknown proposer can be validated while no control cabinet is connected.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..config import Track1Config, VisionConfig, apply_mode, load_config
from ..domain import Detection, Frame, TaskMode
from ..vision import build_vision
from ..vision_runtime import render_preview


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "config" / "track1_task_cd_40class_v11_official.yaml"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs"
DEFAULT_FRAMES = 15


class AcceptanceError(RuntimeError):
    """Raised when read-only acceptance preconditions are not satisfied."""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    """Convert NumPy and non-finite values into strict JSON-safe values."""

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists():
        raise AcceptanceError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _unique_default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = DEFAULT_OUTPUT_ROOT / f"task_d_vision_acceptance_{stamp}"
    candidate = base
    index = 1
    while candidate.exists():
        candidate = Path(f"{base}_{index:02d}")
        index += 1
    return candidate


def assert_read_only_safety(config: Track1Config) -> None:
    """Fail closed unless this capture is explicitly non-actuating Task D."""

    if config.task.mode is not TaskMode.D:
        raise AcceptanceError(
            f"read-only acceptance requires task.mode=D, got {config.task.mode.value}"
        )
    if not config.vision.unknown_proposer.enabled:
        raise AcceptanceError(
            "Task D read-only acceptance requires the unknown proposer to be enabled"
        )
    if config.motion.dry_run is not True:
        raise AcceptanceError(
            "refusing read-only acceptance because motion.dry_run is not true"
        )
    if config.gripper.backend != "mock":
        raise AcceptanceError(
            "refusing read-only acceptance because gripper.backend is not mock"
        )


def _depth_metrics(depth_m: Any, vision: VisionConfig) -> dict[str, Any]:
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or depth.size == 0:
        raise AcceptanceError(
            f"depth frame must be a non-empty 2-D array, got shape={depth.shape}"
        )

    finite = np.isfinite(depth)
    positive = finite & (depth > 0.0)
    lower = float(vision.unknown_proposer.min_depth_m)
    upper = float(vision.unknown_proposer.max_depth_m)
    in_range = positive & (depth >= lower) & (depth <= upper)
    finite_positive_values = depth[positive]
    in_range_values = depth[in_range]

    height, width = depth.shape
    y0 = height // 4
    y1 = height - height // 4
    x0 = width // 4
    x1 = width - width // 4
    central = in_range[y0:y1, x0:x1]

    quantiles: dict[str, float | None] = {}
    if in_range_values.size:
        for name, percentile in (
            ("p01", 1.0),
            ("p10", 10.0),
            ("p50", 50.0),
            ("p90", 90.0),
            ("p99", 99.0),
        ):
            quantiles[name] = float(np.quantile(in_range_values, percentile / 100.0))
    else:
        quantiles = {name: None for name in ("p01", "p10", "p50", "p90", "p99")}

    return {
        "shape": [int(height), int(width)],
        "total_pixels": int(depth.size),
        "finite_positive_pixels": int(positive.sum()),
        "finite_positive_ratio": float(positive.sum() / depth.size),
        "in_configured_range_pixels": int(in_range.sum()),
        "in_configured_range_ratio": float(in_range.sum() / depth.size),
        "central_roi": [int(x0), int(y0), int(x1), int(y1)],
        "central_roi_in_range_pixels": int(central.sum()),
        "central_roi_in_range_ratio": float(central.mean()),
        "configured_range_m": [lower, upper],
        "quantiles_m": quantiles,
        "finite_positive_quantiles_m": (
            {
                "p01": float(np.quantile(finite_positive_values, 0.01)),
                "p50": float(np.quantile(finite_positive_values, 0.50)),
                "p99": float(np.quantile(finite_positive_values, 0.99)),
            }
            if finite_positive_values.size
            else {"p01": None, "p50": None, "p99": None}
        ),
    }


def _track_id(detection: Detection) -> str:
    explicit = str(detection.instance_id or "").strip()
    if explicit:
        return explicit
    raw = detection.attributes.get("track_id")
    return "" if raw is None else str(raw)


def _detection_payload(detection: Detection, unknown_labels: set[str]) -> dict[str, Any]:
    normalized_label = detection.label.strip().casefold()
    return {
        "kind": "unknown" if normalized_label in unknown_labels else "known",
        "label": detection.label,
        "confidence": float(detection.confidence),
        "stable_confidence": float(
            detection.attributes.get("stable_confidence", detection.confidence)
        ),
        "raw_confidence": float(
            detection.attributes.get("raw_confidence", detection.confidence)
        ),
        "bbox_xyxy_px": [float(value) for value in detection.bbox],
        "depth_m": None if detection.depth_m is None else float(detection.depth_m),
        "camera_xyz_m": _json_safe(detection.camera_xyz),
        "object_center_camera_xyz_m": _json_safe(
            detection.object_center_camera_xyz
        ),
        "major_axis_camera": _json_safe(detection.major_axis_camera),
        "minor_axis_camera": _json_safe(detection.minor_axis_camera),
        "major_axis_length_m": _json_safe(detection.major_axis_length_m),
        "minor_axis_length_m": _json_safe(detection.minor_axis_length_m),
        "pose_confidence": _json_safe(detection.pose_confidence),
        "graspability": float(detection.graspability),
        "instance_id": detection.instance_id,
        "track_id": _track_id(detection),
        "attributes": _json_safe(detection.attributes),
    }


def _capture_record(
    index: int,
    frame: Frame,
    *,
    capture_elapsed_ms: float,
    unknown_labels: set[str],
) -> tuple[dict[str, Any], int, int]:
    detection_payloads = [
        _detection_payload(detection, unknown_labels)
        for detection in frame.detections
    ]
    known = [item for item in detection_payloads if item["kind"] == "known"]
    unknown = [item for item in detection_payloads if item["kind"] == "unknown"]
    labels = Counter(item["label"] for item in detection_payloads)
    unknown_track_ids = sorted(
        {
            str(item["track_id"])
            for item in unknown
            if str(item["track_id"]).strip()
        }
    )

    intrinsics = frame.camera_intrinsics
    return (
        {
            "index": int(index),
            "frame_id": frame.frame_id,
            "stamp_s": float(frame.stamp_s),
            "capture_elapsed_ms": float(capture_elapsed_ms),
            "detection_count": len(detection_payloads),
            "known_count": len(known),
            "unknown_count": len(unknown),
            "labels": dict(sorted(labels.items())),
            "known_track_ids": sorted(
                {
                    str(item["track_id"])
                    for item in known
                    if str(item["track_id"]).strip()
                }
            ),
            "unknown_track_ids": unknown_track_ids,
            "detections": detection_payloads,
            "camera_intrinsics": (
                None
                if intrinsics is None
                else {
                    "width": int(intrinsics.width),
                    "height": int(intrinsics.height),
                    "fx": float(intrinsics.fx),
                    "fy": float(intrinsics.fy),
                    "cx": float(intrinsics.cx),
                    "cy": float(intrinsics.cy),
                }
            ),
        },
        len(known),
        len(unknown),
    )


def _bgr_u8(image: Any, *, field: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2).astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, :3].astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] == 3:
        return array.astype(np.uint8, copy=False)
    raise AcceptanceError(
        f"{field} must have shape (H,W), (H,W,3) or (H,W,4), got {array.shape}"
    )


def _depth_color(depth_m: Any, vision: VisionConfig) -> np.ndarray:
    import cv2

    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise AcceptanceError(f"depth must be 2-D, got {depth.shape}")
    lower = float(vision.unknown_proposer.min_depth_m)
    upper = float(vision.unknown_proposer.max_depth_m)
    if not upper > lower:
        raise AcceptanceError("unknown proposer depth range must be increasing")

    valid = np.isfinite(depth) & (depth >= lower) & (depth <= upper)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    scaled = (depth[valid] - lower) * (255.0 / (upper - lower))
    normalized[valid] = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)
    return colored


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    import cv2

    if not cv2.imwrite(
        str(path),
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    ):
        raise AcceptanceError(f"failed to write image: {path}")


def _save_frame_artifacts(
    output_dir: Path,
    record: dict[str, Any],
    frame: Frame,
    vision: VisionConfig,
) -> None:
    image = _bgr_u8(frame.image, field="frame.image")
    if frame.depth_m is None:
        raise AcceptanceError(f"{frame.frame_id} has no aligned depth frame")

    stem = f"frame_{record['index']:03d}"
    rgb_path = output_dir / f"{stem}_rgb.jpg"
    depth_path = output_dir / f"{stem}_depth_color.jpg"
    depth_npz_path = output_dir / f"{stem}_depth_m.npz"
    overlay_path = output_dir / f"{stem}_overlay.jpg"

    _write_jpeg(rgb_path, image)
    _write_jpeg(depth_path, _depth_color(frame.depth_m, vision))
    np.savez_compressed(
        depth_npz_path,
        depth_m=np.asarray(frame.depth_m, dtype=np.float32),
    )

    overlay = render_preview(
        frame,
        status=(
            f"TASK D READ ONLY | frame {record['index']} | "
            f"known={record['known_count']} unknown={record['unknown_count']}"
        ),
        target_class="",
        stale_after_s=vision.stale_after_s,
    )
    _write_jpeg(overlay_path, overlay)

    record["images"] = {
        "rgb": rgb_path.name,
        "depth_color": depth_path.name,
        "depth_m_npz": depth_npz_path.name,
        "detection_overlay": overlay_path.name,
    }


def _copy_representative_files(
    output_dir: Path,
    frame_record: Mapping[str, Any],
) -> dict[str, str]:
    copied: dict[str, str] = {}
    for key, filename in dict(frame_record.get("images", {})).items():
        source = output_dir / str(filename)
        suffix = source.suffix
        target = output_dir / f"representative_{key}{suffix}"
        shutil.copy2(source, target)
        copied[key] = target.name
    return copied


def _aggregate_frames(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known_observations = sum(int(frame["known_count"]) for frame in frames)
    unknown_observations = sum(int(frame["unknown_count"]) for frame in frames)
    depth_ratios = [
        float(frame["depth"]["in_configured_range_ratio"]) for frame in frames
    ]
    center_ratios = [
        float(frame["depth"]["central_roi_in_range_ratio"]) for frame in frames
    ]
    known_labels = sorted(
        {
            str(detection["label"])
            for frame in frames
            for detection in frame["detections"]
            if detection["kind"] == "known"
        }
    )
    unknown_track_ids = sorted(
        {
            str(detection["track_id"])
            for frame in frames
            for detection in frame["detections"]
            if detection["kind"] == "unknown" and str(detection["track_id"]).strip()
        }
    )
    return {
        "frames_with_detections": sum(
            1 for frame in frames if int(frame["detection_count"]) > 0
        ),
        "frames_with_known": sum(
            1 for frame in frames if int(frame["known_count"]) > 0
        ),
        "frames_with_unknown": sum(
            1 for frame in frames if int(frame["unknown_count"]) > 0
        ),
        "known_observations": known_observations,
        "unknown_observations": unknown_observations,
        "unique_known_labels": known_labels,
        "unique_unknown_track_ids": unknown_track_ids,
        "mean_frame_depth_valid_ratio": (
            float(sum(depth_ratios) / len(depth_ratios)) if depth_ratios else None
        ),
        "minimum_frame_depth_valid_ratio": min(depth_ratios) if depth_ratios else None,
        "mean_frame_central_depth_valid_ratio": (
            float(sum(center_ratios) / len(center_ratios)) if center_ratios else None
        ),
        "minimum_frame_central_depth_valid_ratio": (
            min(center_ratios) if center_ratios else None
        ),
    }


def run_acceptance(
    config: Track1Config,
    *,
    output_dir: Path,
    frame_count: int = DEFAULT_FRAMES,
    show: bool = False,
    require_known: bool = False,
    require_unknown: bool = False,
    min_valid_depth_ratio: float = 0.0,
    config_path: Path | None = None,
    build_vision_fn: Callable[..., Any] = build_vision,
) -> dict[str, Any]:
    """Capture Task D vision frames without constructing actuators."""

    assert_read_only_safety(config)
    if frame_count <= 0:
        raise AcceptanceError("frame_count must be positive")
    if not 0.0 <= float(min_valid_depth_ratio) <= 1.0:
        raise AcceptanceError("min_valid_depth_ratio must be in [0, 1]")

    output_dir = _prepare_output_dir(output_dir)
    vision_config = replace(config.vision, show=bool(show))
    unknown_labels = {
        label.strip().casefold()
        for label in config.target.unknown_labels
        if label.strip()
    }
    frames: list[dict[str, Any]] = []
    source = None
    capture_error = ""
    close_error = ""
    first_capture_s: float | None = None
    started_s = time.monotonic()

    log = lambda message: print(message, flush=True)  # noqa: E731
    try:
        source = build_vision_fn(vision_config, log=log)
        for index in range(1, frame_count + 1):
            capture_started_s = time.monotonic()
            frame = source.capture()
            capture_elapsed_ms = 1000.0 * (time.monotonic() - capture_started_s)
            if first_capture_s is None:
                first_capture_s = time.monotonic() - started_s

            record, _known_count, _unknown_count = _capture_record(
                index,
                frame,
                capture_elapsed_ms=capture_elapsed_ms,
                unknown_labels=unknown_labels,
            )
            record["depth"] = _depth_metrics(frame.depth_m, vision_config)
            _save_frame_artifacts(output_dir, record, frame, vision_config)
            frames.append(record)
            print(
                f"[acceptance] frame {index}/{frame_count} "
                f"{frame.frame_id}: known={record['known_count']} "
                f"unknown={record['unknown_count']} "
                f"depth={record['depth']['in_configured_range_ratio']:.2%} "
                f"center={record['depth']['central_roi_in_range_ratio']:.2%}",
                flush=True,
            )
    except Exception as exc:
        capture_error = f"{type(exc).__name__}: {exc}"
        print(f"[acceptance] capture failed: {capture_error}", file=sys.stderr, flush=True)
    finally:
        elapsed_s = time.monotonic() - started_s
        if source is not None:
            close = getattr(source, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # pragma: no cover - hardware dependent
                    close_error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[acceptance] vision close failed: {close_error}",
                        file=sys.stderr,
                        flush=True,
                    )

    aggregate = _aggregate_frames(frames)
    failures: list[str] = []
    if capture_error:
        failures.append(f"capture failed: {capture_error}")
    if close_error:
        failures.append(f"vision close failed: {close_error}")
    if len(frames) != frame_count:
        failures.append(
            f"captured {len(frames)} of {frame_count} requested frames"
        )
    if min_valid_depth_ratio > 0.0:
        for frame in frames:
            ratio = float(frame["depth"]["in_configured_range_ratio"])
            if ratio < min_valid_depth_ratio:
                failures.append(
                    f"{frame['frame_id']} depth ratio {ratio:.4f} is below "
                    f"minimum {min_valid_depth_ratio:.4f}"
                )
    if require_known and int(aggregate["known_observations"]) == 0:
        failures.append("--require-known was set but V11 observed no known target")
    if require_unknown and int(aggregate["unknown_observations"]) == 0:
        failures.append(
            "--require-unknown was set but the Task D proposer observed no unknown"
        )

    representative = None
    representative_images: dict[str, str] = {}
    if frames:
        representative = max(
            frames,
            key=lambda frame: (
                int(frame["unknown_count"]),
                int(frame["detection_count"]),
                int(frame["known_count"]),
                -int(frame["index"]),
            ),
        )
        representative_images = _copy_representative_files(output_dir, representative)

    capture_times = [float(frame["capture_elapsed_ms"]) for frame in frames]
    subsequent_times = capture_times[1:]
    status = "passed" if not failures else "failed"
    notes: list[str] = [
        "No RobotInfo is required and no robot bridge is started.",
        "No motion backend or gripper backend is constructed.",
        "No motion or gripper command is published by this tool.",
        "The normal run_task_d launcher retains its /RobotInfo safety gate.",
    ]
    if not frames:
        notes.append("No frame was captured, so this report is a failure baseline.")
    elif int(aggregate["frames_with_detections"]) == 0:
        notes.append(
            "Zero detections are a valid zero-target baseline only; use "
            "--require-known/--require-unknown for positive-sample acceptance."
        )
    if aggregate["unknown_observations"] == 0:
        notes.append(
            "No unknown positive sample was observed; opaque unknown-object "
            "recall is not yet accepted."
        )
    notes.append(
        "Transparent and mirror-like objects remain a known depth limitation "
        "and are not made graspable by this acceptance tool."
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "task_d_read_only_vision_acceptance",
        "generated_at": _now_iso(),
        "status": status,
        "scope": "vision_only_no_motion_no_gripper",
        "safety": {
            "motion_dry_run": bool(config.motion.dry_run),
            "gripper_backend": config.gripper.backend,
            "robot_info_required": False,
            "robot_bridge_started": False,
            "motion_backend_constructed": False,
            "gripper_backend_constructed": False,
            "motion_commands_published": False,
            "gripper_commands_sent": False,
        },
        "config": {
            "path": "" if config_path is None else str(config_path),
            "task_mode": config.task.mode.value,
            "vision_backend": config.vision.backend,
            "weights": config.vision.weights,
            "weights_sha256": config.vision.weights_sha256,
            "serial": config.vision.serial,
            "width": int(config.vision.width),
            "height": int(config.vision.height),
            "fps": int(config.vision.fps),
            "rotate_180": bool(config.vision.rotate_180),
            "stabilizer_enabled": bool(config.vision.stabilize),
            "unknown_proposer_enabled": bool(config.vision.unknown_proposer.enabled),
            "unknown_labels": sorted(unknown_labels),
            "show_window": bool(show),
        },
        "acceptance": {
            "requested_frames": int(frame_count),
            "captured_frames": len(frames),
            "require_known": bool(require_known),
            "require_unknown": bool(require_unknown),
            "minimum_frame_depth_valid_ratio": float(min_valid_depth_ratio),
            "passed": not failures,
            "failures": failures,
        },
        "capture": {
            "elapsed_s": float(elapsed_s),
            "startup_to_first_capture_s": first_capture_s,
            "first_capture_ms": capture_times[0] if capture_times else None,
            "mean_capture_ms": (
                float(sum(capture_times) / len(capture_times))
                if capture_times
                else None
            ),
            "mean_subsequent_capture_ms": (
                float(sum(subsequent_times) / len(subsequent_times))
                if subsequent_times
                else None
            ),
            "capture_error": capture_error,
            "close_error": close_error,
        },
        "aggregate": aggregate,
        "representative_frame": (
            None
            if representative is None
            else {
                "index": representative["index"],
                "frame_id": representative["frame_id"],
                "images": representative_images,
            }
        ),
        "frames": frames,
        "notes": notes,
    }
    _write_json(output_dir / "summary.json", payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="accept_task_d_vision",
        description=(
            "Read-only D405 + V11 + Task D unknown-proposal acceptance. "
            "Never starts a robot bridge or commands the arm/gripper."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="formal Track 1 YAML (task mode is forced to D)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"number of frames to capture (default: {DEFAULT_FRAMES})",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="new output directory; defaults to logs/task_d_vision_acceptance_<time>",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show the observation-only OpenCV preview window",
    )
    parser.add_argument(
        "--require-known",
        action="store_true",
        help="fail unless at least one stable V11 target is observed",
    )
    parser.add_argument(
        "--require-unknown",
        action="store_true",
        help="fail unless at least one stable Task D unknown is observed",
    )
    parser.add_argument(
        "--min-valid-depth-ratio",
        type=float,
        default=0.0,
        help="fail if any frame's in-range depth ratio is below this value",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    try:
        config = apply_mode(load_config(config_path), TaskMode.D)
        assert_read_only_safety(config)
        config.validate()
    except Exception as exc:
        print(f"[acceptance] configuration rejected: {exc}", file=sys.stderr)
        return 2

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _unique_default_output_dir()
    )
    print(
        "[acceptance] READ ONLY: no RobotInfo, no robot bridge, "
        "no arm motion, no gripper command",
        flush=True,
    )
    print(
        f"[acceptance] config={config_path} frames={args.frames} "
        f"output={output_dir}",
        flush=True,
    )
    try:
        payload = run_acceptance(
            config,
            output_dir=output_dir,
            frame_count=args.frames,
            show=args.show,
            require_known=args.require_known,
            require_unknown=args.require_unknown,
            min_valid_depth_ratio=args.min_valid_depth_ratio,
            config_path=config_path,
        )
    except KeyboardInterrupt:
        print("[acceptance] interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[acceptance] failed: {exc}", file=sys.stderr)
        return 1

    aggregate = payload["aggregate"]
    print(
        "[acceptance] "
        f"status={payload['status']} frames={len(payload['frames'])} "
        f"known={aggregate['known_observations']} "
        f"unknown={aggregate['unknown_observations']} "
        f"depth_mean={aggregate['mean_frame_depth_valid_ratio']} "
        f"center_mean={aggregate['mean_frame_central_depth_valid_ratio']}",
        flush=True,
    )
    print(f"[acceptance] summary: {output_dir / 'summary.json'}", flush=True)
    if payload["status"] != "passed":
        for failure in payload["acceptance"]["failures"]:
            print(f"[acceptance] FAIL: {failure}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
