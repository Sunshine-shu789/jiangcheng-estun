"""Read-only multi-frame verification at the current pick_4 observation pose.

This tool never builds a gripper backend, never publishes motion, never closes
the jaws and never retries.  It captures a bounded set of RGB-D frames, applies
the exact Task B banana waist/short-axis estimator, transforms each measured
waist centre through the calibrated flange camera, and reports whether the
current TCP is aligned with a stable pick_4 observation.
"""

from __future__ import annotations

import argparse
import math
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "config/track1_task_b_five_point_full_live_20261106.yaml"
REFERENCE_POINTS = {
    "pick_1": (-0.041246, 0.490700),
    "pick_2": (-0.124884, 0.521709),
    "pick_3": (-0.108009, 0.417709),
    "pick_4": (0.056674, 0.417881),
    "pick_5": (0.034196, 0.486713),
}
FRAME_COUNT = 20
MIN_STABLE_FRAMES = 17
MIN_FORMAL_FRAMES = 17
MAX_TCP_CENTER_DISTANCE_M = 0.005
MAX_CENTER_SPREAD_M = 0.005
MAX_AXIS_SPREAD_DEG = 15.0
MAX_MAJOR_MINOR_DOT = 0.20


def _axis_angle(axis_xy) -> float:
    return math.atan2(float(axis_xy[1]), float(axis_xy[0])) % math.pi


def _axis_delta(a: float, b: float) -> float:
    return (float(a) - float(b) + 0.5 * math.pi) % math.pi - 0.5 * math.pi


def run(config_path: Path, log_dir: Path) -> int:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    from track1_grasp.config import load_config
    from track1_grasp.motion import build_motion
    from track1_grasp.planner import PosePlanner
    from track1_grasp.vision import build_vision

    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"task_b_pick4_verification_readonly_{stamp}.log"

    def log(message: str) -> None:
        text = str(message)
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    log(f"[start] pick_4 read-only verification; log={log_path}")
    log("[authorization] read-only only; no motion; no gripper; no grasp; no retry")

    cfg = load_config(config_path)
    if cfg.motion.backend != "robotmove":
        raise RuntimeError(f"unexpected motion backend: {cfg.motion.backend}")
    if not bool(cfg.adaptive_inverse.enabled):
        raise RuntimeError("verification requires adaptive_inverse.enabled=true")
    if not bool(cfg.adaptive_inverse.pca.waist_midline_enabled):
        raise RuntimeError("verification requires waist_midline_enabled=true")
    target = cfg.vision.target_class.strip() or "8-Plastic_Banana"
    min_confidence = float(cfg.vision.min_confidence)
    min_pose_confidence = float(cfg.adaptive_inverse.min_pose_confidence)
    log(
        f"[thresholds] target={target} min_confidence={min_confidence:.3f} "
        f"min_pose_confidence={min_pose_confidence:.3f} "
        f"frames={FRAME_COUNT}"
    )

    rclpy.init()
    node = Node("task_b_pick4_verification_readonly")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin,
        name="task-b-pick4-verification-readonly-ros",
        daemon=True,
    )
    spin_thread.start()

    motion = None
    vision = None
    try:
        motion = build_motion(cfg.motion, node=node, log=log)
        state = motion.wait_for_feedback_state(5.0)
        if not state.ready or state.fault or state.pose is None or state.joints is None:
            raise RuntimeError(f"robot state gate failed: {state.message}")
        log(
            f"ROBOT state={state.controller_state} fault={int(bool(state.fault))} "
            f"xyz=({state.pose.xyz[0]:.6f}, {state.pose.xyz[1]:.6f}, "
            f"{state.pose.xyz[2]:.6f}) "
            f"rpy=({state.pose.rpy[0]:.6f}, {state.pose.rpy[1]:.6f}, "
            f"{state.pose.rpy[2]:.6f})"
        )
        log(
            "JOINTS_DEG="
            + str(np.round(np.degrees(np.asarray(state.joints)), 3).tolist())
        )
        log(
            "POINTS "
            + " ".join(
                f"{name}=({xy[0]:.6f}, {xy[1]:.6f})"
                for name, xy in REFERENCE_POINTS.items()
            )
        )

        vision = build_vision(
            cfg.vision,
            log=log,
            adaptive_inverse=cfg.adaptive_inverse,
        )
        planner = PosePlanner(cfg, log=log)
        base_camera = np.asarray(planner.base_camera_transform(state), dtype=float)

        stable_frames = 0
        formal_frames = 0
        centers: list[tuple[float, float]] = []
        centers_xyz: list[tuple[float, float, float]] = []
        short_angles: list[float] = []
        orthogonality_errors: list[float] = []
        nearest_names: list[str] = []
        nearest_distances: list[float] = []
        confidences: list[float] = []
        pose_confidences: list[float] = []

        for index in range(1, FRAME_COUNT + 1):
            frame = vision.capture()
            candidates = tuple(
                detection
                for detection in frame.detections
                if detection.label == target
            )
            image_path = frame.image_path
            if not candidates:
                log(f"FRAME {index:02d} detections={len(frame.detections)} banana=0 image={image_path}")
                continue

            stable_frames += 1
            detection = max(
                candidates,
                key=lambda item: (
                    float(item.pose_confidence or -1.0),
                    float(item.confidence),
                ),
            )
            pose_confidence = detection.pose_confidence
            formal = (
                detection.object_center_camera_xyz is not None
                and detection.major_axis_camera is not None
                and detection.minor_axis_camera is not None
                and pose_confidence is not None
                and float(detection.confidence) >= min_confidence
                and float(pose_confidence) >= min_pose_confidence
                and detection.attributes.get("adaptive_axis_source")
                == "waist_midline"
            )
            if not formal:
                log(
                    f"FRAME {index:02d} detections={len(frame.detections)} banana=1 "
                    f"formal=False conf={detection.confidence:.4f} "
                    f"pose_conf={pose_confidence} image={image_path}"
                )
                continue

            formal_frames += 1
            center_xyz = np.asarray(
                planner.camera_point_to_base(
                    detection.object_center_camera_xyz,
                    state,
                ),
                dtype=float,
            )
            if center_xyz.shape != (3,) or not np.all(np.isfinite(center_xyz)):
                raise RuntimeError(f"frame {index}: invalid base-frame centre")
            major_base = base_camera[:3, :3] @ np.asarray(
                detection.major_axis_camera, dtype=float
            )
            minor_base = base_camera[:3, :3] @ np.asarray(
                detection.minor_axis_camera, dtype=float
            )
            major_xy = major_base[:2] / max(float(np.linalg.norm(major_base[:2])), 1e-12)
            minor_xy = minor_base[:2] / max(float(np.linalg.norm(minor_base[:2])), 1e-12)
            short_angle = _axis_angle(minor_xy)
            orthogonality_error = abs(float(major_xy @ minor_xy))

            distances = {
                name: float(np.linalg.norm(center_xyz[:2] - np.asarray(xy)))
                for name, xy in REFERENCE_POINTS.items()
            }
            ordered = sorted(distances.items(), key=lambda item: item[1])
            nearest_name, nearest_distance = ordered[0]
            margin = ordered[1][1] - nearest_distance

            centers.append((float(center_xyz[0]), float(center_xyz[1])))
            centers_xyz.append(tuple(float(value) for value in center_xyz))
            short_angles.append(short_angle)
            orthogonality_errors.append(orthogonality_error)
            nearest_names.append(nearest_name)
            nearest_distances.append(nearest_distance)
            confidences.append(float(detection.confidence))
            pose_confidences.append(float(pose_confidence))
            log(
                f"FORMAL {index:02d} conf={detection.confidence:.4f} "
                f"pose_conf={pose_confidence:.4f} "
                f"center_base=({center_xyz[0]:.6f},{center_xyz[1]:.6f},"
                f"{center_xyz[2]:.6f}) short_xy=({minor_xy[0]:.4f},"
                f"{minor_xy[1]:.4f}) short_deg={math.degrees(short_angle):.2f} "
                f"nearest={nearest_name}:{nearest_distance:.6f} "
                f"margin={margin:.6f} image={image_path}"
            )

        if not centers:
            raise RuntimeError("no formal banana waist pose was produced")

        centers_array = np.asarray(centers, dtype=float)
        median_xy = np.median(centers_array, axis=0)
        radial_errors = np.linalg.norm(centers_array - median_xy, axis=1)
        center_spread = float(np.max(radial_errors))
        doubled = 2.0 * np.asarray(short_angles, dtype=float)
        mean_short_angle = 0.5 * math.atan2(
            float(np.mean(np.sin(doubled))),
            float(np.mean(np.cos(doubled))),
        ) % math.pi
        axis_spread_deg = math.degrees(
            max(abs(_axis_delta(angle, mean_short_angle)) for angle in short_angles)
        )
        max_orthogonality_error = float(max(orthogonality_errors))
        unique_nearest = sorted(set(nearest_names))
        min_margin = float(
            min(
                (
                    sorted(
                        (
                            float(np.linalg.norm(np.asarray(center) - np.asarray(xy)))
                            for xy in REFERENCE_POINTS.values()
                        )
                    )[1]
                    - sorted(
                        (
                            float(np.linalg.norm(np.asarray(center) - np.asarray(xy)))
                            for xy in REFERENCE_POINTS.values()
                        )
                    )[0]
                )
                for center in centers
            )
        )
        tcp_xy = np.asarray(state.pose.xyz[:2], dtype=float)
        delta_xy = median_xy - tcp_xy
        tcp_center_distance = float(np.linalg.norm(delta_xy))

        log(
            f"SUMMARY frames={FRAME_COUNT} stable_banana_frames={stable_frames} "
            f"formal_frames={formal_frames} formal_centers={len(centers)}"
        )
        log(
            f"CONFIDENCE min={min(confidences):.4f} max={max(confidences):.4f} "
            f"POSE_CONFIDENCE min={min(pose_confidences):.4f} "
            f"max={max(pose_confidences):.4f}"
        )
        log(
            f"CENTER_MEDIAN=({median_xy[0]:.6f},{median_xy[1]:.6f}) "
            f"Z_MEDIAN={float(np.median(np.asarray(centers_xyz)[:, 2])):.6f} "
            f"MAX_RADIAL_SPREAD={center_spread:.6f}"
        )
        log(
            f"SHORT_AXIS median_deg={math.degrees(mean_short_angle):.3f} "
            f"max_spread_deg={axis_spread_deg:.3f} "
            f"max_abs_major_minor_xy_dot={max_orthogonality_error:.4f} "
            f"source=waist_midline"
        )
        log(
            f"ASSIGNMENT nearest={','.join(unique_nearest)} "
            f"min_margin={min_margin:.6f}"
        )
        log(
            f"TCP_TO_CENTER_XY delta=({delta_xy[0]:.6f},{delta_xy[1]:.6f}) "
            f"distance={tcp_center_distance:.6f}"
        )

        checks = {
            "stable_frames": stable_frames >= MIN_STABLE_FRAMES,
            "formal_frames": formal_frames >= MIN_FORMAL_FRAMES,
            "pick_4_only": unique_nearest == ["pick_4"],
            "assignment_margin": min_margin >= 0.005,
            "center_spread": center_spread <= MAX_CENTER_SPREAD_M,
            "short_axis_stable": axis_spread_deg <= MAX_AXIS_SPREAD_DEG,
            "short_axis_orthogonal": max_orthogonality_error
            <= MAX_MAJOR_MINOR_DOT,
            "tcp_center_aligned": tcp_center_distance
            <= MAX_TCP_CENTER_DISTANCE_M,
        }
        log("CHECKS " + " ".join(f"{name}={int(value)}" for name, value in checks.items()))
        if not all(checks.values()):
            failed = ",".join(name for name, value in checks.items() if not value)
            log(f"VERDICT=NOT_READY_FOR_GRASP failed={failed}")
            log(f"LOG={log_path}")
            return 2

        log("[success] stable pick_4 waist centre and short axis verified")
        log("VERDICT=READY_FOR_GRASP_CONFIRMATION")
        log(f"LOG={log_path}")
        return 0
    except BaseException as exc:
        log(f"[ABORT] {type(exc).__name__}: {exc}")
        log("VERDICT=ABORTED_NO_RETRY")
        log(f"LOG={log_path}")
        return 1
    finally:
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass
        if motion is not None:
            try:
                motion.close()
            except Exception:
                pass
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        if spin_thread.is_alive():
            log("[cleanup] ROS executor thread did not stop within 2.0 s")
        try:
            executor.remove_node(node)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    args = parser.parse_args()
    return run(args.config.resolve(), args.log_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
