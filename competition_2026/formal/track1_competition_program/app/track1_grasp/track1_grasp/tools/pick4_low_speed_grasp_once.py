"""One authorized pick_4 grasp, with no release, recovery or retry.

The operator authorized exactly one low-speed pick_4 grasp at 0.12 m/s.  This
tool captures fresh RGB-D frames, requires a stable banana waist/short-axis
observation assigned uniquely to pick_4, prechecks the complete
pregrasp -> grasp -> lift sequence, opens the jaws once, closes once using the
same-frame measured width, verifies the hold, and only then lifts.  It never
releases the object, retreats, returns to an initial pose, retries, or advances
to another pick point.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "config/track1_task_b_five_point_full_live_20261106.yaml"

AUTHORIZED_SPEED_MPS = 0.12
EXPECTED_START_XYZ = (0.038596, 0.440595, 0.256354)
EXPECTED_START_RPY = (-3.121062, 0.049624, -0.522546)
GRASP_Z_M = 0.169903173540
GRASP_RPY = (3.083158245, 0.050086616, -0.347881105)
APPROACH_HEIGHT_M = 0.075
START_POSITION_TOLERANCE_M = 0.003
START_ROTATION_TOLERANCE_RAD = 0.02

FRAME_COUNT = 12
MIN_STABLE_FRAMES = 9
MIN_FORMAL_FRAMES = 9
MAX_TCP_CENTER_DISTANCE_M = 0.006
MAX_CENTER_SPREAD_M = 0.005
MAX_AXIS_SPREAD_DEG = 15.0
MAX_MAJOR_MINOR_DOT = 0.20
MIN_ASSIGNMENT_MARGIN_M = 0.005

REFERENCE_POINTS = {
    "pick_1": (-0.041246, 0.490700),
    "pick_2": (-0.124884, 0.521709),
    "pick_3": (-0.108009, 0.417709),
    "pick_4": (0.056674, 0.417881),
    "pick_5": (0.034196, 0.486713),
}


def _axis_angle(axis_xy) -> float:
    return math.atan2(float(axis_xy[1]), float(axis_xy[0])) % math.pi


def _axis_delta(a: float, b: float) -> float:
    return (float(a) - float(b) + 0.5 * math.pi) % math.pi - 0.5 * math.pi


def _pose_errors(actual, expected_xyz, expected_rpy) -> tuple[float, float]:
    from track1_grasp.kinematics import rotation_angle_error, rpy_to_matrix

    position_error = float(
        np.linalg.norm(np.asarray(actual.xyz) - np.asarray(expected_xyz))
    )
    rotation_error = float(
        rotation_angle_error(
            rpy_to_matrix(np.asarray(actual.rpy)),
            rpy_to_matrix(np.asarray(expected_rpy)),
        )
    )
    return position_error, rotation_error


def _pose_text(pose) -> str:
    return (
        f"xyz=({pose.xyz[0]:.6f}, {pose.xyz[1]:.6f}, {pose.xyz[2]:.6f}) "
        f"rpy=({pose.rpy[0]:.6f}, {pose.rpy[1]:.6f}, {pose.rpy[2]:.6f})"
    )


def _validate_state(label: str, state) -> None:
    if not state.ready:
        raise RuntimeError(f"{label}: controller is not AutoReady: {state.message}")
    if state.fault:
        raise RuntimeError(f"{label}: controller fault: {state.message}")
    if state.pose is None:
        raise RuntimeError(f"{label}: no pose feedback")
    pose_values = np.asarray((*state.pose.xyz, *state.pose.rpy), dtype=float)
    if pose_values.shape != (6,) or not np.all(np.isfinite(pose_values)):
        raise RuntimeError(f"{label}: malformed pose feedback")
    if state.joints is None:
        raise RuntimeError(f"{label}: no six-joint feedback")
    joints = np.asarray(state.joints, dtype=float)
    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
        raise RuntimeError(f"{label}: malformed six-joint feedback")


def _validate_capture_state(before, after, *, frame_id: str, cfg) -> None:
    from track1_grasp.kinematics import rotation_angle_error, rpy_to_matrix

    _validate_state(f"frame {frame_id} before", before)
    _validate_state(f"frame {frame_id} after", after)
    capture = cfg.adaptive_inverse.box_relative_five_point
    joint_delta_deg = math.degrees(
        float(
            np.max(
                np.abs(
                    np.asarray(after.joints, dtype=float)
                    - np.asarray(before.joints, dtype=float)
                )
            )
        )
    )
    if joint_delta_deg > float(capture.max_capture_joint_delta_deg):
        raise RuntimeError(
            f"frame {frame_id}: joint drift {joint_delta_deg:.6f} deg exceeds "
            f"{capture.max_capture_joint_delta_deg:.6f} deg"
        )
    translation_delta_m = float(
        np.linalg.norm(
            np.asarray(after.pose.xyz, dtype=float)
            - np.asarray(before.pose.xyz, dtype=float)
        )
    )
    if translation_delta_m > float(capture.max_capture_pose_translation_m):
        raise RuntimeError(
            f"frame {frame_id}: pose translation drift "
            f"{translation_delta_m:.6f} m exceeds "
            f"{capture.max_capture_pose_translation_m:.6f} m"
        )
    rotation_delta_rad = float(
        rotation_angle_error(
            rpy_to_matrix(before.pose.rpy),
            rpy_to_matrix(after.pose.rpy),
        )
    )
    max_rotation_rad = math.radians(float(capture.max_capture_pose_rotation_deg))
    if rotation_delta_rad > max_rotation_rad:
        raise RuntimeError(
            f"frame {frame_id}: pose rotation drift "
            f"{math.degrees(rotation_delta_rad):.6f} deg exceeds "
            f"{capture.max_capture_pose_rotation_deg:.6f} deg"
        )


def _capture_verified_detection(cfg, motion, vision, planner, log):
    from track1_grasp.domain import CaptureRobotState

    target = cfg.vision.target_class.strip() or "8-Plastic_Banana"
    min_confidence = float(cfg.vision.min_confidence)
    min_pose_confidence = float(cfg.adaptive_inverse.min_pose_confidence)
    stable_frames = 0
    formal_records = []

    for index in range(1, FRAME_COUNT + 1):
        before = motion.get_feedback_state()
        _validate_state(f"frame {index} before capture", before)
        sequence = int(motion.get_feedback_sequence())
        frame = vision.capture()
        after = motion.wait_for_feedback_state(
            max(1.0, float(cfg.motion.robot_info_timeout_s)),
            newer_than=sequence,
        )
        _validate_capture_state(before, after, frame_id=frame.frame_id, cfg=cfg)
        capture = CaptureRobotState(before=before, after=after)

        candidates = tuple(
            detection for detection in frame.detections if detection.label == target
        )
        if len(candidates) != 1:
            log(
                f"FRAME {index:02d} frame={frame.frame_id} banana_count={len(candidates)} "
                f"all_detections={len(frame.detections)} formal=False image={frame.image_path}"
            )
            continue
        stable_frames += 1
        detection = candidates[0]
        pose_confidence = detection.pose_confidence
        formal = (
            detection.object_center_camera_xyz is not None
            and detection.major_axis_camera is not None
            and detection.minor_axis_camera is not None
            and detection.major_axis_length_m is not None
            and detection.minor_axis_length_m is not None
            and pose_confidence is not None
            and float(detection.confidence) >= min_confidence
            and float(pose_confidence) >= min_pose_confidence
            and detection.attributes.get("adaptive_axis_source") == "waist_midline"
            and detection.attributes.get("adaptive_support_plane_valid") is True
        )
        if not formal:
            log(
                f"FRAME {index:02d} frame={frame.frame_id} formal=False "
                f"conf={detection.confidence:.4f} pose_conf={pose_confidence} "
                f"axis_source={detection.attributes.get('adaptive_axis_source')} "
                f"support_plane={detection.attributes.get('adaptive_support_plane_valid')} "
                f"image={frame.image_path}"
            )
            continue

        center_xyz = np.asarray(
            planner.camera_point_to_base(detection.object_center_camera_xyz, before),
            dtype=float,
        )
        if center_xyz.shape != (3,) or not np.all(np.isfinite(center_xyz)):
            raise RuntimeError(f"frame {index}: invalid base-frame banana centre")
        base_camera = np.asarray(planner.base_camera_transform(before), dtype=float)
        major_base = base_camera[:3, :3] @ np.asarray(detection.major_axis_camera, dtype=float)
        minor_base = base_camera[:3, :3] @ np.asarray(detection.minor_axis_camera, dtype=float)
        major_norm = float(np.linalg.norm(major_base[:2]))
        minor_norm = float(np.linalg.norm(minor_base[:2]))
        if major_norm <= 1e-12 or minor_norm <= 1e-12:
            raise RuntimeError(f"frame {index}: banana axis has no horizontal projection")
        major_xy = major_base[:2] / major_norm
        minor_xy = minor_base[:2] / minor_norm
        short_angle = _axis_angle(minor_xy)
        orthogonality_error = abs(float(major_xy @ minor_xy))
        distances = {
            name: float(np.linalg.norm(center_xyz[:2] - np.asarray(xy)))
            for name, xy in REFERENCE_POINTS.items()
        }
        ordered = sorted(distances.items(), key=lambda item: item[1])
        nearest_name, nearest_distance = ordered[0]
        margin = float(ordered[1][1] - nearest_distance)
        tcp_distance = float(
            np.linalg.norm(center_xyz[:2] - np.asarray(before.pose.xyz[:2]))
        )
        record = {
            "index": index,
            "frame": frame,
            "detection": detection,
            "capture": capture,
            "center_xyz": center_xyz,
            "short_angle": short_angle,
            "orthogonality_error": orthogonality_error,
            "nearest_name": nearest_name,
            "nearest_distance": nearest_distance,
            "margin": margin,
            "tcp_distance": tcp_distance,
        }
        formal_records.append(record)
        log(
            f"FORMAL {index:02d} frame={frame.frame_id} "
            f"conf={detection.confidence:.4f} pose_conf={float(pose_confidence):.4f} "
            f"center_base=({center_xyz[0]:.6f},{center_xyz[1]:.6f},{center_xyz[2]:.6f}) "
            f"short_deg={math.degrees(short_angle):.3f} "
            f"nearest={nearest_name}:{nearest_distance:.6f} margin={margin:.6f} "
            f"tcp_distance={tcp_distance:.6f} image={frame.image_path}"
        )

    if not formal_records:
        raise RuntimeError("no formal banana waist pose was captured")
    centers = np.asarray([record["center_xyz"][:2] for record in formal_records])
    median_xy = np.median(centers, axis=0)
    center_spread = float(np.max(np.linalg.norm(centers - median_xy, axis=1)))
    doubled = 2.0 * np.asarray([record["short_angle"] for record in formal_records])
    mean_short_angle = 0.5 * math.atan2(
        float(np.mean(np.sin(doubled))), float(np.mean(np.cos(doubled)))
    ) % math.pi
    axis_spread_deg = math.degrees(
        max(
            abs(_axis_delta(record["short_angle"], mean_short_angle))
            for record in formal_records
        )
    )
    nearest_names = sorted({record["nearest_name"] for record in formal_records})
    min_margin = float(min(record["margin"] for record in formal_records))
    max_tcp_distance = float(max(record["tcp_distance"] for record in formal_records))
    max_orthogonality = float(
        max(record["orthogonality_error"] for record in formal_records)
    )

    # Execute from the freshest formal frame.  The multi-frame ensemble above
    # proves that this frame is not an isolated jitter; the selected frame's
    # own centre is what the planner must align to.
    selected = max(formal_records, key=lambda record: int(record["index"]))
    selected_tcp_distance = float(selected["tcp_distance"])
    checks = {
        "stable_frames": stable_frames >= MIN_STABLE_FRAMES,
        "formal_frames": len(formal_records) >= MIN_FORMAL_FRAMES,
        "pick_4_only": nearest_names == ["pick_4"],
        "assignment_margin": min_margin >= MIN_ASSIGNMENT_MARGIN_M,
        "center_spread": center_spread <= MAX_CENTER_SPREAD_M,
        "short_axis_stable": axis_spread_deg <= MAX_AXIS_SPREAD_DEG,
        "short_axis_orthogonal": max_orthogonality <= MAX_MAJOR_MINOR_DOT,
        "tcp_center_aligned": selected_tcp_distance <= MAX_TCP_CENTER_DISTANCE_M,
    }
    log(
        f"VISION_SUMMARY stable={stable_frames}/{FRAME_COUNT} "
        f"formal={len(formal_records)} median_xy=({median_xy[0]:.6f},{median_xy[1]:.6f}) "
        f"center_spread={center_spread:.6f} short_axis_spread_deg={axis_spread_deg:.3f} "
        f"assignment={','.join(nearest_names)} min_margin={min_margin:.6f} "
        f"selected_tcp_center_distance={selected_tcp_distance:.6f} "
        f"max_observed_tcp_center_distance={max_tcp_distance:.6f}"
    )
    log("VISION_CHECKS " + " ".join(f"{name}={int(value)}" for name, value in checks.items()))
    if not all(checks.values()):
        failed = ",".join(name for name, value in checks.items() if not value)
        raise RuntimeError(f"fresh pick_4 vision gate failed: {failed}")
    return selected


def _validate_plan(cfg, planner, candidate, selected, log):
    from track1_grasp.collision import check_box_clearance
    from track1_grasp.gripper import validate_close_width
    from track1_grasp.kinematics import rpy_to_matrix

    plan = candidate.plan
    expected_names = ("pregrasp", "grasp", "lift", "transfer", "release", "retreat")
    names = tuple(name for name, _pose in plan.ordered_waypoints())
    if names != expected_names:
        raise RuntimeError(f"unexpected grasp waypoint sequence: {names}")
    if candidate.orientation_source != "task_b_banana_waist_short_axis":
        raise RuntimeError(
            f"unexpected orientation source: {candidate.orientation_source!r}"
        )
    if plan.grasp_geometry_mode != "task_b_banana_waist_short_axis":
        raise RuntimeError(
            f"unexpected grasp geometry mode: {plan.grasp_geometry_mode!r}"
        )
    if plan.grasp_close_width_m is None:
        raise RuntimeError("plan has no measured width-aware close width")
    width_m = float(plan.grasp_close_width_m)
    if not np.isfinite(width_m) or width_m <= 0.0:
        raise RuntimeError(f"invalid measured close width {width_m!r}")
    capacity = validate_close_width(cfg.gripper, width_m)
    log(
        f"CLOSE_WIDTH measured={width_m:.6f} m required_opening="
        f"{capacity.required_opening_m:.6f} m maximum_opening="
        f"{capacity.maximum_open_opening_m:.6f} m"
    )

    visual_center = np.asarray(selected["center_xyz"], dtype=float)
    commanded_tip = np.asarray(candidate.commanded_jaw_tip_base_xyz, dtype=float)
    if commanded_tip.shape != (3,) or not np.all(np.isfinite(commanded_tip)):
        raise RuntimeError("candidate has no finite commanded jaw-tip point")
    tip_xy_error = float(np.linalg.norm(commanded_tip[:2] - visual_center[:2]))
    if tip_xy_error > 0.0005:
        raise RuntimeError(
            f"commanded jaw-tip XY differs from the measured waist by "
            f"{tip_xy_error * 1000:.3f} mm"
        )

    # The strict planner built pregrasp and grasp at the same visual XY.  The
    # lift must be directly above that column.
    pregrasp = np.asarray(plan.pregrasp_pose.xyz, dtype=float)
    grasp = np.asarray(plan.grasp_pose.xyz, dtype=float)
    lift = np.asarray(plan.lift_pose.xyz, dtype=float)
    if not np.allclose(pregrasp[:2], grasp[:2], atol=1e-9, rtol=0.0):
        raise RuntimeError("pregrasp is not aligned with the grasp column")
    if not np.allclose(lift[:2], grasp[:2], atol=1e-9, rtol=0.0):
        raise RuntimeError("lift is not aligned with the grasp column")
    if not 0.070 <= float(lift[2] - grasp[2]) <= 0.080:
        raise RuntimeError(
            f"lift height {float(lift[2] - grasp[2]):.6f} m is outside 70-80 mm"
        )

    safety = planner.validate_grasp_safety_plan(
        plan,
        expected_grasp_rpy=tuple(float(value) for value in plan.grasp_pose.rpy),
    )
    if safety.enabled:
        safety_metrics = (
            safety.minimum_route_flange_z_m,
            safety.maximum_route_flange_z_m,
            safety.minimum_route_tip_z_m,
            safety.maximum_route_tip_z_m,
            safety.approach_height_m,
        )
        if any(value is None for value in safety_metrics):
            raise RuntimeError(
                "enabled grasp safety report contains incomplete route metrics"
            )
        log(
            f"GRASP_SAFETY flange_z=[{safety.minimum_route_flange_z_m:.6f},"
            f"{safety.maximum_route_flange_z_m:.6f}] "
            f"tip_z=[{safety.minimum_route_tip_z_m:.6f},"
            f"{safety.maximum_route_tip_z_m:.6f}] "
            f"approach={safety.approach_height_m:.6f}"
        )
    else:
        log(
            "GRASP_SAFETY enabled=False; C/D flange/jaw-tip envelope is not "
            "configured; workspace, route, joint, branch, singularity and "
            "acceleration prechecks remain active"
        )

    clearance = check_box_clearance(
        cfg.adaptive_inverse.box_clearance,
        flange_xy_m=plan.grasp_pose.xyz[:2],
        rotation=rpy_to_matrix(plan.grasp_pose.rpy),
    )
    if not clearance.allowed:
        raise RuntimeError(f"box clearance rejected the grasp: {clearance.reason}")
    log(f"BOX_CLEARANCE allowed=True reason={clearance.reason}")

    log(
        f"PLAN grasp_xyz={tuple(round(float(v), 6) for v in plan.grasp_pose.xyz)} "
        f"grasp_rpy={tuple(round(float(v), 6) for v in plan.grasp_pose.rpy)} "
        f"pregrasp_xyz={tuple(round(float(v), 6) for v in plan.pregrasp_pose.xyz)} "
        f"lift_xyz={tuple(round(float(v), 6) for v in plan.lift_pose.xyz)} "
        f"tip_xy_error={tip_xy_error:.6f}"
    )
    return plan


def _route_metrics(cfg, route):
    sigmas = [
        float(segment.min_jacobian_singular_value)
        for segment in route.segments
        if np.isfinite(float(segment.min_jacobian_singular_value))
    ]
    if len(sigmas) != len(route.segments):
        raise RuntimeError("prechecked route contains a non-finite Jacobian metric")
    minimum_sigma = min(sigmas)
    required_sigma = float(
        cfg.motion.cartesian_min_jacobian_singular_value
        + cfg.adaptive_inverse.selection.min_sigma_margin
    )
    if minimum_sigma < required_sigma:
        raise RuntimeError(
            f"route minimum Jacobian sigma {minimum_sigma:.6f} is below "
            f"required {required_sigma:.6f}"
        )
    return minimum_sigma, required_sigma


def run(config_path: Path, log_dir: Path) -> int:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    from track1_grasp.config import load_config
    from track1_grasp.domain import Pose
    from track1_grasp.gripper import (
        GripperState,
        build_gripper,
        build_hold_verifier,
    )
    from track1_grasp.motion import build_motion
    from track1_grasp.planner import PosePlanner
    from track1_grasp.vision import build_vision

    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"task_b_pick4_grasp_once_{stamp}.log"

    def log(message: str) -> None:
        text = str(message)
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    log(f"[start] one-shot pick_4 grasp; log={log_path}")
    log(
        "[authorization] exactly one pick_4 grasp at 0.12 m/s; no automatic retry; "
        "no release; no retreat; no continuation"
    )

    cfg = load_config(config_path)
    if cfg.motion.backend != "robotmove":
        raise RuntimeError(f"unexpected motion backend: {cfg.motion.backend}")
    if bool(cfg.motion.dry_run):
        raise RuntimeError("one-shot live grasp requires motion.dry_run=false")
    if not math.isclose(
        float(cfg.motion.speed_mps),
        AUTHORIZED_SPEED_MPS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"configured speed {cfg.motion.speed_mps:.6f} m/s is not the "
            f"authorized {AUTHORIZED_SPEED_MPS:.3f} m/s"
        )
    if cfg.gripper.backend != "ms42dc":
        raise RuntimeError(f"unexpected gripper backend: {cfg.gripper.backend}")
    if not bool(cfg.gripper.require_close_travel_evidence):
        raise RuntimeError("width-aware close evidence must be required")
    if not bool(cfg.adaptive_inverse.enabled):
        raise RuntimeError("adaptive inverse planning must be enabled")
    if not bool(cfg.adaptive_inverse.pca.waist_midline_enabled):
        raise RuntimeError("banana waist midline planning must be enabled")
    log(
        f"[gates] target={cfg.vision.target_class} imgsz={cfg.vision.imgsz} "
        f"padding={cfg.vision.inference_padding_px} speed={AUTHORIZED_SPEED_MPS:.3f} "
        f"tcp_gate={MAX_TCP_CENTER_DISTANCE_M:.3f} "
        f"grasp_z={GRASP_Z_M:.9f} approach={APPROACH_HEIGHT_M:.3f}"
    )

    rclpy.init()
    node = Node("task_b_pick4_grasp_once")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin,
        name="task-b-pick4-grasp-once-ros",
        daemon=True,
    )
    spin_thread.start()

    motion = None
    vision = None
    gripper = None
    motion_stage = "PRE_MOTION"
    close_attempted = False
    lift_attempted = False
    try:
        motion = build_motion(cfg.motion, node=node, log=log)
        state = motion.wait_for_feedback_state(5.0)
        _validate_state("startup", state)
        position_error, rotation_error = _pose_errors(
            state.pose, EXPECTED_START_XYZ, EXPECTED_START_RPY
        )
        log(
            f"ROBOT state={state.controller_state} fault={int(bool(state.fault))} "
            f"{_pose_text(state.pose)}"
        )
        log(
            f"JOINTS_DEG={np.round(np.degrees(np.asarray(state.joints)), 3).tolist()}"
        )
        log(
            f"START_CHECK position_error={position_error * 1000:.3f} mm "
            f"rotation_error={rotation_error:.6f} rad"
        )
        if position_error > START_POSITION_TOLERANCE_M:
            raise RuntimeError(
                f"robot is not at the verified pick_4 observation position: "
                f"{position_error * 1000:.3f} mm error"
            )
        if rotation_error > START_ROTATION_TOLERANCE_RAD:
            raise RuntimeError(
                f"robot observation orientation differs by {rotation_error:.6f} rad"
            )

        publisher = getattr(motion, "_publisher", None)
        if publisher is None:
            raise RuntimeError("RobotMove publisher is unavailable")
        waiter = getattr(publisher, "wait_for_matching_subscriptions", None)
        if callable(waiter):
            if not waiter(5.0):
                raise RuntimeError("/RobotMove bridge subscriber not found")
        elif int(publisher.get_subscription_count()) < 1:
            raise RuntimeError("/RobotMove bridge subscriber not found")
        log("[preflight] existing /RobotMove bridge subscriber confirmed")

        planner = PosePlanner(cfg, log=log)
        vision = build_vision(
            cfg.vision,
            log=log,
            adaptive_inverse=cfg.adaptive_inverse,
        )
        selected = _capture_verified_detection(cfg, motion, vision, planner, log)
        detection = selected["detection"]
        visual_state = selected["capture"].before
        try:
            vision.close()
        finally:
            vision = None
        log(f"[vision] selected frame={selected['frame'].frame_id} image={selected['frame'].image_path}")

        candidates = planner.plan_adaptive_candidates(
            detection,
            visual_state,
            place_name="pick_4",
            return_to_pick=True,
            grasp_z_m=GRASP_Z_M,
            grasp_rpy=GRASP_RPY,
            task_b_anchor_pose=None,
            vertical_grasp_approach_height_m=APPROACH_HEIGHT_M,
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"strict pick_4 planner returned {len(candidates)} candidates; expected one"
            )
        candidate = candidates[0]
        plan = _validate_plan(cfg, planner, candidate, selected, log)

        # Only the authorized pregrasp -> grasp -> lift portion is prechecked
        # and later executed.  The unused transfer/release/retreat poses remain
        # absent from this route.
        planned_by_name = dict(plan.ordered_waypoints())
        waypoints = tuple(
            (name, planned_by_name[name]) for name in ("pregrasp", "grasp", "lift")
        )
        log("[precheck] solving the complete three-segment grasp route before any actuator command")
        route = motion.precheck_linear_targets(
            tuple(pose for _name, pose in waypoints),
            step_m=float(cfg.motion.cartesian_step_m),
        )
        if len(route.segments) != len(waypoints):
            raise RuntimeError(
                f"grasp route has {len(route.segments)} segments; expected {len(waypoints)}"
            )
        for (name, pose), segment in zip(waypoints, route.segments):
            if segment.target != pose:
                raise RuntimeError(f"prechecked {name} target changed")
            if len(segment.waypoints) < 2:
                raise RuntimeError(f"prechecked {name} segment has no trajectory")
            if len(segment.joint_waypoints) != len(segment.waypoints):
                raise RuntimeError(f"prechecked {name} segment lacks joint plan")
            log(
                f"ROUTE {name}: distance={segment.distance_m:.6f} m "
                f"waypoints={len(segment.waypoints) - 1} "
                f"max_joint_step={segment.max_joint_step_deg:.3f} deg "
                f"max_joint_accel={segment.max_joint_acceleration_deg:.3f} deg "
                f"min_sigma={segment.min_jacobian_singular_value:.6f}"
            )
        minimum_sigma, required_sigma = _route_metrics(cfg, route)
        log(
            f"[precheck-pass] segments={len(route.segments)} "
            f"min_sigma={minimum_sigma:.6f} required_sigma={required_sigma:.6f}"
        )

        # Ensure the robot did not move after the RGB-D capture or during the
        # read-only IK/singularity precheck.
        sequence = int(motion.get_feedback_sequence())
        fresh = motion.wait_for_feedback_state(3.0, newer_than=sequence)
        _validate_state("post-precheck", fresh)
        fresh_position_error, fresh_rotation_error = _pose_errors(
            fresh.pose, route.start_pose.xyz, route.start_pose.rpy
        )
        if fresh_position_error > 0.001:
            raise RuntimeError(
                f"robot moved after route precheck: {fresh_position_error * 1000:.3f} mm"
            )
        if fresh_rotation_error > 0.005:
            raise RuntimeError(
                f"robot orientation changed after route precheck: "
                f"{fresh_rotation_error:.6f} rad"
            )
        logged_position_error, logged_rotation_error = _pose_errors(
            fresh.pose, visual_state.pose.xyz, visual_state.pose.rpy
        )
        if logged_position_error > 0.002 or logged_rotation_error > 0.0087:
            raise RuntimeError(
                "robot moved between RGB-D capture and grasp execution: "
                f"position={logged_position_error * 1000:.3f} mm "
                f"rotation={logged_rotation_error:.6f} rad"
            )
        log(
            f"[post-precheck-pass] drift_from_route_start="
            f"{fresh_position_error * 1000:.3f} mm/{fresh_rotation_error:.6f} rad "
            f"capture_to_start={logged_position_error * 1000:.3f} mm/"
            f"{logged_rotation_error:.6f} rad"
        )

        if callable(waiter) and not waiter(2.0):
            raise RuntimeError("/RobotMove bridge subscriber disappeared")
        elif not callable(waiter) and int(publisher.get_subscription_count()) < 1:
            raise RuntimeError("/RobotMove bridge subscriber disappeared")

        gripper = build_gripper(cfg.gripper, node=node, log=log)
        if gripper.state != GripperState.OPEN:
            log("[gripper] opening jaws once before entering the approach corridor")
            gripper.open()
        else:
            log("[gripper] jaws already report OPEN; no redundant open command")
        if gripper.state != GripperState.OPEN:
            raise RuntimeError(f"gripper did not open: state={gripper.state}")

        motion_stage = "PREGRASP"
        log("[motion] executing prechecked pregrasp")
        motion.execute_prechecked_segment(
            route.segments[0],
            name="pick4_pregrasp",
            speed_mps=AUTHORIZED_SPEED_MPS,
        )

        motion_stage = "GRASP"
        log("[motion] executing prechecked grasp descent")
        motion.execute_prechecked_segment(
            route.segments[1],
            name="pick4_grasp",
            speed_mps=AUTHORIZED_SPEED_MPS,
        )

        motion_stage = "CLOSE"
        close_attempted = True
        log(
            f"[gripper] closing exactly once for width={float(plan.grasp_close_width_m):.6f} m"
        )
        gripper.close_for_width(float(plan.grasp_close_width_m))
        if gripper.state != GripperState.CLOSED:
            raise RuntimeError(f"gripper close did not report CLOSED: {gripper.state}")

        verifier = build_hold_verifier(cfg.gripper, gripper, log=log)
        held, hold_reason = verifier.verify(detection, plan)
        log(f"HOLD_VERIFY held={int(bool(held))} reason={hold_reason}")
        if not held:
            log("[stop] grasp hold verification failed; lift was not commanded")
            log("VERDICT=PICK4_GRASP_HOLD_FAILED_NO_RETRY_NO_RELEASE")
            log(f"LOG={log_path}")
            return 3

        motion_stage = "LIFT"
        lift_attempted = True
        log("[motion] hold verified; executing prechecked lift")
        motion.execute_prechecked_segment(
            route.segments[2],
            name="pick4_lift",
            speed_mps=AUTHORIZED_SPEED_MPS,
        )

        # No release, retreat, initial-pose return or next-point logic follows.
        time.sleep(0.3)
        sequence = int(motion.get_feedback_sequence())
        final = motion.wait_for_feedback_state(3.0, newer_than=sequence)
        _validate_state("final", final)
        final_position_error, final_rotation_error = _pose_errors(
            final.pose, plan.lift_pose.xyz, plan.lift_pose.rpy
        )
        log(
            f"FINAL_STATE state={final.controller_state} {_pose_text(final.pose)} "
            f"joints_deg={np.round(np.degrees(np.asarray(final.joints)), 3).tolist()}"
        )
        log(
            f"FINAL_CHECK lift_position_error={final_position_error * 1000:.3f} mm "
            f"rotation_error={final_rotation_error:.6f} rad"
        )
        if final_position_error > 0.005:
            raise RuntimeError(
                f"final lift position error {final_position_error * 1000:.3f} mm "
                "exceeds 5 mm"
            )
        if final_rotation_error > 0.01:
            raise RuntimeError(
                f"final lift rotation error {final_rotation_error:.6f} rad exceeds 0.01 rad"
            )
        if gripper.state != GripperState.CLOSED:
            raise RuntimeError(f"gripper no longer reports CLOSED: {gripper.state}")

        log(
            "[success] pick_4 grasped and lifted once; jaws remain CLOSED and no "
            "release/retreat/retry/next point was executed"
        )
        log("VERDICT=PICK4_GRASP_HOLD_SUCCESS_NO_RELEASE")
        log(f"LOG={log_path}")
        return 0
    except BaseException as exc:
        log(f"[ABORT] stage={motion_stage} {type(exc).__name__}: {exc}")
        log(
            f"SAFETY_STOP close_attempted={int(close_attempted)} "
            f"lift_attempted={int(lift_attempted)}; no retry, release or retreat sent"
        )
        log("VERDICT=ABORTED_NO_RETRY")
        log(f"LOG={log_path}")
        return 1
    finally:
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass
        if gripper is not None:
            try:
                gripper.close_backend()
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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement for the one authorized live grasp",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    args = parser.parse_args()
    if not args.execute:
        parser.error("refusing live grasp without --execute")
    return run(args.config.resolve(), args.log_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
