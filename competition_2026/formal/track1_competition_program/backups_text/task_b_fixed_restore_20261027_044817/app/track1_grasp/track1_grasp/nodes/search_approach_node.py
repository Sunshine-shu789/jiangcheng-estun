#!/usr/bin/env python3
"""Live camera search and small-step approach test for Track 1.

Safety model
------------
* detection only is the default and never moves the robot;
* joint-space search requires ``--execute --scan``;
* visual servoing requires ``--execute --approach``;
* every real movement checks ``AutoReady`` and ``fault_flag == 0``;
* only one bounded Cartesian step per servo iteration is published;
* no gripper command is ever sent by this node.

Typical supervised bring-up::

    ros2 run track1_grasp search_approach_node \
      --config <live.yaml>                    # current view, no motion
    ros2 run track1_grasp search_approach_node \
      --config <live.yaml> --execute --scan --scan-only
    ros2 run track1_grasp search_approach_node \
      --config <live.yaml> --execute --approach --steps 1
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import replace

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from track1_grasp.approach import (
    compute_servo_command,
    pose_inside_workspace,
    select_target,
)
from track1_grasp.config import load_config
from track1_grasp.config_paths import package_config_path
from track1_grasp.domain import Detection, Frame, RobotState
from track1_grasp.kinematics import (
    joint_scan_target,
    joints_within_limits,
)
from track1_grasp.motion import build_motion
from track1_grasp.vision import build_vision


DEFAULT_CONFIG = package_config_path("track1_live.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow real robot motion; omitted means dry-run",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="search joint-space pitch/yaw views when the target is not visible",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="stop after finding the target; do not approach",
    )
    parser.add_argument(
        "--approach",
        action="store_true",
        help="run eye-in-hand visual servoing after detection",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="override servo.max_steps (use 1 for the first supervised move)",
    )
    parser.add_argument(
        "--max-step-mm",
        type=float,
        help="override servo.max_step_m in millimetres",
    )
    parser.add_argument(
        "--desired-standoff-mm",
        type=float,
        help="override servo.desired_standoff_m in millimetres",
    )
    parser.add_argument(
        "--min-depth-mm",
        type=float,
        help="override servo.min_depth_m in millimetres",
    )
    parser.add_argument(
        "--max-total-travel-mm",
        type=float,
        help="override servo.max_total_travel_m in millimetres",
    )
    parser.add_argument(
        "--target-class",
        help="override vision.target_class, e.g. 6-Foam_Brick",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            "after the task completes, keep the live camera/YOLO preview "
            "running until Ctrl-C; this never moves the robot or gripper"
        ),
    )
    return parser


class SearchApproachRunner:
    def __init__(self, node: Node, config, args) -> None:
        self.node = node
        self.config = config
        self.args = args
        self._log = lambda message: node.get_logger().info(str(message))
        self.motion = build_motion(config.motion, node=node, log=self._log)
        vision_config = config.vision
        if config.task.mode is not None:
            vision_config = replace(
                vision_config,
                scenario_file=vision_config.scenario_for(config.task.mode),
            )
        self.vision = build_vision(vision_config, log=self._log)
        self.target_class = (
            args.target_class
            if args.target_class is not None
            else config.vision.target_class
        )
        self._set_vision_status("STARTING SEARCH")

    def run(self) -> int:
        self._log(
            "[start] mode="
            f"{'execute' if self.args.execute else 'dry-run'} "
            f"scan={self.args.scan} approach={self.args.approach} "
            f"target={self.target_class or 'any'}"
        )
        initial_state = self.motion.get_state()
        wait_for_state = getattr(self.motion, "wait_for_state", None)
        if wait_for_state is not None:
            initial_state = wait_for_state(
                max(1.0, float(self.config.motion.robot_info_timeout_s))
            )
        self._log(
            f"[robot] {initial_state.message}; "
            f"pose={self._format_pose(initial_state)} "
            f"joints_deg={self._format_joints(initial_state.joints)}"
        )
        self._set_vision_status("SEARCHING: observing current view")

        detection, frame, state = self._find_target(initial_state)
        if detection is None:
            self._log("[result] target not found")
            self._set_vision_status("DONE: target not found")
            return 1
        self._log_detection(detection, frame)
        self._set_vision_status(
            f"TARGET LOCKED: {detection.label} {detection.confidence:.2f}"
        )
        if self.args.scan_only or not self.args.approach:
            self._log(
                "[result] target detected; no approach requested"
                if not self.args.scan_only
                else "[scan-only] target detected; arm remains at detecting view"
            )
            self._set_vision_status("DONE: detection complete; arm idle")
            return 0

        return self._approach(detection, state)

    def _find_target(
        self, initial_state: RobotState
    ) -> tuple[Detection | None, Frame | None, RobotState]:
        detection, frame = self._capture_best()
        if detection is not None or not self.args.scan:
            return detection, frame, self.motion.get_state()

        views = self._search_views()
        if len(views) <= 1:
            return None, frame, self.motion.get_state()
        if not self.args.execute:
            if not initial_state.ready or initial_state.joints is None:
                self._log(
                    "[search] dry-run cannot validate scan IK: "
                    f"{initial_state.message}"
                )
                return None, frame, self.motion.get_state()
            q_initial = np.asarray(initial_state.joints, dtype=float)
            for index, (pitch_deg, yaw_deg) in enumerate(views[1:], start=1):
                try:
                    q_target = joint_scan_target(
                        q_initial,
                        np.asarray(self.config.servo.camera_to_tool, dtype=float),
                        pitch_down_deg=pitch_deg,
                        yaw_deg=yaw_deg,
                    )
                except RuntimeError as exc:
                    self._log(
                        f"[search:dry-run] view {index + 1} pitch_down="
                        f"{pitch_deg:+.1f} yaw={yaw_deg:+.1f}: IK failed: {exc}"
                    )
                    continue
                limits_ok = joints_within_limits(q_target)
                self._log(
                    f"[search:dry-run] view {index + 1} pitch_down="
                    f"{pitch_deg:+.1f} yaw={yaw_deg:+.1f}: "
                    f"IK ok={limits_ok} joints_deg="
                    f"{np.round(np.degrees(q_target), 2).tolist()}"
                )
            return None, frame, self.motion.get_state()
        if not initial_state.ready or initial_state.joints is None:
            raise RuntimeError(
                f"cannot run joint-space search: {initial_state.message}"
            )

        q_initial = np.asarray(initial_state.joints, dtype=float)
        latest = initial_state
        for index, (pitch_deg, yaw_deg) in enumerate(views[1:], start=1):
            try:
                q_target = joint_scan_target(
                    q_initial,
                    np.asarray(self.config.servo.camera_to_tool, dtype=float),
                    pitch_down_deg=pitch_deg,
                    yaw_deg=yaw_deg,
                )
            except RuntimeError as exc:
                self._log(f"[search] stop before view {index + 1}: {exc}")
                break
            if not joints_within_limits(q_target):
                self._log(
                    f"[search] stop before view {index + 1}: joint target "
                    "violates limits"
                )
                break
            latest = self.motion.get_state()
            if not latest.ready:
                raise RuntimeError(f"robot not ready during search: {latest.message}")
            if not hasattr(self.motion, "move_joints"):
                raise RuntimeError("configured motion backend cannot search joints")
            self._set_vision_status(f"MOVING: search view {index + 1}/{len(views)}")
            self.motion.move_joints(
                q_target,
                name=(
                    f"search_{index:02d}_pitch_{pitch_deg:+.0f}deg_"
                    f"yaw_{yaw_deg:+.0f}deg"
                ),
            )
            self._set_vision_status(f"DETECTING: search view {index + 1}/{len(views)}")
            time.sleep(max(0.0, self.config.search.settle_s))
            detection, frame = self._capture_best()
            self._log(
                f"[search] view {index + 1}/{len(views)} "
                f"pitch_down={pitch_deg:+.1f} yaw={yaw_deg:+.1f} "
                f"detections={len(frame.detections) if frame else 0}"
            )
            if detection is not None:
                return detection, frame, self.motion.get_state()

        if self.config.search.return_to_initial_on_miss:
            self._log("[search] target not found; returning to initial joints")
            self._set_vision_status("MOVING: returning to initial search pose")
            try:
                self.motion.move_joints(q_initial, name="search_return")
            except RuntimeError as exc:
                self._log(f"[search] return move failed: {exc}")
        return None, frame, self.motion.get_state()

    def _capture_best(self) -> tuple[Detection | None, Frame]:
        best: Detection | None = None
        latest: Frame | None = None
        for _ in range(max(1, self.config.search.observe_frames)):
            frame = self.vision.capture()
            latest = frame
            candidate = select_target(
                frame,
                target_class=self.target_class,
                min_confidence=self.config.vision.min_confidence,
            )
            if candidate is not None and (
                best is None or candidate.score() > best.score()
            ):
                best = candidate
            time.sleep(0.05)
        if latest is None:
            raise RuntimeError("vision source returned no frame")
        return best, latest

    def _approach(
        self, initial_detection: Detection, initial_state: RobotState
    ) -> int:
        servo = self.config.servo
        if self.args.steps is not None:
            if self.args.steps < 1:
                raise ValueError("--steps must be >= 1")
            servo = replace(servo, max_steps=int(self.args.steps))
        if self.args.max_step_mm is not None:
            max_step_m = float(self.args.max_step_mm) / 1000.0
            if not 0.0 < max_step_m <= 0.05:
                raise ValueError("--max-step-mm must be in (0, 50]")
            servo = replace(servo, max_step_m=max_step_m)

        state = self._fresh_state(initial_state)
        if state.pose is None:
            raise RuntimeError(f"cannot approach without robot pose: {state.message}")
        start_xyz = np.asarray(state.pose.xyz, dtype=float)
        converged_frames = 0
        executed_steps = 0
        detection = initial_detection
        self._set_vision_status("APPROACH: target locked")

        for step in range(servo.max_steps):
            if detection is None:
                self._log(f"[servo] step {step:02d}: target lost")
                return 1
            state = self._fresh_state(state)
            if self.args.execute and not state.ready:
                raise RuntimeError(
                    f"robot must be AutoReady with fault=0 before moving: "
                    f"{state.message}"
                )
            command = compute_servo_command(
                state, detection, servo, motion=self.config.motion
            )
            inside, low, high = pose_inside_workspace(command.target, self.config.motion)
            travel = float(
                np.linalg.norm(np.asarray(command.target.xyz) - start_xyz)
            )
            self._log(
                f"[servo] step={step:02d} "
                f"pixel_error=({command.pixel_error[0]:+.1f},"
                f"{command.pixel_error[1]:+.1f})px "
                f"depth={command.depth_m:.3f}m "
                f"delta_base={np.round(command.delta_base, 4).tolist()} "
                f"travel={travel:.3f}m"
            )
            if command.converged:
                converged_frames += 1
                self._log(
                    f"[servo] converged frame {converged_frames}/"
                    f"{servo.convergence_frames}"
                )
                if converged_frames >= servo.convergence_frames or not self.args.execute:
                    self._log("[result] object centered at approach standoff")
                    self._set_vision_status("APPROACH DONE: target centered")
                    return 0
            else:
                converged_frames = 0

            if travel > servo.max_total_travel_m:
                self._log(
                    f"[abort] total travel {travel:.3f} m exceeds "
                    f"{servo.max_total_travel_m:.3f} m"
                )
                self._set_vision_status("ABORT: total travel limit")
                return 1
            if not inside:
                self._log(
                    f"[abort] target {np.round(command.target.xyz, 4).tolist()} "
                    f"outside workspace low={low.tolist()} high={high.tolist()}"
                )
                self._set_vision_status("ABORT: target outside workspace")
                return 1
            if command.converged:
                time.sleep(max(0.05, servo.settle_s_after_move))
            else:
                name = f"approach_step_{step + 1:02d}"
                self._set_vision_status(f"MOVING: {name}")
                self.motion.move_to(
                    command.target,
                    name=name,
                    speed_mps=servo.speed_mps,
                )
                self._set_vision_status(f"DETECTING: after {name}")
                executed_steps += 1
                if not self.args.execute:
                    self._log(
                        "[result] dry-run: one servo step planned; "
                        "no motion was published"
                    )
                    return 0
                time.sleep(max(0.0, servo.settle_s_after_move))

            detection, frame = self._capture_best()
            if detection is None:
                self._log(f"[servo] step {step:02d}: target lost after move")
                self._set_vision_status("ABORT: target lost after move")
                return 1
            self._log_detection(detection, frame)

        self._log(
            f"[result] approach stopped after {servo.max_steps} step(s); "
            f"executed={executed_steps}, not yet converged"
        )
        return 0 if executed_steps else 1

    def _set_vision_status(self, text: str) -> None:
        set_status = getattr(self.vision, "set_status", None)
        if set_status is None:
            return
        try:
            set_status(text)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._log(f"[vision:live] status update failed: {exc}")

    def begin_watch(self) -> None:
        """Keep continuous vision alive after the task thread has finished."""

        self._set_vision_status("WATCHING: live recognition; arm idle")
        self._log(
            "[watch] live camera/YOLO preview remains active; press Ctrl-C to stop"
        )

    def _fresh_state(self, fallback: RobotState) -> RobotState:
        wait_for_state = getattr(self.motion, "wait_for_state", None)
        if wait_for_state is None:
            return self.motion.get_state() or fallback
        state = wait_for_state(1.0)
        return state if state.pose is not None else fallback

    def _search_views(self) -> list[tuple[float, float]]:
        if not self.config.search.enabled:
            return [(0.0, 0.0)]
        pitches = self.config.search.pitch_down_offsets_deg or (0.0,)
        yaws = self.config.search.yaw_offsets_deg or (0.0,)
        return [
            (float(pitch), float(yaw))
            for pitch in pitches
            for yaw in yaws
        ]

    def _log_detection(self, detection: Detection, frame: Frame) -> None:
        xyz = detection.camera_xyz
        image_state = frame.image_path or ("memory" if frame.image is not None else "none")
        self._log(
            f"[vision] {detection.label} conf={detection.confidence:.3f} "
            f"depth={float(detection.depth_m or 0.0):.3f}m "
            f"camera_xyz={tuple(round(float(v), 4) for v in xyz) if xyz else None} "
            f"image={image_state}"
        )

    @staticmethod
    def _format_joints(joints) -> str:
        if joints is None:
            return "unknown"
        return str(np.round(np.degrees(np.asarray(joints)), 2).tolist())

    @staticmethod
    def _format_pose(state: RobotState) -> str:
        if state.pose is None:
            return "unknown"
        return str(tuple(round(float(v), 4) for v in state.pose.xyz))

    def close(self) -> None:
        for component in (self.vision, self.motion):
            close = getattr(component, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    self._log(f"[shutdown] {type(component).__name__}: {exc}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps is not None and args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.execute and not (args.scan or args.approach):
        raise SystemExit(
            "--execute only enables motion; also pass --scan or --approach"
        )

    config = load_config(args.config)

    servo_overrides = {}
    if args.desired_standoff_mm is not None:
        if args.desired_standoff_mm <= 0.0:
            raise SystemExit("--desired-standoff-mm must be > 0")
        servo_overrides["desired_standoff_m"] = args.desired_standoff_mm / 1000.0
    if args.min_depth_mm is not None:
        if args.min_depth_mm <= 0.0:
            raise SystemExit("--min-depth-mm must be > 0")
        servo_overrides["min_depth_m"] = args.min_depth_mm / 1000.0
    if args.max_total_travel_mm is not None:
        if args.max_total_travel_mm <= 0.0:
            raise SystemExit("--max-total-travel-mm must be > 0")
        servo_overrides["max_total_travel_m"] = (
            args.max_total_travel_mm / 1000.0
        )
    if servo_overrides:
        servo = replace(config.servo, **servo_overrides)
        if servo.desired_standoff_m <= servo.min_depth_m:
            raise SystemExit(
                "--desired-standoff-mm must exceed --min-depth-mm "
                "(or the configured min_depth_m)"
            )
        config = replace(config, servo=servo)

    if args.execute:
        config = replace(config, motion=replace(config.motion, dry_run=False))

    rclpy.init()
    node = Node("track1_search_approach")
    runner = None
    try:
        runner = SearchApproachRunner(node, config, args)
        result: list[int] = []
        error: list[BaseException] = []

        def worker() -> None:
            try:
                result.append(runner.run())
            except BaseException as exc:
                error.append(exc)

        thread = threading.Thread(target=worker, name="search-approach", daemon=True)
        thread.start()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        while rclpy.ok() and thread.is_alive():
            executor.spin_once(timeout_sec=0.1)
        thread.join(timeout=1.0)
        if args.watch and not error:
            runner.begin_watch()
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.1)
        executor.remove_node(node)
        executor.shutdown()
        if error:
            raise error[0]
        return result[0] if result else 1
    except KeyboardInterrupt:
        return 130
    finally:
        if runner is not None:
            runner.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
