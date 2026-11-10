#!/usr/bin/env python3
"""Bounded no-gripper arm motion check.

This entry point is intentionally independent of vision, grasping and the
gripper backend.  It plans a short straight-line TCP move and the matching
return path as one pre-checked route, then executes it only when ``--execute``
is supplied.  No gripper command is ever built or published.

Typical supervised use::

    test_arm_no_gripper --axis x --direction auto --distance-mm 10

The launcher adds ``--execute`` after an operator confirmation dialog.
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import replace

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from track1_grasp.config import load_config
from track1_grasp.config_paths import package_config_path
from track1_grasp.domain import Pose, RobotState
from track1_grasp.motion import build_motion


DEFAULT_CONFIG = package_config_path("track1_live.yaml")
AXES = ("x", "y", "z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow real motion; omitted means dry-run planning only",
    )
    parser.add_argument("--axis", choices=AXES, default="x")
    parser.add_argument(
        "--direction",
        choices=("auto", "+", "-"),
        default="auto",
        help="base-frame direction; auto chooses the side with more workspace",
    )
    parser.add_argument(
        "--distance-mm",
        type=float,
        default=10.0,
        help="outbound TCP distance in millimetres (default 10, maximum 20)",
    )
    parser.add_argument(
        "--step-mm",
        type=float,
        default=2.5,
        help="Cartesian interpolation spacing in millimetres (default 2.5)",
    )
    parser.add_argument(
        "--radial-inward-mm",
        type=float,
        default=None,
        help=(
            "coordinated mode: move this far toward the base vertical axis "
            "in the horizontal plane"
        ),
    )
    parser.add_argument(
        "--up-mm",
        type=float,
        default=None,
        help="coordinated mode: move this far upward in base +Z",
    )
    parser.add_argument(
        "--max-joint1-travel-deg",
        type=float,
        default=None,
        help="abort before motion if the planned Joint1 travel exceeds this",
    )
    parser.add_argument(
        "--max-joint1-step-deg",
        type=float,
        default=None,
        help="abort before motion if one planned Joint1 step exceeds this",
    )
    return parser


def choose_direction(
    current_xyz: np.ndarray,
    axis: int,
    low: float,
    high: float,
    requested: str,
) -> int:
    """Return +1 or -1 without hiding an explicit operator choice."""

    if requested == "+":
        return 1
    if requested == "-":
        return -1
    positive_room = float(high) - float(current_xyz[axis])
    negative_room = float(current_xyz[axis]) - float(low)
    return 1 if positive_room >= negative_room else -1


def bounded_round_trip(
    state: RobotState,
    *,
    axis: int,
    direction: int,
    distance_m: float,
    low: tuple[float, float, float],
    high: tuple[float, float, float],
) -> tuple[Pose, Pose]:
    """Build outbound and return poses while preserving current orientation."""

    if state.pose is None:
        raise ValueError("robot pose is required")
    if axis < 0 or axis > 2:
        raise ValueError("axis index must be 0, 1 or 2")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("distance must be a positive finite value")

    start_xyz = np.asarray(state.pose.xyz, dtype=float)
    target_xyz = start_xyz.copy()
    target_xyz[axis] += float(direction) * float(distance_m)
    if np.any(target_xyz < np.asarray(low)) or np.any(target_xyz > np.asarray(high)):
        raise ValueError(
            "requested target is outside the configured workspace: "
            f"target={target_xyz.tolist()} low={list(low)} high={list(high)}"
        )
    start = Pose(tuple(float(v) for v in start_xyz), tuple(state.pose.rpy))
    target = Pose(tuple(float(v) for v in target_xyz), tuple(state.pose.rpy))
    return target, start


def bounded_radial_up_round_trip(
    state: RobotState,
    *,
    inward_m: float,
    up_m: float,
    low: tuple[float, float, float],
    high: tuple[float, float, float],
) -> tuple[Pose, Pose]:
    """Build a coordinated radial-inward/upward out-and-back test pose.

    A radial horizontal move leaves the base-yaw angle unchanged in the ideal
    case, so Joint1 should only make a very small correction.  Joint2-Joint6
    absorb the vertical and posture changes according to the local IK solution.
    """

    if state.pose is None:
        raise ValueError("robot pose is required")
    inward = float(inward_m)
    up = float(up_m)
    if not np.isfinite(inward) or not np.isfinite(up):
        raise ValueError("radial and vertical distances must be finite")
    if inward < 0.0 or up < 0.0:
        raise ValueError("coordinated test only permits inward and upward motion")
    if inward <= 0.0 and up <= 0.0:
        raise ValueError("at least one coordinated distance must be positive")

    start_xyz = np.asarray(state.pose.xyz, dtype=float)
    radial = np.asarray([start_xyz[0], start_xyz[1]], dtype=float)
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1e-9:
        raise ValueError("cannot define an inward radial direction at the base axis")

    target_xyz = start_xyz.copy()
    target_xyz[:2] -= radial / radial_norm * inward
    target_xyz[2] += up
    if np.any(target_xyz < np.asarray(low)) or np.any(target_xyz > np.asarray(high)):
        raise ValueError(
            "requested coordinated target is outside the configured workspace: "
            f"target={target_xyz.tolist()} low={list(low)} high={list(high)}"
        )
    start = Pose(tuple(float(v) for v in start_xyz), tuple(state.pose.rpy))
    target = Pose(tuple(float(v) for v in target_xyz), tuple(state.pose.rpy))
    return target, start


def route_joint_travel_deg(route, joint_index: int) -> float:
    """Return summed absolute path travel for one joint over a prechecked route."""

    index = int(joint_index)
    if index < 0 or index >= 6:
        raise ValueError("joint index must be in [0, 6)")
    total = 0.0
    for segment in route.segments:
        metrics = segment.joint_metrics
        if metrics is None:
            raise ValueError("prechecked route has no per-axis joint metrics")
        total += float(metrics.travel_deg[index])
    return total


def route_joint_max_step_deg(route, joint_index: int) -> float:
    """Return the largest single waypoint step for one joint on a route."""

    index = int(joint_index)
    if index < 0 or index >= 6:
        raise ValueError("joint index must be in [0, 6)")
    maximum = 0.0
    for segment in route.segments:
        metrics = segment.joint_metrics
        if metrics is None:
            raise ValueError("prechecked route has no per-axis joint metrics")
        maximum = max(maximum, float(metrics.max_step_deg[index]))
    return maximum


class ArmMotionCheckRunner:
    def __init__(self, node: Node, config, args) -> None:
        self.node = node
        self.config = config
        self.args = args
        self._log = lambda message: node.get_logger().info(str(message))
        self.motion = build_motion(config.motion, node=node, log=self._log)

    def run(self) -> int:
        step_m = float(self.args.step_mm) / 1000.0
        if not 0.0 < step_m <= 0.005:
            raise ValueError("--step-mm must be in (0, 5]")

        coordinated = (
            self.args.radial_inward_mm is not None
            or self.args.up_mm is not None
        )
        inward_m = float(self.args.radial_inward_mm or 0.0) / 1000.0
        up_m = float(self.args.up_mm or 0.0) / 1000.0
        if coordinated:
            if inward_m < 0.0 or up_m < 0.0:
                raise ValueError(
                    "--radial-inward-mm and --up-mm must both be non-negative"
                )
            coordinated_distance = float(np.hypot(inward_m, up_m))
            if not 0.0 < coordinated_distance <= 0.020:
                raise ValueError(
                    "combined radial/upward displacement must be in (0, 20] mm"
                )
            distance_m = coordinated_distance
        else:
            distance_m = float(self.args.distance_mm) / 1000.0
            if not 0.0 < distance_m <= 0.020:
                raise ValueError("--distance-mm must be in (0, 20]")

        max_joint1_travel = self.args.max_joint1_travel_deg
        if max_joint1_travel is None and coordinated:
            max_joint1_travel = 0.25
        if max_joint1_travel is not None:
            max_joint1_travel = float(max_joint1_travel)
            if not np.isfinite(max_joint1_travel) or max_joint1_travel <= 0.0:
                raise ValueError("--max-joint1-travel-deg must be positive")
        max_joint1_step = self.args.max_joint1_step_deg
        if max_joint1_step is None and coordinated:
            max_joint1_step = 0.05
        if max_joint1_step is not None:
            max_joint1_step = float(max_joint1_step)
            if not np.isfinite(max_joint1_step) or max_joint1_step <= 0.0:
                raise ValueError("--max-joint1-step-deg must be positive")

        wait_for_state = getattr(self.motion, "wait_for_state", None)
        if wait_for_state is None:
            state = self.motion.get_state()
        else:
            state = wait_for_state(max(1.0, float(self.config.motion.robot_info_timeout_s)))
        if not state.ready or state.pose is None:
            raise RuntimeError(
                "robot must be AutoReady with fault_flag=0 and complete "
                f"pose/joint feedback: {state.message}"
            )

        margin = float(self.config.motion.workspace_margin)
        low = tuple(float(v) + margin for v in self.config.motion.workspace_min)
        high = tuple(float(v) - margin for v in self.config.motion.workspace_max)
        if coordinated:
            target, start = bounded_radial_up_round_trip(
                state,
                inward_m=inward_m,
                up_m=up_m,
                low=low,
                high=high,
            )
            direction = 0
            start_description = (
                f"coordinated radial-in {inward_m * 1000:.1f} mm + "
                f"up {up_m * 1000:.1f} mm ({distance_m * 1000:.1f} mm)"
            )
        else:
            axis = AXES.index(self.args.axis)
            direction = choose_direction(
                np.asarray(state.pose.xyz, dtype=float),
                axis,
                low[axis],
                high[axis],
                self.args.direction,
            )
            target, start = bounded_round_trip(
                state,
                axis=axis,
                direction=direction,
                distance_m=distance_m,
                low=low,
                high=high,
            )
            sign = "+" if direction > 0 else "-"
            start_description = (
                f"base {sign}{self.args.axis} {distance_m * 1000:.1f} mm"
            )

        mode = "EXECUTE" if self.args.execute else "DRY-RUN"
        self._log(
            f"[start] {mode} no-gripper check: {start_description} "
            f"out-and-back, step={step_m * 1000:.1f} mm"
        )
        self._log(
            f"[robot] {state.message}; start_xyz={start.xyz} "
            f"joints_deg={np.round(np.degrees(state.joints), 2).tolist()}"
        )

        precheck = getattr(self.motion, "precheck_linear_targets", None)
        execute = getattr(self.motion, "execute_linear_route", None)
        if precheck is None or execute is None:
            raise RuntimeError("configured motion backend has no linear precheck API")

        # Both directions are checked before the first real waypoint is sent.
        route = precheck((target, start), step_m=step_m)
        self._log(
            f"[precheck] outbound={route.segments[0].distance_m * 1000:.1f} mm "
            f"return={route.segments[1].distance_m * 1000:.1f} mm; all IK, "
            "joint limits and workspace checks passed"
        )
        if max_joint1_travel is not None or max_joint1_step is not None:
            joint1_travel = route_joint_travel_deg(route, 0)
            joint1_step = route_joint_max_step_deg(route, 0)
            self._log(
                f"[precheck] Joint1 travel={joint1_travel:.3f} deg, "
                f"max_step={joint1_step:.3f} deg"
            )
            if (
                max_joint1_travel is not None
                and joint1_travel > max_joint1_travel + 1e-9
            ):
                raise RuntimeError(
                    "planned Joint1 travel "
                    f"{joint1_travel:.3f} deg exceeds the requested limit "
                    f"{max_joint1_travel:.3f} deg; no motion was sent"
                )
            if (
                max_joint1_step is not None
                and joint1_step > max_joint1_step + 1e-9
            ):
                raise RuntimeError(
                    "planned Joint1 step "
                    f"{joint1_step:.3f} deg exceeds the requested limit "
                    f"{max_joint1_step:.3f} deg; no motion was sent"
                )
        execute(
            route,
            names=("no_gripper_out", "no_gripper_return"),
            speed_mps=float(self.config.motion.speed_mps),
        )
        if not self.args.execute:
            self._log("[result] dry-run complete; no motion was sent and gripper stayed untouched")
            return 0

        final_state = (
            self.motion.wait_for_state(1.0)
            if hasattr(self.motion, "wait_for_state")
            else self.motion.get_state()
        )
        if final_state.pose is None:
            raise RuntimeError("no final robot pose after motion check")
        error_mm = float(
            np.linalg.norm(
                np.asarray(final_state.pose.xyz, dtype=float)
                - np.asarray(start.xyz, dtype=float)
            )
            * 1000.0
        )
        self._log(
            f"[result] out-and-back complete; return error={error_mm:.2f} mm; "
            "no gripper command was sent"
        )
        if error_mm > 5.0:
            raise RuntimeError(
                f"return position error {error_mm:.2f} mm exceeds 5 mm"
            )
        return 0

    def close(self) -> None:
        close = getattr(self.motion, "close", None)
        if close is not None:
            close()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.execute:
        config = replace(config, motion=replace(config.motion, dry_run=False))

    rclpy.init()
    node = Node("track1_arm_motion_check")
    runner = None
    try:
        runner = ArmMotionCheckRunner(node, config, args)
        result: list[int] = []
        error: list[BaseException] = []

        def worker() -> None:
            try:
                result.append(runner.run())
            except BaseException as exc:
                error.append(exc)

        thread = threading.Thread(target=worker, name="arm-motion-check", daemon=True)
        thread.start()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        while rclpy.ok() and thread.is_alive():
            executor.spin_once(timeout_sec=0.1)
        thread.join(timeout=1.0)
        executor.remove_node(node)
        executor.shutdown()
        if error:
            raise error[0]
        return result[0] if result else 1
    except KeyboardInterrupt:
        return 130
    except BaseException as exc:
        node.get_logger().error(str(exc))
        return 2
    finally:
        if runner is not None:
            runner.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
