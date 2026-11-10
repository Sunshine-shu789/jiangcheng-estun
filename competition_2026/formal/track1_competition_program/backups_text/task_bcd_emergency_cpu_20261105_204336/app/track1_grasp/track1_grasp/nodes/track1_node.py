#!/usr/bin/env python3
"""ROS 2 entry point for the whole Track 1 (A/B/C/D/E) framework.

    ros2 run track1_grasp track1_node --config <file> --task C
    ros2 run track1_grasp track1_node --config <file> --task B --execute

``--execute`` disables the motion dry-run.  It is required for any real
movement: the default everywhere is dry-run.
"""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import replace

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from track1_grasp.config import Track1Config, apply_mode, load_config
from track1_grasp.config_paths import package_config_path
from track1_grasp.domain import PickStatus, RunReport, TaskMode, VisionReport
from track1_grasp.factory import build_pipeline


DEFAULT_CONFIG = package_config_path("track1_mock.yaml")


def prepare_config(
    config: Track1Config,
    *,
    task: str | None = None,
    max_picks: int | None = None,
    frames: int | None = None,
    execute: bool = False,
) -> Track1Config:
    """Apply command-line overrides and re-run every configuration guard.

    ``load_config`` validates the on-disk YAML, but the node subsequently
    replaces task and safety fields.  Re-validating the prepared value prevents
    an override from bypassing the Task A J1-J4 dual-arc route constraints.
    """

    prepared = config
    if task is not None:
        prepared = apply_mode(prepared, task)
    if max_picks is not None:
        prepared = replace(
            prepared,
            task=replace(prepared.task, max_picks=max_picks),
        )
    if frames is not None:
        prepared = replace(
            prepared,
            task_a=replace(prepared.task_a, max_frames=frames),
        )
    if execute:
        prepared = replace(
            prepared,
            motion=replace(prepared.motion, dry_run=False),
        )
    prepared.validate()
    return prepared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--task", "--mode", dest="task", choices=("A", "B", "C", "D", "E"))
    parser.add_argument(
        "--max-picks",
        type=int,
        help=(
            "legacy count limit for non-clearing modes; ignored by an enabled "
            "Task C/D task_cd clearing run"
        ),
    )
    parser.add_argument(
        "--acceptance-max-successes",
        type=int,
        help=(
            "Task C/D field-acceptance hard stop after N successful picks; "
            "the arm returns to the fixed observation pose before exit"
        ),
    )
    parser.add_argument("--frames", type=int, help="task A: stop after N frames")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="disable motion dry-run (requires an enabled, ready robot)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = prepare_config(
        load_config(args.config),
        task=args.task,
        max_picks=args.max_picks,
        frames=args.frames,
        execute=args.execute,
    )

    rclpy.init()
    node = Node("track1_grasp")
    pipeline = None
    try:
        arbitrary_placement = bool(
            config.adaptive_inverse.enabled
            and config.adaptive_inverse.arbitrary_placement
        )
        box_relative_five_point = bool(
            config.adaptive_inverse.enabled
            and config.adaptive_inverse.box_relative_five_point.enabled
        )
        dynamic_task_b = bool(arbitrary_placement or box_relative_five_point)

        def pause_for_next_point(completed_name: str, completed: int, total: int) -> None:
            if not sys.stdin.isatty():
                raise RuntimeError(
                    "operator pause after each pick requires an interactive terminal"
                )
            target_name = config.task_b.target_class.strip() or "目标物体"
            if config.task_b.return_to_observation_after_each_pick:
                arm_state = "机械臂已返回本轮观察位，准备下一次视觉采集。"
            elif config.task_b.return_to_initial_after_each_pick:
                arm_state = "机械臂已返回初始位。"
            else:
                arm_state = "机械臂已退回本轮抓取点上方。"
            if box_relative_five_point:
                prompt = (
                    f"\n[人工确认] {completed_name} 已完成原位拾放和撤回，"
                    f"{arm_state}"
                    f"\n请将{target_name}放到箱内新的位置；下一轮只使用新的"
                    "当前画面重新拟合箱体位姿，不读取本轮或任何历史抓取位置。"
                    "\n确认物体已放稳、机械臂周围安全后按回车继续。"
                    f"\n当前进度：{completed}/{total}；下一步："
                    f"{completed + 1}/{total}\n"
                )
            elif dynamic_task_b:
                prompt = (
                    f"\n[人工确认] {completed_name} 已完成原位拾放和撤回，"
                    f"{arm_state}"
                    f"\n请将{target_name}随意摆放到新的位置；下一轮只会使用"
                    "新的当前画面，不读取本轮或任何历史抓取位置。"
                    "\n确认物体已放稳、机械臂周围安全后按回车继续。"
                    f"\n当前进度：{completed}/{total}；下一步："
                    f"{completed + 1}/{total}\n"
                )
            else:
                prompt = (
                    f"\n[人工确认] {completed_name} 已完成原位拾放和撤回，"
                    f"{arm_state}"
                    "\n请将香蕉放到下一个点位，确认机械臂周围安全后按回车继续。"
                    f"\n当前进度：{completed}/{total}；下一步："
                    f"{completed + 1}/{total}\n"
                )
            print(prompt, flush=True)
            input("按回车继续 > ")

        pipeline = build_pipeline(
            config,
            node=node,
            log=lambda message: node.get_logger().info(str(message)),
            operator_gate=pause_for_next_point,
        )
        error: list[BaseException] = []
        exit_code = [0]

        def worker() -> None:
            try:
                report = pipeline.run(
                    acceptance_max_successes=args.acceptance_max_successes,
                )
                if isinstance(report, RunReport):
                    clearing = bool(
                        config.task.mode in {TaskMode.C, TaskMode.D}
                        and config.task_cd.enabled
                    )
                    if clearing and report.results:
                        # C/D clearing is successful only when its final
                        # complete observation proves the bin empty.  In
                        # operator-acceptance mode, the only successful result
                        # is reaching the requested success count and ending on
                        # a completed pick; NO_TARGET/rejection is a failure.
                        if args.acceptance_max_successes is not None:
                            acceptance_complete = bool(
                                report.successes
                                >= args.acceptance_max_successes
                                and report.results[-1].status
                                is PickStatus.SUCCESS
                            )
                            if not acceptance_complete:
                                exit_code[0] = 1
                        elif (
                            report.results[-1].status is not PickStatus.NO_TARGET
                        ):
                            exit_code[0] = 1
                    elif (
                        not clearing
                        and report.attempts > 0
                        and report.successes == 0
                    ):
                        exit_code[0] = 1
                if isinstance(report, VisionReport):
                    node.get_logger().info(
                        f"task A: frames={report.frames} passed={report.passed}/"
                        f"{report.total_annotations} pass_rate={report.pass_rate:.1%}"
                    )
                    if report.patrol_distances_m:
                        node.get_logger().info(
                            "task A patrol: "
                            f"moves={len(report.patrol_distances_m)} "
                            f"min_distance={min(report.patrol_distances_m):.3f} m "
                            "target_positions="
                            f"{list(report.patrol_visible_positions)} "
                            f"detected_classes={list(report.patrol_detected_classes)} "
                            f"missing_classes={list(report.patrol_missing_classes)}"
                        )
            except BaseException as exc:  # propagate to main after executor stops
                error.append(exc)

        thread = threading.Thread(target=worker, name="track1-pipeline", daemon=True)
        thread.start()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        while rclpy.ok() and thread.is_alive():
            executor.spin_once(timeout_sec=0.1)
        thread.join(timeout=1.0)
        executor.remove_node(node)
        executor.shutdown()
        if error:
            raise error[0]
        return exit_code[0]
    except KeyboardInterrupt:
        return 130
    finally:
        if pipeline is not None:
            pipeline.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
