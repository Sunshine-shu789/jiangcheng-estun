#!/usr/bin/env python3
"""ROS 2 entry point for the whole Track 1 (A/B/C/D/E) framework.

    ros2 run track1_grasp track1_node --config <file> --task C
    ros2 run track1_grasp track1_node --config <file> --task B --execute

``--execute`` disables the motion dry-run.  It is required for any real
movement: the default everywhere is dry-run.
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import replace

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from track1_grasp.config import apply_mode, load_config
from track1_grasp.config_paths import package_config_path
from track1_grasp.domain import VisionReport
from track1_grasp.factory import build_pipeline


DEFAULT_CONFIG = package_config_path("track1_mock.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--task", "--mode", dest="task", choices=("A", "B", "C", "D", "E"))
    parser.add_argument("--max-picks", type=int)
    parser.add_argument("--frames", type=int, help="task A: stop after N frames")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="disable motion dry-run (requires an enabled, ready robot)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.task:
        config = apply_mode(config, args.task)
    if args.max_picks:
        config = replace(config, task=replace(config.task, max_picks=args.max_picks))
    if args.frames:
        config = replace(config, task_a=replace(config.task_a, max_frames=args.frames))
    if args.execute:
        config = replace(config, motion=replace(config.motion, dry_run=False))

    rclpy.init()
    node = Node("track1_grasp")
    pipeline = None
    try:
        pipeline = build_pipeline(
            config,
            node=node,
            log=lambda message: node.get_logger().info(str(message)),
        )
        error: list[BaseException] = []

        def worker() -> None:
            try:
                report = pipeline.run()
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
                            f"{list(report.patrol_visible_positions)}"
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
        return 0
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
