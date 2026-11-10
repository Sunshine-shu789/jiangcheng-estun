#!/usr/bin/env python3
"""ROS 2 entry point for the task C/D/E pick pipeline."""

from __future__ import annotations

import argparse
import threading
from dataclasses import replace
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from task_cde_grasp.config import apply_mode, load_config
from task_cde_grasp.factory import build_pipeline


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "cde_mock.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=("C", "D", "E"))
    parser.add_argument("--max-picks", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="disable motion dry-run (requires an enabled, ready robot)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.mode:
        config = apply_mode(config, args.mode)
    if args.max_picks:
        config = replace(config, task=replace(config.task, max_picks=args.max_picks))
    if args.execute:
        config = replace(config, motion=replace(config.motion, dry_run=False))

    rclpy.init()
    node = Node("task_cde_grasp")
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
                pipeline.run()
            except BaseException as exc:  # propagate to main after executor stops
                error.append(exc)

        thread = threading.Thread(target=worker, name="cde-pick", daemon=True)
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
