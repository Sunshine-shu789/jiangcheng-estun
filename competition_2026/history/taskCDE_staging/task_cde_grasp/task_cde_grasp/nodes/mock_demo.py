#!/usr/bin/env python3
"""Run a complete no-hardware CDE pick demonstration."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from task_cde_grasp.config import apply_mode, load_config
from task_cde_grasp.factory import build_pipeline


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "cde_mock.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=("C", "D", "E"))
    parser.add_argument("--max-picks", type=int)
    return parser


def main(argv=None) -> int:
    # parse_known_args keeps the demo usable both directly and via `ros2 launch`,
    # which appends its own `--ros-args -r __node:=...` remap arguments.
    args, _unknown = build_parser().parse_known_args(argv)
    config = load_config(args.config)
    if args.mode:
        config = apply_mode(config, args.mode)
    if args.max_picks:
        config = replace(config, task=replace(config.task, max_picks=args.max_picks))

    pipeline = build_pipeline(config, log=print)
    try:
        report = pipeline.run()
    finally:
        pipeline.close()

    print("\n=== CDE mock summary ===")
    print(
        f"mode={config.task.mode.value} requested={report.requested_picks} "
        f"attempts={report.attempts} successes={report.successes} "
        f"rate={report.success_rate:.1%}"
    )
    for index, result in enumerate(report.results, 1):
        label = result.target.label if result.target else "-"
        print(
            f"{index:02d}. status={result.status.value:8s} target={label:12s} "
            f"message={result.message}"
        )
        print("    " + " -> ".join(event.phase.value for event in result.events))
    return 0 if report.successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
