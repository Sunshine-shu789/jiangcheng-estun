#!/usr/bin/env python3
"""Run the whole Track 1 framework without any hardware.

Task A exercises the detection/annotation/IoU path; tasks B/C/D/E exercise the
full pick state machine.  Everything is scripted from the mock scenario files,
so it works before the camera or gripper are mounted.

    mock_demo --config config/track1_mock.yaml --task A
    mock_demo --config config/track1_mock.yaml --task B
    mock_demo --config config/track1_mock.yaml --task E --max-picks 4
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from track1_grasp.config import apply_mode, load_config
from track1_grasp.config_paths import package_config_path
from track1_grasp.domain import TaskMode
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
        "--save-annotated", action="store_true", help="task A: write annotated images"
    )
    return parser


def main(argv=None) -> int:
    # parse_known_args keeps the demo usable both directly and via `ros2 launch`,
    # which appends its own `--ros-args -r __node:=...` remap arguments.
    args, _unknown = build_parser().parse_known_args(argv)
    config = load_config(args.config)
    if args.task:
        config = apply_mode(config, args.task)
    if args.max_picks:
        config = replace(config, task=replace(config.task, max_picks=args.max_picks))
    if args.frames:
        config = replace(config, task_a=replace(config.task_a, max_frames=args.frames))
    if args.save_annotated:
        config = replace(config, task_a=replace(config.task_a, save_annotated=True))

    pipeline = build_pipeline(config, log=print)
    try:
        report = pipeline.run()
    finally:
        pipeline.close()

    if isinstance(report, VisionReport):
        return _print_vision_summary(config, report)
    return _print_pick_summary(config, report)


def _print_vision_summary(config, report: VisionReport) -> int:
    threshold = report.iou_threshold
    print("\n=== Track 1 task A summary ===")
    print(
        f"frames={report.frames} detections={report.detections} "
        f"matched={report.matched}/{report.total_annotations} "
        f"iou_threshold={threshold:.2f}"
    )
    print(
        f"passed={report.passed}/{report.total_annotations} "
        f"pass_rate={report.pass_rate:.1%} mean_iou={report.mean_iou:.3f}"
    )
    for match in report.results:
        label = match.label
        print(f"  {label:14s} iou={match.iou:.3f} {'ok' if match.passed else 'low'}")
    expected = config.task_a.min_positions
    if report.frames < expected:
        print(f"note: only {report.frames} position(s); the competition needs {expected}")
    return 0 if report.passed else 1


def _print_pick_summary(config, report) -> int:
    print("\n=== Track 1 pick summary ===")
    print(
        f"task={config.task.mode.value} requested={report.requested_picks} "
        f"attempts={report.attempts} successes={report.successes} "
        f"rate={report.success_rate:.1%}"
    )
    for index, result in enumerate(report.results, 1):
        label = result.target.label if result.target else "-"
        place = f" -> {result.place_name}" if result.place_name else ""
        print(
            f"{index:02d}. status={result.status.value:8s} target={label:12s}{place} "
            f"message={result.message}"
        )
        print("    " + " -> ".join(event.phase.value for event in result.events))
    return 0 if report.successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
