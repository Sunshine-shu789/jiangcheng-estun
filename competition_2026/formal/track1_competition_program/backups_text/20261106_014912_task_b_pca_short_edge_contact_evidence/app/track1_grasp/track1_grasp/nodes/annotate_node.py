#!/usr/bin/env python3
"""Task A entry point: detection, annotation and IoU scoring.

Runs the vision-only pipeline.  Before the camera is mounted it reads the mock
scenario; after mounting, point ``vision.backend`` at ``taska_realsense`` and
``task_a.ground_truth_file`` at the labelled frames to score IoU per position.
"""

from __future__ import annotations

import sys

from .track1_node import main as track1_main


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(arg in ("--task", "--mode") for arg in args):
        args = ["--task", "A", *args]
    return track1_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
