#!/usr/bin/env python3
"""Replay captured Task C frames through the production support-plane path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "track1_grasp"))

from track1_grasp import adaptive  # noqa: E402
from track1_grasp.config import load_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=ROOT / "logs" / "task_c_geometry_diagnostic_20261107_164100",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "track1_task_c_v11_official_20261102.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs" / "task_c_support_plane_replay_20261107.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = json.loads(
        (args.diagnostic_dir / "summary.json").read_text(encoding="utf-8")
    )
    config = load_config(args.config).task_cd.short_edge_grasp.pca
    frames: list[dict[str, object]] = []
    for frame in summary["frames"]:
        depth = np.load(
            args.diagnostic_dir / frame["images"]["depth_m_npz"]
        )["depth_m"]
        intrinsics = SimpleNamespace(**frame["camera_intrinsics"])
        detections = [
            item
            for item in frame["detections"]
            if item.get("is_known_class")
            and item.get("duplicate_of_detection_index") is None
        ]
        bboxes = tuple(
            tuple(float(value) for value in item["bbox_xyxy_px"])
            for item in detections
        )
        estimate = adaptive.estimate_scene_support_plane(
            depth,
            bboxes,
            intrinsics,
            config,
        )
        clearances = None
        if estimate is not None:
            clearances = adaptive._support_plane_target_clearances(
                depth,
                bboxes,
                np.asarray(estimate.center_camera_xyz, dtype=float),
                np.asarray(estimate.normal_camera, dtype=float),
                intrinsics,
                config,
            )
        frames.append(
            {
                "frame_index": int(frame["index"]),
                "bboxes_xyxy_px": bboxes,
                "labels": [str(item["label"]) for item in detections],
                "passed": estimate is not None and clearances is not None,
                "support_plane": (
                    None
                    if estimate is None
                    else {
                        "center_camera_xyz_m": list(estimate.center_camera_xyz),
                        "normal_camera": list(estimate.normal_camera),
                        "rms_m": float(estimate.rms_m),
                        "valid_points": int(estimate.valid_points),
                        "candidate_points": int(estimate.candidate_points),
                        "inlier_ratio": float(estimate.inlier_ratio),
                        "target_clearances_m": list(clearances or ()),
                    }
                ),
            }
        )

    passed = sum(bool(frame["passed"]) for frame in frames)
    payload = {
        "diagnostic_dir": str(args.diagnostic_dir.resolve()),
        "config": str(args.config.resolve()),
        "frame_count": len(frames),
        "passed_frames": passed,
        "status": "passed" if passed == len(frames) else "failed",
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"{passed}/{len(frames)} frames passed; report={args.output}")
    return 0 if passed == len(frames) else 1


if __name__ == "__main__":
    raise SystemExit(main())
