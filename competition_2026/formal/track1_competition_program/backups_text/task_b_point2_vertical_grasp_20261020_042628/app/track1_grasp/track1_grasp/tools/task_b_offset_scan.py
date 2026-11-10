from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from track1_grasp.config import load_config
from track1_grasp.domain import Detection, Pose
from track1_grasp.kinematics import controller_fk, matrix_to_rpy
from track1_grasp.motion import RobotMoveMotionBackend
from track1_grasp.planner import PosePlanner

BASE = Path(__file__).resolve().parents[4]
CONFIG = Path(os.environ.get(
    "TASK_B_SCAN_CONFIG",
    BASE / "config/track1_task_b_resume2_20261020_live.yaml",
))
VISUAL_XY = {
    "pick_2": (-0.15564433, 0.52594438),
    "pick_3": (-0.141, 0.454),
    "pick_4": (0.063, 0.433),
    "pick_5": (0.034, 0.506),
}
_G = {}


def init_worker() -> None:
    config = load_config(CONFIG)
    config = replace(
        config,
        pose=replace(config.pose, use_detection_base_xyz=True),
        motion=replace(config.motion, dry_run=True),
    )
    _G["config"] = config
    _G["planner"] = PosePlanner(config)


def scan(task):
    point_index, name, dx, dy = task
    config = _G["config"]
    planner = _G["planner"]
    obs_deg = config.task_b.point_observation_joints_deg[point_index]
    q = np.radians(np.asarray(obs_deg, dtype=float))
    transform = controller_fk(q)
    anchor = Pose(
        tuple(float(v) for v in transform[:3, 3]),
        matrix_to_rpy(transform[:3, :3]),
    )
    x, y = VISUAL_XY[name]
    x += dx
    y += dy
    detection = Detection(
        label="8-Plastic_Banana",
        confidence=0.95,
        bbox=(200, 150, 320, 280),
        base_xyz=(x, y, 0.035),
        graspability=1.0,
        instance_id="offset-scan",
    )
    backend = object.__new__(RobotMoveMotionBackend)
    backend.config = config.motion
    backend._log = lambda message: None
    backend._lock = threading.Lock()
    backend._state = "AutoReady"
    backend._end_positions = [*anchor.xyz, *anchor.rpy]
    backend._joint_positions = [float(v) for v in q]
    backend._fault_flag = 0
    backend._stamp = time.monotonic()
    state = backend.get_state()
    try:
        plan = planner.plan(
            detection,
            state,
            return_to_pick=True,
            place_name=name,
            grasp_z_m=config.task_b.point_grasp_z_m[point_index],
            grasp_rpy=config.task_b.point_grasp_rpy[point_index],
            task_b_anchor_pose=anchor,
        )
        route = backend.precheck_linear_targets(
            tuple(pose for _, pose in plan.ordered_waypoints()),
            step_m=config.motion.cartesian_step_m,
        )
        segment_min = [
            float(segment.min_jacobian_singular_value)
            for segment in route.segments
        ]
        start_sigma = None
        # The precheck validates the anchor start posture before all segments.
        # Record it analytically for diagnostics without creating ROS state.
        from track1_grasp.kinematics import weighted_jacobian_singular_values
        start_sigma = float(np.min(weighted_jacobian_singular_values(q)))
        return {
            "point": name,
            "dx_mm": round(dx * 1000.0, 3),
            "dy_mm": round(dy * 1000.0, 3),
            "status": "PASS",
            "min_sigma": min([start_sigma, *segment_min]),
            "start_sigma": start_sigma,
            "segment_min_sigma": segment_min,
            "max_joint_step_deg": max(s.max_joint_step_deg for s in route.segments),
            "waypoints": sum(max(0, len(s.waypoints) - 1) for s in route.segments),
            "error": "",
        }
    except Exception as exc:
        message = str(exc)
        sigma_match = re.search(r"singular value=([0-9.]+)", message)
        target_match = re.search(r"target (\d+)", message)
        return {
            "point": name,
            "dx_mm": round(dx * 1000.0, 3),
            "dy_mm": round(dy * 1000.0, 3),
            "status": "FAIL",
            "min_sigma": float(sigma_match.group(1)) if sigma_match else None,
            "start_sigma": None,
            "segment_min_sigma": [],
            "max_joint_step_deg": None,
            "waypoints": None,
            "target_index": int(target_match.group(1)) if target_match else None,
            "error": message,
        }


def main() -> None:
    config = load_config(CONFIG)
    names = list(config.task_b.place_names)
    offsets = [i * 0.020 for i in range(-4, 5)]
    tasks = []
    for index, name in enumerate(names):
        for dx in offsets:
            for dy in offsets:
                if math.hypot(dx, dy) <= 0.0800001:
                    tasks.append((index, name, dx, dy))
    print(f"scan_tasks={len(tasks)} workers=16 sigma_threshold={config.motion.cartesian_min_jacobian_singular_value}")
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=16, initializer=init_worker
    ) as pool:
        for result in pool.map(scan, tasks, chunksize=1):
            results.append(result)
            if result["status"] == "FAIL":
                print(
                    f"FAIL {result['point']} d=({result['dx_mm']:+.0f},{result['dy_mm']:+.0f}) "
                    f"target={result.get('target_index')} sigma={result['min_sigma']}"
                )
    output = Path(os.environ.get(
        "TASK_B_OFFSET_SCAN_OUTPUT",
        BASE / "logs/task_b_offset_scan.json",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {output}")
    for name in names:
        point_results = [r for r in results if r["point"] == name]
        passed = [r for r in point_results if r["status"] == "PASS"]
        failed = [r for r in point_results if r["status"] == "FAIL"]
        print(f"\n[{name}] pass={len(passed)} fail={len(failed)} total={len(point_results)}")
        if passed:
            best = min(passed, key=lambda r: r["min_sigma"])
            worst_step = max(passed, key=lambda r: r["max_joint_step_deg"])
            print(
                f"  pass_min_sigma={best['min_sigma']:.6f} at "
                f"({best['dx_mm']:+.0f},{best['dy_mm']:+.0f})mm; "
                f"max_joint_step={worst_step['max_joint_step_deg']:.3f}deg at "
                f"({worst_step['dx_mm']:+.0f},{worst_step['dy_mm']:+.0f})mm"
            )
        for r in sorted(failed, key=lambda item: (item["dy_mm"], item["dx_mm"])):
            print(
                f"  FAIL ({r['dx_mm']:+.0f},{r['dy_mm']:+.0f})mm "
                f"target={r.get('target_index')} sigma={r['min_sigma']}"
            )


if __name__ == "__main__":
    main()
