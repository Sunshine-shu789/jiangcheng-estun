from __future__ import annotations

import concurrent.futures
import json
import os
import re
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from track1_grasp.config import load_config
from track1_grasp.domain import Pose
from track1_grasp.kinematics import (
    controller_fk,
    matrix_to_rpy,
    weighted_jacobian_singular_values,
)
from track1_grasp.motion import RobotMoveMotionBackend

BASE = Path('/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序')
CONFIGS = [
    BASE / 'config/track1_task_b_resume2_20261020_live.yaml',
    BASE / 'config/track1_task_b_resume3_5_20261020_live.yaml',
    BASE / 'config/track1_task_b_20261020.yaml',
]


def anchor_for(config, index: int) -> Pose:
    q = np.radians(np.asarray(config.task_b.point_observation_joints_deg[index], dtype=float))
    transform = controller_fk(q)
    return Pose(tuple(float(v) for v in transform[:3, 3]), matrix_to_rpy(transform[:3, :3]))


def search_poses(config, expected_point: tuple[float, ...]) -> tuple[Pose, ...]:
    base_x, base_y, base_z = (float(v) for v in expected_point[:3])
    clearance = float(config.search.observation_clearance_m)
    if config.motion.orientation == 'fixed':
        rpy = tuple(float(v) for v in config.motion.fixed_rpy)
    elif len(expected_point) >= 6:
        rpy = tuple(float(v) for v in expected_point[3:6])
    else:
        raise RuntimeError('expected point lacks configured RPY')
    return tuple(
        Pose((base_x + float(dx), base_y + float(dy), base_z + clearance), rpy)
        for dx, dy in config.search.observation_offsets_xy_m
    )


def check_point(task):
    config_path_text, index = task
    config_path = Path(config_path_text)
    config = load_config(config_path)
    config = replace(config, motion=replace(config.motion, dry_run=True))
    configured_offsets_json = os.environ.get('SEARCH_OFFSETS_JSON')
    if configured_offsets_json:
        configured_offsets = tuple(
            tuple(float(v) for v in pair)
            for pair in json.loads(configured_offsets_json)
        )
        config = replace(
            config,
            search=replace(
                config.search,
                observation_offsets_xy_m=configured_offsets,
            ),
        )
    name = list(config.task_b.place_names)[index]
    offsets = list(config.search.observation_offsets_xy_m)
    anchor = anchor_for(config, index)
    q = np.radians(np.asarray(config.task_b.point_observation_joints_deg[index], dtype=float))
    backend = object.__new__(RobotMoveMotionBackend)
    backend.config = config.motion
    backend._log = lambda message: None
    backend._lock = threading.Lock()
    backend._state = 'AutoReady'
    backend._end_positions = [*anchor.xyz, *anchor.rpy]
    backend._joint_positions = [float(v) for v in q]
    backend._fault_flag = 0
    backend._stamp = time.monotonic()
    expected = tuple(float(v) for v in config.place_poses[index])
    poses = search_poses(config, expected)
    margin = float(config.motion.workspace_margin)
    low = np.asarray(config.motion.workspace_min) + margin
    high = np.asarray(config.motion.workspace_max) - margin
    in_workspace = [
        bool(np.all(np.asarray(p.xyz) >= low) and np.all(np.asarray(p.xyz) <= high))
        for p in poses
    ]
    kept_indices = [i for i, ok in enumerate(in_workspace) if ok]
    kept = [poses[i] for i in kept_indices]
    start_sigma = float(np.min(weighted_jacobian_singular_values(q)))
    row = {
        'point': name,
        'config': config_path.name,
        'anchor_xyz': anchor.xyz,
        'anchor_rpy': anchor.rpy,
        'expected_point': expected,
        'search_views': [
            {
                'index': i + 1,
                'offset_m': offsets[i],
                'xyz': p.xyz,
                'rpy': p.rpy,
                'in_workspace': in_workspace[i],
            }
            for i, p in enumerate(poses)
        ],
        'start_sigma': start_sigma,
        'status': 'PASS',
        'min_sigma': start_sigma,
        'max_joint_step_deg': 0.0,
        'waypoints': 0,
        'segments': [],
        'error': '',
    }
    if not kept:
        row['status'] = 'FAIL'
        row['error'] = 'all configured search views are outside workspace'
        return {'config': str(config_path), 'sigma_threshold': float(config.motion.cartesian_min_jacobian_singular_value), 'row': row}
    try:
        route = backend.precheck_linear_targets(tuple(kept), step_m=float(config.motion.cartesian_step_m))
        if len(route.segments) != len(kept):
            raise RuntimeError(f'segment count {len(route.segments)} != view count {len(kept)}')
        for route_index, segment in enumerate(route.segments):
            view_index = kept_indices[route_index]
            sigma = float(segment.min_jacobian_singular_value)
            row['segments'].append({
                'view_index': view_index + 1,
                'offset_m': offsets[view_index],
                'min_sigma': sigma,
                'max_joint_step_deg': float(segment.max_joint_step_deg),
                'waypoints': max(0, len(segment.waypoints) - 1),
            })
            row['min_sigma'] = min(row['min_sigma'], sigma)
            row['max_joint_step_deg'] = max(row['max_joint_step_deg'], float(segment.max_joint_step_deg))
            row['waypoints'] += max(0, len(segment.waypoints) - 1)
    except Exception as exc:
        row['status'] = 'FAIL'
        row['error'] = str(exc)
        match = re.search(r'singular value=([0-9.]+)', str(exc))
        if match:
            row['min_sigma'] = float(match.group(1))
        target = re.search(r'target (\d+)', str(exc))
        if target and 0 < int(target.group(1)) <= len(kept_indices):
            view_index = kept_indices[int(target.group(1)) - 1]
            row['failed_view'] = view_index + 1
            row['failed_offset_m'] = offsets[view_index]
    return {'config': str(config_path), 'sigma_threshold': float(config.motion.cartesian_min_jacobian_singular_value), 'row': row}


def main():
    tasks = []
    for config_path in CONFIGS:
        config = load_config(config_path)
        tasks.extend((str(config_path), index) for index in range(len(config.task_b.place_names)))
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as pool:
        futures = {pool.submit(check_point, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            row = result['row']
            print(
                f"FINISHED {Path(result['config']).name} {row['point']}: "
                f"{row['status']} min_sigma={row['min_sigma']:.6f}",
                flush=True,
            )
            results.append(result)
    by_config = {}
    for result in results:
        by_config.setdefault(result['config'], {
            'config': result['config'],
            'sigma_threshold': result['sigma_threshold'],
            'rows': [],
        })['rows'].append(result['row'])
    output = {'checks': [by_config[str(path)] for path in CONFIGS]}
    out = Path(os.environ.get(
        'SEARCH_OUTPUT_JSON',
        BASE / 'logs/task_b_search_route_check.json',
    ))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'WROTE {out}')
    for check in output['checks']:
        print(f"\nCONFIG {Path(check['config']).name} threshold={check['sigma_threshold']:.6f}")
        for row in check['rows']:
            print(f"{row['point']}: {row['status']} min_sigma={row['min_sigma']:.6f} max_joint_step={row['max_joint_step_deg']:.3f}deg waypoints={row['waypoints']}")
            if row['status'] != 'PASS':
                print(f"  FAILED view={row.get('failed_view')} offset={row.get('failed_offset_m')} error={row['error']}")
            for segment in row['segments']:
                print(f"  view {segment['view_index']} offset={segment['offset_m']} min_sigma={segment['min_sigma']:.6f} max_step={segment['max_joint_step_deg']:.3f} waypoints={segment['waypoints']}")


if __name__ == '__main__':
    main()
