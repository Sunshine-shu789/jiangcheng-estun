#!/usr/bin/env python3
"""Build gripper-link meshes from the official FAE2MR86M-D STEP extraction.

The extraction is performed by ``extraction/extract_step_components.py``.  This
script only transforms the resulting component STLs from the CAD assembly frame
to the gripper-local frame, groups rigid components, and writes deterministic
visual/collision STLs for URDF links.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_link_meshes import (  # noqa: E402
    matrix_list,
    read_binary_stl,
    sha256,
    transform_triangles,
    voxel_cluster,
    write_binary_stl,
)

DEFAULT_ASSEMBLY = PACKAGE_ROOT / "config" / "gripper_assembly.fae2mr86m_ms42dc.yaml"
DEFAULT_DIMENSIONS = PACKAGE_ROOT / "config" / "fae2mr86m_ms42dc_dimensions.json"
DEFAULT_COMPONENT_DIR = PACKAGE_ROOT / "mesh" / "gripper" / "components"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "mesh" / "gripper"

LINK_GROUPS = {
    "gripper_body_link": "body_components",
    "gripper_left_finger_link": "left_finger_components",
    "gripper_right_finger_link": "right_finger_components",
}


def local_from_cad(dimensions: dict) -> np.ndarray:
    coordinate_system = dimensions["coordinate_system"]
    rotation = np.asarray(coordinate_system["rotation_local_from_cad"], dtype=float)
    origin_mm = np.asarray(coordinate_system["origin_cad_mm"], dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_local_from_cad must be 3x3")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9):
        raise ValueError("rotation_local_from_cad must be orthonormal")
    origin_m = origin_mm / 1000.0
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ origin_m
    return transform


def read_components(component_dir: Path, names: Sequence[str]) -> np.ndarray:
    groups = []
    for name in names:
        path = component_dir / f"{name}.stl"
        if not path.is_file():
            raise FileNotFoundError(path)
        groups.append(read_binary_stl(path))
    if not groups:
        raise ValueError("component list must not be empty")
    return np.concatenate(groups, axis=0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, default=DEFAULT_ASSEMBLY)
    parser.add_argument("--dimensions", type=Path, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--component-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    assembly = yaml.safe_load(args.assembly.read_text(encoding="utf-8"))
    dimensions = json.loads(args.dimensions.read_text(encoding="utf-8"))
    source_step = PACKAGE_ROOT / str(assembly["source"]["step_path"])
    if not source_step.is_file():
        raise FileNotFoundError(source_step)
    if sha256(source_step) != str(assembly["source"]["step_sha256"]):
        raise ValueError("gripper STEP SHA-256 does not match assembly authority")
    if sha256(args.dimensions) != sha256(DEFAULT_DIMENSIONS):
        raise ValueError("non-default dimensions file is not supported without a manifest update")

    mesh_config = assembly["mesh"]
    local_from_cad_matrix = local_from_cad(dimensions)
    cad_from_local_matrix = np.linalg.inv(local_from_cad_matrix)
    visual_dir = args.output_dir / "visual"
    collision_dir = args.output_dir / "collision"
    visual_dir.mkdir(parents=True, exist_ok=True)
    collision_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "schema_version": 1,
        "status": assembly["status"],
        "source_step": str(source_step),
        "source_step_sha256": sha256(source_step),
        "component_directory": str(args.component_dir),
        "component_report": str(args.component_dir / "step_report.json"),
        "component_report_sha256": sha256(args.component_dir / "step_report.json"),
        "cad_from_gripper_local_matrix": matrix_list(cad_from_local_matrix),
        "gripper_local_from_cad_matrix": matrix_list(local_from_cad_matrix),
        "coordinate_definition": dimensions["coordinate_system"],
        "visual_voxel_mm": float(mesh_config["visual_voxel_mm"]),
        "collision_voxel_mm": float(mesh_config["collision_voxel_mm"]),
        "links": {},
        "caveat": "Open-state CAD meshes with voxel decimation; offline collision use only.",
    }

    for link_name, group_key in LINK_GROUPS.items():
        component_names = [str(name) for name in mesh_config[group_key]]
        global_triangles = read_components(args.component_dir, component_names)
        local_triangles = transform_triangles(global_triangles, local_from_cad_matrix)
        visual, visual_stats = voxel_cluster(
            local_triangles,
            voxel_m=float(mesh_config["visual_voxel_mm"]) / 1000.0,
        )
        collision, collision_stats = voxel_cluster(
            local_triangles,
            voxel_m=float(mesh_config["collision_voxel_mm"]) / 1000.0,
        )
        visual_path = visual_dir / f"{link_name}.stl"
        collision_path = collision_dir / f"{link_name}.stl"
        write_binary_stl(
            visual_path,
            visual,
            header=b"FAE2MR86M-D gripper visual mesh",
        )
        write_binary_stl(
            collision_path,
            collision,
            header=b"FAE2MR86M-D gripper collision mesh",
        )
        bounds_min = local_triangles.min(axis=(0, 1))
        bounds_max = local_triangles.max(axis=(0, 1))
        component_hashes = {
            name: sha256(args.component_dir / f"{name}.stl")
            for name in component_names
        }
        report["links"][link_name] = {
            "components": component_names,
            "component_stl_sha256": component_hashes,
            "bounds_m": {
                "min": [float(value) for value in bounds_min],
                "max": [float(value) for value in bounds_max],
            },
            "visual": {
                "path": str(visual_path),
                "sha256": sha256(visual_path),
                "bytes": visual_path.stat().st_size,
                **visual_stats,
            },
            "collision": {
                "path": str(collision_path),
                "sha256": sha256(collision_path),
                "bytes": collision_path.stat().st_size,
                **collision_stats,
            },
        }
        print(
            f"{link_name}: visual {visual_stats['output_triangles']} triangles, "
            f"collision {collision_stats['output_triangles']} triangles"
        )

    report_path = args.output_dir / "generation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "sha256": sha256(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
