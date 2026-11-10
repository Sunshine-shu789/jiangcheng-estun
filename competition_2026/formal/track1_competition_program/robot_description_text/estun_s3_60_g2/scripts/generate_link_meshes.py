#!/usr/bin/env python3
"""Transform STEP component meshes into controller-DH link frames.

The official S3-60-G2 STEP assembly is stored in a fixed global frame.  Its
joint layout matches the controller model at q=0 once one rigid global-to-base
transform is applied.  This script uses that relationship to place every STEP
component in the same link frame as the generated URDF.

Outputs:
  * link-local visual meshes, reduced with conservative vertex clustering;
  * link-local collision meshes, reduced more aggressively for MoveIt/FCL;
  * a JSON report with transforms, hashes and triangle counts.

The original component STLs remain untouched as source evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from typing import Sequence

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KINEMATICS = Path(
    "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/"
    "app/track1_grasp/track1_grasp/kinematics.py"
)
DEFAULT_MESH_DIR = PACKAGE_ROOT / "mesh"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "mesh" / "link_local"

COMPONENT_BY_LINK = {
    "base_link": "NAUO7",
    "link1": "NAUO3",
    "link2": "NAUO1",
    "link3": "NAUO2",
    "link4": "NAUO4",
    "link5": "NAUO5",
    "link6": "NAUO6",
}

# STEP assembly -> controller base at q=0.  The robot in the STEP file is
# lying on its side: base +Z -> STEP +Y, base +Y -> STEP +Z,
# base +X -> STEP -X.  STEP base origin is 4.123711 mm along global +Y.
STEP_TO_BASE = np.asarray(
    (
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.004123711),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=float,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_kinematics(path: Path):
    spec = importlib.util.spec_from_file_location("track1_link_mesh_kinematics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import kinematics module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.eye(4)
    matrix[1:3, 1:3] = ((c, -s), (s, c))
    return matrix


def rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.eye(4)
    matrix[:2, :2] = ((c, -s), (s, c))
    return matrix


def dh_constant_transform(row: Sequence[float]) -> np.ndarray:
    """Return Rx(alpha) Tx(a) Rz(theta) Tz(d) for one modified-DH row."""

    a_mm, alpha_deg, d_mm, theta_deg = (float(value) for value in row)
    translate_x = np.eye(4)
    translate_x[0, 3] = a_mm / 1000.0
    translate_z = np.eye(4)
    translate_z[2, 3] = d_mm / 1000.0
    return (
        rx(math.radians(alpha_deg))
        @ translate_x
        @ rz(math.radians(theta_deg))
        @ translate_z
    )


def link_frames_at_zero(dh: Sequence[Sequence[float]]) -> dict[str, np.ndarray]:
    frames = {"base_link": np.eye(4)}
    transform = np.eye(4)
    for index, row in enumerate(dh, start=1):
        transform = transform @ dh_constant_transform(row)
        frames[f"link{index}"] = transform.copy()
    return frames


def read_binary_stl(path: Path) -> np.ndarray:
    """Return a (N, 3, 3) float64 triangle array in metres."""

    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path}: file is too short to be a binary STL")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    expected = 84 + triangle_count * 50
    if len(raw) != expected:
        raise ValueError(
            f"{path}: expected {expected} bytes for {triangle_count} binary "
            f"triangles, got {len(raw)}"
        )
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    records = np.frombuffer(raw, dtype=dtype, count=triangle_count, offset=84)
    return records["vertices"].astype(np.float64) / 1000.0


def write_binary_stl(
    path: Path,
    triangles: np.ndarray,
    *,
    header: bytes = b"S3-60-G2 controller-DH link-local mesh",
) -> None:
    """Write triangle-soup vertices as a binary STL."""

    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (N, 3, 3)")
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-14
    triangles = triangles[valid]
    normals = normals[valid] / lengths[valid, None]
    if not len(triangles):
        raise ValueError("refusing to write an empty STL")

    records = np.empty(
        len(triangles),
        dtype=np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        ),
    )
    records["normal"] = normals.astype(np.float32)
    records["vertices"] = triangles.astype(np.float32)
    records["attribute"] = 0
    encoded_header = bytes(header)[:80].ljust(80, b"\0")
    with path.open("wb") as handle:
        handle.write(encoded_header)
        handle.write(struct.pack("<I", len(records)))
        handle.write(records.tobytes())


def voxel_cluster(
    triangles: np.ndarray,
    *,
    voxel_m: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Reduce a triangle soup using averaged vertex-cluster decimation.

    The clustering is deliberately simple and deterministic.  It is suitable
    for collision geometry and visualization, but it is not presented as
    manufacturing-grade reverse engineering.
    """

    if voxel_m <= 0.0:
        raise ValueError("voxel size must be positive")
    points = triangles.reshape(-1, 3)
    quantized = np.rint(points / voxel_m).astype(np.int64)
    _unique, inverse = np.unique(quantized, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=int(inverse.max()) + 1)
    sums = np.vstack(
        [np.bincount(inverse, weights=points[:, axis]) for axis in range(3)]
    ).T
    vertices = sums / counts[:, None]

    faces = inverse.reshape(-1, 3)
    nondegenerate = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    faces = faces[nondegenerate]
    if not len(faces):
        raise ValueError("voxel decimation removed every triangle")

    # Deduplicate faces independent of winding, while retaining the first
    # occurrence and therefore a consistent winding for the retained face.
    sorted_faces = np.sort(faces, axis=1)
    _keys, first = np.unique(sorted_faces, axis=0, return_index=True)
    faces = faces[np.sort(first)]
    reduced = vertices[faces]
    areas = np.linalg.norm(
        np.cross(reduced[:, 1] - reduced[:, 0], reduced[:, 2] - reduced[:, 0]),
        axis=1,
    )
    faces = faces[areas > 1e-14]
    reduced = vertices[faces]
    return reduced, {
        "input_triangles": int(len(triangles)),
        "cluster_vertices": int(len(vertices)),
        "output_triangles": int(len(reduced)),
    }


def transform_triangles(triangles: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = triangles.reshape(-1, 3)
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=float)), axis=1
    )
    transformed = (transform @ homogeneous.T).T[:, :3]
    return transformed.reshape(triangles.shape)


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinematics", type=Path, default=DEFAULT_KINEMATICS)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--visual-voxel-mm", type=float, default=1.0)
    parser.add_argument("--collision-voxel-mm", type=float, default=5.0)
    args = parser.parse_args(argv)

    kinematics = import_kinematics(args.kinematics)
    dh = tuple(tuple(float(value) for value in row) for row in kinematics.CONTROLLER_DH)
    if len(dh) != 6:
        raise ValueError("CONTROLLER_DH must contain exactly six rows")

    base_frames = link_frames_at_zero(dh)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "visual"
    collision_dir = args.output_dir / "collision"
    visual_dir.mkdir(parents=True, exist_ok=True)
    collision_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "status": "step_component_meshes_link_local",
        "source_step": str(PACKAGE_ROOT / "source" / "S3-60-G2.STEP"),
        "source_step_sha256": sha256(PACKAGE_ROOT / "source" / "S3-60-G2.STEP"),
        "controller_kinematics": str(args.kinematics),
        "controller_kinematics_sha256": sha256(args.kinematics),
        "step_to_base_matrix": matrix_list(STEP_TO_BASE),
        "step_to_base_derivation": {
            "assembly_pose": "controller q=0",
            "base_x_in_step": [-1.0, 0.0, 0.0],
            "base_y_in_step": [0.0, 0.0, 1.0],
            "base_z_in_step": [0.0, 1.0, 0.0],
            "base_origin_step_m": [0.0, 0.004123711, 0.0],
            "evidence": "Major STEP cylinder axes align with controller joint axes "
            "after this transform; component extents reproduce the assembled chain.",
        },
        "visual_voxel_mm": float(args.visual_voxel_mm),
        "collision_voxel_mm": float(args.collision_voxel_mm),
        "links": {},
        "caveat": "Voxel decimation is for visualization/collision, not "
        "manufacturing acceptance. Exact STEP faces remain in source/S3-60-G2.STEP.",
    }

    for link_name, component in COMPONENT_BY_LINK.items():
        source = args.mesh_dir / f"{component}.stl"
        if not source.is_file():
            raise FileNotFoundError(source)
        global_triangles = read_binary_stl(source)
        global_from_link = STEP_TO_BASE @ base_frames[link_name]
        link_from_global = np.linalg.inv(global_from_link)
        local_triangles = transform_triangles(global_triangles, link_from_global)

        visual, visual_stats = voxel_cluster(
            local_triangles, voxel_m=float(args.visual_voxel_mm) / 1000.0
        )
        collision, collision_stats = voxel_cluster(
            local_triangles, voxel_m=float(args.collision_voxel_mm) / 1000.0
        )
        visual_path = visual_dir / f"{link_name}.stl"
        collision_path = collision_dir / f"{link_name}.stl"
        write_binary_stl(visual_path, visual)
        write_binary_stl(collision_path, collision)

        bounds_min = local_triangles.min(axis=(0, 1))
        bounds_max = local_triangles.max(axis=(0, 1))
        report["links"][link_name] = {
            "step_component": component,
            "source_stl": str(source),
            "source_stl_sha256": sha256(source),
            "global_from_link_matrix": matrix_list(global_from_link),
            "link_local_bounds_m": {
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
            f"{link_name}: {component} -> visual "
            f"{visual_stats['output_triangles']} triangles, collision "
            f"{collision_stats['output_triangles']} triangles"
        )

    report_path = args.output_dir / "generation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "sha256": sha256(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
