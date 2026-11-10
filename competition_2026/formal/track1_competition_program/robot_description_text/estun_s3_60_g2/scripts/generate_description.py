#!/usr/bin/env python3
"""Generate the S3-60-G2 URDF from the controller-matched DH model.

This script deliberately imports ``track1_grasp.kinematics`` instead of
duplicating the DH table.  The generated URDF therefore either matches the
controller model used by live code or generation fails.

The arm component STLs extracted from the official STEP file and the official
FAE2MR86M-D gripper STEP are represented as link-local meshes.  The tool chain
uses the provisional offline assembly authority; it is not a hardware
calibration and never authorizes motion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KINEMATICS = Path(
    "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/"
    "app/track1_grasp/track1_grasp/kinematics.py"
)
DEFAULT_GEOMETRY = PACKAGE_ROOT / "config" / "geometry_estimates.yaml"
DEFAULT_CAMERA = (
    PACKAGE_ROOT / "config" / "camera_to_flange.d405_calibrated_20261025.yaml"
)
DEFAULT_GRIPPER_ASSEMBLY = (
    PACKAGE_ROOT / "config" / "gripper_assembly.fae2mr86m_ms42dc.yaml"
)
DEFAULT_OUTPUT = PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf"

JOINT_EFFORT = (80.0, 100.0, 80.0, 50.0, 50.0, 50.0)
JOINT_VELOCITY = (2.617994, 2.617994, 2.617994, 3.141593, 3.141593, 3.141593)
LINK_NAMES = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6")


def import_kinematics(path: Path):
    spec = importlib.util.spec_from_file_location("track1_controller_kinematics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import kinematics module: {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolves annotations through sys.modules while the module is
    # executing, so registration is required for dynamically loaded sources.
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


def ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.eye(4)
    matrix[[0, 0, 2, 2], [0, 2, 0, 2]] = (c, s, -s, c)
    return matrix


def transform_xyz_rpy(xyz, rpy, label: str) -> np.ndarray:
    """Build a homogeneous transform using URDF fixed-axis RPY convention."""

    translation = _vector3(xyz, f"{label}.xyz")
    angles = _vector3(rpy, f"{label}.rpy")
    transform = rz(float(angles[2])) @ ry(float(angles[1])) @ rx(float(angles[0]))
    transform[:3, 3] = translation
    return transform


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


def matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=float)
    horizontal = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    pitch = math.atan2(-float(rotation[2, 0]), horizontal)
    if horizontal > 1e-9:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    return np.asarray((roll, pitch, yaw), dtype=float)


def rotation_align_z_to(vector: np.ndarray) -> np.ndarray:
    """Return a rotation whose local +Z points along ``vector``."""

    # Copy: callers pass slices of the DH transforms.  Normalising a view in
    # place would corrupt the joint origins later in this same build.
    target = np.array(vector, dtype=float, copy=True)
    length = float(np.linalg.norm(target))
    if length < 1e-12:
        return np.eye(3)
    target /= length
    source = np.asarray((0.0, 0.0, 1.0))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(source, target))
    if sine < 1e-10:
        return np.eye(3) if cosine > 0.0 else np.diag((1.0, -1.0, -1.0))
    skew = np.asarray(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        )
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def number(value: float) -> str:
    if abs(value) < 5e-13:
        value = 0.0
    return f"{float(value):.15g}"


def origin(transform: np.ndarray, indent: str = "      ") -> str:
    xyz = " ".join(number(value) for value in transform[:3, 3])
    rpy = " ".join(number(value) for value in matrix_to_rpy(transform[:3, :3]))
    return f'{indent}<origin xyz="{xyz}" rpy="{rpy}"/>\n'


def material_block(name: str, rgba: str, indent: str = "    ") -> str:
    return (
        f'{indent}<material name="{name}">\n'
        f'{indent}  <color rgba="{rgba}"/>\n'
        f'{indent}</material>\n'
    )


def mesh_block(
    *,
    kind: str,
    filename: str,
    color: str | None = None,
    indent: str = "    ",
) -> str:
    """Return one link-local STEP-derived mesh visual/collision element."""

    return (
        f"{indent}<{kind}>\n"
        + f'{indent}  <geometry>\n'
        + f'{indent}    <mesh filename="{filename}" scale="1 1 1"/>\n'
        + f"{indent}  </geometry>\n"
        + (material_block(color, "0.18 0.42 0.78 1", indent + "  ") if color else "")
        + f"{indent}</{kind}>\n"
    )


def primitive_blocks(
    *,
    kind: str,
    radius: float,
    vector: np.ndarray | None = None,
    color: str | None = None,
    indent: str = "    ",
) -> str:
    blocks: list[str] = []
    if vector is not None and float(np.linalg.norm(vector)) > 1e-9:
        length = float(np.linalg.norm(vector))
        rotation = rotation_align_z_to(vector)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(vector, dtype=float) / 2.0
        blocks.append(
            f"{indent}<{kind}>\n"
            + origin(transform, indent + "  ")
            + f"{indent}  <geometry>\n"
            + f'{indent}    <cylinder radius="{number(radius)}" length="{number(length)}"/>\n'
            + f"{indent}  </geometry>\n"
            + (material_block(color, "0.18 0.42 0.78 1", indent + "  ") if color else "")
            + f"{indent}</{kind}>\n"
        )
    # End spheres make the cylinder a conservative capsule.  They overlap the
    # cylinder and are intentionally not a manufacturing CAD representation.
    for point in (np.zeros(3), np.asarray(vector if vector is not None else np.zeros(3))):
        transform = np.eye(4)
        transform[:3, 3] = point
        blocks.append(
            f"{indent}<{kind}>\n"
            + origin(transform, indent + "  ")
            + f"{indent}  <geometry>\n"
            + f'{indent}    <sphere radius="{number(radius)}"/>\n'
            + f"{indent}  </geometry>\n"
            + (material_block(color, "0.18 0.42 0.78 1", indent + "  ") if color else "")
            + f"{indent}</{kind}>\n"
        )
    return "".join(blocks)


def _matrix4(value, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{label} must be 4x4")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
        raise ValueError(f"{label} must be a homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError(f"{label} rotation is not right-handed")
    return matrix


def _vector3(value, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain three finite values")
    return vector


def _axis(value, label: str) -> np.ndarray:
    axis = _vector3(value, label)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError(f"{label} must be non-zero")
    return axis / norm


def build_gripper_assembly(assembly: dict) -> str:
    """Return links/joints for the provisional offline gripper assembly."""

    mounting = assembly["mounting"]
    mesh = assembly["mesh"]
    finger_joints = assembly["finger_joints"]
    tcp = assembly["tcp"]

    flange_link = str(mounting["flange_link"])
    body_link = str(mounting["body_link"])
    flange_from_body = _matrix4(
        mounting["flange_from_body"]["matrix"], "mounting.flange_from_body.matrix"
    )
    travel = float(finger_joints["limit_m"])
    default = float(finger_joints.get("default_position_m", 0.0))
    velocity = float(finger_joints["velocity_mps"])
    effort = float(finger_joints["effort_n"])
    if not all(math.isfinite(value) for value in (travel, default, velocity, effort)):
        raise ValueError("gripper joint parameters must be finite")
    if travel <= 0.0 or velocity <= 0.0 or effort <= 0.0:
        raise ValueError("gripper travel, velocity and effort must be positive")
    if abs(default) > travel + 1e-12:
        raise ValueError("gripper default position is outside its travel limit")
    if abs(default) > 1e-12:
        raise ValueError("the open/default gripper finger position must be zero")

    visual_dir = str(mesh["visual_dir"]).strip("/")
    collision_dir = str(mesh["collision_dir"]).strip("/")
    finger_specs = (
        ("left", "gripper_silicone"),
        ("right", "gripper_silicone"),
    )

    lines: list[str] = []
    lines.append(f'  <link name="{body_link}">\n')
    for kind, directory in (("visual", visual_dir), ("collision", collision_dir)):
        lines.append(
            mesh_block(
                kind=kind,
                filename=(
                    "package://estun_s3_60_g2_description/"
                    f"{directory}/{body_link}.stl"
                ),
                color="gripper_dark" if kind == "visual" else None,
                indent="    ",
            )
        )
    lines.append('  </link>\n')

    for side, color in finger_specs:
        spec = finger_joints[side]
        child_link = str(spec["child"])
        lines.append(f'  <link name="{child_link}">\n')
        for kind, directory in (("visual", visual_dir), ("collision", collision_dir)):
            lines.append(
                mesh_block(
                    kind=kind,
                    filename=(
                        "package://estun_s3_60_g2_description/"
                        f"{directory}/{child_link}.stl"
                    ),
                    color=color if kind == "visual" else None,
                    indent="    ",
                )
            )
        lines.append('  </link>\n')

    tcp_link = str(tcp["link"])
    lines.append(f'  <link name="{tcp_link}"/>\n')

    body_joint = 'gripper_body_fixed'
    lines.append(f'  <joint name="{body_joint}" type="fixed">\n')
    lines.append(origin(flange_from_body))
    lines.append(f'    <parent link="{flange_link}"/>\n')
    lines.append(f'    <child link="{body_link}"/>\n')
    lines.append('  </joint>\n')

    for side, _ in finger_specs:
        spec = finger_joints[side]
        joint_name = str(spec["name"])
        if str(spec["parent"]) != body_link:
            raise ValueError(f"{joint_name} must be attached to {body_link}")
        child_link = str(spec["child"])
        axis = _axis(spec["axis"], f"finger_joints.{side}.axis")
        if not np.allclose(np.abs(axis), (0.0, 1.0, 0.0), atol=1e-12):
            raise ValueError(f"{joint_name} axis must be aligned with local Y")
        lines.append(f'  <joint name="{joint_name}" type="prismatic">\n')
        lines.append('    <origin xyz="0 0 0" rpy="0 0 0"/>\n')
        lines.append(f'    <parent link="{body_link}"/>\n')
        lines.append(f'    <child link="{child_link}"/>\n')
        lines.append(f'    <axis xyz="{number(axis[0])} {number(axis[1])} {number(axis[2])}"/>\n')
        lines.append(
            f'    <limit lower="{number(-travel)}" upper="{number(default)}" '
            f'effort="{number(effort)}" velocity="{number(velocity)}"/>\n'
        )
        lines.append('  </joint>\n')

    tcp_parent = str(tcp.get("parent", body_link))
    if tcp_parent != body_link:
        raise ValueError(f"{tcp_link} parent must be {body_link} for this assembly")
    tcp_xyz = _vector3(tcp["xyz_m"], "tcp.xyz_m")
    tcp_rpy = _vector3(tcp["rpy_rad"], "tcp.rpy_rad")
    _axis(tcp["approach_axis"], "tcp.approach_axis")
    _axis(tcp["closure_axis"], "tcp.closure_axis")
    _axis(tcp["finger_height_axis"], "tcp.finger_height_axis")
    lines.append(f'  <joint name="grasp_tcp_fixed" type="fixed">\n')
    lines.append(
        '    <origin xyz="'
        + ' '.join(number(value) for value in tcp_xyz)
        + '" rpy="'
        + ' '.join(number(value) for value in tcp_rpy)
        + '"/>\n'
    )
    lines.append(f'    <parent link="{body_link}"/>\n')
    lines.append(f'    <child link="{tcp_link}"/>\n')
    lines.append('  </joint>\n')
    return ''.join(lines)


def build_urdf(
    kinematics, geometry: dict, camera: dict, gripper_assembly: dict | None = None
) -> str:
    dh = tuple(tuple(float(v) for v in row) for row in kinematics.CONTROLLER_DH)
    if len(dh) != 6:
        raise ValueError("CONTROLLER_DH must contain exactly six rows")
    constants = [dh_constant_transform(row) for row in dh]
    lower = np.asarray(kinematics.JOINT_LOW, dtype=float)
    upper = np.asarray(kinematics.JOINT_HIGH, dtype=float)
    if lower.shape != (6,) or upper.shape != (6,):
        raise ValueError("controller joint limits must contain six values")

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>\n')
    lines.append(
        '<!-- GENERATED FILE. Do not hand-edit. Source of truth: '
        'track1_grasp/kinematics.py CONTROLLER_DH. -->\n'
    )
    lines.append(
        '<!-- Collision primitives are STEP-derived conservative envelopes, '
        'not accepted manufacturing geometry. -->\n'
    )
    lines.append(
        '<robot name="estun_s3_60_g2" xmlns:xacro="http://www.ros.org/wiki/xacro">\n'
    )
    lines.append(material_block("estun_blue", "0.10 0.32 0.68 1"))
    lines.append(material_block("estun_dark", "0.16 0.17 0.19 1"))
    lines.append(material_block("d405_black", "0.02 0.02 0.02 1"))
    lines.append(material_block("gripper_dark", "0.12 0.14 0.16 1"))
    lines.append(material_block("gripper_silicone", "0.08 0.52 0.62 1"))

    mesh_representation = geometry.get("mesh_representation", {})
    use_step_meshes = bool(mesh_representation.get("enabled", False))
    visual_mesh_dir = str(
        mesh_representation.get("visual_dir", "mesh/link_local/visual")
    ).strip("/")
    collision_mesh_dir = str(
        mesh_representation.get("collision_dir", "mesh/link_local/collision")
    ).strip("/")

    base_radius = float(geometry["base_link"]["radius_m"])
    joint1_height = float(constants[0][2, 3])
    base_transform = np.eye(4)
    base_transform[2, 3] = joint1_height / 2.0
    lines.append('  <link name="base_link">\n')
    if use_step_meshes:
        lines.append(
            mesh_block(
                kind="visual",
                filename=(
                    "package://estun_s3_60_g2_description/"
                    f"{visual_mesh_dir}/base_link.stl"
                ),
                color="estun_dark",
                indent="    ",
            )
        )
        lines.append(
            mesh_block(
                kind="collision",
                filename=(
                    "package://estun_s3_60_g2_description/"
                    f"{collision_mesh_dir}/base_link.stl"
                ),
                indent="    ",
            )
        )
    else:
        lines.append('    <visual>\n')
        lines.append(origin(base_transform, "      "))
        lines.append('      <geometry>\n')
        lines.append(
            f'        <cylinder radius="{number(base_radius)}" '
            f'length="{number(joint1_height)}"/>\n'
        )
        lines.append('      </geometry>\n')
        lines.append(material_block("estun_dark", "0.16 0.17 0.19 1", "      "))
        lines.append('    </visual>\n')
        lines.append('    <collision>\n')
        lines.append(origin(base_transform, "      "))
        lines.append('      <geometry>\n')
        lines.append(
            f'        <cylinder radius="{number(base_radius)}" '
            f'length="{number(joint1_height)}"/>\n'
        )
        lines.append('      </geometry>\n')
        lines.append('    </collision>\n')
    lines.append('  </link>\n')

    for index in range(1, 7):
        link_name = f"link{index}"
        radius = float(geometry[link_name]["radius_m"])
        vector = constants[index][:3, 3] if index < 6 else np.zeros(3)
        lines.append(f'  <link name="{link_name}">\n')
        if use_step_meshes:
            lines.append(
                mesh_block(
                    kind="visual",
                    filename=(
                        "package://estun_s3_60_g2_description/"
                        f"{visual_mesh_dir}/{link_name}.stl"
                    ),
                    color="estun_blue",
                    indent="    ",
                )
            )
            lines.append(
                mesh_block(
                    kind="collision",
                    filename=(
                        "package://estun_s3_60_g2_description/"
                        f"{collision_mesh_dir}/{link_name}.stl"
                    ),
                    indent="    ",
                )
            )
        else:
            lines.append(
                primitive_blocks(
                    kind="visual",
                    radius=radius,
                    vector=vector,
                    color="estun_blue",
                    indent="    ",
                )
            )
            lines.append(
                primitive_blocks(
                    kind="collision",
                    radius=radius,
                    vector=vector,
                    indent="    ",
                )
            )
        lines.append('  </link>\n')

    for index in range(6):
        line = (
            f'  <joint name="joint{index + 1}" type="revolute">\n'
            + origin(constants[index])
            + f'    <parent link="{LINK_NAMES[index]}"/>\n'
            + f'    <child link="link{index + 1}"/>\n'
            + '    <axis xyz="0 0 1"/>\n'
            + (
                f'    <limit lower="{number(lower[index])}" '
                f'upper="{number(upper[index])}" '
                f'effort="{number(JOINT_EFFORT[index])}" '
                f'velocity="{number(JOINT_VELOCITY[index])}"/>\n'
            )
            + '  </joint>\n'
        )
        lines.append(line)

    lines.append('  <link name="flange_link"/>\n')
    lines.append('  <joint name="flange_fixed" type="fixed">\n')
    lines.append('    <origin xyz="0 0 0" rpy="0 0 0"/>\n')
    lines.append('    <parent link="link6"/>\n')
    lines.append('    <child link="flange_link"/>\n')
    lines.append('  </joint>\n')

    if gripper_assembly is not None:
        lines.append(build_gripper_assembly(gripper_assembly))

    matrix = _matrix4(camera["matrix"], "camera.matrix")
    camera_geometry = geometry["d405_camera_link"]
    if camera_geometry.get("frame_semantics") != "camera_color_optical_frame":
        raise ValueError(
            "d405_camera_link must explicitly represent camera_color_optical_frame"
        )
    visual = camera_geometry["visual"]
    collision = camera_geometry["collision"]
    visual_mesh = PACKAGE_ROOT / str(visual["mesh_path"])
    if not visual_mesh.is_file():
        raise FileNotFoundError(visual_mesh)
    expected_mesh_sha = str(camera_geometry["source"]["packaged_mesh_sha256"])
    if sha256(visual_mesh) != expected_mesh_sha:
        raise ValueError("D405 mesh SHA-256 does not match geometry authority")
    license_path = PACKAGE_ROOT / str(camera_geometry["source"]["packaged_license"])
    if not license_path.is_file():
        raise FileNotFoundError(license_path)
    if sha256(license_path) != str(
        camera_geometry["source"]["packaged_license_sha256"]
    ):
        raise ValueError("D405 license SHA-256 does not match geometry authority")
    visual_transform = transform_xyz_rpy(
        visual["xyz_m"], visual["rpy_rad"], "camera.visual"
    )
    collision_transform = transform_xyz_rpy(
        collision["xyz_m"], collision["rpy_rad"], "camera.collision"
    )
    if collision.get("shape") != "box":
        raise ValueError("D405 collision geometry must be the official box")
    camera_size = [float(v) for v in collision["box_size_m"]]
    if len(camera_size) != 3 or any(value <= 0.0 for value in camera_size):
        raise ValueError("D405 collision box dimensions must be positive")
    mesh_scale = [float(v) for v in visual["mesh_scale"]]
    if len(mesh_scale) != 3 or any(value <= 0.0 for value in mesh_scale):
        raise ValueError("D405 mesh scale must contain three positive values")

    lines.append('  <link name="d405_camera_link">\n')
    lines.append('    <visual>\n')
    lines.append(origin(visual_transform))
    lines.append('      <geometry>\n')
    lines.append(
        '        <mesh filename="package://estun_s3_60_g2_description/'
        f'{str(visual["mesh_path"]).strip("/")}" scale="'
        + " ".join(number(value) for value in mesh_scale)
        + '"/>\n'
    )
    lines.append('      </geometry>\n')
    lines.append(material_block("d405_black", "0.02 0.02 0.02 1", "      "))
    lines.append('    </visual>\n')
    lines.append('    <collision>\n')
    lines.append(origin(collision_transform))
    lines.append('      <geometry>\n')
    lines.append(
        f'        <box size="{number(camera_size[0])} {number(camera_size[1])} '
        f'{number(camera_size[2])}"/>\n'
    )
    lines.append('      </geometry>\n')
    lines.append('    </collision>\n')
    lines.append('  </link>\n')
    lines.append('  <joint name="d405_camera_fixed" type="fixed">\n')
    lines.append(origin(matrix))
    lines.append('    <parent link="flange_link"/>\n')
    lines.append('    <child link="d405_camera_link"/>\n')
    lines.append('  </joint>\n')
    lines.append('</robot>\n')
    return "".join(lines)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinematics", type=Path, default=DEFAULT_KINEMATICS)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument(
        "--gripper-assembly", type=Path, default=DEFAULT_GRIPPER_ASSEMBLY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    kinematics = import_kinematics(args.kinematics)
    geometry = yaml.safe_load(args.geometry.read_text(encoding="utf-8"))
    camera = yaml.safe_load(args.camera.read_text(encoding="utf-8"))
    gripper_assembly = yaml.safe_load(
        args.gripper_assembly.read_text(encoding="utf-8")
    )
    use_step_meshes = bool(geometry.get("mesh_representation", {}).get("enabled", False))
    urdf = build_urdf(kinematics, geometry, camera, gripper_assembly)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(urdf, encoding="utf-8")
    xacro_output = args.output.with_suffix(args.output.suffix + ".xacro")
    xacro_output.write_text(urdf, encoding="utf-8")

    report = {
        "geometry_representation": (
            "step_component_link_local_meshes" if use_step_meshes else "conservative_primitives"
        ),
        "generated_urdf": str(args.output),
        "generated_urdf_sha256": sha256(args.output),
        "generated_xacro": str(xacro_output),
        "generated_xacro_sha256": sha256(xacro_output),
        "controller_kinematics": str(args.kinematics),
        "controller_kinematics_sha256": sha256(args.kinematics),
        "geometry_estimates": str(args.geometry),
        "geometry_estimates_sha256": sha256(args.geometry),
        "camera_extrinsics": str(args.camera),
        "camera_extrinsics_sha256": sha256(args.camera),
        "d405_camera_geometry": {
            "frame_semantics": geometry["d405_camera_link"]["frame_semantics"],
            "status": geometry["d405_camera_link"]["status"],
            "source": geometry["d405_camera_link"]["source"],
            "visual": geometry["d405_camera_link"]["visual"],
            "collision": geometry["d405_camera_link"]["collision"],
        },
        "controller_dh": [list(row) for row in kinematics.CONTROLLER_DH],
        "joint_low_rad": list(map(float, kinematics.JOINT_LOW)),
        "joint_high_rad": list(map(float, kinematics.JOINT_HIGH)),
    }
    gripper_source = PACKAGE_ROOT / str(gripper_assembly["source"]["step_path"])
    gripper_mesh_report = PACKAGE_ROOT / "mesh" / "gripper" / "generation_report.json"
    gripper_component_report = (
        PACKAGE_ROOT / str(gripper_assembly["source"]["component_report"])
    )
    for required_path in (gripper_source, gripper_mesh_report, gripper_component_report):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    if sha256(gripper_source) != str(gripper_assembly["source"]["step_sha256"]):
        raise ValueError("gripper STEP SHA-256 does not match assembly authority")
    if sha256(gripper_mesh_report) != str(
        gripper_assembly["source"].get(
            "mesh_generation_report_sha256",
            sha256(gripper_mesh_report),
        )
    ):
        raise ValueError("gripper mesh report SHA-256 does not match assembly authority")
    report.update(
        {
            "gripper_assembly": str(args.gripper_assembly),
            "gripper_assembly_sha256": sha256(args.gripper_assembly),
            "gripper_assembly_status": str(gripper_assembly["status"]),
            "gripper_model": str(gripper_assembly["model"]["name"]),
            "flange_from_body_matrix": [
                [float(value) for value in row]
                for row in gripper_assembly["mounting"]["flange_from_body"]["matrix"]
            ],
            "gripper_clocking": gripper_assembly["mounting"]["clocking"],
            "gripper_source_step": str(gripper_source),
            "gripper_source_step_sha256": sha256(gripper_source),
            "gripper_component_report": str(gripper_component_report),
            "gripper_component_report_sha256": sha256(gripper_component_report),
            "gripper_mesh_generation_report": str(gripper_mesh_report),
            "gripper_mesh_generation_report_sha256": sha256(gripper_mesh_report),
            "grasp_tcp": gripper_assembly["tcp"],
            "finger_joints": gripper_assembly["finger_joints"],
        }
    )
    if use_step_meshes:
        mesh_report = PACKAGE_ROOT / "mesh" / "link_local" / "generation_report.json"
        report["link_mesh_generation_report"] = str(mesh_report)
        report["link_mesh_generation_report_sha256"] = sha256(mesh_report)
    report_path = args.output.parent / "generation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
