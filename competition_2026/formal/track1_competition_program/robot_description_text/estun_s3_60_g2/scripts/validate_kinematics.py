#!/usr/bin/env python3
"""Validate the generated URDF against the controller's authoritative DH model.

The check deliberately parses the URDF as an external artifact and builds a KDL
chain using the same mapping as ``kdl_parser``.  It does not import helper code
from the generator, so a shared programming mistake cannot make both sides agree.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_URDF = PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf"
DEFAULT_XACRO = PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf.xacro"
DEFAULT_KINEMATICS = (
    DATA_ROOT / "app" / "track1_grasp" / "track1_grasp" / "kinematics.py"
)
DEFAULT_CAMERA = (
    PACKAGE_ROOT / "config" / "camera_to_flange.d405_calibrated_20261025.yaml"
)
POSITION_TOLERANCE_M = 1e-8
ROTATION_TOLERANCE_RAD = 1e-8


def import_kinematics(path: Path):
    spec = importlib.util.spec_from_file_location("track1_validation_kinematics", path)
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


def _numbers(value: str | None, *, default: Sequence[float]) -> tuple[float, ...]:
    if value is None:
        return tuple(float(v) for v in default)
    values = tuple(float(part) for part in value.split())
    if len(values) != 3:
        raise ValueError(f"expected three numbers, got {value!r}")
    return values


def _joint_origin(joint: ET.Element):
    import PyKDL

    origin = joint.find("origin")
    xyz = _numbers(origin.get("xyz") if origin is not None else None, default=(0, 0, 0))
    rpy = _numbers(origin.get("rpy") if origin is not None else None, default=(0, 0, 0))
    return PyKDL.Frame(
        PyKDL.Rotation.RPY(*rpy),
        PyKDL.Vector(*xyz),
    )


def build_kdl_chain(urdf_path: Path, *, base_link: str, tip_link: str):
    """Build a KDL chain exactly as ros/kdl_parser maps URDF origins/axes."""

    import PyKDL

    root = ET.parse(urdf_path).getroot()
    joints_by_child: dict[str, ET.Element] = {}
    links: set[str] = set()
    for link in root.findall("link"):
        name = link.get("name")
        if name:
            links.add(name)
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {joint.get('name')} is missing parent or child")
        parent_link = parent.get("link")
        child_link = child.get("link")
        if not parent_link or not child_link:
            raise ValueError(f"joint {joint.get('name')} has an empty parent/child")
        if parent_link not in links or child_link not in links:
            raise ValueError(
                f"joint {joint.get('name')} references an undeclared link"
            )
        if child_link in joints_by_child:
            raise ValueError(f"link {child_link} has more than one parent joint")
        joints_by_child[child_link] = joint

    path: list[ET.Element] = []
    current = tip_link
    seen: set[str] = set()
    while current != base_link:
        if current in seen:
            raise ValueError("URDF contains a kinematic loop")
        seen.add(current)
        if current not in joints_by_child:
            raise ValueError(f"no path from {base_link!r} to {tip_link!r}")
        joint = joints_by_child[current]
        path.append(joint)
        current = joint.find("parent").get("link")  # type: ignore[union-attr]
    path.reverse()

    chain = PyKDL.Chain()
    active_joint_names: list[str] = []
    for joint in path:
        name = joint.get("name", "")
        kind = joint.get("type", "")
        parent_frame_joint = _joint_origin(joint)
        if kind == "fixed":
            kdl_joint = PyKDL.Joint(name, PyKDL.Joint.Fixed)
        elif kind == "revolute":
            axis_element = joint.find("axis")
            axis_values = _numbers(
                axis_element.get("xyz") if axis_element is not None else None,
                default=(0, 0, 1),
            )
            axis = np.asarray(axis_values, dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm < 1e-12:
                raise ValueError(f"joint {name} has a zero axis")
            axis /= norm
            rotated_axis = parent_frame_joint.M * PyKDL.Vector(*axis)
            kdl_joint = PyKDL.Joint(
                name,
                parent_frame_joint.p,
                rotated_axis,
                PyKDL.Joint.RotAxis,
            )
            active_joint_names.append(name)
        else:
            raise ValueError(f"unsupported joint type {kind!r} on {name!r}")
        chain.addSegment(
            PyKDL.Segment(name, kdl_joint, parent_frame_joint)
        )
    return chain, tuple(active_joint_names)


def kdl_frame_matrix(frame) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(
        [[frame.M[row, column] for column in range(3)] for row in range(3)],
        dtype=float,
    )
    matrix[:3, 3] = np.asarray([frame.p[index] for index in range(3)], dtype=float)
    return matrix


def urdf_fk(urdf_path: Path, q_rad: Sequence[float]) -> np.ndarray:
    import PyKDL

    chain, names = build_kdl_chain(
        urdf_path, base_link="base_link", tip_link="flange_link"
    )
    if len(names) != 6:
        raise ValueError(f"expected six revolute joints, got {len(names)}: {names}")
    q = np.asarray(q_rad, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("q_rad must be a finite six-element vector")
    joint_array = PyKDL.JntArray(6)
    for index, value in enumerate(q):
        joint_array[index] = float(value)
    frame = PyKDL.Frame()
    solver = PyKDL.ChainFkSolverPos_recursive(chain)
    status = solver.JntToCart(joint_array, frame)
    if status < 0:
        raise RuntimeError(f"KDL forward kinematics failed with status {status}")
    return kdl_frame_matrix(frame)


def rotation_error(current: np.ndarray, target: np.ndarray) -> float:
    relative = np.asarray(current, dtype=float)[:3, :3].T @ np.asarray(
        target, dtype=float
    )[:3, :3]
    sine_vector = np.asarray(
        (
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ),
        dtype=float,
    )
    # atan2 of the skew and symmetric parts remains accurate near identity;
    # acos((trace-1)/2) has a ~3e-8 rad floating-point noise floor there.
    return float(np.arctan2(np.linalg.norm(sine_vector) / 2.0, (np.trace(relative) - 1.0) / 2.0))


def compare_fk(urdf_path: Path, kinematics, q_rad: Sequence[float]) -> dict[str, float]:
    expected = np.asarray(kinematics.controller_fk(q_rad), dtype=float)
    actual = urdf_fk(urdf_path, q_rad)
    return {
        "position_error_m": float(
            np.linalg.norm(actual[:3, 3] - expected[:3, 3])
        ),
        "rotation_error_rad": rotation_error(actual, expected),
    }


def validate_joint_limits(urdf_path: Path, kinematics) -> dict[str, float]:
    root = ET.parse(urdf_path).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    lower = np.asarray(kinematics.JOINT_LOW, dtype=float)
    upper = np.asarray(kinematics.JOINT_HIGH, dtype=float)
    max_lower_error = 0.0
    max_upper_error = 0.0
    for index in range(6):
        joint = joints[f"joint{index + 1}"]
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"joint{index + 1} has no limit element")
        max_lower_error = max(
            max_lower_error, abs(float(limit.get("lower")) - lower[index])
        )
        max_upper_error = max(
            max_upper_error, abs(float(limit.get("upper")) - upper[index])
        )
    return {
        "max_lower_error_rad": max_lower_error,
        "max_upper_error_rad": max_upper_error,
    }


def validate_camera(urdf_path: Path, camera_path: Path) -> dict[str, float]:
    root = ET.parse(urdf_path).getroot()
    joint = next(
        (
            item
            for item in root.findall("joint")
            if item.get("name") == "d405_camera_fixed"
        ),
        None,
    )
    if joint is None:
        raise ValueError("d405_camera_fixed joint is missing")
    parent = joint.find("parent")
    child = joint.find("child")
    if parent is None or child is None:
        raise ValueError("d405_camera_fixed is missing parent or child")
    if parent.get("link") != "flange_link":
        raise ValueError("D405 must be attached to flange_link")
    if child.get("link") != "d405_camera_link":
        raise ValueError("D405 child link has the wrong name")
    frame = _joint_origin(joint)
    actual = kdl_frame_matrix(frame)
    payload = yaml.safe_load(camera_path.read_text(encoding="utf-8"))
    expected = np.asarray(payload["matrix"], dtype=float)
    if expected.shape != (4, 4):
        raise ValueError("camera matrix must be 4x4")
    return {
        "position_error_m": float(
            np.linalg.norm(actual[:3, 3] - expected[:3, 3])
        ),
        "rotation_error_rad": rotation_error(actual, expected),
    }


def validate(
    *,
    urdf_path: Path,
    kinematics_path: Path,
    camera_path: Path,
    samples: int = 256,
) -> dict[str, object]:
    kinematics = import_kinematics(kinematics_path)
    if samples < 1:
        raise ValueError("samples must be positive")

    limits = validate_joint_limits(urdf_path, kinematics)
    camera = validate_camera(urdf_path, camera_path)
    known_q = np.radians([90.0, 0.0, 0.0, 0.0, 90.0, 0.0])
    known = compare_fk(urdf_path, kinematics, known_q)
    known_position_expected = np.asarray([-0.106964, 0.123500, 0.751726])
    expected_known = np.asarray(kinematics.controller_fk(known_q), dtype=float)
    known_reference_error = float(
        np.linalg.norm(expected_known[:3, 3] - known_position_expected)
    )

    rng = np.random.default_rng(20261029)
    lower = np.asarray(kinematics.JOINT_LOW, dtype=float)
    upper = np.asarray(kinematics.JOINT_HIGH, dtype=float)
    max_position = 0.0
    max_rotation = 0.0
    for _ in range(samples):
        q = rng.uniform(lower, upper)
        result = compare_fk(urdf_path, kinematics, q)
        max_position = max(max_position, result["position_error_m"])
        max_rotation = max(max_rotation, result["rotation_error_rad"])

    tolerances = {
        "position_m": POSITION_TOLERANCE_M,
        "rotation_rad": ROTATION_TOLERANCE_RAD,
    }
    failures: list[str] = []
    for name, value in (
        ("joint_lower", limits["max_lower_error_rad"]),
        ("joint_upper", limits["max_upper_error_rad"]),
        ("camera_position", camera["position_error_m"]),
        ("camera_rotation", camera["rotation_error_rad"]),
        ("known_position", known["position_error_m"]),
        ("known_rotation", known["rotation_error_rad"]),
        ("known_reference", known_reference_error),
        ("random_position", max_position),
        ("random_rotation", max_rotation),
    ):
        tolerance = (
            tolerances["rotation_rad"]
            if name.endswith("rotation")
            else tolerances["position_m"]
        )
        if value > tolerance:
            failures.append(f"{name}={value:.3e} exceeds {tolerance:.1e}")

    chain, joint_names = build_kdl_chain(
        urdf_path, base_link="base_link", tip_link="flange_link"
    )
    return {
        "status": "pass" if not failures else "fail",
        "urdf": str(urdf_path),
        "kinematics": str(kinematics_path),
        "camera": str(camera_path),
        "kdl_chain_joints": list(joint_names),
        "kdl_joint_count": int(chain.getNrOfJoints()),
        "random_samples": int(samples),
        "tolerances": tolerances,
        "joint_limits": limits,
        "camera_extrinsics": camera,
        "known_configuration": {
            "q_deg": [90.0, 0.0, 0.0, 0.0, 90.0, 0.0],
            **known,
            "expected_position_m": known_position_expected.tolist(),
            "reference_position_error_m": known_reference_error,
        },
        "random_fk": {
            "max_position_error_m": max_position,
            "max_rotation_error_rad": max_rotation,
        },
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--kinematics", type=Path, default=DEFAULT_KINEMATICS)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate(
        urdf_path=args.urdf,
        kinematics_path=args.kinematics,
        camera_path=args.camera,
        samples=args.samples,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
