#!/usr/bin/env python3
"""Write a deterministic provenance manifest for the arm and gripper model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_OUTPUT = PACKAGE_ROOT / "validation" / "model_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _same_inode(first: Path, second: Path) -> bool:
    return (first.stat().st_dev, first.stat().st_ino) == (
        second.stat().st_dev,
        second.stat().st_ino,
    )


def build_manifest() -> dict:
    arm_step = PACKAGE_ROOT / "source" / "S3-60-G2.STEP"
    original_arm_step = Path("/media/sunshine/新加卷/S3-60-G2.STEP")
    gripper_step = PACKAGE_ROOT / "source" / "FAE2MR86M-D.STEP"
    dimensions_path = PACKAGE_ROOT / "config" / "fae2mr86m_ms42dc_dimensions.json"
    dimensions = json.loads(dimensions_path.read_text(encoding="utf-8"))
    original_gripper_step = Path(dimensions["source"]["step_path"])
    kinematics = PROJECT_ROOT / "app" / "track1_grasp" / "track1_grasp" / "kinematics.py"
    urdf = PACKAGE_ROOT / "urdf" / "estun_s3_60_g2.urdf"
    camera = PACKAGE_ROOT / "config" / "camera_to_flange.d405_calibrated_20261025.yaml"
    geometry_estimates = PACKAGE_ROOT / "config" / "geometry_estimates.yaml"
    camera_mesh = PACKAGE_ROOT / "mesh" / "camera" / "d405.stl"
    camera_license = (
        PACKAGE_ROOT / "mesh" / "camera" / "LICENSE.realsense-ros.txt"
    )
    camera_provenance = PACKAGE_ROOT / "mesh" / "camera" / "README.md"
    arm_mesh_report = PACKAGE_ROOT / "mesh" / "link_local" / "generation_report.json"
    gripper_assembly = PACKAGE_ROOT / "config" / "gripper_assembly.fae2mr86m_ms42dc.yaml"
    gripper_mesh_report = PACKAGE_ROOT / "mesh" / "gripper" / "generation_report.json"
    gripper_component_report = (
        PACKAGE_ROOT / "mesh" / "gripper" / "components" / "step_report.json"
    )
    jaw_profile = PACKAGE_ROOT / "config" / "fae2mr86m_ms42dc_jaw_profile.json"
    assembly_preview = PACKAGE_ROOT / "validation" / "step_assembly_q0_preview.png"
    offline_smoke = (
        PROJECT_ROOT
        / "robot_description"
        / "estun_s3_60_g2_moveit_config"
        / "validation"
        / "moveit_offline_smoke.json"
    )
    srdf = (
        PROJECT_ROOT
        / "robot_description"
        / "estun_s3_60_g2_moveit_config"
        / "config"
        / "estun_s3_60_g2.srdf"
    )

    return {
        "schema_version": 1,
        "status": "offline_arm_and_gripper_geometry_validated",
        "safety_scope": "offline_only_no_robot_motion_no_gripper_command",
        "source_step": {
            **file_record(arm_step),
            "original_path": str(original_arm_step),
            "original_is_hardlink_to_packaged_copy": _same_inode(
                arm_step, original_arm_step
            ),
        },
        "gripper_source_step": {
            **file_record(gripper_step),
            "original_path": str(original_gripper_step),
            "original_is_hardlink_to_packaged_copy": _same_inode(
                gripper_step, original_gripper_step
            ),
        },
        "authorities": {
            "controller_kinematics": file_record(kinematics),
            "generated_urdf": file_record(urdf),
            "validated_d405_camera_to_flange": file_record(camera),
            "geometry_estimates": file_record(geometry_estimates),
            "d405_official_mesh": file_record(camera_mesh),
            "d405_license": file_record(camera_license),
            "d405_provenance": file_record(camera_provenance),
            "arm_mesh_generation_report": file_record(arm_mesh_report),
            "gripper_assembly": file_record(gripper_assembly),
            "gripper_dimensions": file_record(dimensions_path),
            "gripper_jaw_profile": file_record(jaw_profile),
            "gripper_component_report": file_record(gripper_component_report),
            "gripper_mesh_generation_report": file_record(gripper_mesh_report),
            "moveit_semantic_description": file_record(srdf),
        },
        "step_to_controller_base": {
            "matrix": [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.004123711],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "derivation": "official STEP q=0 assembly aligned to controller joint axes",
        },
        "step_component_to_link": {
            "base_link": "NAUO7",
            "link1": "NAUO3",
            "link2": "NAUO1",
            "link3": "NAUO2",
            "link4": "NAUO4",
            "link5": "NAUO5",
            "link6": "NAUO6",
        },
        "gripper_assembly": {
            "status": "provisional_offline",
            "flange_from_body_matrix": [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.091751924],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "installed_clocking_status": "unmeasured",
            "shim_thickness_status": "unmeasured",
            "finger_joint_model": "independent prismatic rigid-CAD approximation",
            "finger_travel_m": 0.0362745,
            "tcp_link": "grasp_tcp",
            "tcp_xyz_from_body_m": [0.050, 0.0, 0.0],
            "tcp_approach_axis": [1.0, 0.0, 0.0],
            "tcp_closure_axis": [0.0, 1.0, 0.0],
        },
        "d405_camera_geometry": {
            "frame_semantics": "camera_color_optical_frame",
            "status": "official_d405_case_geometry_rebased_to_optical_frame",
            "upstream_repository": "https://github.com/IntelRealSense/realsense-ros",
            "upstream_commit": "f48ba88c8fc512711247c46af58d3f1f1d12745a",
            "upstream_model": "Intel RealSense D405 aluminum peripheral evaluation case",
            "license": "Apache-2.0",
            "nominal_body_dimensions_m": [0.023, 0.042, 0.042],
            "visual_mesh_scale": [0.001, 0.001, 0.001],
            "visual_xyz_in_optical_m": [0.009, 0.0, 0.0038],
            "visual_rpy_in_optical_rad": [0.0, 0.0, 3.141592653589793],
            "collision_shape": "box",
            "collision_xyz_in_optical_m": [0.009, 0.0, -0.0078],
            "collision_rpy_in_optical_rad": [
                0.0,
                -1.5707963267948966,
                1.5707963267948966,
            ],
            "collision_box_m": [0.023, 0.042, 0.042],
            "not_represented": [
                "USB plug and cable",
                "custom 3D-printed mounting bracket",
                "manufacturing tolerance and protective covers",
            ],
        },
        "mesh_generation": {
            "arm": {
                "visual_voxel_mm": 1.0,
                "collision_voxel_mm": 5.0,
            },
            "gripper": {
                "visual_voxel_mm": 0.8,
                "collision_voxel_mm": 2.0,
            },
            "purpose": "offline visualization and collision checking",
        },
        "validated_offline": {
            "controller_dh_fk_match": True,
            "kdl_ik_round_trip": True,
            "moveit_collision_service": True,
            "arm_and_tool_collision_mesh_loaded": True,
            "d405_official_geometry_loaded": True,
            "gripper_open_and_rigid_close_envelope_checked": True,
            "ompl_rrtconnect_trajectory": True,
            "trajectory_execution_disabled": True,
        },
        "validation_artifacts": {
            "step_assembly_q0_preview": file_record(assembly_preview),
            "moveit_offline_smoke_report": file_record(offline_smoke),
        },
        "not_represented": [
            "installed gripper clocking and shim thickness",
            "flexible four-bar finger linkage and silicone deformation",
            "motor-command-to-jaw-width and clamping-force calibration",
            "tool payload, center of mass and controller tool-frame convention",
            "GraspNet real-robot grasp labels and success labels",
            "manufacturing acceptance geometry",
            "D405 USB plug, cable and custom mounting bracket",
        ],
        "caveats": [
            "Voxel-decimated meshes are not manufacturing acceptance geometry.",
            "The prismatic finger joints are an offline collision approximation; full rigid closure is not a physical collision state.",
            "The controller all-zero pose is not accepted as a planning seed because the STEP assembly overlaps around base_link, link1 and link3.",
            "Successful offline planning does not authorize physical robot motion or a gripper command.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
