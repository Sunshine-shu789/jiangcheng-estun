from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from track1_grasp.collision import GEOMETRY_SOURCE_FIELD_MEASURED
from track1_grasp.tools.validate_task_b_box_clearance_measurement import (
    box_clearance_dict,
    validate_measurement,
)


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = (
    ROOT
    / "config"
    / "track1_task_b_box_clearance_measurement_template.yaml"
)


def _template_payload() -> dict:
    payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_evidence(tmp_path: Path, payload: dict) -> None:
    evidence_paths: list[list[str]] = [payload["evidence"]["overview_photos"]]
    evidence_paths.extend(payload["evidence"]["corner_photos"].values())
    evidence_paths.extend(payload["evidence"]["boss_photos"].values())
    evidence_paths.append(payload["evidence"]["gripper_geometry_sources"])
    for index, paths in enumerate(evidence_paths):
        evidence_file = tmp_path / f"evidence_{index}.jpg"
        evidence_file.write_bytes(b"offline-evidence-fixture")
        paths.append(evidence_file.name)


def _complete_payload(tmp_path: Path) -> dict:
    payload = _template_payload()
    payload["status"] = "reviewed"
    payload["metadata"].update(
        {
            "measured_at": "2026-10-25T23:00:00+08:00",
            "measured_by": "operator",
            "reviewed_by": "safety reviewer",
            "measurement_location": "competition bin station",
            "controller_base_frame": "documented_robot_base",
            "instrument_id": "laser-01 / caliper-01",
            "measurement_method": "static laser spans plus corner survey",
            "uncertainty_m": 0.004,
            "yaw_uncertainty_rad": 0.01,
            "bin_identifier": "task-b-bin-01",
        }
    )
    payload["box"].update(
        {
            "center_xy_m": [0.10, 0.20],
            "yaw_rad": 0.05,
            "interior_size_m": [1.20, 1.00],
        }
    )
    payload["clearances"].update({"wall_m": 0.010, "obstacle_m": 0.015})
    payload["gripper_footprint"].update(
        {
            "source_description": "measured maximum assembly projection",
            "flange_xy_m": [
                [-0.059, -0.0315],
                [0.059, -0.0315],
                [0.059, 0.0315],
                [-0.059, 0.0315],
            ],
        }
    )
    corner_shapes = (
        ((0.46, 0.36), (0.56, 0.36), (0.56, 0.46), (0.46, 0.46)),
        ((0.36, 0.36), (0.56, 0.36), (0.56, 0.40), (0.36, 0.40)),
        ((-0.56, 0.36), (-0.46, 0.36), (-0.46, 0.46), (-0.56, 0.46)),
        ((-0.56, 0.36), (-0.36, 0.36), (-0.36, 0.40), (-0.56, 0.40)),
        ((0.46, -0.46), (0.56, -0.46), (0.56, -0.36), (0.46, -0.36)),
        ((0.36, -0.40), (0.56, -0.40), (0.56, -0.36), (0.36, -0.36)),
        ((-0.56, -0.46), (-0.46, -0.46), (-0.46, -0.36), (-0.56, -0.36)),
        ((-0.56, -0.40), (-0.36, -0.40), (-0.36, -0.36), (-0.56, -0.36)),
    )
    for obstacle, vertices in zip(
        payload["obstacles_m"][:8],
        corner_shapes,
        strict=True,
    ):
        obstacle["vertices_m"] = [list(point) for point in vertices]
        obstacle["notes"] = "static field sketch polygon"
    payload["obstacles_m"][8]["vertices_m"] = [
        [-0.12, 0.43],
        [0.12, 0.43],
        [0.12, 0.50],
        [-0.12, 0.50],
    ]
    payload["obstacles_m"][9]["vertices_m"] = [
        [-0.12, -0.50],
        [0.12, -0.50],
        [0.12, -0.43],
        [-0.12, -0.43],
    ]
    for obstacle in payload["obstacles_m"]:
        obstacle["notes"] = "included in the reviewed obstacle inventory"
    _write_evidence(tmp_path, payload)
    payload["review"].update(
        {
            "measurement_complete": True,
            "dimensions_rechecked": True,
            "all_obstacles_inventoried": True,
            "gripper_footprint_reviewed": True,
            "safety_approved": True,
            "review_notes": "dimensions and every convex piece rechecked",
        }
    )
    return payload


def test_draft_template_is_rejected_with_actionable_errors(tmp_path):
    with pytest.raises(ValueError) as caught:
        validate_measurement(
            _template_payload(),
            source_path=tmp_path / "draft.yaml",
        )

    message = str(caught.value)
    assert "status must be 'reviewed'" in message
    assert "evidence.overview_photos" in message
    assert "review.safety_approved" in message


def test_complete_reviewed_measurement_emits_field_measured_geometry(tmp_path):
    payload = _complete_payload(tmp_path)
    validated = validate_measurement(
        payload,
        source_path=tmp_path / "reviewed.yaml",
    )

    block = box_clearance_dict(validated.config)
    assert validated.evidence_count == 8
    assert block["required"] is True
    assert block["geometry_validated"] is True
    assert block["geometry_source"] == GEOMETRY_SOURCE_FIELD_MEASURED
    assert len(block["obstacles_m"]) >= 10
    assert block["center_xy_m"] == pytest.approx([0.10, 0.20])


def test_missing_l_corner_inventory_is_rejected(tmp_path):
    payload = _complete_payload(tmp_path)
    payload["obstacles_m"] = payload["obstacles_m"][:-3]

    with pytest.raises(ValueError, match="at least eight convex lower_corner"):
        validate_measurement(
            payload,
            source_path=tmp_path / "missing_corner.yaml",
        )


def test_clearance_must_not_be_smaller_than_measurement_uncertainty(tmp_path):
    payload = _complete_payload(tmp_path)
    payload["clearances"]["wall_m"] = 0.001

    with pytest.raises(ValueError, match="wall_m must be at least"):
        validate_measurement(
            payload,
            source_path=tmp_path / "unsafe_margin.yaml",
        )


def test_missing_evidence_file_is_rejected(tmp_path):
    payload = deepcopy(_complete_payload(tmp_path))
    missing = tmp_path / "evidence_0.jpg"
    missing.unlink()

    with pytest.raises(ValueError, match="not a readable file"):
        validate_measurement(
            payload,
            source_path=tmp_path / "missing_evidence.yaml",
        )
