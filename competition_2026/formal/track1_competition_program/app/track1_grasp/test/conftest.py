"""Shared test guards for the intentionally fail-closed C/D profiles.

The four official C/D YAML files require onsite box measurements.  Unit tests
must not pretend those measurements exist in the real profile.  For tests that
exercise planning or perception logic, this fixture builds an explicitly
offline configuration from the same YAML while replacing only the actuator
backend and the missing box-clearance block with clearly labelled synthetic
test geometry.  Raw-YAML tests still inspect the untouched files directly.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

import track1_grasp.config as config_module
from track1_grasp.collision import GEOMETRY_SOURCE_OFFLINE_MOCK_SYNTHETIC


_OFFICIAL_CD_CONFIG_NAMES = frozenset(
    {
        "track1_task_c_v11_official_20261102.yaml",
        "track1_task_d_v11_official_20261102.yaml",
        "track1_task_cd_40class_v11_official.yaml",
        "track1_task_c_v11_official_mustard_single_20261102.yaml",
    }
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "raw_formal_cd_config: inspect or load the untouched official C/D YAML",
    )


def _synthetic_box_clearance() -> dict[str, Any]:
    # Offline test only; never copy these synthetic dimensions into a shipped
    # official configuration or treat them as field measurements.
    return {
        "required": True,
        "geometry_validated": True,
        "geometry_source": GEOMETRY_SOURCE_OFFLINE_MOCK_SYNTHETIC,
        "center_xy_m": [0.0, 0.0],
        "yaw_rad": 0.0,
        "interior_size_m": [2.0, 2.0],
        "wall_clearance_m": 0.0,
        "obstacle_clearance_m": 0.0,
        "gripper_footprint_flange_xy_m": [
            [-0.06, -0.02],
            [0.06, -0.02],
            [0.06, 0.02],
            [-0.06, 0.02],
        ],
        "obstacles_m": [],
    }


def _load_offline_official_config(path: str | Path) -> config_module.Track1Config:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload = deepcopy(payload)
    task_cd = payload.setdefault("task_cd", {})
    task_cd["box_clearance"] = _synthetic_box_clearance()
    motion = payload.setdefault("motion", {})
    motion["backend"] = "mock"
    motion["dry_run"] = True
    gripper = payload.setdefault("gripper", {})
    gripper["backend"] = "mock"
    return config_module.Track1Config.from_dict(
        payload,
        base_dir=config_path.parent,
    )


@pytest.fixture(autouse=True)
def _offline_geometry_for_official_cd_tests(monkeypatch, request):
    """Keep all existing C/D logic tests non-actuating and self-contained."""

    if request.node.get_closest_marker("raw_formal_cd_config") is not None:
        return

    original = config_module.load_config

    def load_config(path: str | Path):
        config_path = Path(path).expanduser().resolve()
        if config_path.name in _OFFICIAL_CD_CONFIG_NAMES:
            return _load_offline_official_config(config_path)
        return original(config_path)

    if getattr(request.module, "load_config", None) is original:
        monkeypatch.setattr(request.module, "load_config", load_config)
    if config_module.load_config is original:
        monkeypatch.setattr(config_module, "load_config", load_config)
