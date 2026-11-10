from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from capture_store import CaptureStore, load_class_catalog
from live_camera_preview import load_capture_controls, load_scene_types


ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
CONFIG = ROOT / "config/track1_capture_web.yaml"
CLASS_MAP = ROOT / "datasets/track1_selfcollected_v1/manifests/class_map.yaml"


def test_multi_object_hint_capture_is_review_only(tmp_path: Path) -> None:
    args = SimpleNamespace(config=str(CONFIG), scene_types_json="")
    catalog = load_class_catalog(CLASS_MAP)
    scenes = load_scene_types(args)
    controls = load_capture_controls(args)

    assert len(catalog.entries) == 40
    assert 99 not in {entry.class_id for entry in catalog.entries}
    hint = controls["multi_object_hint"]
    assert hint["class_id"] == 99
    assert hint["default_scene_type"] == "box_11l_clutter"
    assert hint["default_notes"] == "多物品混放"
    assert hint["require_instance_id"] is False

    store = CaptureStore(
        tmp_path / "photos",
        catalog,
        scene_types=scenes,
        **controls,
    )
    policy = store.capture_policy_api()
    assert policy["multi_object_hint"]["label_index"] is None
    assert policy["multi_object_hint"]["is_collection_hint"] is True
    assert policy["multi_object_hint"]["capture_hint_kind"] == "multi_object"
    assert policy["multi_object_hint"]["requires_per_object_bbox_review"] is True

    result = store.capture(
        session_id="20261031_session_test",
        class_id=99,
        frame=np.zeros((48, 64, 3), dtype=np.uint8),
        depth_raw=np.full((48, 64), 1000, dtype=np.uint16),
        depth_scale_m_per_unit=0.0001,
        intrinsics={
            "width": 64,
            "height": 48,
            "fx": 50.0,
            "fy": 50.0,
            "ppx": 32.0,
            "ppy": 24.0,
        },
        camera_serial="TEST-D405",
        rotate_180=False,
        scene_type="box_11l",
        instance_id="must-be-cleared",
        notes="",
        save_overlay_jpeg=False,
    )
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["class_id"] == 99
    assert metadata["canonical_label"] == "99-Multi_Object"
    assert metadata["label_index"] is None
    assert metadata["scene_type"] == "box_11l_clutter"
    assert metadata["instance_id"] == ""
    assert metadata["notes"] == "多物品混放"
    assert metadata["is_collection_hint"] is True
    assert metadata["capture_hint_kind"] == "multi_object"
    assert metadata["requires_per_object_bbox_review"] is True
