from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from review_core import (
    build_review_index,
    export_reviews,
    load_review,
    load_review_index,
    save_review,
)
from capture_store import load_class_catalog


CLASS_MAP = Path(
    "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/"
    "datasets/track1_selfcollected_v1/manifests/class_map.yaml"
)


def _make_fake_dataset(
    root: Path,
    *,
    include_multi_object_hint: bool = False,
) -> tuple[Path, Path]:
    rgb = root / "rgb"
    depth = root / "depth"
    meta = root / "meta"
    manifests = root / "manifests"
    for directory in (rgb, depth, meta, manifests):
        directory.mkdir(parents=True, exist_ok=True)
    class_map = root / "class_map.yaml"
    shutil.copy2(CLASS_MAP, class_map)
    rows = []
    for index, (split, hint) in enumerate((("train", 11), ("val", 40)), start=1):
        frame_id = f"{index:06d}"
        stem = f"{frame_id}_{hint}-Item"
        image = np.full((48, 64, 3), 80 + index * 40, dtype=np.uint8)
        cv2.imwrite(str(rgb / f"{stem}.png"), image)
        cv2.imwrite(
            str(depth / f"{stem}.depth.png"),
            np.full((48, 64), 5000, dtype=np.uint16),
        )
        metadata = {
            "frame_id": frame_id,
            "class_id": hint,
            "canonical_label": f"{hint}-Item",
            "capture_group": f"burst-{index}",
            "width": 64,
            "height": 48,
            "depth_scale_m_per_unit": 0.0001,
            "intrinsics": {
                "width": 64,
                "height": 48,
                "fx": 50,
                "fy": 50,
                "ppx": 32,
                "ppy": 24,
            },
        }
        (meta / f"{stem}.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        rows.append(
            {
                "source_meta": f"meta/{stem}.json",
                "frame_id": frame_id,
                "class_id": str(hint),
                "capture_role": "positive",
                "effective_capture_group": f"{hint}|capture:burst-{index}",
                "split_unit": f"split-unit-{index:04d}",
                "split": split,
                "rgb": f"rgb/{stem}.png",
                "depth": f"depth/{stem}.depth.png",
                "overlay": "",
            }
        )
    if include_multi_object_hint:
        index = 3
        frame_id = f"{index:06d}"
        stem = f"{frame_id}_99-Multi_Object"
        image = np.full((48, 64, 3), 220, dtype=np.uint8)
        cv2.imwrite(str(rgb / f"{stem}.png"), image)
        cv2.imwrite(
            str(depth / f"{stem}.depth.png"),
            np.full((48, 64), 5000, dtype=np.uint16),
        )
        metadata = {
            "frame_id": frame_id,
            "class_id": 99,
            "canonical_label": "99-Multi_Object",
            "capture_group": "multi-object-burst",
            "width": 64,
            "height": 48,
            "depth_scale_m_per_unit": 0.0001,
            "intrinsics": {
                "width": 64,
                "height": 48,
                "fx": 50,
                "fy": 50,
                "ppx": 32,
                "ppy": 24,
            },
        }
        (meta / f"{stem}.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        rows.append(
            {
                "source_meta": f"meta/{stem}.json",
                "frame_id": frame_id,
                "class_id": "99",
                "capture_role": "multi_object_review",
                "effective_capture_group": "99|capture:multi-object-burst",
                "split_unit": "split-unit-0003",
                "split": "train",
                "rgb": f"rgb/{stem}.png",
                "depth": f"depth/{stem}.depth.png",
                "overlay": "",
            }
        )
    with (manifests / "split_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root, class_map


def test_multi_object_export_and_empty_label(tmp_path: Path) -> None:
    source_root, class_map = _make_fake_dataset(tmp_path / "source")
    review_root = tmp_path / "review"
    index = build_review_index(
        source_root=source_root,
        class_map_path=class_map,
        review_root=review_root,
    )
    assert index["item_count"] == 2
    catalog = load_class_catalog(class_map)
    first = index["items"][0]
    second = index["items"][1]

    review = save_review(
        review_root=review_root,
        source_root=source_root,
        item=first,
        catalog=catalog,
        payload={
            "status": "labeled",
            "boxes": [
                {"class_id": 11, "bbox_xyxy": [2, 3, 20, 25], "source": "manual"},
                {"class_id": 40, "bbox_xyxy": [24, 4, 58, 42], "source": "manual"},
                {"class_id": 11, "bbox_xyxy": [6, 28, 18, 45], "source": "manual"},
            ],
            "notes": "multi-class and multi-instance",
        },
    )
    assert len(review["boxes"]) == 3
    assert load_review(review_root, second["review_id"]) is None
    save_review(
        review_root=review_root,
        source_root=source_root,
        item=second,
        catalog=catalog,
        payload={"status": "empty", "boxes": [], "notes": "empty"},
    )

    result = export_reviews(
        source_root=source_root,
        review_root=review_root,
        export_root=tmp_path / "export",
        index=load_review_index(review_root),
        catalog=catalog,
    )
    output = Path(result["export_root"])
    train_labels = list((output / "labels" / "train").glob("*.txt"))
    val_labels = list((output / "labels" / "val").glob("*.txt"))
    assert len(train_labels) == 1
    assert len(val_labels) == 1
    lines = train_labels[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [int(line.split()[0]) for line in lines] == [10, 39, 10]
    assert val_labels[0].read_bytes() == b""
    data = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert len(data["names"]) == 40
    audit = result["audit"]
    assert audit["box_count"] == 3
    assert audit["empty_background_count"] == 1
    assert audit["unreviewed_count"] == 0
    assert audit["complete_review"] is True
    assert audit["postclasses_merge_ready"] is False
    assert audit["training_ready"] is False
    assert audit["missing_spot_class_ids"] == [17, 26]
    assert audit["required_postclass_class_ids"] == [11, 40]
    assert audit["required_postclass_without_val"] == [11, 40]


def test_multi_object_hint_is_excluded_from_yolo_classes(tmp_path: Path) -> None:
    source_root, class_map = _make_fake_dataset(
        tmp_path / "source",
        include_multi_object_hint=True,
    )
    review_root = tmp_path / "review"
    index = build_review_index(
        source_root=source_root,
        class_map_path=class_map,
        review_root=review_root,
    )
    assert index["item_count"] == 3
    hint_item = next(
        item for item in index["items"] if item["hint_class_id"] == 99
    )
    assert hint_item["hint_class_label"] == "99-Multi_Object"
    assert hint_item["hint_display_name"] == "多物品混放"

    catalog = load_class_catalog(class_map)
    for item in index["items"]:
        if item["hint_class_id"] == 99:
            payload = {
                "status": "labeled",
                "boxes": [
                    {
                        "class_id": 11,
                        "bbox_xyxy": [2, 3, 20, 25],
                        "source": "manual",
                    }
                ],
                "notes": "multi-object hint reviewed as official objects",
            }
        else:
            payload = {"status": "empty", "boxes": [], "notes": "empty"}
        save_review(
            review_root=review_root,
            source_root=source_root,
            item=item,
            catalog=catalog,
            payload=payload,
        )

    result = export_reviews(
        source_root=source_root,
        review_root=review_root,
        export_root=tmp_path / "export",
        index=load_review_index(review_root),
        catalog=catalog,
    )
    output = Path(result["export_root"])
    data = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert len(data["names"]) == 40
    assert 99 not in result["audit"]["required_postclass_class_ids"]
    assert result["audit"]["required_postclass_class_ids"] == [11, 40]
