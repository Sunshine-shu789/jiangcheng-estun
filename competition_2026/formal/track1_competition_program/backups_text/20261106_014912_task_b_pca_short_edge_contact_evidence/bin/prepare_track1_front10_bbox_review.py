#!/usr/bin/env python3
"""Build a dedicated front-10 bbox review source and index.

This script never modifies the source capture dataset. It creates a small
review-source directory with symlinked RGB-D assets, a deduplicated physical
split manifest, the dedicated review config, and the review index.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
DEFAULT_SOURCE = ROOT / "datasets/track1_front10_selfcollected_v1"
DEFAULT_REVIEW_SOURCE = ROOT / "datasets/track1_front10_bbox_review_source_v1"
DEFAULT_REVIEW_ROOT = ROOT / "datasets/track1_front10_bbox_review_v1"
DEFAULT_EXPORT_ROOT = ROOT / "datasets/track1_front10_bbox_export_v1"
DEFAULT_CONFIG = ROOT / "config/track1_front10_bbox_review_v1.yaml"
DEFAULT_TEMPLATE_CONFIG = ROOT / "config/track1_bbox_review.yaml"
DEFAULT_CLASS_MAP = ROOT / "datasets/track1_selfcollected_v1/manifests/class_map.yaml"
TARGET_IDS = tuple(range(1, 11))
TRIPLE_SPLITS = {
    "triple-can-layout-20261029-a": "val",
    "triple-can-layout-20261029-b": "train",
}
ASSET_DIRS = ("rgb", "depth", "meta", "overlays")

CAPTURE_APP = ROOT / "app/track1_capture_web"
REVIEW_APP = ROOT / "app/track1_bbox_review"
for module_dir in (CAPTURE_APP, REVIEW_APP):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
from review_core import build_review_index  # noqa: E402


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"无法读取 RGB：{path}")
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    return resized[:, 1:] > resized[:, :-1]


def hamming(first: np.ndarray, second: np.ndarray) -> int:
    return int(np.count_nonzero(first != second))


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def load_selected(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for raw in (source / "frames.jsonl").read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    selected = [row for row in rows if row.get("review_selected") and row.get("accepted")]
    if not selected:
        raise RuntimeError("前10类数据集中没有 review_selected 代表帧")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_group[str(row["physical_group"])].append(row)
    representatives: list[dict[str, Any]] = []
    for group, group_rows in sorted(by_group.items()):
        representative = max(
            group_rows,
            key=lambda row: (
                float((row.get("quality") or {}).get("rgb_sharpness", -1.0)),
                str(row["frame_id"]),
            ),
        )
        if int(representative["class_id"]) not in TARGET_IDS:
            raise RuntimeError(f"{group} 不属于前10类：{representative['class_id']}")
        representatives.append(representative)
    return rows, representatives


def load_template(template: Path) -> dict[str, Any]:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"候选配置格式错误：{template}")
    return payload


def build_plan(
    representatives: list[dict[str, Any]],
    *,
    source: Path,
    near_duplicate_distance: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    groups = {str(row["physical_group"]): row for row in representatives}
    if len(groups) != len(representatives):
        raise RuntimeError("同一 physical_group 出现多个代表帧")
    hashes = {
        group: dhash(source / str(row["source_files"]["rgb"]))
        for group, row in groups.items()
    }
    union_find = UnionFind(sorted(groups))
    duplicate_pairs: list[dict[str, Any]] = []
    ordered_groups = sorted(
        groups,
        key=lambda group: (str(groups[group].get("captured_at") or ""), group),
    )
    for index, first in enumerate(ordered_groups):
        first_row = groups[first]
        for second in ordered_groups[index + 1 :]:
            second_row = groups[second]
            if int(first_row["class_id"]) != int(second_row["class_id"]):
                continue
            if first in TRIPLE_SPLITS or second in TRIPLE_SPLITS:
                continue
            distance = hamming(hashes[first], hashes[second])
            if distance <= near_duplicate_distance:
                union_find.union(first, second)
                duplicate_pairs.append(
                    {
                        "class_id": int(first_row["class_id"]),
                        "first_group": first,
                        "second_group": second,
                        "first_frame_id": str(first_row["frame_id"]),
                        "second_frame_id": str(second_row["frame_id"]),
                        "original_splits": [str(first_row["split"]), str(second_row["split"])],
                        "dhash_distance": distance,
                    }
                )

    components: dict[str, list[str]] = defaultdict(list)
    for group in sorted(groups):
        components[union_find.find(group)].append(group)

    units: list[dict[str, Any]] = []
    for component_index, component_groups in enumerate(
        sorted(
            components.values(),
            key=lambda values: (
                min(str(groups[group].get("captured_at") or "") for group in values),
                values,
            ),
        ),
        1,
    ):
        component_rows = [groups[group] for group in component_groups]
        class_ids = {int(row["class_id"]) for row in component_rows}
        if len(class_ids) != 1:
            raise RuntimeError(f"去重单元混入多个提示类别：{component_groups}: {class_ids}")
        forced_splits = {TRIPLE_SPLITS[group] for group in component_groups if group in TRIPLE_SPLITS}
        if len(forced_splits) > 1:
            raise RuntimeError(f"去重单元同时包含两个三罐头布局：{component_groups}")
        if forced_splits:
            split = next(iter(forced_splits))
        else:
            original_splits = {str(row["split"]) for row in component_rows}
            split = "val" if "val" in original_splits else "train"
        units.append(
            {
                "split_unit": f"front10-review-unit-{component_index:04d}",
                "split": split,
                "hint_class_id": next(iter(class_ids)),
                "physical_groups": component_groups,
                "representative_frame_ids": [str(row["frame_id"]) for row in component_rows],
                "member_rows": component_rows,
                "is_triple_can_layout": any(group in TRIPLE_SPLITS for group in component_groups),
            }
        )

    split_units: dict[str, list[str]] = defaultdict(list)
    class_split_units: dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
    }
    class_box_hints: dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
    }
    for unit in units:
        split_units[unit["split"]].append(unit["split_unit"])
        class_split_units[unit["split"]][int(unit["hint_class_id"])] += 1
        for class_id in sorted(
            {
                int(value)
                for row in unit["member_rows"]
                for value in (row.get("member_class_ids") or [row["class_id"]])
            }
        ):
            class_box_hints[unit["split"]][class_id] += 1
    # Triple-can layouts contain classes 2, 5 and 10 even though the capture
    # hint is class 2. Record that explicitly for the expected positive audit.
    for unit in units:
        if unit["is_triple_can_layout"]:
            for class_id in (2, 5, 10):
                class_box_hints[unit["split"]][class_id] += 0

    for split in ("train", "val"):
        missing = [class_id for class_id in TARGET_IDS if class_split_units[split][class_id] == 0]
        if missing:
            raise RuntimeError(f"{split} 的采集提示类别无代表帧：{missing}")

    split_manifest_rows: list[dict[str, str]] = []
    for unit in units:
        for row in unit["member_rows"]:
            split_manifest_rows.append(
                {
                    "source_meta": str(row["source_files"]["metadata"]),
                    "frame_id": str(row["frame_id"]),
                    "class_id": str(row["class_id"]),
                    "capture_role": "positive",
                    "effective_capture_group": str(row["physical_group"]),
                    "split_unit": str(unit["split_unit"]),
                    "split": str(unit["split"]),
                    "rgb": str(row["source_files"]["rgb"]),
                    "depth": str(row["source_files"]["depth"]),
                    "overlay": str(row["source_files"]["overlay"]),
                    "physical_group": str(row["physical_group"]),
                }
            )

    triple_units = [
        {
            "split_unit": unit["split_unit"],
            "split": unit["split"],
            "physical_group": unit["physical_groups"][0],
            "frame_id": unit["representative_frame_ids"][0],
            "required_classes_for_bbox": [2, 5, 10],
        }
        for unit in units
        if unit["is_triple_can_layout"]
    ]
    if len(triple_units) != 2:
        raise RuntimeError(f"三罐头布局应有2个独立复核单元，实际 {len(triple_units)}")
    if {item["split"] for item in triple_units} != {"train", "val"}:
        raise RuntimeError(f"三罐头布局未形成 train/val 对照：{triple_units}")

    plan = {
        "schema_version": 1,
        "created_at": now_iso(),
        "source": str(source),
        "source_frame_count": len(representatives),
        "near_duplicate_policy": {
            "algorithm": "dhash",
            "hamming_distance_leq": int(near_duplicate_distance),
            "scope": "same_capture_hint_class_only",
            "triple_can_layouts_never_merged_with_other_groups": True,
        },
        "duplicate_pairs": duplicate_pairs,
        "review_unit_count": len(units),
        "deduplicated_count": len(representatives) - len(units),
        "split_counts": {split: len(split_units[split]) for split in ("train", "val")},
        "hint_class_units": {
            split: {str(class_id): int(class_split_units[split][class_id]) for class_id in TARGET_IDS}
            for split in ("train", "val")
        },
        "triple_can_units": triple_units,
        "units": [
            {
                "split_unit": unit["split_unit"],
                "split": unit["split"],
                "hint_class_id": unit["hint_class_id"],
                "physical_groups": unit["physical_groups"],
                "representative_frame_ids": unit["representative_frame_ids"],
                "is_triple_can_layout": unit["is_triple_can_layout"],
            }
            for unit in units
        ],
    }
    return plan, units, split_manifest_rows


def write_config(
    *,
    config_path: Path,
    template_path: Path,
    review_source: Path,
    review_root: Path,
    export_root: Path,
    class_map: Path,
) -> None:
    payload = load_template(template_path)
    payload["title"] = "赛道一前10类新自采 bbox 复核 v1"
    payload["description"] = (
        "仅复核 2026-10-29 新采集的前10类代表帧。文件名只是采集提示；"
        "三罐头同框图必须分别标注 2-Starkist_Tuna、5-Tomato_Soup、10-meat_can。"
        "本程序不打开相机、不导入 ROS、不控制机器人。"
    )
    payload["paths"] = {
        "project_root": str(ROOT),
        "source_root": str(review_source),
        "review_root": str(review_root),
        "export_root": str(export_root),
        "class_map": str(class_map),
        "world_weights": str(ROOT / "models/yolov8s-worldv2.pt"),
    }
    policy = dict(payload.get("review_policy") or {})
    policy.update(
        {
            "label_all_visible_known_objects": True,
            "allow_multiple_classes_per_image": True,
            "allow_multiple_instances_per_class": True,
            "empty_background_uses_empty_label_file": True,
            "background_is_not_a_yolo_class": True,
            "missing_spots_class_ids": [17, 26],
            "expected_known_class_count": 38,
            "front10_only_positive_classes": list(TARGET_IDS),
            "triple_can_required_classes": [2, 5, 10],
            "triple_can_layout_splits": TRIPLE_SPLITS,
        }
    )
    payload["review_policy"] = policy
    config_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_metadata(
    *,
    review_source: Path,
    review_root: Path,
    plan: dict[str, Any],
    split_manifest_rows: list[dict[str, str]],
    config_path: Path,
    class_map: Path,
) -> None:
    (review_source / "manifests").mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_meta",
        "frame_id",
        "class_id",
        "capture_role",
        "effective_capture_group",
        "split_unit",
        "split",
        "rgb",
        "depth",
        "overlay",
        "physical_group",
    ]
    with (review_source / "manifests/split_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_manifest_rows)
    plan_path = review_source / "manifests/review_split_plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checklist = f"""# 赛道一前10类 bbox 复核 v1

- 源数据：`{plan['source']}`
- 复核项：{plan['review_unit_count']}
- 删除近重复代表帧：{plan['deduplicated_count']}
- train / val：{plan['split_counts']['train']} / {plan['split_counts']['val']}
- 类别：官方 ID 1–10，对应 YOLO 0–9
- 17、26：不参与本前10类复核，最终训练集继续保留空槽
- 三罐头布局 A：`triple-can-layout-20261029-a` -> val
- 三罐头布局 B：`triple-can-layout-20261029-b` -> train
- 每张三罐头图必须分别标注 2、5、10 三个独立 bbox
- 缺目标、夹子遮挡、只拍到局部：标出仍可辨认的已知物品；不能当空背景
- 真正没有已知物品才按“本图无已知物品”导出空 `.txt`
- 小目标不得因面积小删除
- 同类多实例逐个标框
- 本复核只生成前10类 bbox；不要把 11–40 类框进训练标签
"""
    (review_source / "README.md").write_text(checklist, encoding="utf-8")
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "README.md").write_text(checklist, encoding="utf-8")
    (review_source / "manifests/source_manifest.json").write_text(
        json.dumps(
            {
                "created_at": now_iso(),
                "source_dataset": str(plan["source"]),
                "config": str(config_path),
                "class_map": str(class_map),
                "class_map_sha256": sha256_file(class_map),
                "review_root": str(review_root),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    review_source = args.review_source.resolve()
    review_root = args.review_root.resolve()
    config_path = args.config.resolve()
    class_map = args.class_map.resolve()
    if review_source.exists():
        raise RuntimeError(f"复核源目录已存在，拒绝覆盖：{review_source}")
    if config_path.exists():
        raise RuntimeError(f"复核配置已存在，拒绝覆盖：{config_path}")
    if (review_root / "index.json").exists() or (review_root / "reviews").exists():
        raise RuntimeError(f"复核目录已有正式数据，拒绝覆盖：{review_root}")

    _, representatives = load_selected(source)
    plan, _units, split_rows = build_plan(
        representatives,
        source=source,
        near_duplicate_distance=int(args.near_duplicate_distance),
    )

    review_source.mkdir(parents=True, exist_ok=False)
    for name in ASSET_DIRS:
        target = source / name
        if not target.is_dir():
            raise RuntimeError(f"源数据缺少目录：{target}")
        os.symlink(target, review_source / name, target_is_directory=True)

    write_config(
        config_path=config_path,
        template_path=args.template_config.resolve(),
        review_source=review_source,
        review_root=review_root,
        export_root=args.export_root.resolve(),
        class_map=class_map,
    )
    write_metadata(
        review_source=review_source,
        review_root=review_root,
        plan=plan,
        split_manifest_rows=split_rows,
        config_path=config_path,
        class_map=class_map,
    )
    index = build_review_index(
        source_root=review_source,
        class_map_path=class_map,
        review_root=review_root,
    )
    result = {
        "ok": True,
        "source": str(source),
        "review_source": str(review_source),
        "review_root": str(review_root),
        "config": str(config_path),
        "review_units": index["item_count"],
        "split_counts": index["split_counts"],
        "duplicate_pairs": plan["duplicate_pairs"],
        "triple_can_units": plan["triple_can_units"],
    }
    (review_source / "manifests/prepare_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成赛道一前10类独立 bbox 复核源和索引")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--review-source", type=Path, default=DEFAULT_REVIEW_SOURCE)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE_CONFIG)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
