#!/usr/bin/env python3
"""Audit and build a derived Track-1 self-collected RGB-D dataset.

Design constraints:
- Never modify the source capture session.
- Keep official class IDs unchanged.
- Do not use object pixel area as a rejection rule: small objects are valid.
- Treat class_id=0 as the only empty-box/no-target negative class.
- Do not create or report scene subcategories. Legacy scene metadata is never
  used to delete or silently relabel a frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
    import numpy as np
    import yaml
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - launcher supplies the runtime path
    raise SystemExit(
        "缺少审计依赖（cv2/numpy/yaml/Pillow）。请通过 bin 包装脚本运行，"
        f"不要直接调用本文件。原始错误：{exc}"
    )

TARGET_IDS = tuple(range(11, 41))
BACKGROUND_CLASS_ID = 0
ACCEPTED_IDS = (BACKGROUND_CLASS_ID,) + TARGET_IDS
TEMPORARILY_MISSING_IDS = (17, 26)
BACKGROUND_DISPLAY_NAME = "空箱/无目标（负样本）"
BACKGROUND_ENGLISH_NAME = "Empty Box / No Target"
FILE_KINDS = ("rgb", "depth", "metadata", "overlay")
IMAGE_KINDS = ("rgb", "depth", "overlay")
DATASET_NAME = "track1_selfcollected_v3_no_scene_taxonomy"
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480
NEAR_DUP_REPORT_HAMMING = 2
SPLIT_DUP_HAMMING = 1


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_text(row) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cooked: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, tuple, dict)):
                    value = json_text(value)
                cooked[key] = value
            writer.writerow(cooked)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def dhash_hex(value: int | None) -> str:
    return "" if value is None else f"{value:016x}"


def within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def capture_role_for(class_id: int) -> str:
    if class_id == BACKGROUND_CLASS_ID:
        return "background"
    if class_id in TARGET_IDS:
        return "positive"
    return "out_of_scope"


def public_display_name(entry: dict[str, Any]) -> str:
    if int(entry.get("class_id", -1)) == BACKGROUND_CLASS_ID:
        return BACKGROUND_DISPLAY_NAME
    return str(entry.get("display_name") or "")


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    internal_only_fields = {"scene_id", "scene_type", "scene_label", "box_type"}
    row = {
        key: value
        for key, value in record.items()
        if key not in internal_only_fields
    }
    row["capture_role"] = capture_role_for(int(record["class_id"]))
    return row


def load_class_map(path: Path) -> tuple[dict[int, dict[str, Any]], str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = payload.get("classes") or {}
    catalog: dict[int, dict[str, Any]] = {}
    background = payload.get("background")
    if not isinstance(background, dict):
        raise SystemExit("类别表缺少 background 对象")
    background_item = dict(background)
    background_item["class_id"] = int(background_item.get("class_id", -1))
    if background_item["class_id"] != BACKGROUND_CLASS_ID:
        raise SystemExit(
            f"背景类别 ID 必须为 {BACKGROUND_CLASS_ID}，实际 {background_item['class_id']}"
        )
    background_item["display_name"] = BACKGROUND_DISPLAY_NAME
    background_item["english_name"] = BACKGROUND_ENGLISH_NAME
    catalog[BACKGROUND_CLASS_ID] = background_item
    for key, value in classes.items():
        class_id = int(key)
        item = dict(value)
        item["class_id"] = class_id
        catalog[class_id] = item
    missing = [class_id for class_id in TARGET_IDS if class_id not in catalog]
    if missing:
        raise SystemExit(f"类别表缺少后30类中的 ID：{missing}")
    return catalog, sha256_file(path)


def load_frames_jsonl_audit(source: Path) -> dict[str, Any]:
    path = source / "frames.jsonl"
    if not path.is_file():
        return {"available": False, "path": str(path)}
    line_count = 0
    malformed = 0
    ids: Counter[str] = Counter()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                line_count += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                ids[str(item.get("frame_id", ""))] += 1
    except OSError as exc:
        return {"available": False, "path": str(path), "error": str(exc)}
    return {
        "available": True,
        "path": str(path),
        "nonempty_lines": line_count,
        "malformed_lines": malformed,
        "unique_frame_ids": len(ids),
        "duplicate_frame_ids": sum(1 for value in ids.values() if value > 1),
        "duplicate_extra_records": sum(max(0, value - 1) for value in ids.values()),
        "note": "frames.jsonl 是追加历史；清洗以每帧独立 meta/*.json 为权威记录。",
    }


def audit_records(source: Path, class_map_path: Path) -> dict[str, Any]:
    catalog, map_sha256 = load_class_map(class_map_path)
    meta_dir = source / "meta"
    if not meta_dir.is_dir():
        raise SystemExit(f"原始会话缺少 meta 目录：{meta_dir}")

    records: list[dict[str, Any]] = []
    meta_load_errors: list[dict[str, str]] = []
    for meta_path in sorted(meta_dir.glob("*.json")):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            meta_load_errors.append({"file": meta_path.name, "error": str(exc)})
            continue

        try:
            class_id = int(metadata.get("class_id"))
        except (TypeError, ValueError):
            class_id = -1
        files = metadata.get("files") or {}
        source_files: dict[str, str] = {}
        missing: list[str] = []
        unsafe: list[str] = []
        zero_size: list[str] = []
        unreadable: list[str] = []
        quality: dict[str, Any] = {}
        rgb_gray: np.ndarray | None = None
        rgb_sha256 = ""

        for kind in FILE_KINDS:
            rel = str(files.get(kind) or "")
            source_files[kind] = rel
            candidate = source / rel if rel else None
            if candidate is None or not candidate.is_file():
                missing.append(kind)
                continue
            if not within_root(candidate, source):
                unsafe.append(kind)
                continue
            if candidate.stat().st_size <= 0:
                zero_size.append(kind)
                continue
            if kind in IMAGE_KINDS:
                image = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
                if image is None:
                    unreadable.append(kind)
                    continue
                height, width = image.shape[:2]
                quality[f"{kind}_width"] = int(width)
                quality[f"{kind}_height"] = int(height)
                if kind == "rgb":
                    rgb_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    quality["rgb_sharpness_laplacian"] = round(
                        float(cv2.Laplacian(rgb_gray, cv2.CV_64F).var()), 4
                    )
                    rgb_sha256 = sha256_file(candidate)
                elif kind == "depth":
                    depth = image
                    if depth.ndim > 2:
                        depth = depth[:, :, 0]
                    valid_ratio = float((depth > 0).mean())
                    quality["depth_valid_ratio"] = round(valid_ratio, 6)
                    quality["depth_min"] = int(depth.min())
                    quality["depth_max"] = int(depth.max())
                    quality["depth_median_nonzero"] = (
                        int(np.median(depth[depth > 0])) if np.any(depth > 0) else 0
                    )

        if rgb_gray is not None:
            quality["dhash64"] = dhash_hex(dhash64(rgb_gray))

        complete = not (missing or unsafe or zero_size or unreadable)
        frame_id = str(metadata.get("frame_id") or "")
        record = {
            "record_key": meta_path.relative_to(source).as_posix(),
            "source_meta": meta_path.relative_to(source).as_posix(),
            "frame_id": frame_id,
            "scene_id": str(metadata.get("scene_id") or ""),
            "captured_at": str(metadata.get("captured_at") or ""),
            "class_id": class_id,
            "capture_role": capture_role_for(class_id),
            "canonical_label": str(
                metadata.get("canonical_label")
                or (catalog.get(class_id, {}).get("canonical_label", ""))
            ),
            "english_name": (
                BACKGROUND_ENGLISH_NAME
                if class_id == BACKGROUND_CLASS_ID
                else str(
                    metadata.get("english_name")
                    or (catalog.get(class_id, {}).get("english_name", ""))
                )
            ),
            "display_name": (
                BACKGROUND_DISPLAY_NAME
                if class_id == BACKGROUND_CLASS_ID
                else str(
                    metadata.get("display_name")
                    or (catalog.get(class_id, {}).get("display_name", ""))
                )
            ),
            "instance_id": str(metadata.get("instance_id") or ""),
            "scene_type": str(metadata.get("scene_type") or ""),
            "capture_group": str(metadata.get("capture_group") or ""),
            "capture_mode": str(metadata.get("capture_mode") or ""),
            "lighting_id": str(metadata.get("lighting_id") or ""),
            "source_files": source_files,
            "missing_files": missing,
            "unsafe_files": unsafe,
            "zero_size_files": zero_size,
            "unreadable_files": unreadable,
            "complete": complete,
            "quality": quality,
            "rgb_sha256": rgb_sha256,
            "warnings": [],
            "exclusion_reasons": [],
            "hard_excluded": False,
        }
        records.append(record)

    # File path and frame-id conflict detection must consider all source records.
    frame_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    file_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["frame_id"]:
            frame_groups[record["frame_id"]].append(record)
        for kind, rel in record["source_files"].items():
            if rel:
                file_groups[(kind, rel)].append(record)

    duplicate_frame_ids: list[dict[str, Any]] = []
    for frame_id, group in sorted(frame_groups.items()):
        if len(group) <= 1:
            continue
        class_ids = sorted({record["class_id"] for record in group})
        complete_count = sum(1 for record in group if record["complete"])
        conflict = len(class_ids) > 1
        duplicate_frame_ids.append(
            {
                "frame_id": frame_id,
                "record_count": len(group),
                "complete_record_count": complete_count,
                "class_ids": class_ids,
                "conflicting_class_ids": conflict,
                "records": [
                    {
                        "source_meta": record["source_meta"],
                        "class_id": record["class_id"],
                        "complete": record["complete"],
                    }
                    for record in group
                ],
                "policy": (
                    "仅当同一 frame_id 有多个完整且共用同一文件/同一像素却类别不同时排除；"
                    "重启导致的 ID 重复但文件不同，不强删。"
                ),
            }
        )
        if conflict:
            for record in group:
                record["warnings"].append("frame_id_reused_by_different_classes")

    reused_file_conflicts: list[dict[str, Any]] = []
    for (kind, rel), group in sorted(file_groups.items()):
        class_ids = sorted({record["class_id"] for record in group})
        if len(group) > 1 and len(class_ids) > 1:
            reused_file_conflicts.append(
                {
                    "kind": kind,
                    "relative_path": rel,
                    "class_ids": class_ids,
                    "source_metas": [record["source_meta"] for record in group],
                }
            )
            for record in group:
                record["exclusion_reasons"].append(f"file_reused_across_classes:{kind}:{rel}")
                record["hard_excluded"] = True

    # Exact pixel duplicates across different classes are label conflicts.
    complete_target = [
        record
        for record in records
        if record["complete"] and record["class_id"] in ACCEPTED_IDS and record["rgb_sha256"]
    ]
    sha_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in complete_target:
        sha_groups[record["rgb_sha256"]].append(record)

    exact_duplicate_groups: list[dict[str, Any]] = []
    for digest, group in sorted(sha_groups.items()):
        if len(group) <= 1:
            continue
        classes = sorted({record["class_id"] for record in group})
        cross_class = len(classes) > 1
        exact_duplicate_groups.append(
            {
                "sha256": digest,
                "class_ids": classes,
                "cross_class_conflict": cross_class,
                "records": [record["source_meta"] for record in group],
            }
        )
        if cross_class:
            for record in group:
                record["exclusion_reasons"].append("exact_same_pixels_different_class")
                record["hard_excluded"] = True
        else:
            canonical = sorted(group, key=lambda item: item["source_meta"])[0]
            for record in group:
                record["warnings"].append("exact_same_class_duplicate")
                if record is not canonical:
                    record["exclusion_reasons"].append("exact_duplicate_extra_copy")
                    record["hard_excluded"] = True

    # Missing-site classes are omitted explicitly; no reindexing is allowed.
    for record in records:
        if record["class_id"] in TEMPORARILY_MISSING_IDS:
            record["exclusion_reasons"].append("class_not_available_at_site")
        if record["class_id"] not in ACCEPTED_IDS:
            record["exclusion_reasons"].append("outside_task_C_D_class_scope")
        if not record["complete"]:
            if record["missing_files"]:
                record["exclusion_reasons"].append(
                    "missing_files:" + ",".join(record["missing_files"])
                )
            if record["unsafe_files"]:
                record["exclusion_reasons"].append(
                    "unsafe_file_path:" + ",".join(record["unsafe_files"])
                )
            if record["zero_size_files"]:
                record["exclusion_reasons"].append(
                    "zero_size_files:" + ",".join(record["zero_size_files"])
                )
            if record["unreadable_files"]:
                record["exclusion_reasons"].append(
                    "unreadable_files:" + ",".join(record["unreadable_files"])
                )
        if record["class_id"] in TARGET_IDS and record["scene_type"] == "box_11l_empty":
            # 旧采集字段只触发人工复核。绝不据此删图或自动改成空箱负样本。
            record["warnings"].append("legacy_positive_label_needs_manual_review")
            record["quality"]["retention_policy"] = "no_object_area_filter"
        if record["exclusion_reasons"]:
            record["hard_excluded"] = True

    accepted = [record for record in records if not record["hard_excluded"]]
    excluded = [record for record in records if record["hard_excluded"]]

    # Visual similarity is report-only here. It never removes a frame.
    near_duplicate_pairs: list[dict[str, Any]] = []
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in accepted:
        by_class[record["class_id"]].append(record)
    for class_id, group in sorted(by_class.items()):
        for index, left in enumerate(group):
            left_hash = int(left["quality"].get("dhash64") or "0", 16)
            for right in group[index + 1 :]:
                right_hash = int(right["quality"].get("dhash64") or "0", 16)
                distance = (left_hash ^ right_hash).bit_count()
                if distance <= NEAR_DUP_REPORT_HAMMING:
                    near_duplicate_pairs.append(
                        {
                            "class_id": class_id,
                            "left_meta": left["source_meta"],
                            "right_meta": right["source_meta"],
                            "left_frame_id": left["frame_id"],
                            "right_frame_id": right["frame_id"],
                            "left_capture_group": left["capture_group"],
                            "right_capture_group": right["capture_group"],
                            "hamming_distance": distance,
                            "same_capture_group": (
                                bool(left["capture_group"])
                                and left["capture_group"] == right["capture_group"]
                            ),
                        }
                    )

    reason_counts = Counter()
    for record in excluded:
        for reason in record["exclusion_reasons"]:
            reason_counts[reason.split(":", 1)[0]] += 1

    class_counts = Counter(record["class_id"] for record in accepted)
    role_counts = Counter(record["capture_role"] for record in accepted)
    warning_counts = Counter()
    for record in accepted:
        for warning in record["warnings"]:
            warning_counts[warning] += 1

    missing_classes = []
    for class_id in TEMPORARILY_MISSING_IDS:
        missing_classes.append(
            {
                "class_id": class_id,
                "canonical_label": catalog[class_id].get("canonical_label", ""),
                "english_name": catalog[class_id].get("english_name", ""),
                "display_name": catalog[class_id].get("display_name", ""),
                "reason": "物件现场不存在，用户确认暂不补采；不得改为 unknown，不得重编号。",
                "complete_accepted_frames": class_counts.get(class_id, 0),
            }
        )

    summary = {
        "schema_version": 1,
        "audit_at": now_local(),
        "dataset_name": DATASET_NAME,
        "source_session": str(source),
        "class_map": str(class_map_path),
        "class_map_sha256": map_sha256,
        "target_class_ids": list(TARGET_IDS),
        "accepted_class_ids": list(ACCEPTED_IDS),
        "background_class_id": BACKGROUND_CLASS_ID,
        "temporarily_missing_class_ids": list(TEMPORARILY_MISSING_IDS),
        "meta_record_count": len(records),
        "meta_load_error_count": len(meta_load_errors),
        "complete_record_count": sum(1 for record in records if record["complete"]),
        "accepted_record_count": len(accepted),
        "excluded_record_count": len(excluded),
        "accepted_unique_frame_id_count": len({record["frame_id"] for record in accepted}),
        "accepted_class_count": sum(1 for class_id in class_counts if class_id in TARGET_IDS),
        "accepted_label_count": len(class_counts),
        "accepted_positive_frame_count": role_counts.get("positive", 0),
        "accepted_background_frame_count": role_counts.get("background", 0),
        "accepted_capture_role_counts": dict(sorted(role_counts.items())),
        "accepted_class_counts": {str(k): v for k, v in sorted(class_counts.items())},
        "accepted_class_labels": {
            str(k): catalog[k].get("canonical_label", "") for k in sorted(class_counts)
        },
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "duplicate_frame_id_group_count": len(duplicate_frame_ids),
        "conflicting_duplicate_frame_id_group_count": sum(
            1 for item in duplicate_frame_ids if item["conflicting_class_ids"]
        ),
        "reused_file_conflict_count": len(reused_file_conflicts),
        "exact_duplicate_group_count": len(exact_duplicate_groups),
        "near_duplicate_pair_count": len(near_duplicate_pairs),
        "near_duplicate_hamming_threshold": NEAR_DUP_REPORT_HAMMING,
        "small_object_policy": (
            "不按像素面积、前景面积、清晰度低或看起来像空箱来删除；"
            "完整的小目标帧全部保留。"
        ),
        "capture_role_policy": (
            "class_id=0 表示空箱/无目标负样本；class_id=11-40 表示物品正样本。"
            "不创建场景子类，不按目标面积或外观自动删除。"
        ),
        "frames_jsonl_audit": load_frames_jsonl_audit(source),
        "meta_load_errors": meta_load_errors,
        "missing_classes": missing_classes,
    }

    # Keep complete diagnostics out of the summary but available to build writers.
    return {
        "summary": summary,
        "records": records,
        "accepted": accepted,
        "excluded": excluded,
        "catalog": catalog,
        "duplicate_frame_ids": duplicate_frame_ids,
        "reused_file_conflicts": reused_file_conflicts,
        "exact_duplicate_groups": exact_duplicate_groups,
        "near_duplicate_pairs": near_duplicate_pairs,
    }


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in self.parent}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def effective_capture_group(record: dict[str, Any]) -> str:
    capture_group = record["capture_group"]
    if capture_group:
        return f"{record['class_id']}|capture:{capture_group}"
    date = record["captured_at"][:10] or "unknown-date"
    return (
        f"{record['class_id']}|single:{date}:"
        f"{record['frame_id']}:{record['source_meta']}"
    )


def assign_splits(accepted: list[dict[str, Any]], near_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_meta = {record["source_meta"]: record for record in accepted}
    base_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in accepted:
        base_groups[effective_capture_group(record)].append(record)

    uf = UnionFind(base_groups)
    cross_group_links = 0
    for pair in near_pairs:
        if pair["same_capture_group"]:
            continue
        if pair["hamming_distance"] > SPLIT_DUP_HAMMING:
            continue
        left_record = by_meta.get(pair["left_meta"])
        right_record = by_meta.get(pair["right_meta"])
        if not left_record or not right_record or left_record["class_id"] != right_record["class_id"]:
            continue
        left_group = effective_capture_group(left_record)
        right_group = effective_capture_group(right_record)
        if left_group != right_group:
            uf.union(left_group, right_group)
            cross_group_links += 1

    split_units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_key, records in base_groups.items():
        split_units[uf.find(group_key)].extend(records)

    units_by_class: dict[int, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for unit_key, records in split_units.items():
        classes = {record["class_id"] for record in records}
        if len(classes) != 1:
            raise RuntimeError(f"拆分单元跨类别，拒绝继续：{unit_key}: {sorted(classes)}")
        records.sort(key=lambda item: (item["captured_at"], item["source_meta"]))
        units_by_class[records[0]["class_id"]].append((unit_key, records))

    unit_to_split: dict[str, str] = {}
    no_validation_classes: list[int] = []
    for class_id, unit_items in sorted(units_by_class.items()):
        unit_items.sort(
            key=lambda item: (
                min(record["captured_at"] for record in item[1]),
                item[0],
            )
        )
        total = len(unit_items)
        if total < 2:
            for unit_key, _ in unit_items:
                unit_to_split[unit_key] = "train"
            no_validation_classes.append(class_id)
            continue
        val_count = max(1, math.ceil(total * 0.20))
        val_count = min(val_count, total - 1)
        val_indices: set[int] = set()
        for ordinal in range(1, val_count + 1):
            index = math.floor(ordinal * total / (val_count + 1))
            index = min(max(index, 0), total - 1)
            while index in val_indices:
                index = (index + 1) % total
            val_indices.add(index)
        for index, (unit_key, _) in enumerate(unit_items):
            unit_to_split[unit_key] = "val" if index in val_indices else "train"

    unit_ids: dict[str, str] = {}
    for index, unit_key in enumerate(sorted(unit_to_split), start=1):
        unit_ids[unit_key] = f"split-unit-{index:04d}"

    for record in accepted:
        group = effective_capture_group(record)
        unit_key = uf.find(group)
        record["effective_capture_group"] = group
        record["split_unit"] = unit_ids[unit_key]
        record["split"] = unit_to_split[unit_key]

    split_class_counts: Counter[tuple[str, int]] = Counter(
        (record["split"], record["class_id"]) for record in accepted
    )
    split_counts = Counter(record["split"] for record in accepted)
    return {
        "unit_count": len(unit_to_split),
        "cross_group_near_duplicate_links": cross_group_links,
        "split_counts": dict(split_counts),
        "split_class_counts": {
            f"{split}:{class_id}": count
            for (split, class_id), count in sorted(split_class_counts.items())
        },
        "no_validation_classes": no_validation_classes,
    }


def choose_evenly(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(records) <= count:
        return list(records)
    if count <= 1:
        return [records[len(records) // 2]]
    indices = {
        round(index * (len(records) - 1) / (count - 1))
        for index in range(count)
    }
    return [records[index] for index in sorted(indices)]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def make_class_contact_sheets(
    output: Path,
    source: Path,
    accepted: list[dict[str, Any]],
    catalog: dict[int, dict[str, Any]],
) -> list[str]:
    sheet_dir = output / "review" / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in accepted:
        by_class[record["class_id"]].append(record)
    font = load_font(18)
    small_font = load_font(15)
    created: list[str] = []
    for class_id in sorted(by_class):
        records = sorted(by_class[class_id], key=lambda item: (item["captured_at"], item["source_meta"]))
        picked = choose_evenly(records, 12)
        cell_width, cell_height = 440, 350
        columns, rows = 3, 4
        canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "#1d1d1d")
        draw = ImageDraw.Draw(canvas)
        label = catalog[class_id].get("canonical_label", f"{class_id}")
        for index, record in enumerate(picked):
            row, column = divmod(index, columns)
            x0, y0 = column * cell_width, row * cell_height
            rgb_path = source / record["source_files"]["rgb"]
            image = Image.open(rgb_path).convert("RGB")
            image.thumbnail((cell_width - 12, cell_height - 54), Image.Resampling.LANCZOS)
            x = x0 + (cell_width - image.width) // 2
            y = y0 + 42
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1),
                outline="#555555",
            )
            draw.text((x0 + 6, y0 + 5), f"{label}  frame {record['frame_id']}", fill="white", font=font)
            draw.text(
                (x0 + 6, y0 + 25),
                f"role={record['capture_role']}  split={record['split']}",
                fill="#8ccfff",
                font=small_font,
            )
        safe_label = label.replace("/", "_").replace(" ", "_")
        path = sheet_dir / f"{class_id:02d}_{safe_label}.jpg"
        canvas.save(path, quality=92)
        created.append(path.relative_to(output).as_posix())
    return created


def count_rows(records: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    counts = Counter(key_fn(record) for record in records)
    return [{"key": key, "count": count} for key, count in sorted(counts.items(), key=lambda item: str(item[0]))]


def copy_accepted_files(output: Path, source: Path, accepted: list[dict[str, Any]]) -> None:
    for kind in FILE_KINDS:
        (output / kind).mkdir(parents=True, exist_ok=True)
    copied: dict[tuple[str, str], str] = {}
    for record in accepted:
        for kind in FILE_KINDS:
            rel = record["source_files"][kind]
            src = source / rel
            dst = output / rel
            digest = sha256_file(src)
            previous = copied.get((kind, rel))
            if previous and previous != digest:
                raise RuntimeError(f"目标路径冲突且内容不同：{kind}:{rel}")
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied[(kind, rel)] = digest


def write_sha256s(root: Path) -> int:
    checksum_path = root / "SHA256SUMS"
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        rel = path.relative_to(root).as_posix()
        entries.append(f"{sha256_file(path)}  {rel}")
    checksum_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return len(entries)


def render_readme(summary: dict[str, Any], split_info: dict[str, Any]) -> str:
    lines = [
        f"# {DATASET_NAME}",
        "",
        "> 临时版本：现场缺少第 **17 类（蛋形毛绒玩具）** 和 "
        "第 **26 类（塑料草莓）**。这两个 ID 被明确省略，**没有删除、重编号或改成 unknown**。",
        "",
        "## 当前结论",
        "",
        f"- 已生成有效帧：**{summary['accepted_record_count']}**",
        f"- 物品正类帧：**{summary['accepted_positive_frame_count']}**",
        f"- 空箱/无目标负样本帧：**{summary['accepted_background_frame_count']}**",
        f"- 覆盖的后 30 类中的类别数：**{summary['accepted_class_count']} / 28**",
        f"- 排除记录：**{summary['excluded_record_count']}**（主要是原始 RGB 缺失）",
        f"- 训练/验证划分：train={split_info['split_counts'].get('train', 0)}，"
        f"val={split_info['split_counts'].get('val', 0)}",
        "- 高级检测框标注：**尚无覆盖全部帧的 YOLO bbox，因此现在不能直接训练检测器。**",
        "",
        "## 标签规则",
        "",
        "- 有物品：`class_id=11～40`，物品名称必须人工选择。",
        "- 画面没有待识别物品：`class_id=0`，统一表示"
        "“空箱/无目标（负样本）”。",
        "- **不划分、不记录采集场景子类。**",
        "- 不按目标像素面积、前景面积、清晰度或“看起来像空箱”删除图片。",
        "- 小目标、弱前景和遮挡图片全部保留，无法确认时进入人工复核。",
        "",
        f"- 旧采集元数据中有 {summary['warning_counts'].get('legacy_positive_label_needs_manual_review', 0)} 帧"
        "被标为物品正类但带有旧“空箱”提示。这些帧只进入人工复核清单，"
        "不会被自动删除，也不会被自动改成 `class_id=0`。",
        "",
        "## 目录",
        "",
        "- `rgb/`、`depth/`、`meta/`、`overlays/`：只包含有效帧的四件套。",
        "- `manifests/accepted_frames.jsonl`：保留的帧、正类/负样本角色、质量和 split。",
        "- `manifests/excluded_frames.csv`：未复制帧及原因。",
        "- `manifests/class_counts.csv`：按类别统计，不包含场景子类。",
        "- `manifests/capture_role_counts.csv`：正类与空箱/无目标负样本统计。",
        "- `manifests/split_manifest.csv`：按采集组划分的 train/val。",
        "- `review/contact_sheets/`：每类代表性总览，便于复核小目标。",
        "- `dataset_report.md`：完整审计与训练门禁。",
        "- `SHA256SUMS`：派生数据集完整性校验。",
        "",
        "## 类别计数",
        "",
        "| class_id | 类别 | 角色 | 有效帧 |",
        "|---:|---|---|---:|",
    ]
    labels = {int(key): value for key, value in summary.get("accepted_class_labels", {}).items()}
    for class_id, count in sorted(summary["accepted_class_counts"].items(), key=lambda item: int(item[0])):
        cid = int(class_id)
        label = BACKGROUND_DISPLAY_NAME if cid == BACKGROUND_CLASS_ID else labels.get(cid, "")
        role = "空箱/无目标负样本" if cid == BACKGROUND_CLASS_ID else "物品正样本"
        lines.append(f"| {cid} | {label} | {role} | {count} |")
    lines += [
        "",
        "## 缺失类别",
        "",
        "- `17-Eggs_Plush`：0 个有效帧，现场暂不采集。",
        "- `26-Plastic_Strawberry`：0 个有效帧，现场暂不采集。",
        "- 当前临时数据集只能覆盖 28 个物品正类；正式 40 类模型必须补齐数据或采用“前10类模型 + 后28类模型”的组合方案。",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_report(summary: dict[str, Any], split_info: dict[str, Any], accepted: list[dict[str, Any]]) -> str:
    class_role: Counter[tuple[int, str]] = Counter(
        (record["class_id"], record["capture_role"]) for record in accepted
    )
    labels = {
        record["class_id"]: (
            BACKGROUND_DISPLAY_NAME
            if record["class_id"] == BACKGROUND_CLASS_ID
            else record["canonical_label"]
        )
        for record in accepted
    }
    lines = [
        "# 自采临时数据集 v3 审计报告（无场景细分）",
        "",
        f"- 生成时间：{summary['audit_at']}",
        f"- 原始会话：`{summary['source_session']}`",
        f"- 数据集版本：`{DATASET_NAME}`",
        f"- 原始 `meta/*.json` 记录：{summary['meta_record_count']}",
        f"- 完整四件套并通过解码：{summary['complete_record_count']}",
        f"- 入组有效帧：{summary['accepted_record_count']}",
        f"- 其中物品正类：{summary['accepted_positive_frame_count']}",
        f"- 其中空箱/无目标负样本：{summary['accepted_background_frame_count']}",
        f"- 排除记录：{summary['excluded_record_count']}",
        f"- 缺失类别：17、26（ID 保留，不重编号）",
        "",
        "## 标签与训练门禁",
        "",
        "- `class_id=0`：空箱/无目标负样本。",
        "- `class_id=11～40`：物品正样本。",
        "- 不记录场景细分；不按目标大小、前景强弱或外观自动删除。",
        "",
        "**当前不可直接作 YOLO 检测训练集。** 现有自采数据只有每帧类别标签，"
        "没有覆盖目标的 bbox；已有高级标注数量不足且不能替代全量标注。",
        "",
        "下一步必须先生成候选框并进行人工复核，再导出 `images/` + `labels/`。"
        "无目标负样本导出 YOLO 时使用空标签文件，不使用类别 `0` 的检测框。",
        "",
        "## 排除原因",
        "",
        "| 原因 | 数量 |",
        "|---|---:|",
    ]
    for reason, count in sorted(summary["exclusion_reason_counts"].items()):
        lines.append(f"| `{reason}` | {count} |")
    lines += [
        "",
        "排除以文件不完整、无法解码或明确跨类文件冲突为主，不以物体大小或“像空箱”判断。",
        "",
        "## 每类统计（不细分场景）",
        "",
        "| class_id | 类别 | 角色 | 有效帧 |",
        "|---:|---|---|---:|",
    ]
    class_ids = sorted({record["class_id"] for record in accepted})
    for class_id in class_ids:
        role = "空箱/无目标负样本" if class_id == BACKGROUND_CLASS_ID else "物品正样本"
        lines.append(f"| {class_id} | {labels.get(class_id, '')} | {role} | {class_role.get((class_id, capture_role_for(class_id)), 0)} |")
    lines += [
        "",
        "## 重复与采集组",
        "",
        f"- 重复 `frame_id` 组：{summary['duplicate_frame_id_group_count']}；"
        f"其中跨类别冲突组：{summary['conflicting_duplicate_frame_id_group_count']}。",
        "- 原会话存在计数器重启后的 `frame_id` 重复，但完整记录的文件路径不同；"
        "本版本只排除共用同一文件或同一像素却类别不同的真实冲突。",
        f"- 感知哈希近似对（Hamming ≤ {summary['near_duplicate_hamming_threshold']}）："
        f"{summary['near_duplicate_pair_count']}。仅用于泄漏审计，不删除小目标帧。",
        f"- 拆分单元数：{split_info['unit_count']}；"
        f"跨采集组近似重复约束链接：{split_info['cross_group_near_duplicate_links']}。",
        f"- train/val：{split_info['split_counts']}。",
        "",
        "## 建议的下一步",
        "",
        "1. 用 `review/contact_sheets/` 检查每类是否确为目标，重点放大钥匙、钉子、适配器等小目标。",
        "2. 对旧采集中带“空箱”提示的正类帧逐帧人工确认：有目标就保留原类别，确实无目标才人工改为 `class_id=0`。",
        "3. 对每帧生成深度候选框，只标主目标和必要遮挡目标，人工复核类别与框。",
        "4. 无目标负样本导出 YOLO 时写空 label 文件；禁止整幅图伪框。",
        "5. 最终模型仍按官方 40 类 ID 接入；第 17、26 类在正式模型完成前必须明确标为不可用，而不是 unknown。",
        "",
        "## 完整性",
        "",
        "- `SHA256SUMS` 覆盖派生的 RGB、深度、overlay、meta、清单和复核图。",
        "- 原始照片目录未修改。",
        "",
    ]
    return "\n".join(lines)


def write_audit_sidecars(build: dict[str, Any], output: Path) -> dict[str, Any]:
    summary = build["summary"]
    accepted = build["accepted"]
    excluded = build["excluded"]
    catalog = build["catalog"]
    split_info = build["split_info"]

    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    public_accepted = [public_record(record) for record in accepted]
    public_excluded = [public_record(record) for record in excluded]
    write_json(manifests / "audit_summary.json", summary)
    write_jsonl(manifests / "accepted_frames.jsonl", public_accepted)
    write_jsonl(manifests / "excluded_frames.jsonl", public_excluded)
    write_jsonl(manifests / "duplicate_frame_ids.jsonl", build["duplicate_frame_ids"])
    write_jsonl(manifests / "reused_file_conflicts.jsonl", build["reused_file_conflicts"])
    write_jsonl(manifests / "exact_duplicate_groups.jsonl", build["exact_duplicate_groups"])
    write_jsonl(manifests / "near_duplicate_pairs.jsonl", build["near_duplicate_pairs"])

    accepted_fields = [
        "record_key", "frame_id", "class_id", "capture_role", "canonical_label",
        "english_name", "display_name", "instance_id", "captured_at", "capture_group",
        "effective_capture_group", "split_unit", "split", "rgb_sha256",
        "rgb_sharpness_laplacian", "depth_valid_ratio", "warnings", "source_files",
    ]
    accepted_csv_rows = []
    for record in public_accepted:
        row = dict(record)
        row.update(record.get("quality") or {})
        accepted_csv_rows.append(row)
    write_csv(manifests / "accepted_frames.csv", accepted_csv_rows, accepted_fields)

    excluded_rows = []
    for record in public_excluded:
        excluded_rows.append(
            {
                **record,
                "exclusion_reasons": record["exclusion_reasons"],
                "missing_files": record["missing_files"],
                "unreadable_files": record["unreadable_files"],
            }
        )
    write_csv(
        manifests / "excluded_frames.csv",
        excluded_rows,
        [
            "record_key", "frame_id", "class_id", "capture_role", "canonical_label",
            "captured_at", "complete", "missing_files", "unreadable_files",
            "exclusion_reasons",
        ],
    )

    role_counter = Counter(record["capture_role"] for record in accepted)
    role_rows = [
        {
            "capture_role": role,
            "meaning": (
                "空箱/无目标负样本"
                if role == "background"
                else "物品正样本"
            ),
            "count": count,
        }
        for role, count in sorted(role_counter.items())
    ]
    write_csv(
        manifests / "capture_role_counts.csv",
        role_rows,
        ["capture_role", "meaning", "count"],
    )

    class_counter = Counter(record["class_id"] for record in accepted)
    class_rows = []
    for class_id in sorted(class_counter):
        item = catalog[class_id]
        class_rows.append(
            {
                "class_id": class_id,
                "canonical_label": item.get("canonical_label", ""),
                "english_name": (
                    BACKGROUND_ENGLISH_NAME
                    if class_id == BACKGROUND_CLASS_ID
                    else item.get("english_name", "")
                ),
                "display_name": public_display_name(item),
                "capture_role": capture_role_for(class_id),
                "accepted_frames": class_counter[class_id],
                "train_frames": sum(
                    1
                    for record in accepted
                    if record["class_id"] == class_id and record["split"] == "train"
                ),
                "val_frames": sum(
                    1
                    for record in accepted
                    if record["class_id"] == class_id and record["split"] == "val"
                ),
                "legacy_label_review_frames": sum(
                    1
                    for record in accepted
                    if record["class_id"] == class_id
                    and "legacy_positive_label_needs_manual_review" in record["warnings"]
                ),
                "usable_for_detection_now": "no_missing_bbox_labels",
            }
        )
    write_csv(
        manifests / "class_counts.csv",
        class_rows,
        [
            "class_id", "canonical_label", "english_name", "display_name",
            "capture_role", "accepted_frames", "train_frames", "val_frames",
            "legacy_label_review_frames", "usable_for_detection_now",
        ],
    )
    write_csv(
        manifests / "missing_classes.csv",
        summary["missing_classes"],
        ["class_id", "canonical_label", "english_name", "display_name", "reason", "complete_accepted_frames"],
    )

    warning_rows = []
    for record in accepted:
        for warning in record["warnings"]:
            warning_rows.append(
                {
                    "record_key": record["source_meta"],
                    "frame_id": record["frame_id"],
                    "class_id": record["class_id"],
                    "capture_role": record["capture_role"],
                    "warning": warning,
                    "policy": "保留；该警告不构成删除理由",
                }
            )
    write_csv(
        manifests / "quality_warnings.csv",
        warning_rows,
        ["record_key", "frame_id", "class_id", "capture_role", "warning", "policy"],
    )

    capture_group_counter: Counter[str] = Counter()
    capture_group_classes: dict[str, set[int]] = defaultdict(set)
    capture_group_splits: dict[str, set[str]] = defaultdict(set)
    for record in accepted:
        group = record["effective_capture_group"]
        capture_group_counter[group] += 1
        capture_group_classes[group].add(record["class_id"])
        capture_group_splits[group].add(record["split"])
    capture_rows = [
        {
            "effective_capture_group": group,
            "class_ids": sorted(capture_group_classes[group]),
            "splits": sorted(capture_group_splits[group]),
            "frame_count": count,
        }
        for group, count in sorted(capture_group_counter.items())
    ]
    write_csv(
        manifests / "capture_groups.csv",
        capture_rows,
        ["effective_capture_group", "class_ids", "splits", "frame_count"],
    )

    split_rows = [
        {
            "source_meta": record["source_meta"],
            "frame_id": record["frame_id"],
            "class_id": record["class_id"],
            "capture_role": record["capture_role"],
            "effective_capture_group": record["effective_capture_group"],
            "split_unit": record["split_unit"],
            "split": record["split"],
            "rgb": record["source_files"]["rgb"],
            "depth": record["source_files"]["depth"],
            "overlay": record["source_files"]["overlay"],
        }
        for record in sorted(
            accepted,
            key=lambda item: (
                item["class_id"],
                item["captured_at"],
                item["source_meta"],
            ),
        )
    ]
    write_csv(
        manifests / "split_manifest.csv",
        split_rows,
        [
            "source_meta", "frame_id", "class_id", "capture_role",
            "effective_capture_group", "split_unit", "split", "rgb", "depth", "overlay",
        ],
    )
    splits_dir = output / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        lines = [row["rgb"] for row in split_rows if row["split"] == split]
        (splits_dir / f"{split}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    active_classes = []
    for class_id in sorted(class_counter):
        item = dict(catalog[class_id])
        item["class_id"] = class_id
        item["capture_role"] = capture_role_for(class_id)
        item["display_name"] = public_display_name(item)
        item["accepted_frames"] = class_counter[class_id]
        active_classes.append(item)
    write_json(
        manifests / "active_classes.json",
        {
            "schema_version": 1,
            "dataset_name": DATASET_NAME,
            "official_class_ids_preserved": True,
            "background_negative_class_id": BACKGROUND_CLASS_ID,
            "temporarily_missing_class_ids": list(TEMPORARILY_MISSING_IDS),
            "active_class_ids": sorted(class_counter),
            "classes": active_classes,
        },
    )
    write_json(
        manifests / "curation_policy.json",
        {
            "schema_version": 1,
            "dataset_name": DATASET_NAME,
            "accepted_class_ids": list(ACCEPTED_IDS),
            "background_negative_class_id": BACKGROUND_CLASS_ID,
            "positive_class_ids": list(TARGET_IDS),
            "temporarily_missing_class_ids": list(TEMPORARILY_MISSING_IDS),
            "no_reindexing": True,
            "no_unknown_substitution": True,
            "scene_subdivision": "disabled",
            "capture_roles": {
                "background": "空箱/无目标负样本，class_id=0",
                "positive": "物品正样本，class_id=11-40",
            },
            "hard_exclusions": [
                "missing or unreadable RGB/depth/meta/overlay",
                "unsafe paths",
                "same file or same pixels used with different class labels",
                "extra exact duplicate with the same class label",
                "class outside class_id=0 or 11-40 scope",
            ],
            "soft_warnings_not_used_for_deletion": [
                "legacy positive label with old empty-box hint",
                "small object area / weak foreground",
                "low sharpness when image is decodable",
                "reused frame_id when files and pixels are distinct",
                "burst / near-duplicate framing",
            ],
            "object_area_filter": "disabled",
            "visual_empty_box_filter": "disabled",
            "automatic_relabel_to_background": "disabled",
            "negative_yolo_export": "empty label file with no boxes",
            "detection_bbox_annotation_status": "missing",
        },
    )

    target_class_map = manifests / "class_map.yaml"
    shutil.copy2(Path(summary["class_map"]), target_class_map)
    if sha256_file(target_class_map) != summary["class_map_sha256"]:
        raise RuntimeError("复制后的 class_map.yaml 哈希不一致")

    metadata = {
        "schema_version": 1,
        "dataset_name": DATASET_NAME,
        "created_at": summary["audit_at"],
        "source_session": summary["source_session"],
        "source_class_map": summary["class_map"],
        "class_map_sha256": summary["class_map_sha256"],
        "accepted_frames": summary["accepted_record_count"],
        "accepted_positive_frames": summary["accepted_positive_frame_count"],
        "accepted_background_negative_frames": summary["accepted_background_frame_count"],
        "excluded_frames": summary["excluded_record_count"],
        "accepted_positive_classes": summary["accepted_class_count"],
        "accepted_label_ids": summary["accepted_class_counts"],
        "background_negative_class_id": BACKGROUND_CLASS_ID,
        "temporarily_missing_class_ids": list(TEMPORARILY_MISSING_IDS),
        "scene_subdivision": "disabled",
        "split": split_info,
        "detection_training_ready": False,
        "detection_training_blocker": "全量 bbox 标注不存在；禁止整图伪框训练。",
    }
    write_json(output / "dataset_metadata.json", metadata)
    (output / "dataset_metadata.yaml").write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return metadata


def build_dataset(source: Path, output: Path, class_map_path: Path, force: bool = False) -> dict[str, Any]:
    class_map_path = class_map_path.resolve()
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise SystemExit(f"原始会话不存在：{source}")
    if not class_map_path.is_file():
        raise SystemExit(f"类别表不存在：{class_map_path}")
    if output.exists():
        if not force:
            raise SystemExit(
                f"目标已存在，为保护数据拒绝覆盖：{output}\n"
                "如需重建，请先人工确认旧目录无用并显式加 --force。"
            )
        raise SystemExit(
            "本工具不执行自动删除。请手动归档旧版本后使用新的输出版本名，避免误删。"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}-{int(time.time())}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        print(f"[1/7] 审计原始会话：{source}", flush=True)
        build = audit_records(source, class_map_path)
        accepted = build["accepted"]
        if not accepted:
            raise RuntimeError("没有找到可接受帧，停止构建")
        print(
            f"      完整 {build['summary']['complete_record_count']}，"
            f"入组 {len(accepted)}，排除 {len(build['excluded'])}",
            flush=True,
        )

        print("[2/7] 按采集组划分 train/val", flush=True)
        split_info = assign_splits(accepted, build["near_duplicate_pairs"])
        build["split_info"] = split_info
        build["summary"]["split_info"] = split_info
        print(f"      {split_info}", flush=True)

        print("[3/7] 复制有效 RGB-D 四件套", flush=True)
        copy_accepted_files(staging, source, accepted)

        print("[4/7] 写入清单与审计报告", flush=True)
        metadata = write_audit_sidecars(build, staging)
        (staging / "README.md").write_text(
            render_readme(build["summary"], split_info), encoding="utf-8"
        )
        (staging / "dataset_report.md").write_text(
            render_report(build["summary"], split_info, accepted), encoding="utf-8"
        )
        write_jsonl(staging / "frames.jsonl", [public_record(record) for record in accepted])

        print("[5/7] 生成每类复核总览图", flush=True)
        sheets = make_class_contact_sheets(staging, source, accepted, build["catalog"])
        write_jsonl(
            staging / "manifests" / "contact_sheets.jsonl",
            [{"path": path} for path in sheets],
        )
        (staging / "annotations").mkdir(exist_ok=True)
        (staging / "annotations" / "README.md").write_text(
            "# bbox 标注门禁\n\n"
            "本目录暂不写 YOLO 标签。当前只有图像级类别标签，没有目标 bbox。\n\n"
            "必须先用深度候选框/人工框标注目标，人工复核后再导出 YOLO labels。\n"
            "禁止把整幅图当作目标框训练检测器。\n",
            encoding="utf-8",
        )

        print("[6/7] 写入版本元数据并校验复制文件", flush=True)
        # Re-run the summary after split is known so audit_summary.json includes it.
        write_json(staging / "manifests" / "audit_summary.json", build["summary"])
        write_json(staging / "manifests" / "build_metadata.json", metadata)
        for record in accepted:
            for kind in FILE_KINDS:
                src = source / record["source_files"][kind]
                dst = staging / record["source_files"][kind]
                if not dst.is_file() or src.stat().st_size != dst.stat().st_size:
                    raise RuntimeError(f"复制校验失败：{dst}")
                if sha256_file(src) != sha256_file(dst):
                    raise RuntimeError(f"SHA256 复制校验失败：{dst}")

        print("[7/7] 生成 SHA256SUMS 并发布版本目录", flush=True)
        # Refresh the enriched frames file after split assignment.
        write_jsonl(staging / "frames.jsonl", [public_record(record) for record in accepted])
        file_count = write_sha256s(staging)
        # Do not include the checksum file itself in its own list, so publish after creation.
        if output.exists():
            raise RuntimeError(f"发布前发现目标目录被占用：{output}")
        staging.rename(output)
        print(f"      已发布：{output}", flush=True)
        print(f"      SHA256SUMS 覆盖文件数：{file_count}", flush=True)
        return {
            "output": str(output),
            "accepted": len(accepted),
            "excluded": len(build["excluded"]),
            "classes": build["summary"]["accepted_class_count"],
            "split": split_info,
            "sha256_entries": file_count,
        }
    except Exception:
        # Keep the failed staging directory for diagnosis; never touch source or old versions.
        print(f"构建失败，诊断目录保留在：{staging}", file=sys.stderr, flush=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="赛道一自采数据集审计/构建")
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_source = Path("/media/sunshine/新加卷/赛道一_C_D自采照片/20261027_session_001")
    default_class_map = root / "datasets/track1_selfcollected_v1/manifests/class_map.yaml"
    default_output = root / f"datasets/{DATASET_NAME}"

    audit = subparsers.add_parser("audit", help="只审计，不修改原始目录")
    audit.add_argument("--source", type=Path, default=default_source)
    audit.add_argument("--class-map", type=Path, default=default_class_map)
    audit.add_argument("--json", action="store_true", help="输出 JSON 摘要")

    build = subparsers.add_parser("build", help="构建新的派生数据集版本")
    build.add_argument("--source", type=Path, default=default_source)
    build.add_argument("--class-map", type=Path, default=default_class_map)
    build.add_argument("--output", type=Path, default=default_output)
    build.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "audit":
        build = audit_records(args.source.resolve(), args.class_map.resolve())
        summary = build["summary"]
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"原始记录：{summary['meta_record_count']}")
            print(f"完整四件套：{summary['complete_record_count']}")
            print(f"可入组：{summary['accepted_record_count']}")
            print(f"排除：{summary['excluded_record_count']}")
            print(f"覆盖类别：{summary['accepted_class_count']} / 28（缺 17、26）")
            print(f"正类帧：{summary['accepted_positive_frame_count']}")
            print(f"空箱/无目标负样本帧：{summary['accepted_background_frame_count']}")
            print(f"类别计数：{summary['accepted_class_counts']}")
            print(f"排除原因：{summary['exclusion_reason_counts']}")
            print(f"警告：{summary['warning_counts']}")
            print(f"小目标规则：{summary['small_object_policy']}")
            print(f"正类/负样本规则：{summary['capture_role_policy']}")
        return 0

    result = build_dataset(
        args.source.resolve(),
        args.output.resolve(),
        args.class_map.resolve(),
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
