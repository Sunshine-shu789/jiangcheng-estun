#!/usr/bin/env python3
"""Core storage and export helpers for Track-1 multi-object bbox review.

The capture class is only a hint.  One reviewed image may contain any number of
boxes and classes.  Explicitly reviewed background images are exported as
empty YOLO label files, never as a synthetic class.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml

from capture_store import CatalogEntry, ClassCatalog, load_class_catalog


REVIEW_STATUSES = {"labeled", "empty", "skipped"}
MULTI_OBJECT_HINT_ID = 99
MULTI_OBJECT_HINT_LABEL = "99-Multi_Object"
MULTI_OBJECT_HINT_DISPLAY_NAME = "多物品混放"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML 根节点必须是对象：{path}")
    return payload


def _resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _image_sharpness(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return -math.inf
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    brightness = float(image.mean())
    penalty = 0.0
    if brightness < 35:
        penalty += (35.0 - brightness) * 5.0
    elif brightness > 225:
        penalty += (brightness - 225.0) * 5.0
    return sharpness - penalty


def _select_representative(
    rows: Sequence[dict[str, str]],
    source_root: Path,
) -> dict[str, str]:
    scored: list[tuple[float, int, dict[str, str]]] = []
    middle = len(rows) // 2
    for index, row in enumerate(rows):
        image_path = source_root / row["rgb"]
        score = _image_sharpness(image_path)
        # Prefer the middle frame only as a deterministic tie-breaker.
        tie_break = -abs(index - middle)
        scored.append((score, tie_break, row))
    scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
    return scored[0][2]


def build_review_index(
    *,
    source_root: Path,
    class_map_path: Path,
    review_root: Path,
) -> dict[str, Any]:
    """Select one representative frame from every existing split unit.

    The original train/val split is preserved exactly.  No burst is split
    across train and validation by this function.
    """
    source_root = source_root.resolve()
    class_map_path = class_map_path.resolve()
    review_root = review_root.resolve()
    split_path = source_root / "manifests" / "split_manifest.csv"
    if not split_path.is_file():
        raise ValueError(f"缺少划分清单：{split_path}")
    catalog = load_class_catalog(class_map_path)
    by_id = {entry.class_id: entry for entry in catalog.entries}
    if MULTI_OBJECT_HINT_ID in by_id:
        raise ValueError(f"官方类别不得占用采集提示 ID {MULTI_OBJECT_HINT_ID}")
    by_id[MULTI_OBJECT_HINT_ID] = CatalogEntry(
        class_id=MULTI_OBJECT_HINT_ID,
        canonical_label=MULTI_OBJECT_HINT_LABEL,
        english_name="Multi Object",
        display_name=MULTI_OBJECT_HINT_DISPLAY_NAME,
        subset=None,
        model_class_names=(),
        is_collection_hint=True,
        capture_hint_kind="multi_object",
    )

    with split_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("split_manifest.csv 为空")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        split_unit = str(row.get("split_unit") or "").strip()
        if not split_unit:
            raise ValueError("split_manifest.csv 存在空 split_unit")
        grouped[split_unit].append(row)

    items: list[dict[str, Any]] = []
    for split_unit in sorted(grouped, key=_natural_unit_key):
        unit_rows = grouped[split_unit]
        splits = {str(row.get("split") or "") for row in unit_rows}
        class_ids = {int(row["class_id"]) for row in unit_rows}
        if len(splits) != 1 or "" in splits:
            raise ValueError(f"{split_unit} 的 train/val 划分不一致：{splits}")
        if len(class_ids) != 1:
            raise ValueError(f"{split_unit} 的采集提示类别不一致：{class_ids}")
        hint_class_id = next(iter(class_ids))
        if hint_class_id not in by_id:
            raise ValueError(f"{split_unit} 的类别编号无效：{hint_class_id}")
        representative = _select_representative(unit_rows, source_root)
        frame_id = str(representative.get("frame_id") or "")
        if not frame_id:
            raise ValueError(f"{split_unit} 的代表帧缺少 frame_id")
        meta_path = source_root / representative["source_meta"]
        if not meta_path.is_file():
            raise ValueError(f"代表帧元数据不存在：{meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"代表帧尺寸无效：{meta_path}")

        entry = by_id[hint_class_id]
        items.append(
            {
                "review_id": split_unit,
                "split_unit": split_unit,
                "split": next(iter(splits)),
                "hint_class_id": hint_class_id,
                "hint_class_label": entry.canonical_label,
                "hint_display_name": entry.display_name,
                "selected_frame_id": frame_id,
                "source_meta": representative["source_meta"],
                "source_rgb": representative["rgb"],
                "source_depth": representative["depth"],
                "width": width,
                "height": height,
                "frame_count": len(unit_rows),
                "frame_ids": [str(row["frame_id"]) for row in unit_rows],
            }
        )

    counts = Counter(item["split"] for item in items)
    hint_counts = Counter(str(item["hint_class_id"]) for item in items)
    index = {
        "schema_version": 1,
        "created_at": now_iso(),
        "source_root": str(source_root),
        "class_map": str(class_map_path),
        "review_root": str(review_root),
        "selection_policy": (
            "每个既有 split_unit 选 1 张清晰度最高的代表帧；"
            "split_unit 的 train/val 归属原样保留。"
        ),
        "multi_object_policy": (
            "文件名/采集类别只是提示；一张图必须标出全部可辨认的已知物品，"
            "同类多实例分别标框。"
        ),
        "item_count": len(items),
        "split_counts": dict(sorted(counts.items())),
        "hint_class_counts": dict(sorted(hint_counts.items(), key=lambda x: int(x[0]))),
        "items": items,
    }
    atomic_write_json(review_root / "index.json", index)
    return index


def _natural_unit_key(value: str) -> tuple[str, int, str]:
    prefix, _, suffix = value.rpartition("-")
    try:
        return prefix, int(suffix), value
    except ValueError:
        return value, 0, value


def load_review_index(review_root: Path) -> dict[str, Any]:
    path = review_root / "index.json"
    if not path.is_file():
        raise ValueError(f"复核索引不存在：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"复核索引格式错误：{path}")
    return payload


def review_path(review_root: Path, review_id: str) -> Path:
    return review_root / "reviews" / f"{review_id}.json"


def load_review(review_root: Path, review_id: str) -> dict[str, Any] | None:
    path = review_path(review_root, review_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0,
        float(box[3]) - float(box[1]),
    )


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def validate_review_payload(
    *,
    payload: Mapping[str, Any],
    item: Mapping[str, Any],
    catalog: ClassCatalog,
) -> dict[str, Any]:
    status = str(payload.get("status") or "").strip()
    if status not in REVIEW_STATUSES:
        raise ValueError("status 必须是 labeled、empty 或 skipped")
    raw_boxes = payload.get("boxes", [])
    if not isinstance(raw_boxes, list):
        raise ValueError("boxes 必须是列表")
    width = int(item["width"])
    height = int(item["height"])
    class_ids = {entry.class_id for entry in catalog.entries}
    boxes: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for index, raw_box in enumerate(raw_boxes, start=1):
        if not isinstance(raw_box, dict):
            raise ValueError(f"第 {index} 个框不是对象")
        class_id = int(raw_box.get("class_id", 0))
        if class_id not in class_ids:
            raise ValueError(f"第 {index} 个框类别编号无效：{class_id}")
        bbox = raw_box.get("bbox_xyxy")
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise ValueError(f"第 {index} 个框缺少 bbox_xyxy")
        values = [float(value) for value in bbox]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"第 {index} 个框坐标不是有限数值")
        x1 = min(max(values[0], 0.0), float(width))
        y1 = min(max(values[1], 0.0), float(height))
        x2 = min(max(values[2], 0.0), float(width))
        y2 = min(max(values[3], 0.0), float(height))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            raise ValueError(f"第 {index} 个框小于 2 像素")
        rounded = tuple(round(value * 10) for value in (x1, y1, x2, y2))
        signature = (class_id, *rounded)
        if signature in seen:
            raise ValueError(f"第 {index} 个框与已有框完全重复")
        seen.add(signature)
        boxes.append(
            {
                "class_id": class_id,
                "class_label": catalog.by_id(class_id).canonical_label,
                "bbox_xyxy": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
                "source": str(raw_box.get("source") or "manual")[:32],
            }
        )

    if status == "labeled" and not boxes:
        raise ValueError("labeled 状态至少需要一个框；空画面请选择 empty")
    if status == "empty" and boxes:
        raise ValueError("empty 状态不能带标注框")
    if status == "skipped" and boxes:
        raise ValueError("skipped 状态不能带标注框")

    return {
        "schema_version": 1,
        "review_id": str(item["review_id"]),
        "source_frame_id": str(item["selected_frame_id"]),
        "split": str(item["split"]),
        "hint_class_id": int(item["hint_class_id"]),
        "status": status,
        "reviewed_at": now_iso(),
        "boxes": boxes,
        "notes": str(payload.get("notes") or "")[:2000],
        "policy_note": (
            "采集提示类别不是本图唯一标签；本记录已按同框全部可辨认已知物品复核。"
        ),
    }


def render_review_overlay(
    *,
    source_root: Path,
    item: Mapping[str, Any],
    review: Mapping[str, Any],
    catalog: ClassCatalog,
    output_path: Path,
) -> None:
    image_path = source_root / str(item["source_rgb"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取代表帧：{image_path}")
    height, width = image.shape[:2]
    colors = (
        (80, 220, 150),
        (85, 180, 255),
        (255, 190, 80),
        (235, 120, 240),
        (100, 220, 240),
        (255, 230, 90),
    )
    for index, box in enumerate(review.get("boxes", [])):
        x1, y1, x2, y2 = (int(round(float(v))) for v in box["bbox_xyxy"])
        class_id = int(box["class_id"])
        entry = catalog.by_id(class_id)
        color = colors[index % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_id}-{entry.english_name}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        top = max(0, y1 - text_height - baseline - 6)
        right = min(width - 1, x1 + text_width + 8)
        cv2.rectangle(image, (x1, top), (right, top + text_height + baseline + 6), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 4, top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (15, 20, 25),
            1,
            cv2.LINE_AA,
        )
    if review.get("status") == "empty":
        cv2.rectangle(image, (0, 0), (width - 1, 38), (35, 80, 150), -1)
        cv2.putText(
            image,
            "REVIEWED EMPTY: no known objects",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 250, 255),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise ValueError(f"无法写入复核预览：{output_path}")


def save_review(
    *,
    review_root: Path,
    source_root: Path,
    item: Mapping[str, Any],
    payload: Mapping[str, Any],
    catalog: ClassCatalog,
) -> dict[str, Any]:
    review = validate_review_payload(payload=payload, item=item, catalog=catalog)
    path = review_path(review_root, str(item["review_id"]))
    atomic_write_json(path, review)
    render_review_overlay(
        source_root=source_root,
        item=item,
        review=review,
        catalog=catalog,
        output_path=review_root / "overlays" / f"{item['review_id']}.jpg",
    )
    return review


def review_summary(
    *,
    review_root: Path,
    index: Mapping[str, Any],
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    box_counts: Counter[int] = Counter()
    items: list[dict[str, Any]] = []
    for item in index["items"]:
        review = load_review(review_root, str(item["review_id"]))
        status = str(review.get("status")) if review else "pending"
        status_counts[status] += 1
        boxes = review.get("boxes", []) if review else []
        for box in boxes:
            box_counts[int(box["class_id"])] += 1
        items.append(
            {
                "review_id": item["review_id"],
                "split": item["split"],
                "hint_class_id": item["hint_class_id"],
                "hint_display_name": item["hint_display_name"],
                "selected_frame_id": item["selected_frame_id"],
                "frame_count": item["frame_count"],
                "status": status,
                "box_count": len(boxes),
            }
        )
    total = len(items)
    reviewed = status_counts["labeled"] + status_counts["empty"]
    return {
        "total": total,
        "reviewed": reviewed,
        "pending": status_counts["pending"],
        "labeled": status_counts["labeled"],
        "empty": status_counts["empty"],
        "skipped": status_counts["skipped"],
        "completion": reviewed / total if total else 0.0,
        "status_counts": dict(status_counts),
        "box_counts_by_class_id": dict(sorted(box_counts.items())),
        "items": items,
    }


def _yolo_line(box: Mapping[str, Any], width: int, height: int) -> str:
    x1, y1, x2, y2 = (float(v) for v in box["bbox_xyxy"])
    center_x = (x1 + x2) * 0.5 / width
    center_y = (y1 + y2) * 0.5 / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    class_index = int(box["class_id"]) - 1
    return (
        f"{class_index} {center_x:.6f} {center_y:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )


def _next_empty_export_dir(path: Path) -> Path:
    if not path.exists():
        return path
    if path.is_dir() and not any(path.iterdir()):
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}_{stamp}")
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{stamp}_{suffix}")
        suffix += 1
    return candidate


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def export_reviews(
    *,
    source_root: Path,
    review_root: Path,
    export_root: Path,
    index: Mapping[str, Any],
    catalog: ClassCatalog,
    missing_spot_class_ids: Iterable[int] = (17, 26),
) -> dict[str, Any]:
    output = _next_empty_export_dir(export_root.resolve())
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary = review_summary(review_root=review_root, index=index)
    by_id = {str(item["review_id"]): item for item in index["items"]}
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    class_split_counts: dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
    }
    empty_count = 0
    total_boxes = 0

    for summary_item in summary["items"]:
        review_id = str(summary_item["review_id"])
        item = by_id[review_id]
        review = load_review(review_root, review_id)
        if review is None or review.get("status") not in {"labeled", "empty"}:
            continue
        split = str(item["split"])
        if split not in {"train", "val"}:
            raise ValueError(f"{review_id} 的 split 无效：{split}")
        source_image = source_root / str(item["source_rgb"])
        if not source_image.is_file():
            raise ValueError(f"缺少代表帧：{source_image}")
        suffix = source_image.suffix.lower() or ".png"
        stem = f"{review_id}__{item['selected_frame_id']}"
        target_image = output / "images" / split / f"{stem}{suffix}"
        target_label = output / "labels" / split / f"{stem}.txt"
        _link_or_copy(source_image, target_image)

        boxes = list(review.get("boxes", []))
        if review.get("status") == "empty":
            if boxes:
                raise ValueError(f"{review_id} 标记空背景但存在标注框")
            empty_count += 1
            label_text = ""
        else:
            if not boxes:
                raise ValueError(f"{review_id} 标记有目标但不存在标注框")
            lines: list[str] = []
            for box in boxes:
                lines.append(_yolo_line(box, int(item["width"]), int(item["height"])))
                class_id = int(box["class_id"])
                class_split_counts[split][class_id] += 1
            label_text = "\n".join(lines) + "\n"
        target_label.write_text(label_text, encoding="utf-8")
        total_boxes += len(boxes)
        records.append(
            {
                "review_id": review_id,
                "split": split,
                "source_frame_id": item["selected_frame_id"],
                "source_rgb": str(source_image),
                "image": str(target_image.relative_to(output)),
                "label": str(target_label.relative_to(output)),
                "capture_hint_class_id": item["hint_class_id"],
                "status": review.get("status"),
                "box_count": len(boxes),
                "boxes": boxes,
            }
        )

        for box_index, first in enumerate(boxes):
            for second in boxes[box_index + 1 :]:
                if int(first["class_id"]) != int(second["class_id"]):
                    continue
                iou = _box_iou(first["bbox_xyxy"], second["bbox_xyxy"])
                if iou >= 0.85:
                    warnings.append(
                        f"{review_id}: 同类框高度重叠 IoU={iou:.3f}"
                    )
            area_fraction = _box_area(first["bbox_xyxy"]) / (
                int(item["width"]) * int(item["height"])
            )
            if area_fraction >= 0.65:
                warnings.append(
                    f"{review_id}: 框占全图 {area_fraction:.1%}，疑似整图伪框"
                )

    names = [entry.canonical_label for entry in catalog.entries]
    data_yaml = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    manifest_path = output / "review_export_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    completed_ids = {record["review_id"] for record in records}
    unreviewed_ids = [
        item["review_id"]
        for item in index["items"]
        if item["review_id"] not in completed_ids
    ]
    skipped_ids = [
        str(item["review_id"])
        for item in index["items"]
        if (load_review(review_root, str(item["review_id"])) or {}).get("status")
        == "skipped"
    ]
    labeled_classes = sorted(
        set(class_split_counts["train"]) | set(class_split_counts["val"])
    )
    train_classes = set(class_split_counts["train"])
    val_classes = set(class_split_counts["val"])
    required_postclass_ids = sorted(
        {
            int(item["hint_class_id"])
            for item in index["items"]
            if 1 <= int(item["hint_class_id"]) <= 40
        }
    )
    missing_required = [
        class_id
        for class_id in required_postclass_ids
        if class_id not in labeled_classes
    ]
    required_without_train = [
        class_id
        for class_id in required_postclass_ids
        if class_id not in train_classes
    ]
    required_without_val = [
        class_id
        for class_id in required_postclass_ids
        if class_id not in val_classes
    ]
    no_train = sorted(val_classes - train_classes)
    no_val = sorted(train_classes - val_classes)
    missing_spots = sorted({int(value) for value in missing_spot_class_ids})
    forbidden_labeled = sorted(set(missing_spots) & set(labeled_classes))
    all_official_missing = [
        class_id for class_id in range(1, 41) if class_id not in labeled_classes
    ]
    front_class_ids_present = sorted(set(range(1, 11)) & set(labeled_classes))
    complete = len(unreviewed_ids) == 0
    has_whole_image_warning = any(
        "疑似整图伪框" in warning for warning in warnings
    )
    merge_ready = bool(
        complete
        and total_boxes > 0
        and not missing_required
        and not required_without_train
        and not required_without_val
        and not forbidden_labeled
        and not has_whole_image_warning
    )
    audit = {
        "schema_version": 1,
        "created_at": now_iso(),
        "source_root": str(source_root),
        "review_root": str(review_root),
        "export_root": str(output),
        "item_count": int(index["item_count"]),
        "exported_image_count": len(records),
        "empty_background_count": empty_count,
        "box_count": total_boxes,
        "unreviewed_count": len(unreviewed_ids),
        "unreviewed_ids": unreviewed_ids,
        "skipped_count": len(skipped_ids),
        "skipped_ids": skipped_ids,
        "complete_review": complete,
        "postclasses_merge_ready": merge_ready,
        "training_ready": False,
        "training_ready_reason": (
            "当前文件仅为后类 bbox 导出；最终训练还必须合并官方/自采前 10 类 bbox，"
            "并再次执行合并审计。"
        ),
        "class_split_counts": {
            split: {str(key): value for key, value in sorted(counts.items())}
            for split, counts in class_split_counts.items()
        },
        "labeled_class_ids": labeled_classes,
        "required_postclass_class_ids": required_postclass_ids,
        "missing_required_postclass_class_ids": missing_required,
        "required_postclass_without_train": required_without_train,
        "required_postclass_without_val": required_without_val,
        "classes_without_train": no_train,
        "classes_without_val": no_val,
        "missing_spot_class_ids": missing_spots,
        "forbidden_labeled_class_ids": forbidden_labeled,
        "all_official_missing_class_ids": all_official_missing,
        "front_class_ids_present_as_extra_objects": front_class_ids_present,
        "warnings": warnings,
        "label_mapping": "official class_id 1-40 -> YOLO index 0-39",
        "empty_label_policy": "empty 状态输出 0 字节 .txt，不创建空背景类别",
        "scope": "仅包含已有后类照片的多物品 bbox；不是最终 40 槽训练集",
    }
    (output / "audit_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = f"""# Track-1 后类多物品 bbox 导出

- 来源：`{source_root}`
- 复核：`{review_root}`
- 导出图片：{len(records)}
- 空背景负样本：{empty_count}
- bbox 总数：{total_boxes}
- 未复核/未导出：{len(unreviewed_ids)}
- 跳过：{len(skipped_ids)}
- 完整复核：{complete}
- 后类导出可进入合并：{merge_ready}
- 最终训练就绪：False
- 类别映射：官方 ID 1-40 对应 YOLO 0-39
- 空背景：空 `.txt`，不设置第 41 类

这不是最终训练集。必须把本导出与前 10 类自采 bbox 合并，保留 40 个
YOLO 类别槽位（17、26 暂为空槽），并再次通过逐类 train/val 审计，
才能开始最终训练。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {
        "ok": True,
        "export_root": str(output),
        "audit": audit,
    }
