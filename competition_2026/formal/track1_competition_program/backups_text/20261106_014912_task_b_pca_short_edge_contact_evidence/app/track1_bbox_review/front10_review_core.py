#!/usr/bin/env python3
"""Validation and export helpers for the self-collected front-10 dataset.

The shared review UI/export code is intentionally untouched.  This module
adds a strict front-10 path that filters the catalog to official classes 1-10,
rejects labels outside those slots, and verifies the two three-can layouts.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from capture_store import ClassCatalog
from review_core import export_reviews, load_review


FRONT10_IDS = tuple(range(1, 11))
TRIPLE_CAN_IDS = (2, 5, 10)
TRIPLE_IOU_LIMIT = 0.85
MAX_BOX_AREA_FRACTION = 0.65


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0,
        float(box[3]) - float(box[1]),
    )


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def filter_front10_catalog(
    catalog: ClassCatalog,
    allowed_class_ids: Iterable[int] = FRONT10_IDS,
) -> ClassCatalog:
    """Return an in-memory catalog containing only the allowed front classes."""
    allowed = tuple(sorted({int(value) for value in allowed_class_ids}))
    if allowed != FRONT10_IDS:
        raise ValueError(f"前10类类别集合必须为 1-10，实际为 {allowed}")
    by_id = {entry.class_id: entry for entry in catalog.entries}
    missing = [class_id for class_id in allowed if class_id not in by_id]
    if missing:
        raise ValueError(f"类别表缺少前10类：{missing}")
    entries = tuple(by_id[class_id] for class_id in allowed)
    return ClassCatalog(
        dataset_name=catalog.dataset_name,
        entries=entries,
        background=catalog.background,
        model_support=dict(catalog.model_support),
        source={
            **dict(catalog.source),
            "front10_review_filter": "official class_id 1-10 only",
        },
    )


def load_review_plan(source_root: Path) -> dict[str, Any]:
    path = source_root / "manifests" / "review_split_plan.json"
    if not path.is_file():
        raise ValueError(f"缺少前10类复核划分计划：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"复核划分计划格式错误：{path}")
    return payload


def _validated_box(
    raw: Mapping[str, Any],
    *,
    width: int,
    height: int,
    review_id: str,
    box_number: int,
) -> dict[str, Any]:
    class_id = int(raw.get("class_id", -1))
    bbox = raw.get("bbox_xyxy")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
        raise ValueError(f"{review_id} 第 {box_number} 个框缺少 bbox_xyxy")
    if len(bbox) != 4:
        raise ValueError(f"{review_id} 第 {box_number} 个框不是四坐标")
    values = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{review_id} 第 {box_number} 个框含非有限值")
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError(f"{review_id} 第 {box_number} 个框越界：{values}")
    area_fraction = box_area(values) / float(width * height)
    if area_fraction >= MAX_BOX_AREA_FRACTION:
        raise ValueError(
            f"{review_id} 第 {box_number} 个框占全图 {area_fraction:.1%}，疑似整图伪框"
        )
    return {
        "class_id": class_id,
        "bbox_xyxy": values,
        "area_fraction": area_fraction,
        "label": raw.get("label"),
    }


def validate_front10_reviews(
    *,
    source_root: Path,
    review_root: Path,
    index: Mapping[str, Any],
    required_class_ids: Iterable[int] = FRONT10_IDS,
    triple_class_ids: Iterable[int] = TRIPLE_CAN_IDS,
    expected_triple_splits: Mapping[str, str] | None = None,
    triple_iou_limit: float = TRIPLE_IOU_LIMIT,
) -> dict[str, Any]:
    """Fully validate every review before any export directory is created."""
    required = tuple(sorted({int(value) for value in required_class_ids}))
    triple_required = tuple(sorted({int(value) for value in triple_class_ids}))
    if required != FRONT10_IDS:
        raise ValueError(f"required_class_ids 必须为 1-10，实际为 {required}")
    if triple_required != TRIPLE_CAN_IDS:
        raise ValueError(f"三罐头类别必须为 2、5、10，实际为 {triple_required}")

    items = list(index.get("items") or [])
    if not items:
        raise ValueError("复核索引没有图片")
    by_id = {str(item["review_id"]): item for item in items}
    plan = load_review_plan(source_root)
    plan_triples = list(plan.get("triple_can_units") or [])
    if len(plan_triples) != 2:
        raise ValueError(f"复核计划应有两个三罐头布局，实际为 {len(plan_triples)}")
    triple_by_id = {str(item["split_unit"]): item for item in plan_triples}

    class_split_counts: dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
    }
    status_counts: Counter[str] = Counter()
    positive_item_counts = {"train": 0, "val": 0}
    empty_item_counts = {"train": 0, "val": 0}
    box_counts_by_class: Counter[int] = Counter()
    warnings: list[str] = []
    errors: list[str] = []
    reviews_by_id: dict[str, dict[str, Any]] = {}

    for item in items:
        review_id = str(item.get("review_id") or "")
        split = str(item.get("split") or "")
        if not review_id:
            errors.append("复核索引存在空 review_id")
            continue
        if split not in {"train", "val"}:
            errors.append(f"{review_id}: split 无效：{split!r}")
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            errors.append(f"{review_id}: 图片尺寸无效：{width}x{height}")
            continue
        review = load_review(review_root, review_id)
        if review is None:
            status_counts["missing"] += 1
            errors.append(f"{review_id}: 复核记录不存在")
            continue
        status = str(review.get("status") or "")
        status_counts[status] += 1
        if status not in {"labeled", "empty"}:
            errors.append(f"{review_id}: 状态为 {status or '空'}，必须完成复核")
            continue
        raw_boxes = review.get("boxes", [])
        if not isinstance(raw_boxes, list):
            errors.append(f"{review_id}: boxes 不是列表")
            continue
        if status == "empty":
            if raw_boxes:
                errors.append(f"{review_id}: 空背景却含标注框")
            empty_item_counts[split] += 1
            reviews_by_id[review_id] = review
            continue
        if not raw_boxes:
            errors.append(f"{review_id}: 有目标状态但没有框")
            continue
        positive_item_counts[split] += 1
        validated_boxes: list[dict[str, Any]] = []
        for box_number, raw_box in enumerate(raw_boxes, start=1):
            if not isinstance(raw_box, Mapping):
                errors.append(f"{review_id}: 第 {box_number} 个框不是对象")
                continue
            try:
                box = _validated_box(
                    raw_box,
                    width=width,
                    height=height,
                    review_id=review_id,
                    box_number=box_number,
                )
            except (TypeError, ValueError) as error:
                errors.append(str(error))
                continue
            class_id = int(box["class_id"])
            if class_id not in required:
                errors.append(
                    f"{review_id}: 出现前10类之外的官方类别 {class_id}"
                )
                continue
            validated_boxes.append(box)
            class_split_counts[split][class_id] += 1
            box_counts_by_class[class_id] += 1

        for first_index, first in enumerate(validated_boxes):
            for second in validated_boxes[first_index + 1 :]:
                if int(first["class_id"]) != int(second["class_id"]):
                    continue
                iou = box_iou(first["bbox_xyxy"], second["bbox_xyxy"])
                if iou >= 0.85:
                    warnings.append(
                        f"{review_id}: 同类框高度重叠 IoU={iou:.3f}"
                    )
        reviews_by_id[review_id] = review

    missing_ids = sorted(set(by_id) - set(reviews_by_id))
    if missing_ids:
        warnings.append(f"尚未形成有效导出的复核项：{missing_ids[:10]}")

    for split in ("train", "val"):
        missing_classes = [
            class_id
            for class_id in required
            if class_split_counts[split][class_id] == 0
        ]
        if missing_classes:
            errors.append(f"{split}: 缺少类别正标签：{missing_classes}")

    triple_audits: list[dict[str, Any]] = []
    for plan_item in plan_triples:
        review_id = str(plan_item.get("split_unit") or "")
        expected_split = str(plan_item.get("split") or "")
        physical_group = str(plan_item.get("physical_group") or "")
        required_for_triple = tuple(
            sorted(
                int(value)
                for value in plan_item.get("required_classes_for_bbox", triple_required)
            )
        )
        item = by_id.get(review_id)
        review = reviews_by_id.get(review_id)
        if item is None:
            errors.append(f"{review_id}: 三罐头复核项不在索引中")
            continue
        if str(item.get("split")) != expected_split:
            errors.append(
                f"{review_id}: 三罐头 split 不一致，索引={item.get('split')}，计划={expected_split}"
            )
        if expected_triple_splits and expected_triple_splits.get(physical_group) != expected_split:
            errors.append(
                f"{review_id}: 三罐头布局 {physical_group} 的 split 与配置不一致"
            )
        if str(item.get("selected_frame_id")) != str(plan_item.get("frame_id")):
            errors.append(
                f"{review_id}: 三罐头代表帧不一致，索引={item.get('selected_frame_id')}，计划={plan_item.get('frame_id')}"
            )
        if review is None or str(review.get("status")) != "labeled":
            errors.append(f"{review_id}: 三罐头图片必须标记为有目标并完成复核")
            continue
        raw_boxes = review.get("boxes", [])
        boxes: list[dict[str, Any]] = []
        for box_number, raw_box in enumerate(raw_boxes, start=1):
            try:
                box = _validated_box(
                    raw_box,
                    width=int(item["width"]),
                    height=int(item["height"]),
                    review_id=review_id,
                    box_number=box_number,
                )
            except (TypeError, ValueError) as error:
                errors.append(str(error))
                continue
            if int(box["class_id"]) not in required_for_triple:
                errors.append(
                    f"{review_id}: 三罐头图出现额外类别 {box['class_id']}；本图应只有 2、5、10 三个实例"
                )
            else:
                boxes.append(box)
        class_box_counts = Counter(int(box["class_id"]) for box in boxes)
        expected_counts = {class_id: 1 for class_id in required_for_triple}
        if dict(sorted(class_box_counts.items())) != expected_counts:
            errors.append(
                f"{review_id}: 三罐头应各标 1 个框，实际类别计数为 {dict(class_box_counts)}"
            )
        pair_ious: dict[str, float] = {}
        for first_index, first_class in enumerate(required_for_triple):
            for second_class in required_for_triple[first_index + 1 :]:
                first_box = next(
                    (box for box in boxes if int(box["class_id"]) == first_class),
                    None,
                )
                second_box = next(
                    (box for box in boxes if int(box["class_id"]) == second_class),
                    None,
                )
                if first_box is None or second_box is None:
                    continue
                iou = box_iou(first_box["bbox_xyxy"], second_box["bbox_xyxy"])
                pair_ious[f"{first_class}-{second_class}"] = round(iou, 6)
                if iou >= triple_iou_limit:
                    errors.append(
                        f"{review_id}: 类别 {first_class} 与 {second_class} 的框高度重叠 IoU={iou:.3f}"
                    )
        max_iou = max(pair_ious.values(), default=0.0)
        triple_audits.append(
            {
                "review_id": review_id,
                "split": expected_split,
                "physical_group": physical_group,
                "frame_id": str(plan_item.get("frame_id") or ""),
                "required_class_ids": list(required_for_triple),
                "class_box_counts": {
                    str(key): value for key, value in sorted(class_box_counts.items())
                },
                "pairwise_iou": pair_ious,
                "max_pairwise_iou": max_iou,
                "passed": dict(sorted(class_box_counts.items())) == expected_counts
                and max_iou < triple_iou_limit,
            }
        )

    if len(triple_audits) != 2 or not all(item["passed"] for item in triple_audits):
        if len(triple_audits) != 2:
            errors.append(f"三罐头审计数量应为 2，实际为 {len(triple_audits)}")

    if errors:
        raise ValueError(
            "前10类复核审计失败（前 30 条）：\n- " + "\n- ".join(errors[:30])
        )

    return {
        "item_count": len(items),
        "reviewed_count": sum(1 for review in reviews_by_id.values() if review.get("status") in {"labeled", "empty"}),
        "positive_item_counts": positive_item_counts,
        "empty_item_counts": empty_item_counts,
        "status_counts": dict(status_counts),
        "class_split_counts": {
            split: {str(key): value for key, value in sorted(counts.items())}
            for split, counts in class_split_counts.items()
        },
        "box_counts_by_class_id": {
            str(key): value for key, value in sorted(box_counts_by_class.items())
        },
        "box_count": sum(box_counts_by_class.values()),
        "triple_can_audits": triple_audits,
        "warnings": warnings,
        "required_class_ids": list(required),
        "triple_can_class_ids": list(triple_required),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
        records.append(payload)
    return records


def export_front10_reviews(
    *,
    source_root: Path,
    review_root: Path,
    export_root: Path,
    index: Mapping[str, Any],
    catalog: ClassCatalog,
    path_catalog_id: str = "front10",
    expected_triple_splits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Export a strictly front-10 YOLO dataset and its merge manifest."""
    front_catalog = filter_front10_catalog(catalog)
    validation = validate_front10_reviews(
        source_root=source_root,
        review_root=review_root,
        index=index,
        expected_triple_splits=expected_triple_splits,
    )
    generic_result = export_reviews(
        source_root=source_root,
        review_root=review_root,
        export_root=export_root,
        index=index,
        catalog=front_catalog,
        missing_spot_class_ids=(),
    )
    output = Path(str(generic_result["export_root"])).resolve()
    records = _load_jsonl(output / "review_export_manifest.jsonl")
    if len(records) != int(index["item_count"]):
        raise ValueError(
            f"导出图片数 {len(records)} 与复核项 {index['item_count']} 不一致"
        )

    expected_names = [entry.canonical_label for entry in front_catalog.entries]
    dataset_path = output / "dataset.yaml"
    dataset_payload = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_payload, dict):
        raise ValueError(f"dataset.yaml 格式错误：{dataset_path}")
    names = dataset_payload.get("names")
    if isinstance(names, dict):
        names = [str(names[index]) for index in sorted(names)]
    if names != expected_names:
        raise ValueError(f"前10类导出 names 不正确：{names}")
    dataset_payload["nc"] = len(names)
    dataset_path.write_text(
        yaml.safe_dump(dataset_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    manifest_path = output / "manifest.csv"
    group_splits: dict[str, set[str]] = defaultdict(set)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "physical_group", "image"],
        )
        writer.writeheader()
        for record in records:
            split = str(record["split"])
            physical_group = str(record["review_id"])
            if split not in {"train", "val"}:
                raise ValueError(f"导出记录 split 无效：{split}")
            group_splits[physical_group].add(split)
            writer.writerow(
                {
                    "split": split,
                    "physical_group": physical_group,
                    "image": str(record["image"]),
                }
            )
    crossing = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    if crossing:
        raise ValueError(f"前10类 physical_group 跨 train/val：{crossing}")

    for label_path in (output / "labels").glob("*/*.txt"):
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not raw_line.strip():
                continue
            parts = raw_line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_path}:{line_number} 不是 YOLO 5 列")
            yolo_index = int(parts[0])
            if not 0 <= yolo_index <= 9:
                raise ValueError(
                    f"{label_path}:{line_number} 出现前10类之外的 YOLO 索引 {yolo_index}"
                )

    names = [str(value) for value in expected_names]
    audit = {
        "schema_version": 1,
        "created_at": now_iso(),
        "scope": "赛道一前10类新自采 bbox 专用导出",
        "source_root": str(source_root),
        "review_root": str(review_root),
        "export_root": str(output),
        "dataset_yaml": str(dataset_path),
        "manifest_csv": str(manifest_path),
        "item_count": validation["item_count"],
        "exported_image_count": len(records),
        "reviewed_count": validation["reviewed_count"],
        "positive_image_counts": validation["positive_item_counts"],
        "empty_background_counts": validation["empty_item_counts"],
        "status_counts": validation["status_counts"],
        "box_count": validation["box_count"],
        "box_counts_by_class_id": validation["box_counts_by_class_id"],
        "class_split_counts": validation["class_split_counts"],
        "nc": len(names),
        "names": names,
        "label_mapping": "official class_id 1-10 -> YOLO index 0-9",
        "uses_official_front10_images": False,
        "all_front10_classes_have_train_and_val_positives": True,
        "triple_can_audits": validation["triple_can_audits"],
        "triple_can_layouts_passed": len(validation["triple_can_audits"]) == 2
        and all(item["passed"] for item in validation["triple_can_audits"]),
        "warnings": validation["warnings"],
        "front10_merge_ready": True,
        "training_ready": False,
        "training_ready_reason": "前10类导出已就绪；还必须与后28类 bbox 导出合并并通过 40 槽审计。",
        "path_catalog_id": path_catalog_id,
    }
    (output / "audit_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = f"""# 赛道一前10类新自采 bbox 导出

- 来源：`{source_root}`
- 复核：`{review_root}`
- 导出图片：{len(records)}
- bbox 总数：{validation['box_count']}
- 类别槽位：10（官方 ID 1-10 对应 YOLO 0-9）
- 三罐头布局：2/2 通过
- 前10类导出可进入合并：True
- 最终训练就绪：False
- 官方前10类图片：未使用

## 必须遵守的规则

1. 三个罐头同框图分别标注 2-Starkist_Tuna、5-Tomato_Soup 和 10-meat_can，
   每个实例一个框，不能画整组大框。
2. 空背景只写空 `.txt`，不能创建第 41 类或 `box_11L_empty`。
3. 这个导出只包含前10类，不是最终的 40 槽训练集；下一步必须与后28类导出合并。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {
        "ok": True,
        "export_root": str(output),
        "audit": audit,
    }
