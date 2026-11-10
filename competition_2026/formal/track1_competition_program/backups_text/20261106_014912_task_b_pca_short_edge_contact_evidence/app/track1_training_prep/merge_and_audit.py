#!/usr/bin/env python3
"""Build and audit the 40-slot Track-1 training dataset without copying pixels."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

SPLITS = ("train", "val")
MISSING_SPOT_INDICES = {16, 25}  # official class IDs 17 and 26 -> YOLO indices
EXPECTED_ACTIVE_INDICES = set(range(40)) - MISSING_SPOT_INDICES
POST_ACTIVE_INDICES = set(range(10, 40)) - MISSING_SPOT_INDICES
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_PROJECT_ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")


class PrepError(RuntimeError):
    pass


@dataclass(frozen=True)
class Record:
    source_kind: str
    source_split: str
    source_image: Path
    source_label: Path
    output_image_rel: str
    output_label_rel: str
    group: str
    class_ids: tuple[int, ...]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PrepError(f"缺少 YAML：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PrepError(f"YAML 顶层不是字典：{path}")
    return data


def parse_names(payload: dict[str, Any], path: Path) -> list[str]:
    names = payload.get("names")
    if isinstance(names, dict):
        try:
            ordered = [str(names[i]) for i in range(len(names))]
        except KeyError as exc:
            raise PrepError(f"{path} 的 names 不是连续索引") from exc
    elif isinstance(names, list):
        ordered = [str(value) for value in names]
    else:
        raise PrepError(f"{path} 缺少 names 列表")
    if not ordered:
        raise PrepError(f"{path} 的 names 为空")
    return ordered


def parse_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    errors: list[str] = []
    rows: list[tuple[int, float, float, float, float]] = []
    if not path.is_file():
        return rows, [f"标签不存在：{path}"]
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{line_no} 不是 5 列")
            continue
        try:
            class_id = int(parts[0])
            x, y, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{line_no} 含非数字")
            continue
        if not 0 <= class_id < 40:
            errors.append(f"{path.name}:{line_no} 类别索引越界：{class_id}")
            continue
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            errors.append(f"{path.name}:{line_no} 坐标不在 0..1：{x,y,width,height}")
            continue
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{path.name}:{line_no} 宽高必须大于 0")
            continue
        eps = 1e-6
        if x - width / 2 < -eps or x + width / 2 > 1 + eps or y - height / 2 < -eps or y + height / 2 > 1 + eps:
            errors.append(f"{path.name}:{line_no} 框超出图片边界")
            continue
        if width >= 0.98 and height >= 0.98:
            errors.append(f"{path.name}:{line_no} 疑似整图伪框：w={width:.4f}, h={height:.4f}")
            continue
        rows.append((class_id, x, y, width, height))
    return rows, errors


def discover_split(root: Path, split: str) -> tuple[dict[str, Path], dict[str, Path]]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.is_dir():
        raise PrepError(f"图片目录不存在：{image_dir}")
    if not label_dir.is_dir():
        raise PrepError(f"标签目录不存在：{label_dir}")
    images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    labels = {p.stem: p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
    image_names = [p.name for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    label_names = [p.name for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    if len(image_names) != len(images):
        raise PrepError(f"{image_dir} 存在重复 stem")
    if len(label_names) != len(labels):
        raise PrepError(f"{label_dir} 存在重复 stem")
    missing_labels = sorted(set(images) - set(labels))
    extra_labels = sorted(set(labels) - set(images))
    if missing_labels or extra_labels:
        raise PrepError(
            f"{root.name}/{split} 图片标签不配对；缺标签 {missing_labels[:5]}，"
            f"多标签 {extra_labels[:5]}，数量 {len(missing_labels)}/{len(extra_labels)}"
        )
    return images, labels


def load_front_manifest(front_root: Path, images_by_split: dict[str, dict[str, Path]]) -> dict[tuple[str, str], str]:
    manifest = front_root / "manifest.csv"
    if not manifest.is_file():
        raise PrepError(f"前 10 类缺少 manifest.csv：{manifest}")
    import csv

        # utf-8-sig handles the BOM in the existing manifest.
    groups: dict[tuple[str, str], str] = {}
    seen_images: set[tuple[str, str]] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise PrepError("前 10 类 manifest.csv 没有表头")
        required = {"split", "physical_group", "image"}
        if not required.issubset(set(reader.fieldnames)):
            raise PrepError(f"前 10 类 manifest.csv 缺少字段：{sorted(required - set(reader.fieldnames))}")
        for row in reader:
            split = str(row["split"]).strip()
            group = str(row["physical_group"]).strip()
            image_name = str(row["image"]).strip()
            if split not in SPLITS:
                raise PrepError(f"前 10 类 manifest split 非法：{split}")
            if not group:
                raise PrepError(f"前 10 类图片缺少 physical_group：{image_name}")
            image_stem = Path(image_name).stem
            if image_stem not in images_by_split[split]:
                raise PrepError(f"前 10 类 manifest 指向不存在的图片：{split}/{image_name}")
            key = (split, image_stem)
            if key in seen_images:
                raise PrepError(f"前 10 类 manifest 重复图片：{split}/{image_name}")
            seen_images.add(key)
            groups[(split, image_stem)] = group
            group_splits[group].add(split)
    expected = {(split, stem) for split, images in images_by_split.items() for stem in images}
    if seen_images != expected:
        missing = sorted(expected - seen_images)
        extra = sorted(seen_images - expected)
        raise PrepError(f"前 10 类 manifest 与文件不一致；缺 {missing[:5]}，多 {extra[:5]}")
    crossing = {group: sorted(values) for group, values in group_splits.items() if len(values) > 1}
    if crossing:
        raise PrepError(f"前 10 类 physical_group 跨 train/val：{list(crossing.items())[:5]}")
    return groups


def inspect_source(name: str, root: Path, expected_indices: set[int]) -> tuple[list[Record], dict[str, Any]]:
    if not root.is_dir():
        raise PrepError(f"{name} 数据目录不存在：{root}")
    yaml_path = root / ("data.yaml" if (root / "data.yaml").is_file() else "dataset.yaml")
    yaml_payload = load_yaml(yaml_path)
    names = parse_names(yaml_payload, yaml_path)

    if name == "front10":
        if len(names) != 10:
            raise PrepError(f"前 10 类 names 应为 10 个，实际 {len(names)}")
    else:
        if len(names) != 40:
            raise PrepError(f"后 28 类 names 应为 40 个槽位，实际 {len(names)}")

    images_by_split: dict[str, dict[str, Path]] = {}
    labels_by_split: dict[str, dict[str, Path]] = {}
    for split in SPLITS:
        images_by_split[split], labels_by_split[split] = discover_split(root, split)

    front_groups: dict[tuple[str, str], str] = {}
    if name == "front10":
        front_groups = load_front_manifest(root, images_by_split)

    records: list[Record] = []
    class_counts = {split: Counter() for split in SPLITS}
    empty_counts = Counter()
    group_splits: dict[str, set[str]] = defaultdict(set)
    total_rows = 0
    errors: list[str] = []

    prefix = f"{name}__"
    for split in SPLITS:
        for stem, image_path in sorted(images_by_split[split].items()):
            label_path = labels_by_split[split][stem]
            rows, label_errors = parse_label(label_path)
            errors.extend(label_errors)
            class_ids = [row[0] for row in rows]
            if not rows:
                empty_counts[split] += 1
            used = set(class_ids)
            illegal = sorted(used - expected_indices)
            if illegal:
                errors.append(f"{name}/{split}/{label_path.name} 出现不允许的类别索引：{illegal}")
            for class_id in class_ids:
                class_counts[split][class_id] += 1
            total_rows += len(rows)

            if name == "front10":
                group = front_groups[(split, stem)]
            else:
                group = stem.split("__", 1)[0]
                if not group:
                    errors.append(f"{name}/{split}/{image_path.name} 无法解析 burst")
            group_splits[group].add(split)

            output_stem = prefix + stem
            image_suffix = image_path.suffix.lower()
            records.append(
                Record(
                    source_kind=name,
                    source_split=split,
                    source_image=image_path.resolve(),
                    source_label=label_path.resolve(),
                    output_image_rel=f"images/{split}/{output_stem}{image_suffix}",
                    output_label_rel=f"labels/{split}/{output_stem}.txt",
                    group=group,
                    class_ids=tuple(class_ids),
                )
            )

    crossing = {group: sorted(values) for group, values in group_splits.items() if len(values) > 1}
    if crossing:
        errors.append(f"{name} 有分组跨 train/val：{list(crossing.items())[:5]}")

    if name != "front10":
        if names[16] != "17-Eggs_Plush" or names[25] != "26-Plastic_Strawberry":
            raise PrepError("后类 names 的 17/26 槽位与预期不一致")
        if any(class_counts[split][index] for split in SPLITS for index in MISSING_SPOT_INDICES):
            errors.append("17/26 空槽出现了标签")

    for split in SPLITS:
        missing = sorted(index for index in expected_indices if class_counts[split][index] == 0)
        if missing:
            errors.append(f"{name}/{split} 缺少实际类别正的标签：{missing}")

    if errors:
        raise PrepError(f"{name} 数据审计失败（前 20 条）：\n" + "\n".join(errors[:20]))

    summary = {
        "root": str(root),
        "yaml": str(yaml_path),
        "names": names,
        "images": {split: len(images_by_split[split]) for split in SPLITS},
        "empty_labels": dict(empty_counts),
        "box_count": total_rows,
        "class_counts": {split: {str(k): v for k, v in sorted(class_counts[split].items())} for split in SPLITS},
        "groups": len(group_splits),
    }
    return records, summary


def validate_sources(front_root: Path, post_root: Path) -> tuple[list[Record], dict[str, Any]]:
    post_audit_path = post_root / "audit_summary.json"
    if not post_audit_path.is_file():
        raise PrepError(f"后类导出缺少审计文件：{post_audit_path}")
    post_audit = json.loads(post_audit_path.read_text(encoding="utf-8"))
    if post_audit.get("complete_review") is not True:
        raise PrepError("后类复核未完成：complete_review != true")
    if post_audit.get("postclasses_merge_ready") is not True:
        raise PrepError("后类导出不可进入合并：postclasses_merge_ready != true")
    if post_audit.get("unreviewed_count") not in (0, None) or post_audit.get("skipped_count") not in (0, None):
        raise PrepError("后类导出仍有未复核或跳过项")

    front_records, front_summary = inspect_source("front10", front_root, set(range(10)))
    post_records, post_summary = inspect_source("post28", post_root, POST_ACTIVE_INDICES)

    front_names = front_summary["names"]
    post_names = post_summary["names"]
    if front_names != post_names[:10]:
        raise PrepError(f"前 10 类名称与官方槽位不一致：{front_names} != {post_names[:10]}")

    records = front_records + post_records
    duplicate_outputs = [p for p, count in Counter(r.output_image_rel for r in records).items() if count > 1]
    if duplicate_outputs:
        raise PrepError(f"合并后输出图片名冲突：{duplicate_outputs[:10]}")

    merged_class_counts = {split: Counter() for split in SPLITS}
    for record in records:
        for class_id in record.class_ids:
            merged_class_counts[record.source_split][class_id] += 1
    for split in SPLITS:
        missing = sorted(index for index in EXPECTED_ACTIVE_INDICES if merged_class_counts[split][index] == 0)
        if missing:
            raise PrepError(f"合并后 {split} 缺少实际类别正标签：{missing}")
        forbidden = sorted(index for index in MISSING_SPOT_INDICES if merged_class_counts[split][index])
        if forbidden:
            raise PrepError(f"合并后 {split} 的 17/26 空槽出现标签：{forbidden}")

    summary = {
        "front10": front_summary,
        "post28": post_summary,
        "images": {split: sum(1 for r in records if r.source_split == split) for split in SPLITS},
        "empty_labels": {
            split: sum(1 for r in records if r.source_split == split and not r.class_ids) for split in SPLITS
        },
        "box_count": sum(len(r.class_ids) for r in records),
        "class_counts": {split: {str(k): v for k, v in sorted(merged_class_counts[split].items())} for split in SPLITS},
        "class_slots": 40,
        "active_classes": len(EXPECTED_ACTIVE_INDICES),
        "missing_spot_indices": sorted(MISSING_SPOT_INDICES),
        "missing_spot_official_ids": [17, 26],
        "group_crossing": [],
    }
    return records, summary


def write_dataset(records: list[Record], source_summary: dict[str, Any], output: Path) -> dict[str, Any]:
    output_parent = output.parent
    if output.exists():
        raise PrepError(f"输出目录已存在，拒绝覆盖：{output}")
    if not output_parent.is_dir():
        raise PrepError(f"输出父目录不存在：{output_parent}")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output_parent))
    try:
        for split in SPLITS:
            (temp_dir / "images" / split).mkdir(parents=True, exist_ok=False)
            (temp_dir / "labels" / split).mkdir(parents=True, exist_ok=False)

        for record in records:
            out_image = temp_dir / record.output_image_rel
            out_label = temp_dir / record.output_label_rel
            os.symlink(record.source_image, out_image)
            os.symlink(record.source_label, out_label)

        names = source_summary["post28"]["names"]
        dataset_yaml = {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "nc": 40,
            "names": names,
        }
        (temp_dir / "dataset.yaml").write_text(
            yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        manifest_path = temp_dir / "merge_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in records:
                item = {
                    "source_kind": record.source_kind,
                    "source_split": record.source_split,
                    "source_image": str(record.source_image),
                    "source_label": str(record.source_label),
                    "output_image": record.output_image_rel,
                    "output_label": record.output_label_rel,
                    "group": record.group,
                    "class_ids": list(record.class_ids),
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        materialization = Counter()
        for record in records:
            for rel in (record.output_image_rel, record.output_label_rel):
                path = temp_dir / rel
                if path.is_symlink():
                    materialization["symlink"] += 1
                elif path.is_file():
                    materialization["file"] += 1
        if materialization != Counter({"symlink": len(records) * 2}):
            raise PrepError(f"合并文件物化方式异常：{dict(materialization)}")

        output_audit = audit_output(temp_dir, names, source_summary)
        audit = {
            "schema_version": 1,
            "created_at": now_iso(),
            "dataset_root": str(output),
            "dataset_yaml": str(output / "dataset.yaml"),
            "materialization": "absolute_symlink",
            "source_summary": source_summary,
            "output_audit": output_audit,
            "training_ready": True,
            "training_started": False,
            "checks": {
                "complete_review": True,
                "postclasses_merge_ready": True,
                "front10_present": True,
                "all_38_active_classes_have_train_and_val": True,
                "missing_slots_17_26_empty": True,
                "class_index_range_0_39": True,
                "bbox_coordinates_valid": True,
                "no_full_image_pseudo_boxes": True,
                "no_group_or_burst_split_crossing": True,
                "image_label_pairing_complete": True,
                "nc_40": True,
            },
        }
        (temp_dir / "audit_summary.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        readme = f"""# 赛道一 40 槽最终训练数据集

- 前 10 类：`{source_summary['front10']['root']}`
- 后 28 类：`{source_summary['post28']['root']}`
- 图片总数：{output_audit['images']['train'] + output_audit['images']['val']}
  - train：{output_audit['images']['train']}
  - val：{output_audit['images']['val']}
- bbox 总数：{output_audit['box_count']}
- 类别槽位：40
- 实际有训练数据类别：38
- 空槽：17（蛋形毛绒玩具）、26（塑料草莓）
- 图片与标签均使用绝对符号链接，不复制照片，不占用本地内存。
- 本目录只是训练数据，脚本不会自动开始训练。

关键审计结果见 `audit_summary.json`，逐图来源见 `merge_manifest.jsonl`。
"""
        (temp_dir / "README.md").write_text(readme, encoding="utf-8")
        (temp_dir / ".complete").write_text(now_iso() + "\n", encoding="utf-8")

        temp_dir.rename(output)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    final_audit = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    return final_audit


def audit_output(root: Path, names: list[str], source_summary: dict[str, Any]) -> dict[str, Any]:
    if len(names) != 40:
        raise PrepError("输出 names 不是 40 个类别槽位")
    class_counts = {split: Counter() for split in SPLITS}
    images_count: dict[str, int] = {}
    label_count: dict[str, int] = {}
    empty_count = Counter()
    box_count = 0
    pair_errors: list[str] = []
    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        labels = {p.stem: p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
        images_count[split] = len(images)
        label_count[split] = len(labels)
        if set(images) != set(labels):
            pair_errors.append(f"{split} 图片/标签 stem 不配对")
        for stem, image_path in images.items():
            if not image_path.resolve().is_file():
                pair_errors.append(f"图片链接失效：{image_path}")
            label_path = labels.get(stem)
            if label_path is None:
                continue
            if not label_path.resolve().is_file():
                pair_errors.append(f"标签链接失效：{label_path}")
                continue
            rows, errors = parse_label(label_path)
            pair_errors.extend(errors)
            if not rows:
                empty_count[split] += 1
            for class_id, *_ in rows:
                class_counts[split][class_id] += 1
                box_count += 1
    if pair_errors:
        raise PrepError("输出数据集审计失败（前 20 条）：\n" + "\n".join(pair_errors[:20]))

    for split in SPLITS:
        missing = sorted(index for index in EXPECTED_ACTIVE_INDICES if class_counts[split][index] == 0)
        forbidden = sorted(index for index in MISSING_SPOT_INDICES if class_counts[split][index])
        if missing:
            raise PrepError(f"输出 {split} 缺少实际类别：{missing}")
        if forbidden:
            raise PrepError(f"输出 {split} 的 17/26 空槽有标签：{forbidden}")

    return {
        "images": images_count,
        "labels": label_count,
        "empty_labels": dict(empty_count),
        "box_count": box_count,
        "class_counts": {split: {str(k): v for k, v in sorted(class_counts[split].items())} for split in SPLITS},
        "active_class_count": len(EXPECTED_ACTIVE_INDICES),
        "empty_slot_indices": sorted(MISSING_SPOT_INDICES),
        "names_count": len(names),
        "pairing_errors": 0,
        "full_image_pseudo_boxes": 0,
    }


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="合并前 10 类与后 28 类并生成 40 槽 YOLO 训练集")
    parser.add_argument(
        "--front-root",
        type=Path,
        default=Path("/media/sunshine/新加卷/taskA/datasets/real_ft_v25_clean_20261016"),
    )
    parser.add_argument(
        "--post-root",
        type=Path,
        default=project_root / "datasets" / "track1_postclasses_bbox_export_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "datasets" / "track1_training_merged_v1",
    )
    parser.add_argument("--dry-run", action="store_true", help="只审计来源，不创建数据集")
    return parser


def main(argv: list[str] | None = None) -> int:
    project_root = Path(os.environ.get("TRACK1_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).resolve()
    args = build_parser(project_root).parse_args(argv)
    front_root = args.front_root.resolve()
    post_root = args.post_root.resolve()
    output = args.output.resolve()
    records, source_summary = validate_sources(front_root, post_root)
    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "source_summary": source_summary}, ensure_ascii=False, indent=2))
        return 0
    audit = write_dataset(records, source_summary, output)
    print(json.dumps({"ok": True, "dataset_root": str(output), "audit": audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
