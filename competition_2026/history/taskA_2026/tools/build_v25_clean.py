#!/usr/bin/env python3
"""从 v22 审计清单构建 v25 干净数据集。

原则：
  - 只读 v22；新数据写入 real_ft_v25_clean_*，绝不覆盖 v22；
  - 精确重复图只保留一个 SHA256 代表；
  - 按审计得到的 physical_group 整体切 train/val，禁止同组跨 split；
  - 旧 v5 弯曲带斑香蕉先隔离，不进入 Plastic_Banana 正样本；
  - 默认只生成计划，显式传 --copy 才复制文件。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_AUDIT = Path("/media/sunshine/新加卷/taskA/reports/v25_audit_20261016.csv")
DEFAULT_OUT = Path("/media/sunshine/新加卷/taskA/datasets/real_ft_v25_clean_20261016")
DEFAULT_REPORTDIR = Path("/media/sunshine/新加卷/taskA/reports")
CLASS_NAMES = (
    "1-Cheez-it",
    "2-Starkist_Tuna",
    "3-Scissors",
    "4-Frenchs_Mustard",
    "5-Tomato_Soup",
    "6-Foam_Brick",
    "7-Clamp",
    "8-Plastic_Banana",
    "9-Mug",
    "10-meat_can",
)

# 由 v25 审计 CSV 选出的物理组。选择目标为约 22% 图像，并优先覆盖每个类别和姿态来源；
# 组 ID 对应 v25_audit_20261016.csv 的 physical_group。
VAL_GROUPS = frozenset(
    {
        "pg0001", "pg0005", "pg0008", "pg0016", "pg0018", "pg0020", "pg0021", "pg0028",
        "pg0033", "pg0040", "pg0045", "pg0052", "pg0056", "pg0058", "pg0063", "pg0064",
        "pg0065", "pg0067", "pg0075", "pg0078", "pg0082", "pg0086", "pg0093", "pg0097",
        "pg0099", "pg0100", "pg0103", "pg0104", "pg0105", "pg0113", "pg0121", "pg0122",
        "pg0124", "pg0125", "pg0130", "pg0132", "pg0133", "pg0134", "pg0138", "pg0139",
        "pg0143", "pg0146", "pg0147", "pg0152", "pg0154", "pg0155", "pg0156", "pg0157",
        "pg0158", "pg0159", "pg0163", "pg0165", "pg0167", "pg0169", "pg0170", "pg0172",
        "pg0173", "pg0179", "pg0181", "pg0182", "pg0183", "pg0188", "pg0193", "pg0194",
        "pg0195", "pg0196", "pg0199", "pg0200", "pg0209", "pg0212", "pg0216", "pg0217",
        "pg0219", "pg0229", "pg0230",
    }
)

EXPECTED_VAL_IMAGES = {
    "1-Cheez-it": 29,
    "2-Starkist_Tuna": 27,
    "3-Scissors": 37,
    "4-Frenchs_Mustard": 49,
    "5-Tomato_Soup": 55,
    "6-Foam_Brick": 12,
    "7-Clamp": 55,
    "8-Plastic_Banana": 22,
    "9-Mug": 12,
    "10-meat_can": 62,
}


@dataclass(frozen=True)
class Item:
    sha256: str
    image_path: Path
    label_path: Path
    image_name: str
    logical_name: str
    physical_group: str
    classes: tuple[str, ...]
    original_split: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_classes(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(";") if part)


def load_items(audit_path: Path, quarantine_v5_banana: bool) -> tuple[list[Item], list[Item]]:
    if not audit_path.is_file():
        raise SystemExit(f"缺少审计清单: {audit_path}")
    by_hash: dict[str, Item] = {}
    with audit_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            digest = row["sha256"]
            if digest in by_hash:
                continue
            by_hash[digest] = Item(
                sha256=digest,
                image_path=Path(row["image_path"]),
                label_path=Path(row["label_path"]),
                image_name=row["image"],
                logical_name=row["logical_name"],
                physical_group=row["physical_group"],
                classes=split_classes(row["class_names"]),
                original_split=row["split"],
            )
    items = sorted(by_hash.values(), key=lambda item: (item.physical_group, item.image_name))
    quarantine: list[Item] = []
    if quarantine_v5_banana:
        quarantine = [item for item in items if "v5_banana" in item.logical_name]
        items = [item for item in items if "v5_banana" not in item.logical_name]
    return items, quarantine


def group_items(items: list[Item]) -> dict[str, list[Item]]:
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        groups[item.physical_group].append(item)
    return dict(groups)


def stats(items: list[Item]) -> dict[str, object]:
    class_files: Counter[str] = Counter()
    class_instances: Counter[str] = Counter()
    groups: set[str] = set()
    for item in items:
        groups.add(item.physical_group)
        for class_name in item.classes:
            class_files[class_name] += 1
            class_instances[class_name] += 1
    return {
        "images": len(items),
        "groups": len(groups),
        "class_files": dict(sorted(class_files.items())),
        "class_instances": dict(sorted(class_instances.items())),
    }


def write_plan(plan_path: Path, groups: dict[str, list[Item]]) -> None:
    with plan_path.open("w", newline="", encoding="utf-8-sig") as fh:
        fields = ["physical_group", "split", "images", "classes", "class_image_counts", "example_image"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for group_id in sorted(groups):
            group = groups[group_id]
            class_counts = Counter(cls for item in group for cls in item.classes)
            writer.writerow(
                {
                    "physical_group": group_id,
                    "split": "val" if group_id in VAL_GROUPS else "train",
                    "images": len(group),
                    "classes": ";".join(sorted(class_counts)),
                    "class_image_counts": ";".join(f"{key}:{value}" for key, value in sorted(class_counts.items())),
                    "example_image": group[0].image_name,
                }
            )


def build_dataset(
    items: list[Item],
    quarantine: list[Item],
    out: Path,
    reportdir: Path,
    copy_files: bool,
    force: bool,
) -> None:
    groups = group_items(items)
    train = [item for item in items if item.physical_group not in VAL_GROUPS]
    val = [item for item in items if item.physical_group in VAL_GROUPS]
    unknown = sorted(group for group in VAL_GROUPS if group not in groups)
    if unknown:
        raise SystemExit(f"VAL_GROUPS 引用了不存在的物理组: {unknown}")
    train_stats = stats(train)
    val_stats = stats(val)

    # 严格校验切分覆盖；若源数据或分组变化，拒绝生成一个看似正确的新数据集。
    for class_name, expected in EXPECTED_VAL_IMAGES.items():
        actual = int(val_stats["class_files"].get(class_name, 0))  # type: ignore[union-attr]
        if actual != expected:
            raise SystemExit(f"val 类别覆盖错误: {class_name} expected={expected} actual={actual}")
    if train_stats["images"] + val_stats["images"] != len(items):
        raise SystemExit("train/val 图片总数不守恒")
    if len(train) + len(val) != len(items):
        raise SystemExit("train/val 文件总数不守恒")
    if any(item.physical_group in VAL_GROUPS for item in train):
        raise SystemExit("发现同一物理组跨 train/val")

    out = out.expanduser().resolve()
    reportdir = reportdir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    reportdir.mkdir(parents=True, exist_ok=True)
    plan_path = reportdir / "v25_clean_split_plan_20261016.csv"
    write_plan(plan_path, groups)

    summary = {
        "source": "derived from v25 audit",
        "out": str(out),
        "copy_files": copy_files,
        "train": train_stats,
        "val": val_stats,
        "quarantine_v5_banana_images": len(quarantine),
        "val_groups": sorted(VAL_GROUPS),
        "class_names": list(CLASS_NAMES),
    }

    if not copy_files:
        print(f"plan={plan_path}")
        print(f"train_images={train_stats['images']} train_groups={train_stats['groups']}")
        print(f"val_images={val_stats['images']} val_groups={val_stats['groups']}")
        print(f"quarantine_v5_banana_images={len(quarantine)}")
        print("dry-run: 未复制图片和标签")
        return

    if out.exists() and any(out.iterdir()) and not force:
        raise SystemExit(f"目标非空，拒绝覆盖: {out}; 需要显式 --force")
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []
    for split_name, split_items in (("train", train), ("val", val)):
        for item in split_items:
            if not item.image_path.is_file() or not item.label_path.is_file():
                raise SystemExit(f"源文件缺失: {item.image_path} / {item.label_path}")
            actual_sha = sha256_file(item.image_path)
            if actual_sha != item.sha256:
                raise SystemExit(f"SHA 与审计不一致: {item.image_path}")
            dest_image = out / "images" / split_name / item.image_name
            dest_label = out / "labels" / split_name / f"{item.image_path.stem}.txt"
            shutil.copy2(item.image_path, dest_image)
            shutil.copy2(item.label_path, dest_label)
            manifest_rows.append(
                {
                    "split": split_name,
                    "physical_group": item.physical_group,
                    "image": item.image_name,
                    "sha256": item.sha256,
                    "classes": ";".join(item.classes),
                    "original_split": item.original_split,
                    "source_image": str(item.image_path),
                }
            )

    (out / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out}",
                "train: images/train",
                "val: images/val",
                "nc: 10",
                "names:",
                *[f"- {name}" for name in CLASS_NAMES],
                "",
            ]
        ),
        encoding="utf-8",
    )

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        fields = ["split", "physical_group", "image", "sha256", "classes", "original_split", "source_image"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (out / "quarantine_v5_banana.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        fields = ["physical_group", "image", "sha256", "classes", "source_image", "reason"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for item in quarantine:
            writer.writerow(
                {
                    "physical_group": item.physical_group,
                    "image": item.image_name,
                    "sha256": item.sha256,
                    "classes": ";".join(item.classes),
                    "source_image": str(item.image_path),
                    "reason": "旧 v5 弯曲带斑香蕉，疑似非比赛塑料香蕉，先隔离复核",
                }
            )

    (out / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dataset={out}")
    print(f"train_images={train_stats['images']} train_groups={train_stats['groups']}")
    print(f"val_images={val_stats['images']} val_groups={val_stats['groups']}")
    print(f"quarantine_v5_banana_images={len(quarantine)}")
    print(f"plan={plan_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reportdir", type=Path, default=DEFAULT_REPORTDIR)
    parser.add_argument("--copy", action="store_true", help="真正复制图片和标签；默认只生成计划")
    parser.add_argument("--force", action="store_true", help="允许写入非空目标目录（仍然拒绝覆盖源数据集）")
    parser.add_argument("--include-v5-banana", action="store_true", help="将旧 v5 香蕉也作为 Plastic_Banana 正样本")
    args = parser.parse_args()

    items, quarantine = load_items(args.audit.expanduser().resolve(), not args.include_v5_banana)
    if len(items) + len(quarantine) != 1628:
        print(f"warning: unique items={len(items)} quarantine={len(quarantine)} expected_total=1628")
    build_dataset(
        items=items,
        quarantine=quarantine,
        out=args.out,
        reportdir=args.reportdir,
        copy_files=args.copy,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
