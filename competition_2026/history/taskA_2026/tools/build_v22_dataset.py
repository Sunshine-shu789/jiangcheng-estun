#!/usr/bin/env python3
"""构建 real_ft_v22_20261016：v21 底座 + 新汤罐 210 张 + 杂物负样本。

规则:
  - 旧汤罐: v21 train 243 张随机保留 50 张；val 5 张全保留。
  - 新汤罐: 14 个姿态组各留最后 1 张做 val，其余 196 张进 train。
  - 负样本: v21 已有 14 张留作底座；新采集的 60 张杂物场景按 4 个拍摄段
            各取 10 张训练、5 张留出复测。另有早期空桌 12 张留出。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from pathlib import Path

T = Path("/media/sunshine/新加卷/taskA")
V21 = T / "datasets/real_ft_v21_20261016"
OUT = T / "datasets/real_ft_v22_20261016"
RAW_SOUP = T / "real_raw/5-Tomato_Soup"
RAW_EMPTY = T / "real_raw/empty_background"
SOUP_LABELS = Path("/home/sunshine/codex/easton/taskA/prelabels_soup/labels_final")
GROUPS_JSON = Path("/home/sunshine/codex/easton/taskA/scratch/soup_groups.json")
SCRATCH = Path("/home/sunshine/codex/easton/taskA/scratch")
SEED = 22
KEEP_OLD_SOUP = 50
NCLUSTERS = {
    "clutter_035": ("20261016_0350", "20261016_0351"),
    "clutter_0408": ("20261016_0408", "20261016_0409"),
    "clutter_0410": ("20261016_0410", "20261016_0411"),
    "clutter_0411": ("20261016_0411", "20261016_0412"),
}


def image_label_files(dataset: Path, split: str) -> tuple[list[Path], list[Path]]:
    images = sorted((dataset / "images" / split).iterdir())
    labels = sorted((dataset / "labels" / split).glob("*.txt"))
    if len(images) != len(labels):
        raise SystemExit(f"{dataset}/{split}: image/label count mismatch")
    return images, labels


def find_empty_holdout() -> list[str]:
    raw_names = sorted(p.name for p in RAW_EMPTY.glob("empty_background_*.jpg"))
    initial = [n for n in raw_names if "20261016_0154" in n]
    v21_train_empty = {
        p.stem
        for p in (V21 / "labels/train").glob("*.txt")
        if not p.read_text(encoding="utf-8").strip()
    }
    return [n for n in initial if Path(n).stem not in v21_train_empty]


def clutter_names() -> dict[str, list[str]]:
    raw_names = sorted(p.name for p in RAW_EMPTY.glob("empty_background_*.jpg"))
    found: dict[str, list[str]] = {}
    for key, (start, end) in NCLUSTERS.items():
        names = [
            n
            for n in raw_names
            if start <= n.split("_")[2] + "_" + n.split("_")[3][:4] < end
        ]
        # 用完整时间戳筛选，避免误抓相邻分钟
        names = [
            n for n in raw_names
            if n.split("_")[2] + "_" + n.split("_")[3][:4] == start
        ]
        if len(names) < 15:
            # 同一段跨分钟时按时间窗口补齐
            lo = start
            hi = end
            names = [
                n
                for n in raw_names
                if lo <= n.split("_")[2] + "_" + n.split("_")[3][:4] < hi
            ]
        found[key] = names
    if any(len(v) != 15 for v in found.values()):
        raise SystemExit(f"clutter clusters invalid: {[(k, len(v)) for k, v in found.items()]}")
    return found


def stats(dataset: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        classes: collections.Counter[int] = collections.Counter()
        empty = 0
        for label in (dataset / "labels" / split).glob("*.txt"):
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty += 1
            for line in lines:
                classes[int(line.split()[0])] += 1
        result[split] = {
            "images": len(list((dataset / "images" / split).iterdir())),
            "labels": len(list((dataset / "labels" / split).glob("*.txt"))),
            "empty": empty,
            **{f"class{k}": classes.get(k, 0) for k in range(10)},
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not V21.is_dir():
        raise SystemExit(f"missing v21: {V21}")
    if not SOUP_LABELS.is_dir():
        raise SystemExit(f"missing final soup labels: {SOUP_LABELS}")
    groups = json.loads(GROUPS_JSON.read_text(encoding="utf-8"))
    if len(groups) != 14 or sum(len(v) for v in groups.values()) != 210:
        raise SystemExit("soup group definition is not 14x15")

    # 新汤罐和标签完整性
    for group, names in groups.items():
        for name in names:
            if not (RAW_SOUP / name).is_file():
                raise SystemExit(f"missing raw soup: {name}")
            if not (SOUP_LABELS / f"{Path(name).stem}.txt").is_file():
                raise SystemExit(f"missing soup label: {name}")

    # 旧汤罐与基础数据
    v21_images, v21_labels = image_label_files(V21, "train")
    v21_val_images, v21_val_labels = image_label_files(V21, "val")
    old_soup = [
        label.stem
        for label in v21_labels
        if any(line.split()[0] == "4" for line in label.read_text(encoding="utf-8").splitlines() if line.strip())
    ]
    if len(old_soup) != 243:
        raise SystemExit(f"expected 243 old soup train images, got {len(old_soup)}")
    kept_old = set(random.Random(SEED).sample(old_soup, KEEP_OLD_SOUP))
    drop_old = set(old_soup) - kept_old

    # 负样本
    clutter = clutter_names()
    neg_train: list[str] = []
    neg_holdout: list[str] = []
    for names in clutter.values():
        neg_train.extend(names[:10])
        neg_holdout.extend(names[10:])
    neg_holdout.extend(find_empty_holdout())
    if len(neg_train) != 40 or len(neg_holdout) != 32:
        raise SystemExit(f"negative split invalid: train={len(neg_train)} holdout={len(neg_holdout)}")

    # 新汤罐切分: 每组最后一张做 val
    soup_val = [names[-1] for names in groups.values()]
    soup_train = [name for names in groups.values() for name in names[:-1]]
    if len(soup_train) != 196 or len(soup_val) != 14:
        raise SystemExit("soup split invalid")

    print(f"[旧汤罐] train {len(old_soup)} -> {len(kept_old)}，丢弃 {len(drop_old)}；val 5 全保留")
    print(f"[新汤罐] train {len(soup_train)} / val {len(soup_val)}")
    print(f"[负样本] 新杂物 train {len(neg_train)} / 留出 {len(neg_holdout)}（含早期空桌留出）")

    if args.dry_run:
        print("[dry-run] 未写数据盘")
        return 0

    if OUT.exists():
        raise SystemExit(f"目标已存在，拒绝覆盖: {OUT}")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    # 1) 拷贝 v21 底座，剔除降采样掉的旧汤罐 train
    copied = {"train": 0, "val": 0}
    for split, images, labels in (
        ("train", v21_images, v21_labels),
        ("val", v21_val_images, v21_val_labels),
    ):
        for image, label in zip(images, labels):
            if split == "train" and image.stem in drop_old:
                continue
            shutil.copy2(image, OUT / "images" / split / image.name)
            shutil.copy2(label, OUT / "labels" / split / label.name)
            copied[split] += 1

    # 2) 新汤罐
    for split, names in (("train", soup_train), ("val", soup_val)):
        for name in names:
            stem = Path(name).stem
            shutil.copy2(RAW_SOUP / name, OUT / "images" / split / name)
            shutil.copy2(SOUP_LABELS / f"{stem}.txt", OUT / "labels" / split / f"{stem}.txt")

    # 3) 新杂物负样本（空标签）
    for name in neg_train:
        shutil.copy2(RAW_EMPTY / name, OUT / "images/train" / name)
        (OUT / "labels/train" / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")

    # 4) data.yaml
    (OUT / "data.yaml").write_text(
        f"""path: {OUT}
train: images/train
val: images/val
nc: 10
names:
- 1-Cheez-it
- 2-Starkist_Tuna
- 3-Scissors
- 4-Frenchs_Mustard
- 5-Tomato_Soup
- 6-Foam_Brick
- 7-Clamp
- 8-Plastic_Banana
- 9-Mug
- 10-meat_can
""",
        encoding="utf-8",
    )

    counts = stats(OUT)
    expected_train = 2943
    expected_val = 238
    if counts["train"]["images"] != expected_train:
        raise SystemExit(f"train images {counts['train']['images']} != {expected_train}")
    if counts["val"]["images"] != expected_val:
        raise SystemExit(f"val images {counts['val']['images']} != {expected_val}")
    if counts["train"]["class4"] != 246 or counts["val"]["class4"] != 19:
        raise SystemExit(f"soup class count invalid: {counts}")
    if counts["train"]["empty"] != 54:
        raise SystemExit(f"train empty count {counts['train']['empty']} != 54")

    manifest = {
        "out": str(OUT),
        "seed": SEED,
        "base": str(V21),
        "kept_old_soup": sorted(kept_old),
        "dropped_old_soup": sorted(drop_old),
        "new_soup_train": soup_train,
        "new_soup_val": soup_val,
        "new_negative_train": neg_train,
        "negative_holdout": neg_holdout,
        "stats": counts,
    }
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "v22_build_manifest_20261016.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (SCRATCH / "v22_negative_holdout_20261016.txt").write_text(
        "\n".join(neg_holdout) + "\n", encoding="utf-8"
    )
    print(f"[完成] train={counts['train']['images']} val={counts['val']['images']} -> {OUT}")
    print(f"[类别] {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
