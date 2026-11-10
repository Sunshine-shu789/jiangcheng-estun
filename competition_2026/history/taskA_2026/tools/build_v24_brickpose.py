#!/usr/bin/env python3
"""构建 v24 泡沫砖侧棱姿态补强数据集。

以 v22 完整数据集为底座，将新采的 12 张侧棱/角度图按 9:3 切开：
  - 9 张训练图各复制 6 次，提高极少数新增姿态对头部更新的影响；
  - 3 张留出图只放 val，用于训练后检查侧棱漏检是否修复。
不覆盖任何已有数据集。
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

T = Path("/media/sunshine/新加卷/taskA")
BASE = T / "datasets/real_ft_v22_20261016"
SCENES = Path("/home/sunshine/codex/easton/taskA/datasets/random_real_scenes_near_supp_20261013")
OUT = T / "datasets/real_ft_v24_brickpose_20261016"
MANIFEST = Path("/home/sunshine/codex/easton/taskA/scratch/v24_build_manifest_20261016.json")
REPEATS = 6
HOLDOUT = {
    "scene_20261016_052610_918105_f54b49",
    "scene_20261016_052712_003682_47a5b9",
    "scene_20261016_052929_597036_851afd",
}
NEW_STEMS = [
    "scene_20261016_052517_539982_3a26ff",
    "scene_20261016_052610_918105_f54b49",
    "scene_20261016_052644_653603_f1a86c",
    "scene_20261016_052712_003682_47a5b9",
    "scene_20261016_052734_382559_17f1af",
    "scene_20261016_052756_351109_5effae",
    "scene_20261016_052818_237569_05f733",
    "scene_20261016_052846_898678_64ad44",
    "scene_20261016_052908_427702_505135",
    "scene_20261016_052929_597036_851afd",
    "scene_20261016_053039_076950_7f85ca",
    "scene_20261016_053116_637807_1334cc",
]


def stats(root: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for split in ("train", "val"):
        classes: Counter[int] = Counter()
        empty = 0
        image_paths = sorted((root / "images" / split).iterdir())
        label_paths = sorted((root / "labels" / split).glob("*.txt"))
        if len(image_paths) != len(label_paths):
            raise SystemExit(f"{split}: image/label mismatch")
        for label in label_paths:
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty += 1
            for line in lines:
                classes[int(line.split()[0])] += 1
        result[split] = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "empty": empty,
            **{f"class{k}": classes.get(k, 0) for k in range(10)},
        }
    return result


def main() -> int:
    if not BASE.is_dir():
        raise SystemExit(f"missing base dataset: {BASE}")
    if OUT.exists():
        raise SystemExit(f"refuse to overwrite existing dataset: {OUT}")
    if set(NEW_STEMS) != HOLDOUT | (set(NEW_STEMS) - HOLDOUT):
        raise SystemExit("new stem list invalid")
    for stem in NEW_STEMS:
        if not (SCENES / "images" / f"{stem}.jpg").is_file():
            raise SystemExit(f"missing scene image: {stem}")
        label = SCENES / "labels" / f"{stem}.txt"
        if not label.is_file():
            raise SystemExit(f"missing scene label: {stem}")
        lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 1 or lines[0].split()[0] != "5":
            raise SystemExit(f"unexpected scene label for {stem}: {lines}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        for image in sorted((BASE / "images" / split).iterdir()):
            if not image.is_file():
                continue
            shutil.copy2(image, OUT / "images" / split / image.name)
            label = BASE / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise SystemExit(f"missing base label: {label}")
            shutil.copy2(label, OUT / "labels" / split / label.name)

    mapping: dict[str, list[str]] = {"train": [], "val": []}
    for stem in NEW_STEMS:
        src_image = SCENES / "images" / f"{stem}.jpg"
        src_label = SCENES / "labels" / f"{stem}.txt"
        split = "val" if stem in HOLDOUT else "train"
        copies = 1 if split == "val" else REPEATS
        for index in range(1, copies + 1):
            suffix = "" if copies == 1 else f"_r{index:02d}"
            out_stem = f"brickpose_{stem}{suffix}"
            shutil.copy2(src_image, OUT / "images" / split / f"{out_stem}.jpg")
            shutil.copy2(src_label, OUT / "labels" / split / f"{out_stem}.txt")
            mapping[split].append(out_stem)

    data_yaml = f"""path: {OUT}
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
"""
    (OUT / "data.yaml").write_text(data_yaml, encoding="utf-8")

    counts = stats(OUT)
    expected = {
        "train": {"images": 2943 + 9 * REPEATS, "class5": 90 + 9 * REPEATS},
        "val": {"images": 241, "class5": 13},
    }
    for split, expected_values in expected.items():
        for key, value in expected_values.items():
            if counts[split][key] != value:
                raise SystemExit(f"{split} {key}: {counts[split][key]} != {value}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "base": str(BASE),
                "scenes": str(SCENES),
                "out": str(OUT),
                "repeats": REPEATS,
                "holdout": sorted(HOLDOUT),
                "new_stems": NEW_STEMS,
                "mapping": mapping,
                "stats": counts,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False, indent=1))
    print(f"[完成] {OUT}")
    print(f"[清单] {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
