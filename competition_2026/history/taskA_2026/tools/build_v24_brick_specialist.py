#!/usr/bin/env python3
"""构建泡沫砖专家数据集：保留 10 类定义，但只训练 class5(Foam_Brick)。

旧类权重不再参与任何训练，因此补泡沫砖不会修改旧类模型。
所有不含 class5 的旧数据集图片均作为背景负样本；新增 12 张按 9:3 切开。
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

T = Path("/media/sunshine/新加卷/taskA")
BASE = T / "datasets/real_ft_v22_20261016"
SCENES = Path("/home/sunshine/codex/easton/taskA/datasets/random_real_scenes_near_supp_20261013")
OUT = T / "datasets/real_ft_v24_brick_specialist_20261016"
MANIFEST = Path("/home/sunshine/codex/easton/taskA/scratch/v24_brick_specialist_manifest_20261016.json")
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


def filtered_label_lines(path: Path) -> list[str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [line for line in lines if int(line.split()[0]) == 5]


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
    if len(set(NEW_STEMS)) != 12 or not HOLDOUT.issubset(NEW_STEMS):
        raise SystemExit("new stem list invalid")
    for stem in NEW_STEMS:
        image = SCENES / "images" / f"{stem}.jpg"
        label = SCENES / "labels" / f"{stem}.txt"
        if not image.is_file() or not label.is_file():
            raise SystemExit(f"missing scene: {stem}")
        lines = filtered_label_lines(label)
        if len(lines) != 1:
            raise SystemExit(f"new scene must have exactly one Foam_Brick label: {stem}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    # 旧数据全部作为泡沫砖专家模型的背景或泡沫砖正样本。
    for split in ("train", "val"):
        for image in sorted((BASE / "images" / split).iterdir()):
            if not image.is_file():
                continue
            label = BASE / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise SystemExit(f"missing base label: {label}")
            shutil.copy2(image, OUT / "images" / split / image.name)
            (OUT / "labels" / split / label.name).write_text(
                "\n".join(filtered_label_lines(label)) + ("\n" if filtered_label_lines(label) else ""),
                encoding="utf-8",
            )

    mapping: dict[str, list[str]] = {"train": [], "val": []}
    for stem in NEW_STEMS:
        split = "val" if stem in HOLDOUT else "train"
        copies = 1 if split == "val" else REPEATS
        for index in range(1, copies + 1):
            suffix = "" if copies == 1 else f"_r{index:02d}"
            out_stem = f"brickpose_{stem}{suffix}"
            shutil.copy2(SCENES / "images" / f"{stem}.jpg", OUT / "images" / split / f"{out_stem}.jpg")
            (OUT / "labels" / split / f"{out_stem}.txt").write_text(
                "\n".join(filtered_label_lines(SCENES / "labels" / f"{stem}.txt")) + "\n",
                encoding="utf-8",
            )
            mapping[split].append(out_stem)

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
    if counts["train"] != {
        "images": 2997,
        "labels": 2997,
        "empty": 2853,
        **{f"class{k}": (144 if k == 5 else 0) for k in range(10)},
    }:
        raise SystemExit(f"unexpected train counts: {counts['train']}")
    if counts["val"] != {
        "images": 241,
        "labels": 241,
        "empty": 228,
        **{f"class{k}": (13 if k == 5 else 0) for k in range(10)},
    }:
        raise SystemExit(f"unexpected val counts: {counts['val']}")

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
