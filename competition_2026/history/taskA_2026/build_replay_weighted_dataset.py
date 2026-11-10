#!/usr/bin/env python3
"""Create a replay-weighted dataset from an existing grouped fine-tune set.

The grouped dataset keeps validation capture groups separate. This builder
does not change those groups; it only repeats selected training subsets under
new filenames so Ultralytics sees them more often.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def classify_image(name: str) -> str:
    if "_v5_" in name:
        return "real_v5"
    if "_v9_" in name:
        return "real_v9"
    if name.startswith("sim_"):
        return "sim"
    return "other"


def link(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"refusing to replace existing link: {dst}")
    os.symlink(src.resolve(), dst)


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def stage_image(
    image_path: Path,
    label_dir: Path,
    output: Path,
    split: str,
    suffix: str,
) -> None:
    label_path = label_dir / f"{image_path.stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    stem = f"{image_path.stem}{suffix}"
    link(image_path, output / "images" / split / f"{stem}{image_path.suffix.lower()}")
    link(label_path, output / "labels" / split / f"{stem}.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--v5-repeat", type=int, default=1)
    parser.add_argument("--v9-repeat", type=int, default=1)
    parser.add_argument("--sim-repeat", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.v5_repeat, args.v9_repeat, args.sim_repeat) < 1:
        raise SystemExit("repeat factors must be at least 1")

    source = Path(args.source).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing dataset: {output}")

    required = [
        source / "data.yaml",
        source / "classes.txt",
        source / "images" / "train",
        source / "labels" / "train",
        source / "images" / "val",
        source / "labels" / "val",
        source / "images" / "real_val",
        source / "images" / "sim_val",
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")

    for split in ("train", "val", "real_val", "sim_val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    repeat = {
        "real_v5": args.v5_repeat,
        "real_v9": args.v9_repeat,
        "sim": args.sim_repeat,
        "other": 1,
    }
    train_counts: Counter[str] = Counter()
    seen: set[str] = set()
    for image_path in image_files(source / "images" / "train"):
        category = classify_image(image_path.name)
        for index in range(repeat[category]):
            suffix = "" if repeat[category] == 1 else f"__r{index + 1:02d}"
            staged_name = f"{image_path.stem}{suffix}{image_path.suffix.lower()}"
            if staged_name in seen:
                raise RuntimeError(f"duplicate staged name: {staged_name}")
            seen.add(staged_name)
            stage_image(
                image_path,
                source / "labels" / "train",
                output,
                "train",
                suffix,
            )
            train_counts[category] += 1

    val_counts: Counter[str] = Counter()
    for image_path in image_files(source / "images" / "val"):
        category = classify_image(image_path.name)
        stage_image(
            image_path,
            source / "labels" / "val",
            output,
            "val",
            "",
        )
        val_counts[category] += 1

    for split in ("real_val", "sim_val"):
        source_label_split = "real_val" if split == "real_val" else "sim_val"
        for image_path in image_files(source / "images" / split):
            stage_image(
                image_path,
                source / "labels" / source_label_split,
                output,
                split,
                "",
            )

    class_names = [
        line.strip()
        for line in (source / "classes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (output / "classes.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )
    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(class_names)}",
                "names:",
                *[f"- {name}" for name in class_names],
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"replay-weighted dataset -> {output}")
    print(
        "  train: "
        + ", ".join(f"{name}={train_counts[name]}" for name in sorted(train_counts))
        + f", total={sum(train_counts.values())}"
    )
    print(
        "  val: "
        + ", ".join(f"{name}={val_counts[name]}" for name in sorted(val_counts))
        + f", total={sum(val_counts.values())}"
    )
    print(
        "  repeats: "
        f"real_v5={args.v5_repeat}, real_v9={args.v9_repeat}, sim={args.sim_repeat}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
