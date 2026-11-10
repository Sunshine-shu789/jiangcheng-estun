#!/usr/bin/env python3
"""Stage the reviewed v5 captures as a clean real-data source set."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


CLASS_NAMES = [
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
]

SOURCES = (
    ("cheezit", "1-Cheez-it", 0),
    ("scissors", "3-Scissors", 2),
    ("mustard", "4-Frenchs_Mustard", 3),
    ("clamp", "7-Clamp", 6),
    ("banana", "8-Plastic_Banana", 7),
    ("meat_can", "10-meat_can", 9),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def validate_label(path: Path, expected_class: int) -> None:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) != 1:
        raise ValueError(f"{path}: expected one box, found {len(lines)}")

    fields = lines[0].split()
    if len(fields) != 5:
        raise ValueError(f"{path}: expected 5 fields, found {len(fields)}")
    try:
        class_id = int(fields[0])
        cx, cy, width, height = (float(value) for value in fields[1:])
    except ValueError as exc:
        raise ValueError(f"{path}: non-numeric label") from exc

    if class_id != expected_class:
        raise ValueError(
            f"{path}: class {class_id} does not match expected {expected_class}"
        )
    if not all(math.isfinite(value) for value in (cx, cy, width, height)):
        raise ValueError(f"{path}: non-finite coordinate")
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        raise ValueError(f"{path}: center coordinate outside [0, 1]")
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError(f"{path}: size outside (0, 1]")
    if cx - width / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6:
        raise ValueError(f"{path}: horizontal box outside image")
    if cy - height / 2 < -1e-6 or cy + height / 2 > 1 + 1e-6:
        raise ValueError(f"{path}: vertical box outside image")


def main() -> int:
    args = parse_args()
    batch_root = Path(args.batch_root).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing staging directory: {out}")

    image_dir = out / "images"
    label_dir = out / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    approved_root = batch_root / "labeled_group_color_v1"
    rows: list[dict[str, str | int]] = []
    counts = {class_id: 0 for class_id in range(len(CLASS_NAMES))}

    for source, class_name, class_id in SOURCES:
        class_root = approved_root / class_name
        image_root = class_root / "images"
        label_root = class_root / "labels"
        images = sorted(
            path
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(images) != 60:
            raise SystemExit(
                f"{class_name}: expected 60 approved images, found {len(images)}"
            )

        for image_path in images:
            label_path = label_root / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise SystemExit(f"missing approved label: {label_path}")
            validate_label(label_path, class_id)

            suffix = image_path.suffix.lower()
            staged_image_name = f"real_c{class_id}_{source}_{image_path.stem}{suffix}"
            staged_label_name = f"real_c{class_id}_{source}_{label_path.stem}.txt"
            os.symlink(image_path, image_dir / staged_image_name)
            os.symlink(label_path, label_dir / staged_label_name)
            rows.append(
                {
                    "source": source,
                    "class_id": class_id,
                    "class_name": class_name,
                    "image": image_path.name,
                    "label": label_path.name,
                }
            )
            counts[class_id] += 1

    (out / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n", encoding="utf-8"
    )
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "class_id", "class_name", "image", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"staged reviewed v5 samples -> {out}")
    for class_id, count in counts.items():
        if count:
            print(f"  class {class_id} {CLASS_NAMES[class_id]}: {count}")
    print(f"  total: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
