#!/usr/bin/env python3
"""Stage the reviewed v5 and v9 real captures as one clean source set.

Only the explicitly promoted ``labeled_group_color_v1`` directories are read.
Every source image must have exactly one complete, correctly classified YOLO
box before it is linked into the output directory.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Capture:
    batch: str
    key: str
    class_name: str
    class_id: int
    expected_count: int


CAPTURES = (
    Capture("v5", "cheezit", "1-Cheez-it", 0, 60),
    Capture("v5", "scissors", "3-Scissors", 2, 60),
    Capture("v5", "mustard", "4-Frenchs_Mustard", 3, 60),
    Capture("v5", "clamp", "7-Clamp", 6, 60),
    Capture("v5", "banana", "8-Plastic_Banana", 7, 60),
    Capture("v5", "meat_can", "10-meat_can", 9, 60),
    Capture("v9", "cheezit", "1-Cheez-it", 0, 48),
    Capture("v9", "scissors", "3-Scissors", 2, 84),
    Capture("v9", "mustard", "4-Frenchs_Mustard", 3, 96),
    Capture("v9", "clamp", "7-Clamp", 6, 72),
    Capture("v9", "banana", "8-Plastic_Banana", 7, 48),
    Capture("v9", "meat_can", "10-meat_can", 9, 48),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v5-batch-root",
        default="/media/sunshine/新加卷/taskA/real_batch_v5_20260926",
    )
    parser.add_argument(
        "--v9-batch-root",
        default="/media/sunshine/新加卷/taskA/real_batch_v9_20260927",
    )
    parser.add_argument(
        "--approved-subdir",
        default="labeled_group_color_v1",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_label(
    label_path: Path,
    expected_class: int,
) -> tuple[float, float, float, float]:
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(f"{label_path}: expected one box, found {len(lines)}")

    fields = lines[0].split()
    if len(fields) != 5:
        raise ValueError(f"{label_path}: expected 5 fields, found {len(fields)}")
    try:
        class_id = int(fields[0])
        center_x, center_y, width, height = (
            float(value) for value in fields[1:]
        )
    except ValueError as exc:
        raise ValueError(f"{label_path}: non-numeric label") from exc

    if class_id != expected_class:
        raise ValueError(
            f"{label_path}: class {class_id} does not match {expected_class}"
        )

    values = (center_x, center_y, width, height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label_path}: non-finite coordinate")
    if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
        raise ValueError(f"{label_path}: center outside [0, 1]")
    if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
        raise ValueError(f"{label_path}: size outside (0, 1]")
    if (
        center_x - width / 2 < -1e-6
        or center_x + width / 2 > 1.0 + 1e-6
        or center_y - height / 2 < -1e-6
        or center_y + height / 2 > 1.0 + 1e-6
    ):
        raise ValueError(f"{label_path}: box outside image")
    return center_x, center_y, width, height


def source_root(
    batch_root: Path,
    approved_subdir: str,
    capture: Capture,
) -> Path:
    return batch_root / approved_subdir / capture.class_name


def stage_capture(
    capture: Capture,
    batch_root: Path,
    approved_subdir: str,
    image_dir: Path,
    label_dir: Path,
    seen_stems: set[str],
) -> list[dict[str, str | int]]:
    class_root = source_root(batch_root, approved_subdir, capture)
    image_root = class_root / "images"
    label_root = class_root / "labels"
    images = image_files(image_root)
    labels = sorted(path for path in label_root.iterdir() if path.suffix == ".txt")

    if len(images) != capture.expected_count:
        raise SystemExit(
            f"{capture.batch}/{capture.class_name}: expected "
            f"{capture.expected_count} approved images, found {len(images)}"
        )
    if {path.stem for path in images} != {path.stem for path in labels}:
        raise SystemExit(
            f"{capture.batch}/{capture.class_name}: image/label filename mismatch"
        )

    source_key = f"{capture.batch}_{capture.key}"
    rows: list[dict[str, str | int]] = []
    for image_path in images:
        label_path = label_root / f"{image_path.stem}.txt"
        parse_label(label_path, capture.class_id)

        staged_stem = f"real_c{capture.class_id}_{source_key}_{image_path.stem}"
        if staged_stem in seen_stems:
            raise SystemExit(f"duplicate staged stem: {staged_stem}")
        seen_stems.add(staged_stem)

        staged_image_name = f"{staged_stem}{image_path.suffix.lower()}"
        staged_label_name = f"{staged_stem}.txt"
        os.symlink(image_path.resolve(), image_dir / staged_image_name)
        os.symlink(label_path.resolve(), label_dir / staged_label_name)
        rows.append(
            {
                "source": source_key,
                "batch": capture.batch,
                "class_id": capture.class_id,
                "class_name": capture.class_name,
                "image": image_path.name,
                "label": label_path.name,
                "staged_image": staged_image_name,
                "staged_label": staged_label_name,
            }
        )
    return rows


def verify_staging(
    out: Path,
    expected_total: int,
    expected_classes: dict[int, int],
) -> tuple[dict[int, int], int]:
    image_dir = out / "images"
    label_dir = out / "labels"
    images = image_files(image_dir)
    labels = sorted(path for path in label_dir.iterdir() if path.suffix == ".txt")
    if {path.stem for path in images} != {path.stem for path in labels}:
        raise SystemExit("staged image/label filename mismatch")
    if len(images) != expected_total:
        raise SystemExit(
            f"staged output: expected {expected_total} images, found {len(images)}"
        )

    class_counts = {class_id: 0 for class_id in expected_classes}
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(lines) != 1:
            raise SystemExit(
                f"{label_path}: expected one box, found {len(lines)}"
            )
        fields = lines[0].split()
        if len(fields) != 5:
            raise SystemExit(
                f"{label_path}: expected 5 fields, found {len(fields)}"
            )
        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise SystemExit(f"{label_path}: non-integer class id") from exc
        if class_id not in expected_classes:
            raise SystemExit(f"{label_path}: unexpected class id {class_id}")
        parse_label(label_path, class_id)
        class_counts[class_id] += 1

    if class_counts != expected_classes:
        raise SystemExit(
            f"staged class counts differ: expected {expected_classes}, "
            f"found {class_counts}"
        )
    return class_counts, len(images)


def main() -> int:
    args = parse_args()
    v5_batch_root = Path(args.v5_batch_root).resolve()
    v9_batch_root = Path(args.v9_batch_root).resolve()
    out = Path(args.out).resolve()
    work_dir = out.with_name(f".{out.name}.tmp")
    if out.exists() or out.is_symlink():
        raise SystemExit(f"refusing to overwrite existing staging directory: {out}")
    if work_dir.exists() or work_dir.is_symlink():
        raise SystemExit(f"refusing to overwrite existing temporary directory: {work_dir}")

    expected_classes = {class_id: 0 for class_id in range(len(CLASS_NAMES))}
    for capture in CAPTURES:
        expected_classes[capture.class_id] += capture.expected_count
    expected_total = sum(expected_classes.values())

    image_dir = work_dir / "images"
    label_dir = work_dir / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    rows: list[dict[str, str | int]] = []
    seen_stems: set[str] = set()
    for capture in CAPTURES:
        batch_root = v5_batch_root if capture.batch == "v5" else v9_batch_root
        rows.extend(
            stage_capture(
                capture,
                batch_root,
                args.approved_subdir,
                image_dir,
                label_dir,
                seen_stems,
            )
        )

    (work_dir / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n",
        encoding="utf-8",
    )
    with (work_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "batch",
                "class_id",
                "class_name",
                "image",
                "label",
                "staged_image",
                "staged_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    class_counts, total = verify_staging(
        work_dir,
        expected_total,
        expected_classes,
    )
    batch_counts = {"v5": 0, "v9": 0}
    source_counts: dict[str, int] = {}
    for row in rows:
        batch = str(row["batch"])
        source = str(row["source"])
        batch_counts[batch] += 1
        source_counts[source] = source_counts.get(source, 0) + 1

    os.replace(work_dir, out)
    print(f"staged reviewed v5+v9 real samples -> {out}")
    for class_id, count in class_counts.items():
        if count:
            print(f"  class {class_id} {CLASS_NAMES[class_id]}: {count}")
    for source in sorted(source_counts):
        print(f"  {source}: {source_counts[source]}")
    print(f"  batch v5: {batch_counts['v5']}")
    print(f"  batch v9: {batch_counts['v9']}")
    print(f"  total: {total}")
    print("  verification: one box/image, class id, five fields, finite/in-frame box")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
