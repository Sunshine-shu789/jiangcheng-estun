#!/usr/bin/env python3
"""Promote reviewed grouped labels into one flat image/label directory."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--class-id", required=True, type=int)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    return parser.parse_args()


def validate_label(
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
    if width <= 0.0 or width > 1.0 or height <= 0.0 or height > 1.0:
        raise ValueError(f"{label_path}: size outside (0, 1]")
    if (
        center_x - width / 2 < -1e-6
        or center_x + width / 2 > 1.0 + 1e-6
        or center_y - height / 2 < -1e-6
        or center_y + height / 2 > 1.0 + 1e-6
    ):
        raise ValueError(f"{label_path}: box outside image")
    return center_x, center_y, width, height


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    destination = Path(args.dest).resolve()
    if not source.is_dir():
        raise SystemExit(f"missing source directory: {source}")
    if destination.exists():
        raise SystemExit(f"refusing to overwrite existing directory: {destination}")

    image_groups = sorted(
        path for path in source.glob("*/images") if path.is_dir()
    )
    if not image_groups:
        raise SystemExit(f"no grouped image directories found under: {source}")

    samples: list[tuple[Path, Path, str]] = []
    seen_stems: set[str] = set()
    for group_image_dir in image_groups:
        group_dir = group_image_dir.parent
        group_label_dir = group_dir / "labels"
        if not group_label_dir.is_dir():
            raise SystemExit(f"missing label directory: {group_label_dir}")

        image_paths = sorted(
            path
            for path in group_image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise SystemExit(f"no images found in: {group_image_dir}")

        image_stems = {path.stem for path in image_paths}
        label_paths = sorted(
            path for path in group_label_dir.iterdir() if path.suffix == ".txt"
        )
        label_stems = {path.stem for path in label_paths}
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)
            extra_labels = sorted(label_stems - image_stems)
            raise SystemExit(
                f"filename mismatch in {group_dir}: "
                f"missing labels={missing_labels}, extra labels={extra_labels}"
            )

        for image_path in image_paths:
            if image_path.stem in seen_stems:
                raise SystemExit(f"duplicate image stem: {image_path.stem}")
            seen_stems.add(image_path.stem)

            label_path = group_label_dir / f"{image_path.stem}.txt"
            center_x, center_y, width, height = validate_label(
                label_path,
                args.class_id,
            )

            image = cv2.imread(str(image_path))
            if image is None:
                raise SystemExit(f"cannot read image: {image_path}")
            height_px, width_px = image.shape[:2]
            if (
                args.expected_width is not None
                and width_px != args.expected_width
            ) or (
                args.expected_height is not None
                and height_px != args.expected_height
            ):
                raise SystemExit(
                    f"{image_path}: expected "
                    f"{args.expected_width}x{args.expected_height}, "
                    f"found {width_px}x{height_px}"
                )

            box_left = round((center_x - width / 2) * width_px)
            box_top = round((center_y - height / 2) * height_px)
            box_right = round((center_x + width / 2) * width_px)
            box_bottom = round((center_y + height / 2) * height_px)
            if (
                box_left < 0
                or box_top < 0
                or box_right > width_px
                or box_bottom > height_px
            ):
                raise SystemExit(f"{label_path}: pixel box outside image")
            if box_right <= box_left or box_bottom <= box_top:
                raise SystemExit(f"{label_path}: empty pixel box")

            samples.append((image_path, label_path, group_dir.name))

    if args.expected_count is not None and len(samples) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} samples, found {len(samples)}"
        )

    destination_image_dir = destination / "images"
    destination_label_dir = destination / "labels"
    destination_image_dir.mkdir(parents=True)
    destination_label_dir.mkdir(parents=True)

    for image_path, label_path, _ in samples:
        shutil.copy2(image_path, destination_image_dir / image_path.name)
        shutil.copy2(label_path, destination_label_dir / label_path.name)

    classes_path = source / "classes.txt"
    if classes_path.is_file():
        shutil.copy2(classes_path, destination / "classes.txt")

    print(f"promoted {len(samples)} samples -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
