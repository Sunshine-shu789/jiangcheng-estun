#!/usr/bin/env python3
"""Verify a flat YOLO image/label directory without modifying it."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--class-id", required=True, type=int)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    return parser.parse_args()


def parse_single_box(
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
    root = Path(args.root).resolve()
    image_dir = root / "images"
    label_dir = root / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(f"missing images or labels directory under: {root}")

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    label_paths = sorted(
        path for path in label_dir.iterdir() if path.suffix == ".txt"
    )
    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}
    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)
        extra_labels = sorted(label_stems - image_stems)
        raise SystemExit(
            f"filename mismatch: missing labels={missing_labels}, "
            f"extra labels={extra_labels}"
        )
    if args.expected_count is not None and len(image_paths) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} samples, found {len(image_paths)}"
        )

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        center_x, center_y, box_width, box_height = parse_single_box(
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
                f"{image_path}: expected {args.expected_width}x"
                f"{args.expected_height}, found {width_px}x{height_px}"
            )

        box_left = round((center_x - box_width / 2) * width_px)
        box_top = round((center_y - box_height / 2) * height_px)
        box_right = round((center_x + box_width / 2) * width_px)
        box_bottom = round((center_y + box_height / 2) * height_px)
        if (
            box_left < 0
            or box_top < 0
            or box_right > width_px
            or box_bottom > height_px
        ):
            raise SystemExit(f"{label_path}: pixel box outside image")
        if box_right <= box_left or box_bottom <= box_top:
            raise SystemExit(f"{label_path}: empty pixel box")

    print(f"verified {len(image_paths)} samples -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
