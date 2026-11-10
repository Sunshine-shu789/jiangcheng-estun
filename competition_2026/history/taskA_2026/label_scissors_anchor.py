#!/usr/bin/env python3
"""Label scissors bursts by translating a pose-specific box from a color anchor."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASS_ID = 2
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


@dataclass(frozen=True)
class GroupSpec:
    name: str
    anchor_roi: tuple[int, int, int, int]
    center_offset: tuple[float, float]
    object_size: tuple[int, int]


# The blue handle segment is a stable anchor. Each pose uses a fixed object
# size and center offset because the camera and scissors geometry stay fixed
# within a burst.
GROUP_SPECS = [
    GroupSpec(
        "front_closed",
        (280, 40, 470, 290),
        (2.0, -35.0),
        (82, 170),
    ),
    GroupSpec(
        "back_closed",
        (280, 40, 470, 290),
        (8.0, -30.0),
        (86, 171),
    ),
    GroupSpec(
        "side_closed",
        (150, 205, 430, 275),
        (-42.0, 2.0),
        (190, 14),
    ),
    GroupSpec(
        "front_tip_down",
        (280, 40, 470, 330),
        (-2.0, 31.0),
        (80, 170),
    ),
    GroupSpec(
        "back_tip_down",
        (280, 40, 470, 330),
        (1.0, 28.0),
        (78, 168),
    ),
    GroupSpec(
        "front_open",
        (270, 40, 500, 300),
        (6.0, -22.0),
        (106, 170),
    ),
    GroupSpec(
        "back_open",
        (270, 40, 500, 300),
        (15.0, -33.0),
        (109, 172),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--padding", type=int, default=6)
    parser.add_argument("--hue-low", type=int, default=85)
    parser.add_argument("--hue-high", type=int, default=150)
    parser.add_argument("--sat-min", type=int, default=35)
    parser.add_argument("--val-min", type=int, default=25)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=25)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.padding < 0:
        raise SystemExit("--padding cannot be negative")
    if not 0 <= args.hue_low < args.hue_high <= 179:
        raise SystemExit("hue range is invalid")
    if not 0 <= args.sat_min <= 255 or not 0 <= args.val_min <= 255:
        raise SystemExit("saturation and value minima must be in [0, 255]")
    if args.close_kernel <= 0 or args.close_kernel % 2 == 0:
        raise SystemExit("--close-kernel must be a positive odd number")
    if args.min_area <= 0:
        raise SystemExit("--min-area must be positive")


def validate_roi(
    roi: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SystemExit(f"ROI outside image: {roi}; image is {width}x{height}")
    return roi


def find_anchor(
    image: np.ndarray,
    spec: GroupSpec,
    args: argparse.Namespace,
) -> tuple[float, float, tuple[int, int, int, int], float]:
    x1, y1, x2, y2 = validate_roi(spec.anchor_roi, image.shape)
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([args.hue_low, args.sat_min, args.val_min], dtype=np.uint8),
        np.array([args.hue_high, 255, 255], dtype=np.uint8),
    )
    kernel = np.ones((args.close_kernel, args.close_kernel), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[int, int, int, int, int]] = []
    for component_id in range(1, count):
        component_x, component_y, width, height, area = (
            int(value) for value in stats[component_id]
        )
        if area < args.min_area or width < 8 or height < 3:
            continue
        candidates.append(
            (
                area,
                component_x + x1,
                component_y + y1,
                width,
                height,
            )
        )
    if not candidates:
        raise SystemExit(
            f"{spec.name}: no blue-handle anchor found inside {spec.anchor_roi}"
        )

    area, anchor_x, anchor_y, anchor_width, anchor_height = max(candidates)
    anchor_center_x = anchor_x + (anchor_width - 1) / 2
    anchor_center_y = anchor_y + (anchor_height - 1) / 2
    return (
        anchor_center_x,
        anchor_center_y,
        (anchor_x, anchor_y, anchor_width, anchor_height),
        float(area),
    )


def make_object_box(
    image_shape: tuple[int, ...],
    anchor_center: tuple[float, float],
    spec: GroupSpec,
    padding: int,
) -> tuple[int, int, int, int, bool]:
    height, width = image_shape[:2]
    center_x = anchor_center[0] + spec.center_offset[0]
    center_y = anchor_center[1] + spec.center_offset[1]
    box_width = spec.object_size[0] + 2 * padding
    box_height = spec.object_size[1] + 2 * padding

    x1 = round(center_x - (box_width - 1) / 2)
    y1 = round(center_y - (box_height - 1) / 2)
    x2 = x1 + box_width - 1
    y2 = y1 + box_height - 1
    clipped = x1 < 0 or y1 < 0 or x2 >= width or y2 >= height
    return (
        max(0, x1),
        max(0, y1),
        min(width - 1, x2),
        min(height - 1, y2),
        clipped,
    )


def make_contact_sheet(
    tiles: list[np.ndarray],
    output: Path,
    columns: int = 4,
) -> None:
    if not tiles:
        raise SystemExit("cannot create an empty review sheet")
    tile_height, tile_width = tiles[0].shape[:2]
    blank = np.full((tile_height, tile_width, 3), 28, dtype=np.uint8)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    sheet = np.vstack(
        [
            np.hstack(tiles[index : index + columns])
            for index in range(0, len(tiles), columns)
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"cannot write review sheet: {output}")


def main() -> int:
    args = parse_args()
    validate_args(args)

    source = Path(args.images_root).resolve()
    output = Path(args.out).resolve()
    if not source.is_dir():
        raise SystemExit(f"missing image root: {source}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, object]] = []

    for spec in GROUP_SPECS:
        group_dir = source / spec.name
        image_paths = sorted(
            path
            for path in group_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not image_paths:
            raise SystemExit(f"missing burst images: {group_dir}")

        destination = output / spec.name
        image_dir = destination / "images"
        label_dir = destination / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        tiles: list[np.ndarray] = []
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                raise SystemExit(f"cannot read image: {path}")
            anchor_x, anchor_y, anchor_bbox, anchor_area = find_anchor(
                image,
                spec,
                args,
            )
            x1, y1, x2, y2, clipped = make_object_box(
                image.shape,
                (anchor_x, anchor_y),
                spec,
                args.padding,
            )
            if x2 <= x1 or y2 <= y1:
                raise SystemExit(f"invalid object box for {path}")

            height, width = image.shape[:2]
            center_x = (x1 + x2) / 2 / width
            center_y = (y1 + y2) / 2 / height
            box_width = (x2 - x1 + 1) / width
            box_height = (y2 - y1 + 1) / height
            label_path = label_dir / f"{path.stem}.txt"
            label_path.write_text(
                f"{CLASS_ID} {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}\n",
                encoding="utf-8",
            )
            shutil.copy2(path, image_dir / path.name)

            annotated = image.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (54, 211, 153), 2)
            cv2.circle(
                annotated,
                (round(anchor_x), round(anchor_y)),
                3,
                (0, 215, 255),
                -1,
            )
            tile = cv2.resize(annotated, (320, 240))
            cv2.rectangle(tile, (0, 0), (319, 25), (16, 20, 24), -1)
            cv2.putText(
                tile,
                path.stem[-18:],
                (7, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (245, 248, 250),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)

            rows.append(
                {
                    "group": spec.name,
                    "image": path.name,
                    "class_id": CLASS_ID,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "anchor_x1": anchor_bbox[0],
                    "anchor_y1": anchor_bbox[1],
                    "anchor_width": anchor_bbox[2],
                    "anchor_height": anchor_bbox[3],
                    "anchor_area": round(anchor_area, 1),
                    "warning": "clipped-box" if clipped else "",
                }
            )

        make_contact_sheet(
            tiles,
            destination / "review" / "contact_sheet.jpg",
        )

    with (output / "review.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    warnings = sum(bool(row["warning"]) for row in rows)
    print(f"scissors labels: {len(rows)} images, {warnings} warnings -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
