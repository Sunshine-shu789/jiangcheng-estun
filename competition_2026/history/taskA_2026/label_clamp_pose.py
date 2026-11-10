#!/usr/bin/env python3
"""Label clamp bursts from the dark body with pose-specific fixed boxes."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASS_ID = 6
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
    roi: tuple[int, int, int, int]
    object_size: tuple[int, int]
    include_orange_tip: bool


# The black body is the position anchor. The clamp is silhouetted against a
# much lighter floor, so a low grayscale threshold gives a stable bbox while
# avoiding the floor reflection. Side views add the orange jaw tip; the front
# views keep only the dark body because their orange reflection extends down.
GROUP_SPECS = [
    GroupSpec(
        "closed_side_jaw_left",
        (230, 70, 480, 280),
        (168, 130),
        True,
    ),
    GroupSpec(
        "closed_side_jaw_right",
        (180, 90, 420, 300),
        (148, 116),
        True,
    ),
    GroupSpec(
        "closed_side_opposite",
        (225, 105, 460, 310),
        (164, 125),
        True,
    ),
    GroupSpec(
        "closed_side_opposite_jaw_left",
        (210, 75, 440, 285),
        (157, 120),
        True,
    ),
    GroupSpec(
        "closed_jaw_toward_camera",
        (240, 200, 410, 275),
        (117, 49),
        False,
    ),
    GroupSpec(
        "closed_jaw_away_from_camera",
        (230, 200, 450, 275),
        (166, 45),
        False,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--padding", type=int, default=3)
    parser.add_argument("--dark-threshold", type=int, default=80)
    parser.add_argument("--min-dark-area", type=int, default=20)
    parser.add_argument("--color-merge-distance", type=int, default=24)
    parser.add_argument("--hue-max", type=int, default=25)
    parser.add_argument("--sat-min", type=int, default=90)
    parser.add_argument("--val-min", type=int, default=70)
    parser.add_argument("--expected-per-group", type=int, default=12)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.padding < 0:
        raise SystemExit("--padding cannot be negative")
    if not 0 < args.dark_threshold < 256:
        raise SystemExit("--dark-threshold must be in [1, 255]")
    if args.min_dark_area <= 0:
        raise SystemExit("--min-dark-area must be positive")
    if args.color_merge_distance < 0:
        raise SystemExit("--color-merge-distance cannot be negative")
    if not 0 < args.hue_max < 90:
        raise SystemExit("--hue-max must be in [1, 89]")
    if not 0 <= args.sat_min <= 255 or not 0 <= args.val_min <= 255:
        raise SystemExit("--sat-min and --val-min must be in [0, 255]")
    if args.expected_per_group <= 0:
        raise SystemExit("--expected-per-group must be positive")


def validate_roi(
    roi: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SystemExit(f"ROI outside image: {roi}; image is {width}x{height}")
    return roi


def component_boxes(
    mask: np.ndarray,
    min_area: int,
) -> list[tuple[int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes: list[tuple[int, int, int, int]] = []
    for component_id in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[component_id]
        )
        if area < min_area:
            continue
        boxes.append((x, y, x + width - 1, y + height - 1))
    return boxes


def union_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    if not boxes:
        raise ValueError("cannot union an empty box list")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def rectangle_distance(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    horizontal = max(first[0] - second[2], second[0] - first[2], 0)
    vertical = max(first[1] - second[3], second[1] - first[3], 0)
    return float(np.hypot(horizontal, vertical))


def find_anchor_box(
    image: np.ndarray,
    spec: GroupSpec,
    args: argparse.Namespace,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    x1, y1, x2, y2 = validate_roi(spec.roi, image.shape)
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = np.where(gray < args.dark_threshold, 255, 0).astype(np.uint8)
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )

    dark_boxes = component_boxes(dark, args.min_dark_area)
    if not dark_boxes:
        raise SystemExit(f"{spec.name}: no dark clamp body found")
    dark_local = union_boxes(dark_boxes)
    dark_box = (
        dark_local[0] + x1,
        dark_local[1] + y1,
        dark_local[2] + x1,
        dark_local[3] + y1,
    )

    anchor_box = dark_box
    if spec.include_orange_tip:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red_low = cv2.inRange(
            hsv,
            np.array([0, args.sat_min, args.val_min], dtype=np.uint8),
            np.array([args.hue_max, 255, 255], dtype=np.uint8),
        )
        red_high = cv2.inRange(
            hsv,
            np.array([180 - args.hue_max, args.sat_min, args.val_min], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        orange = cv2.bitwise_or(red_low, red_high)
        orange = cv2.morphologyEx(
            orange,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        orange_boxes = [
            box
            for box in component_boxes(orange, 2)
            if rectangle_distance(box, dark_local) <= args.color_merge_distance
        ]
        if orange_boxes:
            anchor_box = union_boxes([dark_local, *orange_boxes])
            anchor_box = (
                anchor_box[0] + x1,
                anchor_box[1] + y1,
                anchor_box[2] + x1,
                anchor_box[3] + y1,
            )

    return anchor_box, dark_box


def make_object_box(
    image_shape: tuple[int, ...],
    anchor_box: tuple[int, int, int, int],
    spec: GroupSpec,
    padding: int,
) -> tuple[int, int, int, int, bool]:
    height, width = image_shape[:2]
    center_x = (anchor_box[0] + anchor_box[2]) / 2
    center_y = (anchor_box[1] + anchor_box[3]) / 2
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
        if len(image_paths) != args.expected_per_group:
            raise SystemExit(
                f"{spec.name}: expected {args.expected_per_group} images, "
                f"found {len(image_paths)}"
            )

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
            anchor_box, dark_box = find_anchor_box(image, spec, args)
            x1, y1, x2, y2, clipped = make_object_box(
                image.shape,
                anchor_box,
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
                    "anchor_x1": anchor_box[0],
                    "anchor_y1": anchor_box[1],
                    "anchor_x2": anchor_box[2],
                    "anchor_y2": anchor_box[3],
                    "dark_x1": dark_box[0],
                    "dark_y1": dark_box[1],
                    "dark_x2": dark_box[2],
                    "dark_y2": dark_box[3],
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
    print(f"clamp labels: {len(rows)} images, {warnings} warnings -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
