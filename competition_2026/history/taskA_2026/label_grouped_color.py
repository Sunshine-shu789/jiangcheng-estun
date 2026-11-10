#!/usr/bin/env python3
"""Label grouped captures by tracking a colored object inside each group ROI.

The camera pose may change between groups, but must stay fixed within a group.
Each group gets its own ROI and a per-frame HSV mask. The largest connected
foreground component inside that ROI becomes the detection.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    area: int
    fill_ratio: float
    touches_roi: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument(
        "--roi",
        action="append",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
        help="Repeat once per capture group.",
    )
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--hue-low", type=int, default=10)
    parser.add_argument("--hue-high", type=int, default=45)
    parser.add_argument(
        "--hue-range",
        action="append",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        help="Repeatable inclusive hue interval. Overrides --hue-low/--hue-high.",
    )
    parser.add_argument("--sat-min", type=int, default=90)
    parser.add_argument("--val-min", type=int, default=60)
    parser.add_argument("--padding", type=int, default=6)
    parser.add_argument("--close-kernel", type=int, default=17)
    parser.add_argument(
        "--merge-distance",
        type=int,
        default=0,
        help="Merge nearby same-color components into the object bounding box.",
    )
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--max-area-ratio", type=float, default=0.85)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def validate_roi(
    roi: list[int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SystemExit(
            f"ROI outside image: {roi}; image size is {width}x{height}"
        )
    return x1, y1, x2, y2


def detect_component(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    hue_ranges: list[tuple[int, int]],
    sat_min: int,
    val_min: int,
    close_kernel: int,
    merge_distance: int,
    min_area_ratio: float,
    max_area_ratio: float,
    padding: int,
) -> tuple[Detection | None, np.ndarray]:
    x1, y1, x2, y2 = roi
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hue_low, hue_high in hue_ranges:
        lower = np.array([hue_low, sat_min, val_min], dtype=np.uint8)
        upper = np.array([hue_high, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel, close_kernel),
        ),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    crop_height, crop_width = mask.shape
    crop_area = float(crop_width * crop_height)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[float, int]] = []
    component_boxes: dict[int, tuple[int, int, int, int]] = {}
    for component_id in range(1, count):
        component_x, component_y, component_width, component_height, area = (
            stats[component_id]
        )
        area_ratio = float(area) / crop_area
        if not min_area_ratio <= area_ratio <= max_area_ratio:
            continue
        if component_width < 12 or component_height < 12:
            continue

        center_x, center_y = centroids[component_id]
        distance = np.hypot(
            (center_x - crop_width / 2) / crop_width,
            (center_y - crop_height / 2) / crop_height,
        )
        score = float(area) * float(np.exp(-0.8 * distance))
        candidates.append((score, component_id))
        component_boxes[component_id] = (
            component_x,
            component_y,
            component_width,
            component_height,
        )

    if not candidates:
        return None, mask

    _, core_id = max(candidates)
    core_x, core_y, core_width, core_height = component_boxes[core_id]
    group_x1 = core_x - merge_distance
    group_y1 = core_y - merge_distance
    group_x2 = core_x + core_width - 1 + merge_distance
    group_y2 = core_y + core_height - 1 + merge_distance
    component_x1 = core_x
    component_y1 = core_y
    component_x2 = core_x + core_width - 1
    component_y2 = core_y + core_height - 1
    for _, component_id in candidates:
        if component_id == core_id:
            continue
        component_x, component_y, component_width, component_height = (
            component_boxes[component_id]
        )
        component_right = component_x + component_width - 1
        component_bottom = component_y + component_height - 1
        if (
            component_right < group_x1
            or component_x > group_x2
            or component_bottom < group_y1
            or component_y > group_y2
        ):
            continue
        component_x1 = min(component_x1, component_x)
        component_y1 = min(component_y1, component_y)
        component_x2 = max(component_x2, component_right)
        component_y2 = max(component_y2, component_bottom)

    area = int(
        np.count_nonzero(
            mask[component_y1 : component_y2 + 1, component_x1 : component_x2 + 1]
        )
    )
    absolute_x1 = x1 + int(component_x1)
    absolute_y1 = y1 + int(component_y1)
    absolute_x2 = x1 + int(component_x2)
    absolute_y2 = y1 + int(component_y2)
    padded_x1 = max(0, absolute_x1 - padding)
    padded_y1 = max(0, absolute_y1 - padding)
    padded_x2 = min(image.shape[1] - 1, absolute_x2 + padding)
    padded_y2 = min(image.shape[0] - 1, absolute_y2 + padding)
    box_width = padded_x2 - padded_x1 + 1
    box_height = padded_y2 - padded_y1 + 1
    detection = Detection(
        x1=padded_x1,
        y1=padded_y1,
        x2=padded_x2,
        y2=padded_y2,
        area=int(area),
        fill_ratio=float(area) / float(box_width * box_height),
        touches_roi=(
            absolute_x1 <= x1
            or absolute_y1 <= y1
            or absolute_x2 >= x2 - 1
            or absolute_y2 >= y2 - 1
        ),
    )
    return detection, mask


def make_contact_sheet(
    rows: list[tuple[Path, np.ndarray, Detection | None, int]],
    output: Path,
    group_size: int,
    columns: int = 5,
    tile_width: int = 320,
) -> None:
    if not rows:
        return
    tile_height = int(tile_width * 3 / 4)
    tiles: list[np.ndarray] = []
    for path, image, detection, group_index in rows:
        tile = cv2.resize(image, (tile_width, tile_height))
        if detection is not None:
            scale_x = tile_width / image.shape[1]
            scale_y = tile_height / image.shape[0]
            x1 = round(detection.x1 * scale_x)
            y1 = round(detection.y1 * scale_y)
            x2 = round(detection.x2 * scale_x)
            y2 = round(detection.y2 * scale_y)
            cv2.rectangle(tile, (x1, y1), (x2, y2), (54, 211, 153), 2)
        cv2.rectangle(tile, (0, 0), (tile_width, 25), (16, 20, 24), -1)
        cv2.putText(
            tile,
            f"G{group_index + 1} {path.stem[-21:]}",
            (7, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (245, 248, 250),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    blank = np.full((tile_height, tile_width, 3), 28, dtype=np.uint8)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    sheet = np.vstack(
        [
            np.hstack(tiles[index : index + columns])
            for index in range(0, len(tiles), columns)
        ]
    )
    for group_index in range(1, len(rows) // group_size):
        separator_y = group_index * group_size // columns * tile_height
        cv2.line(sheet, (0, separator_y), (sheet.shape[1] - 1, separator_y), (0, 215, 255), 4)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Failed to write contact sheet: {output}")


def main() -> int:
    args = parse_args()
    if not 0 <= args.class_id < len(CLASS_NAMES):
        raise SystemExit(
            f"--class-id must be between 0 and {len(CLASS_NAMES) - 1}"
        )
    if args.group_size <= 0:
        raise SystemExit("--group-size must be positive")
    if not 0 <= args.hue_low < args.hue_high <= 179:
        raise SystemExit("--hue-low and --hue-high are invalid")
    hue_ranges = (
        [tuple(item) for item in args.hue_range]
        if args.hue_range
        else [(args.hue_low, args.hue_high)]
    )
    for hue_low, hue_high in hue_ranges:
        if not 0 <= hue_low < hue_high <= 179:
            raise SystemExit(f"Invalid hue range: {hue_low}..{hue_high}")
    if not 0 <= args.sat_min <= 255 or not 0 <= args.val_min <= 255:
        raise SystemExit("--sat-min and --val-min must be between 0 and 255")
    if args.close_kernel <= 0 or args.close_kernel % 2 == 0:
        raise SystemExit("--close-kernel must be a positive odd number")
    if args.merge_distance < 0:
        raise SystemExit("--merge-distance cannot be negative")
    if args.padding < 0:
        raise SystemExit("--padding cannot be negative")
    if not 0 < args.min_area_ratio < args.max_area_ratio <= 1:
        raise SystemExit("area ratio range is invalid")

    source = Path(args.images).resolve()
    output = Path(args.out).resolve()
    files = image_files(source)
    if not files:
        raise SystemExit(f"No images found under {source}")
    groups = (len(files) + args.group_size - 1) // args.group_size
    if len(args.roi) != groups:
        raise SystemExit(
            f"Expected {groups} --roi values, received {len(args.roi)}"
        )

    for directory in (output / "images", output / "labels", output / "review"):
        directory.mkdir(parents=True, exist_ok=True)
    if args.debug:
        (output / "masks").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    previews: list[tuple[Path, np.ndarray, Detection | None, int]] = []
    accepted = 0
    warnings = 0
    for index, path in enumerate(files):
        group_index = index // args.group_size
        image = cv2.imread(str(path))
        if image is None:
            continue
        roi = validate_roi(args.roi[group_index], image.shape)
        detection, mask = detect_component(
            image=image,
            roi=roi,
            hue_ranges=hue_ranges,
            sat_min=args.sat_min,
            val_min=args.val_min,
            close_kernel=args.close_kernel,
            merge_distance=args.merge_distance,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            padding=args.padding,
        )

        height, width = image.shape[:2]
        label_path = output / "labels" / f"{path.stem}.txt"
        row: dict[str, object] = {
            "image": path.name,
            "group": group_index + 1,
            "accepted": 0,
            "warning": "",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "area": "",
            "fill_ratio": "",
        }
        if detection is None:
            label_path.write_text("", encoding="utf-8")
            warnings += 1
            row["warning"] = "no-detection"
        else:
            center_x = (detection.x1 + detection.x2) / 2 / width
            center_y = (detection.y1 + detection.y2) / 2 / height
            box_width = (detection.x2 - detection.x1) / width
            box_height = (detection.y2 - detection.y1) / height
            label_path.write_text(
                f"{args.class_id} {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}\n",
                encoding="utf-8",
            )
            row.update(
                {
                    "accepted": 1,
                    "x1": detection.x1,
                    "y1": detection.y1,
                    "x2": detection.x2,
                    "y2": detection.y2,
                    "area": detection.area,
                    "fill_ratio": round(detection.fill_ratio, 4),
                }
            )
            if detection.touches_roi:
                warnings += 1
                row["warning"] = "touches-roi"
            accepted += 1
            shutil.copy2(path, output / "images" / path.name)

        rows.append(row)
        previews.append((path, image, detection, group_index))
        if args.debug:
            cv2.imwrite(str(output / "masks" / f"{path.stem}.png"), mask)

    with (output / "review.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (output / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n",
        encoding="utf-8",
    )
    make_contact_sheet(
        rows=previews,
        output=output / "review" / "contact_sheet.jpg",
        group_size=args.group_size,
    )
    print(
        f"Grouped color labels complete: {accepted}/{len(files)} accepted, "
        f"{warnings} warnings -> {output}"
    )
    print(f"Review sheet: {output / 'review' / 'contact_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
