#!/usr/bin/env python3
"""Label static capture groups with one reviewed bounding box per group."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument(
        "--box",
        action="append",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
        help="按采集组顺序重复",
    )
    return parser.parse_args()


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def validate_box(
    box: list[int], shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SystemExit(f"box outside image bounds: {box}, image={width}x{height}")
    return x1, y1, x2, y2


def make_contact_sheet(
    rows: list[tuple[Path, np.ndarray, tuple[int, int, int, int]]],
    output: Path,
    columns: int = 5,
    tile_width: int = 320,
) -> None:
    tile_height = int(tile_width * 3 / 4)
    tiles: list[np.ndarray] = []
    for path, image, box in rows:
        tile = cv2.resize(image, (tile_width, tile_height))
        scale_x = tile_width / image.shape[1]
        scale_y = tile_height / image.shape[0]
        x1, y1, x2, y2 = box
        cv2.rectangle(
            tile,
            (round(x1 * scale_x), round(y1 * scale_y)),
            (round(x2 * scale_x), round(y2 * scale_y)),
            (54, 211, 153),
            3,
        )
        cv2.rectangle(tile, (0, 0), (tile_width, 25), (16, 20, 24), -1)
        cv2.putText(
            tile,
            path.stem[-17:],
            (7, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
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
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"cannot write contact sheet: {output}")


def main() -> int:
    args = parse_args()
    if args.class_id < 0:
        raise SystemExit("--class-id must be non-negative")
    if args.group_size <= 0:
        raise SystemExit("--group-size must be positive")

    source = Path(args.images).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")

    files = image_files(source)
    if not files:
        raise SystemExit(f"no images found in {source}")
    group_count = (len(files) + args.group_size - 1) // args.group_size
    if len(args.box) != group_count:
        raise SystemExit(
            f"expected {group_count} boxes for {len(files)} images, "
            f"received {len(args.box)}"
        )

    for directory in (
        output / "images",
        output / "labels",
        output / "review",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    previews: list[tuple[Path, np.ndarray, tuple[int, int, int, int]]] = []
    for index, path in enumerate(files):
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"cannot read image: {path}")
        box = validate_box(args.box[index // args.group_size], image.shape)
        x1, y1, x2, y2 = box
        height, width = image.shape[:2]
        center_x = (x1 + x2) / 2 / width
        center_y = (y1 + y2) / 2 / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        (output / "labels" / f"{path.stem}.txt").write_text(
            f"{args.class_id} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}\n",
            encoding="utf-8",
        )
        shutil.copy2(path, output / "images" / path.name)
        previews.append((path, image, box))
        rows.append(
            {
                "image": path.name,
                "accepted": 1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "group": index // args.group_size + 1,
            }
        )

    with (output / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "accepted",
                "x1",
                "y1",
                "x2",
                "y2",
                "group",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(previews, output / "review" / "contact_sheet.jpg")
    print(f"fixed-box labels: {len(files)} images -> {output}")
    print(output / "review" / "contact_sheet.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
