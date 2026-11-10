#!/usr/bin/env python3
"""Create a contact sheet with pixel-coordinate guides for capture review."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=int, default=80)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("images", nargs="+")
    return parser.parse_args()


def draw_grid(image: np.ndarray, step: int) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for x in range(0, width, step):
        cv2.line(canvas, (x, 0), (x, height - 1), (50, 220, 255), 1)
        cv2.putText(
            canvas,
            str(x),
            (x + 3, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(x),
            (x + 3, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (50, 220, 255),
            1,
            cv2.LINE_AA,
        )
    for y in range(0, height, step):
        cv2.line(canvas, (0, y), (width - 1, y), (50, 220, 255), 1)
        cv2.putText(
            canvas,
            str(y),
            (3, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(y),
            (3, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (50, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> int:
    args = parse_args()
    if args.step <= 0 or args.columns <= 0:
        raise SystemExit("--step and --columns must be positive")

    tiles = []
    width = height = 0
    for image_name in args.images:
        path = Path(image_name)
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"cannot read image: {path}")
        if not tiles:
            height, width = image.shape[:2]
        elif image.shape[:2] != (height, width):
            raise SystemExit(f"image size differs from first tile: {path}")
        tile = draw_grid(image, args.step)
        cv2.rectangle(tile, (0, 0), (width, 24), (20, 20, 20), -1)
        cv2.putText(
            tile,
            path.stem,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    blank = np.full((height, width, 3), 35, dtype=np.uint8)
    while len(tiles) % args.columns:
        tiles.append(blank.copy())
    rows = [
        np.hstack(tiles[index : index + args.columns])
        for index in range(0, len(tiles), args.columns)
    ]
    sheet = np.vstack(rows)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"cannot write contact sheet: {output}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
