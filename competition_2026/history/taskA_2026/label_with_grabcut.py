#!/usr/bin/env python3
"""Label grouped single-object captures with ROI-guided GrabCut."""

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
    mask_area: int
    score: float


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
        help="按采集组顺序重复；坐标使用原图左上角原点",
    )
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-area-ratio", type=float, default=0.75)
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
            f"ROI 超出图像范围：{roi}，图像尺寸为 {width}x{height}"
        )
    return x1, y1, x2, y2


def segment_roi(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    iterations: int,
    min_area_ratio: float,
    max_area_ratio: float,
    padding: int,
) -> tuple[Detection | None, np.ndarray]:
    x1, y1, x2, y2 = roi
    crop = image[y1:y2, x1:x2]
    height, width = crop.shape[:2]
    frame_area = float(height * width)
    mask = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        crop,
        mask,
        (1, 1, width - 2, height - 2),
        background_model,
        foreground_model,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )
    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )

    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground)
    candidates: list[Detection] = []
    for component_id in range(1, count):
        component_x, component_y, component_width, component_height, area = (
            stats[component_id]
        )
        area_ratio = float(area) / frame_area
        if not min_area_ratio <= area_ratio <= max_area_ratio:
            continue
        if component_width < 15 or component_height < 15:
            continue

        center_x, center_y = centers[component_id]
        distance = np.hypot(
            (center_x - width / 2) / width,
            (center_y - height / 2) / height,
        )
        score = float(area) / (1.0 + 0.7 * distance)
        absolute_x1 = x1 + int(component_x)
        absolute_y1 = y1 + int(component_y)
        absolute_x2 = absolute_x1 + int(component_width) - 1
        absolute_y2 = absolute_y1 + int(component_height) - 1
        candidates.append(
            Detection(
                x1=max(0, absolute_x1 - padding),
                y1=max(0, absolute_y1 - padding),
                x2=min(image.shape[1] - 1, absolute_x2 + padding),
                y2=min(image.shape[0] - 1, absolute_y2 + padding),
                mask_area=int(area),
                score=score,
            )
        )

    return max(candidates, key=lambda item: item.score, default=None), foreground


def make_contact_sheet(
    rows: list[tuple[Path, np.ndarray, Detection | None]],
    output: Path,
    columns: int = 5,
    tile_width: int = 320,
) -> None:
    if not rows:
        return
    tile_height = int(tile_width * 3 / 4)
    tiles: list[np.ndarray] = []
    for path, image, detection in rows:
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
        raise RuntimeError(f"无法写入总览图：{output}")


def main() -> int:
    args = parse_args()
    if not 0 <= args.class_id < len(CLASS_NAMES):
        raise SystemExit(f"--class-id 必须在 0 到 {len(CLASS_NAMES) - 1} 之间")
    if args.group_size <= 0:
        raise SystemExit("--group-size 必须大于 0")
    if args.iterations <= 0:
        raise SystemExit("--iterations 必须大于 0")
    if args.padding < 0:
        raise SystemExit("--padding 不能小于 0")
    if not 0 < args.min_area_ratio < args.max_area_ratio <= 1:
        raise SystemExit("面积比例范围无效")

    source = Path(args.images).resolve()
    output = Path(args.out).resolve()
    files = image_files(source)
    if not files:
        raise SystemExit(f"{source} 下没有图片")
    groups = (len(files) + args.group_size - 1) // args.group_size
    if len(args.roi) < groups:
        raise SystemExit(
            f"共有 {groups} 个采集组，但只提供了 {len(args.roi)} 个 --roi"
        )

    for directory in (
        output / "images",
        output / "labels",
        output / "review",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if args.debug:
        (output / "masks").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    previews: list[tuple[Path, np.ndarray, Detection | None]] = []
    accepted = 0
    for index, path in enumerate(files):
        image = cv2.imread(str(path))
        if image is None:
            continue
        roi = validate_roi(args.roi[index // args.group_size], image.shape)
        detection, foreground = segment_roi(
            image,
            roi,
            args.iterations,
            args.min_area_ratio,
            args.max_area_ratio,
            args.padding,
        )

        height, width = image.shape[:2]
        label_path = output / "labels" / f"{path.stem}.txt"
        if detection is None:
            label_path.write_text("", encoding="utf-8")
            rows.append(
                {
                    "image": path.name,
                    "accepted": 0,
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "mask_area": "",
                    "score": "",
                    "roi": ",".join(str(value) for value in roi),
                }
            )
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
            rows.append(
                {
                    "image": path.name,
                    "accepted": 1,
                    "x1": detection.x1,
                    "y1": detection.y1,
                    "x2": detection.x2,
                    "y2": detection.y2,
                    "mask_area": detection.mask_area,
                    "score": round(detection.score, 1),
                    "roi": ",".join(str(value) for value in roi),
                }
            )
            accepted += 1
            shutil.copy2(path, output / "images" / path.name)

        previews.append((path, image, detection))
        if args.debug:
            cv2.imwrite(
                str(output / "masks" / f"{path.stem}.png"),
                foreground,
            )

    with (output / "review.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "accepted",
                "x1",
                "y1",
                "x2",
                "y2",
                "mask_area",
                "score",
                "roi",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    (output / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n",
        encoding="utf-8",
    )
    make_contact_sheet(
        previews,
        output / "review" / "contact_sheet.jpg",
    )
    print(f"ROI + GrabCut 标注完成：{accepted}/{len(files)} 张 -> {output}")
    print(f"复核总览：{output / 'review' / 'contact_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
