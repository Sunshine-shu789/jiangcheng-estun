#!/usr/bin/env python3
"""Create YOLO boxes for fixed-camera captures using a clean background frame.

Use this only when the camera pose and exposure stay stable across every
capture group. When the camera moves, use label_with_grabcut.py instead.
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
    area: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.003)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def normalize_exposure(background: np.ndarray, image: np.ndarray) -> np.ndarray:
    background_float = background.astype(np.float32)
    image_float = image.astype(np.float32)
    channel_shift = np.median(image_float - background_float, axis=(0, 1))
    adjusted = background_float + channel_shift
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def difference_mask(
    background: np.ndarray,
    image: np.ndarray,
    threshold: float,
) -> np.ndarray:
    background = cv2.GaussianBlur(background, (5, 5), 0)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    adjusted = normalize_exposure(background, image)
    diff = cv2.absdiff(adjusted, image)
    strength = diff.max(axis=2)
    mask = np.where(strength >= threshold, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    return cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )


def choose_detection(
    mask: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
) -> Detection | None:
    height, width = mask.shape
    frame_area = float(width * height)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates: list[Detection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area_ratio * frame_area <= area <= max_area_ratio * frame_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 16 or box_height < 16:
            continue
        center_x = x + box_width / 2
        center_y = y + box_height / 2
        distance = np.hypot(
            (center_x - width / 2) / width,
            (center_y - height / 2) / height,
        )
        score = area * float(np.exp(-1.5 * distance))
        candidates.append(
            Detection(
                x1=x,
                y1=y,
                x2=x + box_width - 1,
                y2=y + box_height - 1,
                area=area,
                score=score,
            )
        )
    return max(candidates, key=lambda item: item.score, default=None)


def expand_box(
    detection: Detection,
    width: int,
    height: int,
    padding: int,
) -> Detection:
    return Detection(
        x1=max(0, detection.x1 - padding),
        y1=max(0, detection.y1 - padding),
        x2=min(width - 1, detection.x2 + padding),
        y2=min(height - 1, detection.y2 + padding),
        area=detection.area,
        score=detection.score,
    )


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
    if args.threshold <= 0:
        raise SystemExit("--threshold 必须大于 0")
    if not 0 < args.min_area_ratio < args.max_area_ratio <= 1:
        raise SystemExit("面积比例范围无效")

    source = Path(args.images).resolve()
    output = Path(args.out).resolve()
    background = cv2.imread(str(Path(args.background).resolve()))
    if background is None:
        raise SystemExit(f"无法读取背景图：{args.background}")
    files = image_files(source)
    if not files:
        raise SystemExit(f"{source} 下没有图片")

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
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            continue
        if image.shape != background.shape:
            raise SystemExit(
                f"尺寸不一致：{path.name} 为 {image.shape}，"
                f"背景为 {background.shape}"
            )

        mask = difference_mask(background, image, args.threshold)
        detection = choose_detection(
            mask,
            args.min_area_ratio,
            args.max_area_ratio,
        )
        if detection is not None:
            detection = expand_box(
                detection,
                image.shape[1],
                image.shape[0],
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
                    "area": "",
                    "score": "",
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
                    "area": round(detection.area, 1),
                    "score": round(detection.score, 1),
                }
            )
            accepted += 1
            shutil.copy2(path, output / "images" / path.name)

        previews.append((path, image, detection))
        if args.debug:
            cv2.imwrite(str(output / "masks" / f"{path.stem}.png"), mask)

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
                "area",
                "score",
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
    print(f"背景差分标注完成：{accepted}/{len(files)} 张 -> {output}")
    print(f"复核总览：{output / 'review' / 'contact_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
