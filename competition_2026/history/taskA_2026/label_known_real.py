#!/usr/bin/env python3
"""Label single-object real captures with a generic detector and task class."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


TASK_CLASSES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-class-id", type=int, required=True)
    parser.add_argument("--source-class", default="bottle")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def make_contact_sheet(
    rows: list[tuple[Path, np.ndarray]],
    output: Path,
    columns: int = 5,
    tile_width: int = 320,
) -> None:
    if not rows:
        return
    tile_height = int(tile_width * 3 / 4)
    tiles: list[np.ndarray] = []
    for path, image in rows:
        tile = cv2.resize(image, (tile_width, tile_height))
        cv2.rectangle(tile, (0, 0), (tile_width, 25), (16, 20, 24), -1)
        cv2.putText(
            tile,
            path.stem[-17:],
            (7, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 248, 250),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    blank = np.full((tile_height, tile_width, 3), 28, dtype=np.uint8)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    grid_rows = [
        np.hstack(tiles[index : index + columns])
        for index in range(0, len(tiles), columns)
    ]
    sheet = np.vstack(grid_rows)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"无法写入总览图：{output}")


def main() -> int:
    args = parse_args()
    if not 0 <= args.target_class_id < len(TASK_CLASSES):
        raise SystemExit("--target-class-id 超出任务类别范围")
    if not 0 < args.conf <= 1:
        raise SystemExit("--conf 必须在 0 到 1 之间")

    source = Path(args.images).resolve()
    output = Path(args.out).resolve()
    image_out = output / "images"
    label_out = output / "labels"
    review_out = output / "review"
    for directory in (image_out, label_out, review_out):
        directory.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path
        for path in source.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise SystemExit(f"{source} 下没有图片")

    model = YOLO(args.model)
    rows: list[dict[str, object]] = []
    review_rows: list[tuple[Path, np.ndarray]] = []
    accepted = 0
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            continue

        result = model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            device=args.device,
            verbose=False,
        )[0]
        candidates: list[tuple[float, float, np.ndarray]] = []
        for box, class_id, confidence in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            class_name = str(result.names[int(class_id)])
            if class_name != args.source_class:
                continue
            x1, y1, x2, y2 = box
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            candidates.append((float(confidence), area, box))

        if not candidates:
            rows.append(
                {
                    "image": path.name,
                    "accepted": 0,
                    "confidence": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                }
            )
            continue

        confidence, _, box = max(candidates, key=lambda item: (item[0], item[1]))
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in box)
        x1 = max(0.0, min(x1, width - 1))
        y1 = max(0.0, min(y1, height - 1))
        x2 = max(0.0, min(x2, width - 1))
        y2 = max(0.0, min(y2, height - 1))
        if x2 - x1 < 12 or y2 - y1 < 24:
            rows.append(
                {
                    "image": path.name,
                    "accepted": 0,
                    "confidence": round(confidence, 4),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                }
            )
            continue

        center_x = (x1 + x2) / 2 / width
        center_y = (y1 + y2) / 2 / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        label_path = label_out / f"{path.stem}.txt"
        label_path.write_text(
            f"{args.target_class_id} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}\n",
            encoding="utf-8",
        )
        shutil.copy2(path, image_out / path.name)

        annotated = image.copy()
        cv2.rectangle(
            annotated,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (54, 211, 153),
            3,
        )
        cv2.putText(
            annotated,
            f"{TASK_CLASSES[args.target_class_id]} {confidence:.1%}",
            (int(round(x1)), max(22, int(round(y1)) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"{TASK_CLASSES[args.target_class_id]} {confidence:.1%}",
            (int(round(x1)), max(22, int(round(y1)) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 240, 170),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(review_out / path.name), annotated)
        review_rows.append((path, annotated))
        accepted += 1
        rows.append(
            {
                "image": path.name,
                "accepted": 1,
                "confidence": round(confidence, 4),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )

    (output / "classes.txt").write_text(
        "\n".join(TASK_CLASSES) + "\n",
        encoding="utf-8",
    )
    with (output / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "accepted",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(review_rows, output / "review_sheet.jpg")
    print(f"自动标注完成：{accepted}/{len(files)} 张 -> {output}")
    print(f"总览：{output / 'review_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
