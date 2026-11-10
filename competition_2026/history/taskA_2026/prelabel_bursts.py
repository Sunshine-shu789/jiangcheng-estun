#!/usr/bin/env python3
"""Create reviewable YOLO label candidates for grouped burst captures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def group_images(root: Path) -> list[tuple[str, list[Path]]]:
    groups: list[tuple[str, list[Path]]] = []
    root_files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if root_files:
        groups.append((root.name, root_files))

    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if files:
            groups.append((directory.name, files))
    return groups


def write_sheet(
    tiles: list[np.ndarray],
    output: Path,
    columns: int = 4,
) -> None:
    if not tiles:
        return
    height, width = tiles[0].shape[:2]
    blank = np.full((height, width, 3), 28, dtype=np.uint8)
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


def annotate(
    image: np.ndarray,
    boxes: list[tuple[int, float, tuple[float, float, float, float]]],
    expected_class_id: int,
    selected_box: tuple[float, float, float, float] | None,
) -> np.ndarray:
    canvas = image.copy()
    for class_id, confidence, box in boxes:
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        color = (80, 220, 140) if class_id == expected_class_id else (60, 60, 235)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[class_id]} {confidence:.1%}"
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (15, 15, 15),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    if selected_box is None:
        cv2.putText(
            canvas,
            f"MISSING {CLASS_NAMES[expected_class_id]}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"MISSING {CLASS_NAMES[expected_class_id]}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (60, 60, 235),
            2,
            cv2.LINE_AA,
        )
    return canvas


def make_tile(
    image: np.ndarray,
    path: Path,
    status: str,
    confidence: float | None,
    tile_width: int = 320,
) -> np.ndarray:
    tile_height = round(tile_width * image.shape[0] / image.shape[1])
    tile = cv2.resize(image, (tile_width, tile_height))
    cv2.rectangle(tile, (0, 0), (tile_width, 25), (16, 20, 24), -1)
    confidence_text = "missing" if confidence is None else f"{confidence:.1%}"
    text = f"{status} {confidence_text} {path.stem[-18:]}"
    cv2.putText(
        tile,
        text,
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (245, 248, 250),
        1,
        cv2.LINE_AA,
    )
    return tile


def main() -> int:
    args = parse_args()
    if not 0 <= args.class_id < len(CLASS_NAMES):
        raise SystemExit("class id is outside the task range")
    if not 0 < args.conf <= 1:
        raise SystemExit("confidence must be between 0 and 1")

    images_root = Path(args.images_root).resolve()
    output = Path(args.out).resolve()
    if not images_root.is_dir():
        raise SystemExit(f"missing image root: {images_root}")
    groups = group_images(images_root)
    if not groups:
        raise SystemExit(f"no grouped images under {images_root}")

    (output / "groups").mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    rows: list[dict[str, object]] = []

    for group_name, paths in groups:
        group_out = output / "groups" / group_name
        image_out = group_out / "images"
        label_out = group_out / "labels"
        review_out = group_out / "review"
        for directory in (image_out, label_out, review_out):
            directory.mkdir(parents=True, exist_ok=True)

        tiles: list[np.ndarray] = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise SystemExit(f"cannot read image: {path}")

            result = model.predict(
                image,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                max_det=100,
                verbose=False,
            )[0]
            boxes: list[tuple[int, float, tuple[float, float, float, float]]] = []
            for raw_box, raw_class, raw_confidence in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            ):
                class_id = int(raw_class)
                boxes.append(
                    (
                        class_id,
                        float(raw_confidence),
                        tuple(float(value) for value in raw_box),
                    )
                )

            target_boxes = [
                item for item in boxes if item[0] == args.class_id
            ]
            target_boxes.sort(key=lambda item: item[1], reverse=True)
            selected = target_boxes[0] if target_boxes else None
            non_target_count = len(boxes) - len(target_boxes)

            label_path = label_out / f"{path.stem}.txt"
            if selected is None:
                label_path.write_text("", encoding="utf-8")
                image_out_name = f"{path.stem}.png"
                status = "missing"
                confidence = None
                box_values: tuple[float, float, float, float] | None = None
            else:
                _, confidence, selected_box = selected
                x1, y1, x2, y2 = selected_box
                height, width = image.shape[:2]
                center_x = (x1 + x2) / 2 / width
                center_y = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                label_path.write_text(
                    f"{args.class_id} {center_x:.6f} {center_y:.6f} "
                    f"{box_width:.6f} {box_height:.6f}\n",
                    encoding="utf-8",
                )
                image_out_name = f"{path.stem}.png"
                status = "multiple" if len(target_boxes) > 1 else "ok"
                if non_target_count:
                    status = f"{status}+wrong-class"
                box_values = selected_box

            annotated = annotate(
                image,
                boxes,
                args.class_id,
                selected_box=box_values,
            )
            cv2.imwrite(str(image_out / image_out_name), annotated)
            tiles.append(
                make_tile(
                    annotated,
                    path,
                    status,
                    confidence,
                )
            )
            rows.append(
                {
                    "group": group_name,
                    "image": path.name,
                    "status": status,
                    "confidence": (
                        "" if confidence is None else f"{confidence:.6f}"
                    ),
                    "target_boxes": len(target_boxes),
                    "other_boxes": non_target_count,
                }
            )

        write_sheet(tiles, review_out / "review_sheet.jpg")

    with (output / "review.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "image",
                "status",
                "confidence",
                "target_boxes",
                "other_boxes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    missing = sum(row["status"] == "missing" for row in rows)
    suspicious = sum(
        row["status"] != "ok" and row["status"] != "missing" for row in rows
    )
    print(
        f"prelabels: {len(rows)} images, {missing} missing, "
        f"{suspicious} suspicious -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
