#!/usr/bin/env python3
"""Compose random YOLO scenes from RGBA cutouts.

Every label is computed from the same alpha mask that is rendered into the
image.  Placement rejects frame clipping, significant mask overlap, and
significant bounding-box overlap.  The generated labels are decoded again
and compared with the source alpha boxes; generation fails if any instance
does not reproduce its box.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

import cepb


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

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class CutoutRecord:
    path: Path
    class_id: int
    class_name: str
    camera: str
    scene: int
    light: str


@dataclass(frozen=True)
class RenderedObject:
    class_id: int
    class_name: str
    cutout: CutoutRecord
    rgba: np.ndarray
    x: int
    y: int
    angle: float
    scale: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class BackgroundRecord:
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutouts", required=True, help="extract_sim_cutouts.py output")
    parser.add_argument("--backgrounds", required=True)
    parser.add_argument(
        "--val-backgrounds",
        default=None,
        help="explicit background holdout directory; prevents train/val background reuse",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--objects-min", type=int, default=3)
    parser.add_argument("--objects-max", type=int, default=7)
    parser.add_argument("--empty-prob", type=float, default=0.03)
    parser.add_argument("--target-long-min", type=int, default=60)
    parser.add_argument("--target-long-max", type=int, default=175)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--max-mask-overlap", type=float, default=0.01)
    parser.add_argument("--max-box-iou", type=float, default=0.02)
    parser.add_argument("--placement-attempts", type=int, default=300)
    parser.add_argument("--scene-attempts", type=int, default=30)
    parser.add_argument("--rotation-deg", type=float, default=22.0)
    parser.add_argument("--shadow-prob", type=float, default=0.65)
    parser.add_argument("--min-label-iou", type=float, default=0.999)
    parser.add_argument(
        "--image-format",
        choices=("png", "jpg"),
        default="png",
        help="use jpg for large datasets; labels are unchanged",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--allow-missing-val-class",
        action="store_true",
        help="allow a class that has no source-scene holdout cutout in val",
    )
    parser.add_argument("--seed", type=int, default=20260926)
    return parser.parse_args()


def parse_fraction(value: str) -> float:
    return float(value)


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_cutouts(root: Path) -> list[CutoutRecord]:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest}")
    rows: list[CutoutRecord] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = root / "cutouts" / row["output_name"]
            if not path.is_file():
                raise FileNotFoundError(path)
            source_class_id = int(row["class_id"])
            class_id = source_class_id - 1
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(
                    f"cutout class_id must be in 1..{len(CLASS_NAMES)}, "
                    f"got {source_class_id} for {row['output_name']}"
                )
            rows.append(
                CutoutRecord(
                    path=path,
                    class_id=class_id,
                    class_name=row["class_name"],
                    camera=row["camera"],
                    scene=int(row["scene"]),
                    light=row["light"],
                )
            )
    if not rows:
        raise ValueError(f"no cutouts in {manifest}")
    return rows


def split_sources(
    records: Sequence[CutoutRecord],
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[set[int], set[int]]:
    scenes = sorted({record.scene for record in records})
    if val_fraction <= 0 or len(scenes) < 2:
        return set(scenes), set()
    val_count = max(1, min(len(scenes) - 1, round(len(scenes) * val_fraction)))
    classes = sorted({record.class_id for record in records})
    scenes_per_class: dict[int, set[int]] = {
        class_id: {
            record.scene for record in records if record.class_id == class_id
        }
        for class_id in classes
    }

    best_pair: tuple[set[int], set[int]] | None = None
    best_score: tuple[int, int, int] | None = None
    # A source scene is the leakage boundary. Retry the holdout assignment to
    # avoid the common failure where a rare class appears only in train.
    for _ in range(4096):
        shuffled = scenes.copy()
        rng.shuffle(shuffled)
        val_scenes = set(shuffled[:val_count])
        train_scenes = set(scenes) - val_scenes
        shared = sum(
            bool(scene_ids & train_scenes) and bool(scene_ids & val_scenes)
            for scene_ids in scenes_per_class.values()
        )
        val_present = sum(bool(scene_ids & val_scenes) for scene_ids in scenes_per_class.values())
        train_present = sum(
            bool(scene_ids & train_scenes) for scene_ids in scenes_per_class.values()
        )
        score = (shared, min(train_present, val_present), val_present + train_present)
        if best_score is None or score > best_score:
            best_score = score
            best_pair = (train_scenes, val_scenes)
        if shared == len(classes):
            break
    assert best_pair is not None
    return best_pair


def split_backgrounds(
    backgrounds: Sequence[BackgroundRecord],
    val_fraction: float,
) -> tuple[list[BackgroundRecord], list[BackgroundRecord], bool]:
    if len(backgrounds) < 2:
        return list(backgrounds), list(backgrounds), True
    count = max(1, min(len(backgrounds) - 1, round(len(backgrounds) * val_fraction)))
    return list(backgrounds[:-count]), list(backgrounds[-count:]), False


def augment_background(
    image: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    scale = max(width / source_w, height / source_h) * rng.uniform(1.0, 1.12)
    resized_w = max(width, round(source_w * scale))
    resized_h = max(height, round(source_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    max_x = resized_w - width
    max_y = resized_h - height
    x = int(rng.integers(0, max_x + 1)) if max_x else 0
    y = int(rng.integers(0, max_y + 1)) if max_y else 0
    crop = resized[y : y + height, x : x + width].copy()

    if rng.random() < 0.5:
        crop = cv2.flip(crop, 1)
    contrast = rng.uniform(0.90, 1.12)
    brightness = rng.uniform(-14.0, 14.0)
    crop = np.clip(crop.astype(np.float32) * contrast + brightness, 0, 255).astype(
        np.uint8
    )
    if rng.random() < 0.20:
        sigma = rng.uniform(0.2, 0.7)
        crop = cv2.GaussianBlur(crop, (0, 0), sigma)
    if rng.random() < 0.30:
        noise = rng.normal(0.0, rng.uniform(1.0, 4.0), crop.shape)
        crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return crop


def augment_cutout(
    rgba: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    source = rgba.astype(np.float32)
    rgb = source[:, :, :3]
    contrast = rng.uniform(0.88, 1.12)
    brightness = rng.uniform(-12.0, 12.0)
    rgb = np.clip((rgb - 128.0) * contrast + 128.0 + brightness, 0, 255)

    if rng.random() < 0.15:
        sigma = rng.uniform(0.25, 0.65)
        blurred = cv2.GaussianBlur(rgb, (0, 0), sigma)
        rgb = blurred
    output = source.copy()
    output[:, :, :3] = rgb
    return output.astype(np.uint8)


def rotate_rgba(
    rgba: np.ndarray,
    angle: float,
) -> np.ndarray:
    if abs(angle) < 1e-6:
        return rgba
    height, width = rgba.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = max(1, int(math.ceil(height * sine + width * cosine)))
    new_height = max(1, int(math.ceil(height * cosine + width * sine)))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return cv2.warpAffine(
        rgba,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def resize_rgba(
    rgba: np.ndarray,
    target_long: int,
) -> tuple[np.ndarray, float]:
    height, width = rgba.shape[:2]
    scale = target_long / max(height, width)
    target = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(rgba, target, interpolation=interpolation), float(scale)


def alpha_box(rgba: np.ndarray, x: int, y: int) -> tuple[int, int, int, int]:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if xs.size == 0:
        raise ValueError("transparent object")
    return (
        x + int(xs.min()),
        y + int(ys.min()),
        x + int(xs.max()) + 1,
        y + int(ys.max()) + 1,
    )


def cutout_overlap(
    occupied: np.ndarray,
    rgba: np.ndarray,
    x: int,
    y: int,
) -> tuple[float, float]:
    height, width = rgba.shape[:2]
    alpha = rgba[:, :, 3] > 0
    roi = occupied[y : y + height, x : x + width] > 0
    intersection = int(np.count_nonzero(alpha & roi))
    area = int(np.count_nonzero(alpha))
    return (intersection / area if area else 1.0), float(area)


def box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    return cepb.iou_aabb(first, second)


def paste_rgba(
    canvas: np.ndarray,
    rgba: np.ndarray,
    x: int,
    y: int,
) -> None:
    height, width = rgba.shape[:2]
    roi = canvas[y : y + height, x : x + width].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    foreground = rgba[:, :, :3].astype(np.float32)
    blended = roi * (1.0 - alpha) + foreground * alpha
    canvas[y : y + height, x : x + width] = np.clip(blended, 0, 255).astype(np.uint8)


def paste_shadow(
    canvas: np.ndarray,
    rgba: np.ndarray,
    x: int,
    y: int,
    rng: np.random.Generator,
) -> None:
    alpha = rgba[:, :, 3]
    kernel = int(rng.integers(7, 16))
    if kernel % 2 == 0:
        kernel += 1
    blurred = cv2.GaussianBlur(alpha, (kernel, kernel), 0)
    shadow_alpha = np.clip(blurred.astype(np.float32) * rng.uniform(0.12, 0.28), 0, 110)
    offset_x, offset_y = (int(rng.integers(-5, 10)), int(rng.integers(2, 10)))
    height, width = rgba.shape[:2]
    sx1 = max(0, x + offset_x)
    sy1 = max(0, y + offset_y)
    sx2 = min(canvas.shape[1], x + offset_x + width)
    sy2 = min(canvas.shape[0], y + offset_y + height)
    if sx1 >= sx2 or sy1 >= sy2:
        return
    mx1 = sx1 - (x + offset_x)
    my1 = sy1 - (y + offset_y)
    shadow = shadow_alpha[my1 : my1 + sy2 - sy1, mx1 : mx1 + sx2 - sx1]
    roi = canvas[sy1:sy2, sx1:sx2].astype(np.float32)
    alpha3 = shadow[:, :, None] / 255.0
    shadow_rgb = np.zeros_like(roi)
    blended = roi * (1.0 - alpha3) + shadow_rgb * alpha3
    canvas[sy1:sy2, sx1:sx2] = np.clip(blended, 0, 255).astype(np.uint8)


def draw_label(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    text: str,
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (55, 220, 160), 2)
    label_y = max(13, y1 - 5)
    cv2.putText(
        image,
        text,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def yolo_line(class_id: int, box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {box_width:.6f} {box_height:.6f}"


def decode_yolo(line: str, width: int, height: int) -> tuple[int, tuple[float, float, float, float]]:
    fields = line.split()
    class_id = int(float(fields[0]))
    cx, cy, box_width, box_height = map(float, fields[1:5])
    x1 = (cx - box_width / 2.0) * width
    y1 = (cy - box_height / 2.0) * height
    x2 = (cx + box_width / 2.0) * width
    y2 = (cy + box_height / 2.0) * height
    return class_id, (x1, y1, x2, y2)


def choose_classes(
    available: Sequence[int],
    count: int,
    rng: np.random.Generator,
    usage: dict[int, int],
) -> list[int]:
    if not available:
        raise ValueError("no classes available for this split")
    ordered = sorted(
        available,
        key=lambda class_id: (usage.get(class_id, 0), rng.random()),
    )
    selected = ordered[: min(count, len(ordered))]
    while len(selected) < count:
        selected.append(ordered[len(selected) % len(ordered)])
    rng.shuffle(selected)
    return selected


def place_scene(
    background: np.ndarray,
    cutouts: Sequence[CutoutRecord],
    classes: Sequence[int],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[RenderedObject]]:
    canvas = background.copy()
    occupied = np.zeros((args.height, args.width), dtype=np.uint8)
    rendered: list[RenderedObject] = []

    by_class: dict[int, list[CutoutRecord]] = {}
    for record in cutouts:
        by_class.setdefault(record.class_id, []).append(record)

    for class_id in classes:
        choices = by_class.get(class_id)
        if not choices:
            continue
        record = choices[int(rng.integers(0, len(choices)))]
        rgba = cv2.imread(str(record.path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise RuntimeError(f"invalid RGBA cutout: {record.path}")
        rgba = augment_cutout(rgba, rng)
        angle = float(rng.uniform(-args.rotation_deg, args.rotation_deg))
        rgba = rotate_rgba(rgba, angle)
        target_long = int(rng.integers(args.target_long_min, args.target_long_max + 1))
        rgba, scale = resize_rgba(rgba, target_long)
        if rgba.shape[0] + 2 * args.margin > args.height:
            continue
        if rgba.shape[1] + 2 * args.margin > args.width:
            continue

        placed = False
        for _ in range(args.placement_attempts):
            x = int(rng.integers(args.margin, args.width - rgba.shape[1] - args.margin + 1))
            y = int(rng.integers(args.margin, args.height - rgba.shape[0] - args.margin + 1))
            mask_overlap, _ = cutout_overlap(occupied, rgba, x, y)
            if mask_overlap > args.max_mask_overlap:
                continue
            box = alpha_box(rgba, x, y)
            if any(box_iou(box, other.box) > args.max_box_iou for other in rendered):
                continue

            if rng.random() < args.shadow_prob:
                paste_shadow(canvas, rgba, x, y, rng)
            paste_rgba(canvas, rgba, x, y)
            occupied[y : y + rgba.shape[0], x : x + rgba.shape[1]][rgba[:, :, 3] > 0] = 255
            rendered.append(
                RenderedObject(
                    class_id=class_id,
                    class_name=record.class_name,
                    cutout=record,
                    rgba=rgba,
                    x=x,
                    y=y,
                    angle=angle,
                    scale=scale,
                    box=box,
                )
            )
            placed = True
            break
        if not placed:
            continue
    return canvas, rendered


def make_scene(
    backgrounds: Sequence[BackgroundRecord],
    cutouts: Sequence[CutoutRecord],
    args: argparse.Namespace,
    rng: np.random.Generator,
    class_usage: dict[int, int],
) -> tuple[np.ndarray, list[RenderedObject]]:
    available = sorted({record.class_id for record in cutouts})
    if not available:
        raise ValueError("no cutout classes available")
    target_count = (
        0
        if rng.random() < args.empty_prob
        else int(rng.integers(args.objects_min, args.objects_max + 1))
    )
    classes = (
        []
        if target_count == 0
        else choose_classes(available, target_count, rng, class_usage)
    )
    for _ in range(args.scene_attempts):
        background_record = backgrounds[int(rng.integers(0, len(backgrounds)))]
        background = cv2.imread(str(background_record.path), cv2.IMREAD_COLOR)
        if background is None:
            raise RuntimeError(f"cannot read background: {background_record.path}")
        background = augment_background(background, args.width, args.height, rng)
        image, rendered = place_scene(background, cutouts, classes, args, rng)
        if target_count == 0 or len(rendered) == target_count:
            for class_id in classes:
                class_usage[class_id] = class_usage.get(class_id, 0) + 1
            return image, rendered
    raise RuntimeError(
        f"could not place {target_count} objects after {args.scene_attempts} attempts"
    )


def build_data_yaml(out: Path) -> None:
    (out / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(CLASS_NAMES)}",
                "names:",
                *[f"- {name}" for name in CLASS_NAMES],
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_classes_txt(out: Path) -> None:
    (out / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n",
        encoding="utf-8",
    )


def split_class_ids(records: Sequence[CutoutRecord]) -> set[int]:
    return {record.class_id for record in records}


def write_contact_sheet(
    rows: list[dict[str, object]],
    out: Path,
    width: int = 640,
    height: int = 480,
) -> None:
    tiles: list[np.ndarray] = []
    for row in rows[:40]:
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = 0.5
        tile = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        for item in row["objects"]:
            box = item["box"]
            x1, y1, x2, y2 = (round(int(value) * scale) for value in box)
            cv2.rectangle(tile, (x1, y1), (max(x1 + 1, x2 - 1), max(y1 + 1, y2 - 1)), (55, 220, 160), 1)
        label = str(row["image_path"].stem)
        cv2.rectangle(tile, (0, 0), (190, 24), (20, 24, 28), -1)
        cv2.putText(
            tile,
            label,
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return
    blank = np.full_like(tiles[0], 26)
    columns = 2
    while len(tiles) % columns:
        tiles.append(blank.copy())
    sheet = np.vstack(
        [
            np.hstack(tiles[index : index + columns])
            for index in range(0, len(tiles), columns)
        ]
    )
    cv2.imwrite(
        str(out / "contact_sheet.jpg"),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if not 0 <= args.val_fraction < 1:
        raise SystemExit("--val-fraction must be in [0, 1)")
    if args.objects_min < 1 or args.objects_max < args.objects_min:
        raise SystemExit("invalid --objects-min/--objects-max")
    if args.target_long_min < 8 or args.target_long_max < args.target_long_min:
        raise SystemExit("invalid target long-side range")
    if not 0 <= args.max_mask_overlap <= 1:
        raise SystemExit("--max-mask-overlap must be in [0, 1]")
    if not 0 <= args.max_box_iou <= 1:
        raise SystemExit("--max-box-iou must be in [0, 1]")
    if not 0 <= args.min_label_iou <= 1:
        raise SystemExit("--min-label-iou must be in [0, 1]")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be in [1, 100]")

    cutouts_root = Path(args.cutouts).resolve()
    background_root = Path(args.backgrounds).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    cutouts = load_cutouts(cutouts_root)
    background_paths = image_files(background_root)
    if not background_paths:
        raise SystemExit(f"no backgrounds in {background_root}")
    backgrounds = [BackgroundRecord(path) for path in background_paths]

    rng = np.random.default_rng(args.seed)
    train_scenes, val_scenes = split_sources(cutouts, args.val_fraction, rng)
    train_cutouts = [record for record in cutouts if record.scene in train_scenes]
    val_cutouts = [record for record in cutouts if record.scene in val_scenes]
    if args.val_backgrounds:
        val_background_root = Path(args.val_backgrounds).resolve()
        val_background_paths = image_files(val_background_root)
        if not val_background_paths:
            raise SystemExit(f"no backgrounds in {val_background_root}")
        train_background_ids = {path.resolve() for path in background_paths}
        val_background_ids = {path.resolve() for path in val_background_paths}
        overlap = sorted(train_background_ids & val_background_ids)
        if overlap:
            raise SystemExit(
                f"train/val backgrounds overlap: {[str(path) for path in overlap[:3]]}"
            )
        train_backgrounds = backgrounds
        val_backgrounds = [BackgroundRecord(path) for path in val_background_paths]
        shared_backgrounds = False
    else:
        train_backgrounds, val_backgrounds, shared_backgrounds = split_backgrounds(
            backgrounds, args.val_fraction
        )
    if not train_cutouts:
        raise SystemExit("no training cutouts after source-scene split")
    if val_scenes and not val_cutouts:
        raise SystemExit("no validation cutouts after source-scene split")
    available_classes = sorted({record.class_id for record in cutouts})
    train_classes = split_class_ids(train_cutouts)
    val_classes = split_class_ids(val_cutouts)
    missing_train = sorted(set(available_classes) - train_classes)
    missing_val_sources = sorted(set(available_classes) - val_classes)
    if missing_train:
        raise SystemExit(
            f"training split is missing classes {missing_train}; increase source scenes"
        )
    if val_scenes and missing_val_sources and not args.allow_missing_val_class:
        raise SystemExit(
            f"validation source holdout is missing classes {missing_val_sources}; "
            "change --seed/--val-fraction or pass --allow-missing-val-class"
        )

    split_labels = ["train"] * args.count
    val_count = round(args.count * args.val_fraction)
    if val_count:
        split_labels[:val_count] = ["val"] * val_count
        rng.shuffle(split_labels)
    elif args.val_fraction > 0:
        split_labels[0] = "val"
    if "val" in split_labels and not val_cutouts:
        split_labels = ["train"] * args.count
        val_count = 0

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    build_data_yaml(output)
    build_classes_txt(output)

    manifest_rows: list[dict[str, object]] = []
    verification_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    class_counts = {
        "train": {class_id: 0 for class_id in available_classes},
        "val": {class_id: 0 for class_id in available_classes},
    }
    class_usage = {
        "train": {class_id: 0 for class_id in available_classes},
        "val": {class_id: 0 for class_id in available_classes},
    }

    for index, split in enumerate(split_labels, 1):
        if split == "val":
            split_cutouts = val_cutouts
            split_background_pool = val_backgrounds
        else:
            split_cutouts = train_cutouts
            split_background_pool = train_backgrounds
        if not split_background_pool:
            split_background_pool = backgrounds

        image = None
        rendered: list[RenderedObject] = []
        last_error: Exception | None = None
        for _ in range(8):
            try:
                image, rendered = make_scene(
                    split_background_pool,
                    split_cutouts,
                    args,
                    rng,
                    class_usage[split],
                )
                break
            except RuntimeError as exc:
                last_error = exc
        if image is None:
            raise RuntimeError(f"failed to generate scene {index}: {last_error}")

        stem = f"synth_{index:06d}"
        image_name = f"{stem}.{args.image_format}"
        image_path = output / "images" / split / image_name
        label_path = output / "labels" / split / f"{stem}.txt"
        image_params = (
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            if args.image_format == "jpg"
            else [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )
        if not cv2.imwrite(str(image_path), image, image_params):
            raise RuntimeError(f"cannot write {image_path}")

        label_lines: list[str] = []
        rendered_manifest: list[dict[str, object]] = []
        for instance_index, item in enumerate(rendered):
            line = yolo_line(item.class_id, item.box, args.width, args.height)
            label_lines.append(line)
            decoded_class, decoded_box = decode_yolo(line, args.width, args.height)
            alpha_iou = cepb.iou_aabb(item.box, decoded_box)
            if decoded_class != item.class_id or alpha_iou < args.min_label_iou:
                raise RuntimeError(
                    f"{stem}: label mismatch for class {item.class_id}, "
                    f"IoU={alpha_iou} < {args.min_label_iou}"
                )
            instance_id = f"{stem}_{instance_index:02d}"
            class_counts[split][item.class_id] += 1
            row = {
                "split": split,
                "scene": stem,
                "instance_id": instance_id,
                "class_id": item.class_id,
                "class_name": item.class_name,
                "source_cutout": item.cutout.path.name,
                "source_scene": item.cutout.scene,
                "source_camera": item.cutout.camera,
                "source_light": item.cutout.light,
                "x1": item.box[0],
                "y1": item.box[1],
                "x2": item.box[2],
                "y2": item.box[3],
                "alpha_area": int(np.count_nonzero(item.rgba[:, :, 3])),
                "rotation_deg": f"{item.angle:.3f}",
                "scale": f"{item.scale:.6f}",
                "alpha_box_iou": f"{alpha_iou:.6f}",
            }
            manifest_rows.append(row)
            rendered_manifest.append({"box": item.box, "class_id": item.class_id})
            verification_rows.append(row)
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        contact_rows.append(
            {
                "image_path": image_path,
                "objects": rendered_manifest,
            }
        )
        if index % 25 == 0 or index == len(split_labels):
            print(f"generated {index}/{len(split_labels)}")

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "scene",
            "instance_id",
            "class_id",
            "class_name",
            "source_cutout",
            "source_scene",
            "source_camera",
            "source_light",
            "x1",
            "y1",
            "x2",
            "y2",
            "alpha_area",
            "rotation_deg",
            "scale",
            "alpha_box_iou",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    generated_val_classes = {
        class_id
        for class_id, count in class_counts["val"].items()
        if count > 0
    }
    missing_val_images = sorted(set(available_classes) - generated_val_classes)
    if val_scenes and missing_val_images and not args.allow_missing_val_class:
        raise SystemExit(
            f"generated validation images are missing classes {missing_val_images}; "
            "increase --count/--objects-min, change --seed, or pass "
            "--allow-missing-val-class"
        )

    ious = np.asarray([float(row["alpha_box_iou"]) for row in verification_rows])
    min_iou = float(ious.min()) if ious.size else 1.0
    report = [
        "# Random synthetic scene generation",
        "",
        f"- cutouts: `{cutouts_root}`",
        f"- backgrounds: `{background_root}`",
        f"- images: {args.count} ({args.width}x{args.height})",
        f"- image format: {args.image_format}",
        f"- split: train {split_labels.count('train')} / val {split_labels.count('val')}",
        f"- source-scene holdout: train {len(train_scenes)} / val {len(val_scenes)}",
        f"- background holdout: {not shared_backgrounds}",
        f"- instances: {len(manifest_rows)}",
        f"- alpha-box vs decoded YOLO box minimum IoU: {min_iou:.6f}",
        f"- label IoU gate: {args.min_label_iou:.6f}",
        f"- objects per scene: {args.objects_min}..{args.objects_max}, "
        f"empty probability {args.empty_prob:.3f}",
        "",
        "| class_id | class | train | val |",
        "| ---: | --- | ---: | ---: |",
    ]
    for class_id, class_name in enumerate(CLASS_NAMES):
        report.append(
            f"| {class_id} | {class_name} | "
            f"{class_counts['train'].get(class_id, 0)} | "
            f"{class_counts['val'].get(class_id, 0)} |"
        )
    report += [
        "",
        "Every image label is generated from the final transformed alpha mask. "
        "The stored `alpha_box_iou` is a decode-and-compare check, not a model "
        "prediction metric.",
        "",
    ]
    if shared_backgrounds:
        report += [
            "> Only one background is available, so train and validation reuse it. "
            "Use a separate real-machine validation set before claiming model IoU.",
            "",
        ]
    if missing_val_sources or missing_val_images:
        missing = sorted(set(missing_val_sources) | set(missing_val_images))
        report += [
            f"> Validation has incomplete class coverage for classes {missing}. "
            "The generated validation split is incomplete.",
            "",
        ]
    (output / "generation_report.md").write_text("\n".join(report), encoding="utf-8")
    write_contact_sheet(contact_rows, output, args.width, args.height)
    print("\n".join(report))
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
