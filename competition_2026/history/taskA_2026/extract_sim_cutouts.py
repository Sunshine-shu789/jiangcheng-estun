#!/usr/bin/env python3
"""Extract clean RGBA object cutouts from the CEPB segmentation masks.

The extractor intentionally uses the exact segmentation colors published by
CEPB.  It verifies those colors against each object's corrected projected
cuboid before accepting a crop.  Only highly visible objects that stay away
from the image border and contain one dominant mask component are exported.

Typical smoke test:

    python extract_sim_cutouts.py \
        --scenes 12 --camera left --lights directional_light \
        --out /tmp/cepb_cutouts_smoke
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

import cepb


CLASS_NAMES = {
    1: "1-Cheez-it",
    2: "2-Starkist_Tuna",
    3: "3-Scissors",
    4: "4-Frenchs_Mustard",
    5: "5-Tomato_Soup",
    6: "6-Foam_Brick",
    7: "7-Clamp",
    8: "8-Plastic_Banana",
    9: "9-Mug",
    10: "10-meat_can",
}

# Exact BGR values stored in CEPB subset1 segmentation PNG files.  The rest
# of the script keeps OpenCV's native BGR order rather than converting it.
SEGMENTATION_COLORS = {
    1: (178, 178, 178),
    2: (127, 255, 255),
    3: (255, 127, 255),
    4: (255, 255, 127),
    5: (0, 127, 255),
    6: (255, 0, 127),
    7: (127, 255, 0),
    8: (0, 0, 255),
    9: (0, 255, 0),
    10: (255, 0, 0),
}


@dataclass(frozen=True)
class Cutout:
    class_id: int
    class_name: str
    camera: str
    scene: int
    light: str
    visibility: float
    source_rgb: str
    output_name: str
    bbox: tuple[int, int, int, int]
    mask_area: int
    component_ratio: float
    gt_box_iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=100)
    parser.add_argument(
        "--camera",
        default="left",
        choices=list(cepb.CAMERAS) + ["all"],
    )
    parser.add_argument(
        "--lights",
        default="directional_light",
        help="comma-separated lights or 'all'",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-visibility", type=float, default=0.95)
    parser.add_argument("--min-side", type=int, default=24)
    parser.add_argument("--max-side", type=int, default=800)
    parser.add_argument("--edge-margin", type=int, default=2)
    parser.add_argument("--min-gt-iou", type=float, default=0.50)
    parser.add_argument("--min-component-ratio", type=float, default=0.80)
    parser.add_argument("--keep-empty", action="store_true")
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260926)
    return parser.parse_args()


def build_scenes(total: int, count: int) -> list[int]:
    if count < 1 or count > total:
        raise ValueError(f"--scenes must be in 1..{total}, got {count}")
    if count == 1:
        return [1]
    step = total / count
    return sorted({int(round(1 + i * step)) for i in range(count)})


def parse_lights(value: str) -> list[str]:
    if value == "all":
        return list(cepb.LIGHTS)
    lights = [part.strip() for part in value.split(",") if part.strip()]
    invalid = sorted(set(lights) - set(cepb.LIGHTS))
    if invalid:
        raise ValueError(f"invalid lights: {invalid}")
    if not lights:
        raise ValueError("--lights cannot be empty")
    return lights


def class_id_from_name(name: str) -> int:
    match = re.match(r"^(\d+)[-_]", name)
    if not match:
        raise ValueError(f"object name has no numeric prefix: {name}")
    class_id = int(match.group(1))
    if class_id not in CLASS_NAMES:
        raise ValueError(f"unexpected class id {class_id}: {name}")
    return class_id


def rgb_equal(image_rgb: np.ndarray, color: Sequence[int]) -> np.ndarray:
    return np.all(image_rgb == np.asarray(color, dtype=np.uint8), axis=2)


def bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def iou_aabb_int(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    # Convert inclusive pixel bounds to continuous boxes for IoU comparison.
    a = (first[0], first[1], first[2] + 1, first[3] + 1)
    b = (second[0], second[1], second[2] + 1, second[3] + 1)
    return cepb.iou_aabb(a, b)


def dominant_component_ratio(mask: np.ndarray) -> float:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = int(areas.sum())
    return float(areas.max()) / total if total else 0.0


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def crop_rgba(
    rgb: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop_rgb = rgb[y1 : y2 + 1, x1 : x2 + 1]
    crop_alpha = mask[y1 : y2 + 1, x1 : x2 + 1]
    return np.dstack([crop_rgb, crop_alpha])


def add_checkerboard(image: np.ndarray, square: int = 12) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.indices((height, width))
    light = np.full((height, width), 220, dtype=np.uint8)
    dark = np.full((height, width), 170, dtype=np.uint8)
    board = np.where(((yy // square) + (xx // square)) % 2 == 0, light, dark)
    background = np.repeat(board[:, :, None], 3, axis=2)

    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    foreground = image[:, :, :3].astype(np.float32)
    return np.clip(
        foreground * alpha + background.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def write_contact_sheet(
    rows: list[Cutout],
    output_root: Path,
    tile_width: int = 180,
    columns: int = 5,
) -> None:
    by_class: dict[int, list[Cutout]] = {}
    for row in rows:
        by_class.setdefault(row.class_id, []).append(row)

    for class_id, class_rows in sorted(by_class.items()):
        tiles: list[np.ndarray] = []
        for row in class_rows[:50]:
            path = output_root / "cutouts" / row.output_name
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            height, width = image.shape[:2]
            scale = min(tile_width / width, tile_width / height, 1.0)
            resized = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            canvas = np.zeros((tile_width, tile_width, 4), dtype=np.uint8)
            canvas[:, :, :3] = 32
            canvas[:, :, 3] = 255
            y = (tile_width - resized.shape[0]) // 2
            x = (tile_width - resized.shape[1]) // 2
            canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            tile = add_checkerboard(canvas)
            cv2.putText(
                tile,
                path.stem[-25:],
                (5, tile_width - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)

        if not tiles:
            continue
        blank = np.full((tile_width, tile_width, 3), 32, dtype=np.uint8)
        while len(tiles) % columns:
            tiles.append(blank.copy())
        sheet = np.vstack(
            [
                np.hstack(tiles[index : index + columns])
                for index in range(0, len(tiles), columns)
            ]
        )
        cv2.imwrite(
            str(output_root / f"contact_{class_id:02d}_{CLASS_NAMES[class_id]}.jpg"),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )


def entry_names(
    scenes: Iterable[int],
    cameras: Iterable[str],
    lights: Iterable[str],
) -> list[str]:
    names: list[str] = []
    for camera in cameras:
        for scene in scenes:
            names.append(cepb.gt_name(camera, scene))
            names.append(cepb.segmentation_name(camera, scene))
            for light in lights:
                names.append(cepb.rgb_name(camera, light, scene))
    return names


def extract_entries(names: list[str], target: Path, batch_size: int = 1500) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(names), batch_size):
        batch = names[start : start + batch_size]
        subprocess.run(
            ["unzip", "-o", "-j", "-d", str(target), cepb.ARCHIVE, *batch],
            check=True,
            capture_output=True,
        )


def main() -> int:
    args = parse_args()
    if not 0 <= args.min_visibility <= 1:
        raise SystemExit("--min-visibility must be in [0, 1]")
    if args.min_side < 1 or args.max_side < args.min_side:
        raise SystemExit("invalid --min-side/--max-side")
    if not 0 <= args.min_component_ratio <= 1:
        raise SystemExit("--min-component-ratio must be in [0, 1]")

    scenes = build_scenes(5000, args.scenes)
    cameras = list(cepb.CAMERAS) if args.camera == "all" else [args.camera]
    lights = parse_lights(args.lights)
    output = Path(args.out).resolve()
    if output.exists() and not args.keep_empty:
        raise SystemExit(f"output already exists: {output}")
    (output / "cutouts").mkdir(parents=True, exist_ok=True)

    names = entry_names(scenes, cameras, lights)
    print(
        f"Extracting {len(names)} archive entries: {len(scenes)} scenes x "
        f"{len(cameras)} camera(s) x {len(lights)} light(s)"
    )
    temp = Path(tempfile.mkdtemp(prefix="cepb_cutouts_", dir=str(cepb.WORKDIR / "tmp")))
    rows: list[Cutout] = []
    failures: list[tuple[int, str, str]] = []
    try:
        extract_entries(names, temp)
        for camera in cameras:
            for scene in scenes:
                gt_path = temp / cepb.gt_name(camera, scene)
                seg_path = temp / cepb.segmentation_name(camera, scene)
                if not gt_path.is_file() or not seg_path.is_file():
                    failures.append((scene, camera, "missing GT or segmentation"))
                    continue

                frame = cepb.parse_gt_frame(camera, scene, gt_path.read_bytes())
                segmentation = cv2.imread(str(seg_path), cv2.IMREAD_COLOR)
                if segmentation is None:
                    failures.append((scene, camera, "cannot read segmentation"))
                    continue

                for obj in frame.objects:
                    if obj.visibility < args.min_visibility:
                        continue
                    try:
                        class_id = class_id_from_name(obj.name)
                    except ValueError as exc:
                        failures.append((scene, camera, str(exc)))
                        continue

                    color = SEGMENTATION_COLORS[class_id]
                    raw_mask = dominant_component_ratio(
                        rgb_equal(segmentation, color).astype(np.uint8) * 255
                    )
                    mask = clean_mask(
                        rgb_equal(segmentation, color).astype(np.uint8) * 255
                    )
                    bbox = bbox_of_mask(mask)
                    if bbox is None:
                        failures.append((scene, camera, f"{obj.name}: empty mask"))
                        continue

                    x1, y1, x2, y2 = bbox
                    width, height = x2 - x1 + 1, y2 - y1 + 1
                    if min(width, height) < args.min_side or max(width, height) > args.max_side:
                        failures.append(
                            (scene, camera, f"{obj.name}: invalid size {width}x{height}")
                        )
                        continue
                    margin = args.edge_margin
                    if (
                        x1 <= margin
                        or y1 <= margin
                        or x2 >= cepb.IMAGE_WIDTH - 1 - margin
                        or y2 >= cepb.IMAGE_HEIGHT - 1 - margin
                    ):
                        failures.append((scene, camera, f"{obj.name}: touches frame"))
                        continue

                    gt_box = tuple(round(value) for value in obj.aabb)
                    gt_iou = iou_aabb_int(bbox, gt_box)
                    if gt_iou < args.min_gt_iou:
                        failures.append(
                            (
                                scene,
                                camera,
                                f"{obj.name}: mask/GT IoU {gt_iou:.3f}",
                            )
                        )
                        continue
                    if raw_mask < args.min_component_ratio:
                        failures.append(
                            (
                                scene,
                                camera,
                                f"{obj.name}: component ratio {raw_mask:.3f}",
                            )
                        )
                        continue

                    for light in lights:
                        rgb_path = temp / cepb.rgb_name(camera, light, scene)
                        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                        if rgb is None:
                            failures.append((scene, camera, f"cannot read {rgb_path.name}"))
                            continue
                        cutout = crop_rgba(rgb, mask, bbox)
                        output_name = (
                            f"c{class_id:02d}_s{scene:04d}_{camera}_{light}.png"
                        )
                        if not cv2.imwrite(
                            str(output / "cutouts" / output_name),
                            cutout,
                            [cv2.IMWRITE_PNG_COMPRESSION, 3],
                        ):
                            raise RuntimeError(f"cannot write {output_name}")
                        rows.append(
                            Cutout(
                                class_id=class_id,
                                class_name=CLASS_NAMES[class_id],
                                camera=camera,
                                scene=scene,
                                light=light,
                                visibility=obj.visibility,
                                source_rgb=cepb.rgb_name(camera, light, scene),
                                output_name=output_name,
                                bbox=bbox,
                                mask_area=int(np.count_nonzero(mask)),
                                component_ratio=float(raw_mask),
                                gt_box_iou=float(gt_iou),
                            )
                        )
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    if args.limit_per_class:
        by_class: dict[int, list[Cutout]] = {}
        for row in rows:
            by_class.setdefault(row.class_id, []).append(row)
        rng = np.random.default_rng(args.seed)
        kept: list[Cutout] = []
        for class_id in sorted(by_class):
            items = by_class[class_id]
            rng.shuffle(items)
            kept.extend(items[: args.limit_per_class])
        keep_names = {row.output_name for row in kept}
        for cutout_path in (output / "cutouts").glob("*.png"):
            if cutout_path.name not in keep_names:
                cutout_path.unlink()
        rows = kept

    rows.sort(key=lambda row: (row.class_id, row.scene, row.camera, row.light))
    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "output_name",
                "class_id",
                "class_name",
                "camera",
                "scene",
                "light",
                "visibility",
                "source_rgb",
                "x1",
                "y1",
                "x2",
                "y2",
                "mask_area",
                "component_ratio",
                "gt_box_iou",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "output_name": row.output_name,
                    "class_id": row.class_id,
                    "class_name": row.class_name,
                    "camera": row.camera,
                    "scene": row.scene,
                    "light": row.light,
                    "visibility": f"{row.visibility:.4f}",
                    "source_rgb": row.source_rgb,
                    "x1": row.bbox[0],
                    "y1": row.bbox[1],
                    "x2": row.bbox[2],
                    "y2": row.bbox[3],
                    "mask_area": row.mask_area,
                    "component_ratio": f"{row.component_ratio:.4f}",
                    "gt_box_iou": f"{row.gt_box_iou:.6f}",
                }
            )

    class_counts: dict[int, int] = {}
    for row in rows:
        class_counts[row.class_id] = class_counts.get(row.class_id, 0) + 1
    report = [
        "# CEPB RGBA cutout extraction",
        "",
        f"- archive: `{cepb.ARCHIVE}`",
        f"- scenes: {len(scenes)} ({scenes[0]}..{scenes[-1]})",
        f"- cameras: {', '.join(cameras)}",
        f"- lights: {', '.join(lights)}",
        f"- cutouts: {len(rows)}",
        f"- filters: visibility >= {args.min_visibility:.2f}, "
        f"side {args.min_side}..{args.max_side}, "
        f"component ratio >= {args.min_component_ratio:.2f}, "
        f"mask/GT IoU >= {args.min_gt_iou:.2f}",
        "",
        "| class_id | class | cutouts |",
        "| ---: | --- | ---: |",
    ]
    for class_id, class_name in CLASS_NAMES.items():
        report.append(f"| {class_id} | {class_name} | {class_counts.get(class_id, 0)} |")
    report += ["", f"Rejected candidates: {len(failures)}", ""]
    if failures:
        report += ["## First rejections", ""]
        report += [
            f"- scene {scene} / {camera}: {reason}"
            for scene, camera, reason in failures[:30]
        ]
        report.append("")
    (output / "extract_report.md").write_text("\n".join(report), encoding="utf-8")
    write_contact_sheet(rows, output)
    print("\n".join(report[:20]))
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
