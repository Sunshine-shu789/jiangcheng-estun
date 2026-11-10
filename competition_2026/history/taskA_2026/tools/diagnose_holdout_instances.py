#!/usr/bin/env python3
"""Per-instance diagnosis for selected classes on an evaluation holdout.

Outputs CSV + Markdown + prediction/GT overlays + failure montages. Read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

CLASS_NAMES = (
    "1-Cheez-it", "2-Starkist_Tuna", "3-Scissors", "4-Frenchs_Mustard",
    "5-Tomato_Soup", "6-Foam_Brick", "7-Clamp", "8-Plastic_Banana",
    "9-Mug", "10-meat_can",
)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def read_gt(label_path: Path, width: int, height: int):
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(value) for value in parts[1:5])
        out.append((cls, ((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height)))
    return out


def infer(model: YOLO, images: list[Path], imgsz: int, conf: float, batch: int):
    all_preds = []
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        frames = [cv2.imread(str(path)) for path in chunk]
        if any(frame is None for frame in frames):
            raise RuntimeError(f"could not read images: {chunk}")
        results = model.predict(source=frames, imgsz=imgsz, conf=conf, iou=0.7, max_det=300, verbose=False)
        for result in results:
            predictions = []
            boxes = result.boxes
            if boxes is not None:
                for box, cls, score in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                    predictions.append((int(cls), float(score), tuple(float(v) for v in box)))
            all_preds.append(predictions)
    return all_preds


def tag_for(image: str) -> str:
    if image.startswith("sim_") or "_sim_" in image:
        return "sim"
    if "20260926" in image:
        return "v5_0926"
    if "20260927" in image:
        return "v9_0927"
    if "20261016" in image:
        return "v25_1016"
    return "other"


def draw_overlay(image_path: Path, gts, predictions, target_id: int, out_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for cls, gt in gts:
        color = (0, 220, 0) if cls == target_id else (120, 255, 120)
        draw.rectangle(gt, outline=color, width=3)
        draw.text((gt[0] + 4, max(0, gt[1] - 15)), f"GT {CLASS_NAMES[cls]}", fill=color)
    for cls, conf, box in predictions:
        color = (30, 120, 255) if cls == target_id else (255, 150, 0)
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 4, box[1] + 4), f"{CLASS_NAMES[cls]} {conf:.2f}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)


def montage(items, out_path: Path, title: str):
    if not items:
        return
    tile_h = 260
    tiles = []
    for name, image in items:
        tile = image.resize((int(image.width * tile_h / image.height), tile_h))
        tiles.append((name, tile))
    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    tile_w = max(tile.width for _, tile in tiles)
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + 22)), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, (name, tile) in enumerate(tiles):
        x = (index % cols) * tile_w
        y = (index // cols) * (tile_h + 22)
        canvas.paste(tile, (x, y + 22))
        draw.text((x + 4, y + 3), f"{title} | {name[-62:]}", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=88)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="+")
    parser.add_argument("--data", default="/media/sunshine/新加卷/taskA/datasets/real_ft_v25_hist_holdout_20261016")
    parser.add_argument("--classes", default="1-Cheez-it,3-Scissors")
    parser.add_argument("--imgsz", default="640")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", default="/media/sunshine/新加卷/taskA/reports/v25_hist_holdout_diag_20261016")
    args = parser.parse_args()

    root = Path(args.data).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    requested = [name for name in args.classes.split(",") if name]
    class_ids = {CLASS_NAMES.index(name): name for name in requested}
    images = sorted((root / "images/val").glob("*.jpg")) + sorted((root / "images/val").glob("*.png"))
    image_info = {}
    for image in images:
        frame = cv2.imread(str(image))
        if frame is None:
            raise RuntimeError(f"cannot read: {image}")
        h, w = frame.shape[:2]
        gts = read_gt(root / "labels/val" / f"{image.stem}.txt", w, h)
        image_info[image] = gts

    sizes = [int(value) for value in args.imgsz.split(",")]
    rows = []
    for weight_text in args.weights:
        weight = Path(weight_text).expanduser().resolve()
        if not weight.is_file():
            raise SystemExit(f"missing weight: {weight}")
        model_name = weight.parent.parent.name
        model = YOLO(str(weight))
        for imgsz in sizes:
            predictions = infer(model, images, imgsz, args.conf, args.batch)
            failures = defaultdict(list)
            passes = defaultdict(list)
            for image, preds in zip(images, predictions):
                gts = image_info[image]
                target_gts = [(cls, box) for cls, box in gts if cls in class_ids]
                for instance_index, (target_id, gt_box) in enumerate(target_gts):
                    same = []
                    any_box = []
                    for cls, conf, box in preds:
                        value = iou(gt_box, box)
                        any_box.append((value, cls, conf, box))
                        if cls == target_id:
                            same.append((value, cls, conf, box))
                    best_same = max(same, default=(0.0, target_id, 0.0, None), key=lambda item: item[0])
                    best_any = max(any_box, default=(0.0, -1, 0.0, None), key=lambda item: item[0])
                    # For failure classification use the highest-confidence detection best overlapping GT,
                    # otherwise the highest-confidence prediction in the image.
                    overlapping = [item for item in any_box if item[0] > 0]
                    if overlapping:
                        chosen = max(overlapping, key=lambda item: (item[0], item[2]))
                    else:
                        chosen = max(((0.0, cls, conf, box) for cls, conf, box in preds), default=(0.0, -1, 0.0, None), key=lambda item: item[2])
                    if best_same[0] >= 0.75:
                        failure = "pass"
                    elif best_same[0] >= 0.5:
                        failure = "same_class_iou_0.50_0.75"
                    elif best_same[0] > 0:
                        failure = "same_class_iou_below_0.50"
                    elif chosen[1] >= 0 and chosen[2] >= 0.25:
                        failure = f"wrong_or_partial_class:{CLASS_NAMES[chosen[1]]}"
                    elif chosen[1] >= 0 and chosen[2] > 0:
                        failure = f"low_conf:{CLASS_NAMES[chosen[1]]}"
                    else:
                        failure = "missing_no_gt_overlap"
                    row = {
                        "model": model_name,
                        "weights": str(weight),
                        "imgsz": imgsz,
                        "target_class": CLASS_NAMES[target_id],
                        "image": image.name,
                        "physical_group": next((entry["physical_group"] for entry in csv.DictReader((root / "manifest.csv").open(newline="", encoding="utf-8")) if entry["image"] == image.name), ""),
                        "batch": tag_for(image.name),
                        "instance_index": instance_index,
                        "gt": json.dumps([round(v, 2) for v in gt_box]),
                        "best_same_iou": round(best_same[0], 4),
                        "best_same_conf": round(best_same[2], 4),
                        "best_any_iou": round(best_any[0], 4),
                        "best_any_class": CLASS_NAMES[best_any[1]] if best_any[1] >= 0 else "none",
                        "best_any_conf": round(best_any[2], 4),
                        "chosen_overlap_class": CLASS_NAMES[chosen[1]] if chosen[1] >= 0 else "none",
                        "chosen_overlap_conf": round(chosen[2], 4),
                        "failure": failure,
                        "passed_iou75": int(best_same[0] >= 0.75),
                    }
                    rows.append(row)
                    if imgsz == sizes[0]:
                        overlay = Image.open(image).convert("RGB")
                        draw_overlay(image, gts, preds, target_id, out / "overlays" / model_name / CLASS_NAMES[target_id] / image.name)
                        if row["passed_iou75"]:
                            passes[CLASS_NAMES[target_id]].append((image.stem, overlay))
                        else:
                            failures[CLASS_NAMES[target_id]].append((image.stem, overlay))
            for class_name in requested:
                montage(failures[class_name], out / "montages" / f"{model_name}_{class_name}_imgsz{imgsz}_failures.jpg", f"{model_name} {class_name} FAIL")
                montage(passes[class_name], out / "montages" / f"{model_name}_{class_name}_imgsz{imgsz}_passes.jpg", f"{model_name} {class_name} PASS")
        del model

    fields = list(rows[0])
    with (out / "instances.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# 历史未见 holdout 逐实例诊断", "", f"- 数据：`{root}`", f"- 类别：{', '.join(requested)}", f"- imgsz：{', '.join(str(v) for v in sizes)}", f"- 入口 conf：{args.conf}", ""]
    for model_name in sorted({row["model"] for row in rows}):
        for imgsz in sizes:
            lines.extend([f"## {model_name} | imgsz={imgsz}", "", "| 类别 | GT | IoU75 通过 | 通过率 | IoU50 到 0.75 | 同类别 IoU<0.5 | 错类/部分框 | 漏检 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
            for class_name in requested:
                group = [row for row in rows if row["model"] == model_name and row["imgsz"] == imgsz and row["target_class"] == class_name]
                failures = Counter(row["failure"] for row in group)
                p50 = failures["same_class_iou_0.50_0.75"]
                under = failures["same_class_iou_below_0.50"]
                wrong_partial = sum(count for key, count in failures.items() if key.startswith("wrong_or_partial_class:"))
                missing = failures["missing_no_gt_overlap"] + sum(count for key, count in failures.items() if key.startswith("low_conf:"))
                passed = sum(row["passed_iou75"] for row in group)
                lines.append(f"| {class_name} | {len(group)} | {passed} | {passed / len(group):.3f} | {p50} | {under} | {wrong_partial} | {missing} |")
            lines.append("")
            lines.extend(["| 类别 | 批次 | GT | 通过 | 通过率 |", "|---|---|---:|---:|---:|"])
            for class_name in requested:
                for batch in sorted({row["batch"] for row in rows if row["target_class"] == class_name}):
                    group = [row for row in rows if row["model"] == model_name and row["imgsz"] == imgsz and row["target_class"] == class_name and row["batch"] == batch]
                    passed = sum(row["passed_iou75"] for row in group)
                    lines.append(f"| {class_name} | {batch} | {len(group)} | {passed} | {passed / len(group):.3f} |")
            lines.append("")
    lines.extend(["## 说明", "", "- 该表使用每个 GT 的最佳同类别框；错误分类或只有部分覆盖的框会落入相应失败模式。", "- `missing_no_gt_overlap` 包含完全无框或所有框与 GT 零重叠。", ""])
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out / 'instances.csv'}")
    print(f"[saved] {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
