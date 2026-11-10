#!/usr/bin/env python3
"""Render IoU failure frames with GT and base/TTA prediction boxes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iou-json", type=Path, required=True)
    p.add_argument("--instances-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--class-id", type=int, required=True)
    return p.parse_args()


def font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ub = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (ua + ub - inter) if ua + ub - inter > 0 else 0.0


def best(gt, dets, class_id=None):
    choices = []
    for d in dets:
        if class_id is not None and int(d["class_id"]) != class_id:
            continue
        choices.append((iou(gt, d["xyxy"]), d))
    return max(choices, key=lambda x: x[0]) if choices else (0.0, None)


def draw_box(img, box, color, text):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    y = max(y1, th + 6)
    cv2.rectangle(img, (x1, y - th - 5), (x1 + tw + 6, y), color, -1)
    cv2.putText(img, text, (x1 + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10,10,10), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    iou_data = json.loads(args.iou_json.read_text(encoding="utf-8"))
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    pred_rows = {r["file"]: r for r in summary["rows"]}
    instance_rows = list(csv.DictReader(args.instances_csv.open(encoding="utf-8")))
    failures = [r for r in instance_rows if not int(r["union_pass"])]
    if not failures:
        print("no failures")
        return 0

    tiles = []
    for r in failures:
        filename = r["file"]
        img = cv2.imread(str(args.image_dir / filename))
        if img is None:
            raise FileNotFoundError(args.image_dir / filename)
        gt = [float(v) for v in json.loads(r["gt_xyxy"])]
        pred = pred_rows[filename]
        bi, bd = best(gt, list(pred.get("detections") or []), args.class_id)
        ti, td = best(gt, list(pred.get("tta_detections") or []), args.class_id)
        ai, ad = best(gt, list(pred.get("detections") or []) + list(pred.get("tta_detections") or []), None)
        draw_box(img, gt, (60, 255, 60), "GT")
        if bd is not None:
            draw_box(img, bd["xyxy"], (40, 40, 255), f'BASE {bi:.2f} {bd["conf"]:.2f}')
        if td is not None:
            draw_box(img, td["xyxy"], (255, 80, 40), f'TTA {ti:.2f} {td["conf"]:.2f}')
        if ad is not None:
            draw_box(img, ad["xyxy"], (0, 210, 255), f'{ad["name"]} {ai:.2f} {ad["conf"]:.2f}')
        cv2.putText(img, f'{filename} | union={r["union_iou"]}', (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)
        tiles.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    cols, tw, th, cap = 2, 640, 480, 8
    rows = (len(tiles)+cols-1)//cols
    sheet = Image.new("RGB", (cols*tw, rows*(th+cap)), (20,20,20))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i%cols)*tw, (i//cols)*(th+cap)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=94)
    print(f"{len(failures)} failures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
