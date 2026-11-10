#!/usr/bin/env python3
"""Create reviewable single-instance GT candidates with MobileSAM.

This is deliberately a candidate generator, not an auto-GT claim. It uses the
saved base/TTA detections (including wrong classes) as box prompts, and falls
back to a center point prompt when no detection exists. A human must inspect
the overlay contact sheet before the labels are used for competition scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import SAM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--class-id", type=int, required=True)
    p.add_argument("--sam-weights", type=Path, default=Path("/home/sunshine/codex/easton/taskA/mobile_sam.pt"))
    p.add_argument("--device", default="0")
    p.add_argument("--min-area-ratio", type=float, default=0.002)
    p.add_argument("--max-area-ratio", type=float, default=0.70)
    p.add_argument("--min-ratio-to-hint", type=float, default=0.12)
    p.add_argument("--max-ratio-to-hint", type=float, default=2.8)
    p.add_argument("--min-hint-iou", type=float, default=0.25)
    return p.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def image_center_distance(box: list[float], width: int, height: int) -> float:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return math.hypot(cx - width / 2, cy - height / 2) / max(1.0, math.hypot(width, height))


def mask_bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def biggest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == index).astype(np.uint8)


def choose_mask(result: Any, hint_box: list[float] | None, width: int, height: int,
                args: argparse.Namespace) -> tuple[np.ndarray, list[float], dict[str, float]] | None:
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return None
    image_area = max(1.0, width * height)
    best = None
    for raw in result.masks.data:
        mask = biggest_component(raw.detach().cpu().numpy())
        bbox = mask_bbox(mask)
        if bbox is None:
            continue
        mask_area = float(mask.sum())
        bbox_area = box_area(bbox)
        area_ratio = mask_area / image_area
        bbox_ratio = bbox_area / image_area
        if not (args.min_area_ratio <= area_ratio <= args.max_area_ratio):
            continue
        if not (args.min_area_ratio <= bbox_ratio <= args.max_area_ratio):
            continue
        hint_iou = 1.0
        ratio_to_hint = 1.0
        if hint_box is not None and box_area(hint_box) > 0:
            hint_iou = iou_xyxy(bbox, hint_box)
            ratio_to_hint = bbox_area / box_area(hint_box)
            if not (args.min_ratio_to_hint <= ratio_to_hint <= args.max_ratio_to_hint):
                continue
            if hint_iou < args.min_hint_iou:
                continue
        score = hint_iou - 0.05 * abs(math.log(max(ratio_to_hint, 1e-6)))
        metrics = {
            "mask_area_ratio": area_ratio,
            "bbox_area_ratio": bbox_ratio,
            "hint_iou": hint_iou,
            "ratio_to_hint": ratio_to_hint,
            "score": score,
        }
        if best is None or score > best[2]["score"]:
            best = (mask, bbox, metrics)
    return best


def overlay(image: np.ndarray, mask: np.ndarray, box: list[float], text: str) -> np.ndarray:
    out = image.copy()
    mask = mask.astype(bool)
    tint = np.zeros_like(out)
    tint[:, :] = (40, 220, 40)
    out[mask] = (0.55 * out[mask] + 0.45 * tint[mask]).astype(np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y = max(y1, th + 8)
    cv2.rectangle(out, (x1, y - th - 7), (x1 + tw + 8, y), (0, 255, 255), -1)
    cv2.putText(out, text, (x1 + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)
    return out


def make_sheet(rows: list[dict[str, Any]], overlay_dir: Path, out_dir: Path, label: str) -> None:
    if not rows:
        return
    cols, tw, th, cap = 4, 480, 360, 48
    rows_count = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, 70 + rows_count * (th + cap)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), f"{label} GT candidates | 绿色SAM掩码 黄框=候选GT | 必须人工复核", fill=(255, 255, 255), font=font(31))
    for i, row in enumerate(rows):
        col, line = i % cols, i // cols
        x, y = col * tw, 70 + line * (th + cap)
        thumb = Image.open(overlay_dir / f"{Path(row['file']).stem}_gtcandidate.jpg").convert("RGB")
        thumb.thumbnail((tw, th), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tw, th), (28, 28, 28))
        canvas.paste(thumb, ((tw - thumb.width) // 2, (th - thumb.height) // 2))
        sheet.paste(canvas, (x, y))
        color = (70, 220, 70) if row["status"] == "ok" else (255, 90, 90)
        draw.rectangle((x, y + th, x + tw - 1, y + th + cap - 1), fill=color)
        draw.text((x + 6, y + th + 3), f"{row['hint_tag']} {row['hint_conf']:.2f}", fill=(10, 10, 10), font=font(17))
        draw.text((x + 6, y + th + 25), Path(row["file"]).name[:43], fill=(10, 10, 10), font=font(14))
    sheet.save(out_dir / f"{label}_GT候选总览.jpg", quality=93)


def main() -> int:
    args = parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    labels_dir, overlays_dir = args.out / "labels", args.out / "overlays"
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(args.summary.read_text(encoding="utf-8"))
    class_names = report.get("names") or {}
    model = SAM(str(args.sam_weights))
    rows_out: list[dict[str, Any]] = []

    for row in report.get("rows", []):
        file = str(row["file"])
        image_path = args.image_dir / file
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]

        hints: list[dict[str, Any]] = []
        for source_key, tag in (("detections", "base"), ("tta_detections", "tta")):
            for det in row.get(source_key) or []:
                box = [float(v) for v in det["xyxy"]]
                cls = int(det["class_id"])
                hints.append({
                    "box": box,
                    "class_id": cls,
                    "name": det.get("name", str(cls)),
                    "conf": float(det.get("conf", 0.0)),
                    "source": tag,
                    "target": cls == args.class_id,
                    "center_dist": image_center_distance(box, width, height),
                })
        # Target-class detections first. Other detections are useful when the
        # model put the correct object into a wrong class (e.g. Mug -> meat_can).
        hints.sort(key=lambda h: (not h["target"], h["center_dist"], -h["conf"]))
        chosen = None
        failures: list[str] = []
        for hint in hints:
            result = model(
                str(image_path), bboxes=[hint["box"]], device=args.device,
                verbose=False,
            )[0]
            picked = choose_mask(result, hint["box"], width, height, args)
            if picked is not None:
                mask, box, metrics = picked
                chosen = {
                    "mask": mask, "box": box, "hint": hint, "metrics": metrics,
                    "hint_tag": f'{hint["source"]}:{hint["name"]}', "status": "ok",
                }
                break
            failures.append(f'{hint["source"]}:{hint["name"]}')

        if chosen is None:
            # Center point fallback. This is most useful for complete, roughly
            # centered single-object frames where the detector produced nothing.
            cx, cy = width / 2, height / 2
            result = model(
                str(image_path), points=[[cx, cy]], labels=[1], device=args.device,
                verbose=False,
            )[0]
            picked = choose_mask(result, None, width, height, args)
            if picked is not None:
                mask, box, metrics = picked
                hint = {
                    "box": [cx - 1, cy - 1, cx + 1, cy + 1], "class_id": -1,
                    "name": "center-point", "conf": 0.0, "source": "point",
                    "target": False, "center_dist": 0.0,
                }
                chosen = {
                    "mask": mask, "box": box, "hint": hint, "metrics": metrics,
                    "hint_tag": "point:center", "status": "ok",
                }

        if chosen is None:
            # Empty label is explicit: the reviewer must draw this frame.
            (labels_dir / f"{Path(file).stem}.txt").write_text("", encoding="utf-8")
            row_out = {
                "file": file, "status": "needs_manual_box", "hint_tag": "none",
                "hint_conf": 0.0, "bbox_area_ratio": 0.0, "mask_area_ratio": 0.0,
                "hint_iou": 0.0, "failures": ";".join(failures),
            }
            rows_out.append(row_out)
            blank = image.copy()
            cv2.putText(blank, "NO VALID CANDIDATE - MANUAL BOX REQUIRED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(overlays_dir / f"{Path(file).stem}_gtcandidate.jpg"), blank)
            continue

        box = chosen["box"]
        x1 = max(0.0, min(width - 1.0, box[0]))
        y1 = max(0.0, min(height - 1.0, box[1]))
        x2 = max(x1 + 1.0, min(float(width), box[2]))
        y2 = max(y1 + 1.0, min(float(height), box[3]))
        box = [x1, y1, x2, y2]
        cx_n = (x1 + x2) / 2 / width
        cy_n = (y1 + y2) / 2 / height
        w_n = (x2 - x1) / width
        h_n = (y2 - y1) / height
        (labels_dir / f"{Path(file).stem}.txt").write_text(
            f"{args.class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n",
            encoding="utf-8",
        )
        hint = chosen["hint"]
        metrics = chosen["metrics"]
        text = f'{chosen["hint_tag"]} {hint["conf"]:.2f} | bbox {metrics["bbox_area_ratio"]:.3f}'
        cv2.imwrite(
            str(overlays_dir / f"{Path(file).stem}_gtcandidate.jpg"),
            overlay(image, chosen["mask"], box, text),
        )
        rows_out.append({
            "file": file,
            "status": "ok",
            "hint_tag": chosen["hint_tag"],
            "hint_conf": round(float(hint["conf"]), 4),
            "bbox_area_ratio": round(float(metrics["bbox_area_ratio"]), 6),
            "mask_area_ratio": round(float(metrics["mask_area_ratio"]), 6),
            "hint_iou": round(float(metrics["hint_iou"]), 6),
            "failures": ";".join(failures),
        })

    label = str(report.get("label", args.image_dir.name))
    with (args.out / "review.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["file", "status", "hint_tag", "hint_conf", "bbox_area_ratio", "mask_area_ratio", "hint_iou", "failures"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    (args.out / "classes.txt").write_text(f"{args.class_id} {label}\n", encoding="utf-8")
    summary = {
        "label": label, "class_id": args.class_id, "total": len(rows_out),
        "ok": sum(r["status"] == "ok" for r in rows_out),
        "needs_manual_box": sum(r["status"] != "ok" for r in rows_out),
        "rows": rows_out,
        "warning": "These are SAM-assisted candidates. Human verification is required before scoring.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_sheet(rows_out, overlays_dir, args.out, label)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
