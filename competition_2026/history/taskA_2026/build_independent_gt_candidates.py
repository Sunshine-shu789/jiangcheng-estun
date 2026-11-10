#!/usr/bin/env python3
"""Build an independent full-object GT proposal using YOLO-World.

This is an evaluation-only annotation aid. It does not train or alter the
competition detector. For each existing candidate GT box, it selects the best
nearby YOLO-World detection that plausibly encloses the object, then writes a
v2 label and review overlays. The output still requires human/visual review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

PROMPTS: dict[str, list[str]] = {
    "1-Cheez-it": ["cheez-it box", "cracker box"],
    "2-Starkist_Tuna": ["tuna can", "tin can", "canned tuna", "canned fish", "fish can"],
    "3-Scissors": ["scissors"],
    "4-Frenchs_Mustard": ["mustard bottle", "yellow mustard bottle"],
    "5-Tomato_Soup": ["tomato soup can", "soup can", "tin can", "food can"],
    "6-Foam_Brick": ["foam brick", "foam block", "sponge block", "foam"],
    "7-Clamp": ["clip", "clothespin", "plastic clamp", "clamp"],
    "8-Plastic_Banana": ["plastic banana", "banana"],
    "9-Mug": ["mug", "coffee mug"],
    "10-meat_can": ["canned meat", "meat can", "tin can", "food can"],
}

CLASS_IDS = {
    "1-Cheez-it": 0,
    "2-Starkist_Tuna": 1,
    "3-Scissors": 2,
    "4-Frenchs_Mustard": 3,
    "5-Tomato_Soup": 4,
    "6-Foam_Brick": 5,
    "7-Clamp": 6,
    "8-Plastic_Banana": 7,
    "9-Mug": 8,
    "10-meat_can": 9,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--old-gt-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--world-weights", type=Path, required=True)
    p.add_argument("--classes", default=",".join(PROMPTS))
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.015)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=4)
    return p.parse_args()


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + ab - inter) if aa + ab - inter > 0 else 0.0


def containment(inner: list[float], outer: list[float]) -> float:
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1 = max(x1, ox1), max(y1, oy1)
    ix2, iy2 = min(x2, ox2), min(y2, oy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / area if area > 0 else 0.0


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def load_old_label(path: Path, width: int, height: int) -> list[float] | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    parts = lines[0].split()
    if len(parts) < 5:
        return None
    _, cx, cy, w, h = (float(v) for v in parts[:5])
    return [(cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height]


def write_yolo(path: Path, class_id: int, box: list[float], width: int, height: int) -> None:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    path.write_text(f"{class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n", encoding="utf-8")


def font(size: int) -> ImageFont.ImageFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_box(image: np.ndarray, box: list[float], color: tuple[int, int, int], text: str) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, text, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)


def make_sheet(label: str, rows: list[dict[str, Any]], out: Path) -> None:
    tw, th, cap = 320, 240, 52
    cols = 4
    cell_h = th + cap
    sheet = Image.new("RGB", (cols * tw, ((len(rows) + cols - 1) // cols) * cell_h), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    f = font(14)
    for i, row in enumerate(rows):
        image = cv2.imread(str(row["image"]))
        if image is None:
            continue
        image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
        sx, sy = tw / row["width"], th / row["height"]
        if row["old"] is not None:
            b = [int(row["old"][0] * sx), int(row["old"][1] * sy),
                 int(row["old"][2] * sx), int(row["old"][3] * sy)]
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 1)
        for prop in row["proposals"]:
            b = [int(prop["box"][0] * sx), int(prop["box"][1] * sy),
                 int(prop["box"][2] * sx), int(prop["box"][3] * sy)]
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 220, 0), 1)
        if row["selected"] is not None:
            b = [int(row["selected"]["box"][0] * sx), int(row["selected"]["box"][1] * sy),
                 int(row["selected"]["box"][2] * sx), int(row["selected"]["box"][3] * sy)]
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        x, y = (i % cols) * tw, (i // cols) * cell_h
        canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        sheet.paste(canvas, (x, y))
        status = row["status"]
        color = (80, 220, 80) if status == "selected" else (240, 190, 70)
        d.rectangle((x, y + th, x + tw - 1, y + th + cap - 1), fill=color)
        selected = row["selected"]
        detail = "kept old"
        if selected is not None:
            detail = f'{selected["name"]} {selected["conf"]:.3f} exp={selected["expansion"]:.2f}'
        d.text((x + 5, y + th + 4), f"{status} | {detail}", fill=(5, 5, 5), font=f)
        d.text((x + 5, y + th + 27), row["file"][:46], fill=(5, 5, 5), font=f)
    sheet.save(out, quality=93)


def main() -> int:
    args = parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output dir is not empty: {args.out}")
    labels_dir = args.out / "labels"
    overlays_dir = args.out / "overlays"
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.world_weights))
    all_summary: dict[str, Any] = {}
    sheet_rows: dict[str, list[dict[str, Any]]] = {}

    for label in args.classes.split(","):
        if label not in PROMPTS:
            raise SystemExit(f"unknown class: {label}")
        class_id = CLASS_IDS[label]
        image_paths = sorted((args.capture_root / label).glob("*.jpg"))
        model.set_classes(PROMPTS[label])
        results = model.predict(
            [str(p) for p in image_paths], conf=args.conf, iou=0.55,
            imgsz=args.imgsz, device=args.device, batch=args.batch,
            verbose=False,
        )
        rows: list[dict[str, Any]] = []
        for image_path, result in zip(image_paths, results):
            with Image.open(image_path) as im:
                width, height = im.size
            stem = image_path.stem
            old = load_old_label(args.old_gt_root / label / "labels" / f"{stem}.txt", width, height)
            proposals: list[dict[str, Any]] = []
            if result.boxes is not None:
                for det in result.boxes:
                    box = [float(v) for v in det.xyxy[0].tolist()]
                    conf = float(det.conf[0])
                    cls = int(det.cls[0])
                    proposals.append({
                        "box": box,
                        "conf": conf,
                        "name": PROMPTS[label][cls] if 0 <= cls < len(PROMPTS[label]) else str(cls),
                    })
            # Deduplicate nearly identical prompts.
            unique: list[dict[str, Any]] = []
            for prop in sorted(proposals, key=lambda x: -x["conf"]):
                if all(iou(prop["box"], prev["box"]) < 0.92 for prev in unique):
                    unique.append(prop)
            proposals = unique

            selected = None
            if old is not None:
                candidates: list[dict[str, Any]] = []
                for prop in proposals:
                    box = prop["box"]
                    b_iou = iou(old, box)
                    contain = containment(old, box)
                    if contain < 0.82 and b_iou < 0.55:
                        continue
                    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                    old_area = max(0.0, old[2] - old[0]) * max(0.0, old[3] - old[1])
                    if area < old_area * 0.80 or area > width * height * 0.75:
                        continue
                    ocx, ocy = center(old)
                    bcx, bcy = center(box)
                    dist = ((ocx - bcx) ** 2 + (ocy - bcy) ** 2) ** 0.5
                    norm_dist = dist / ((width ** 2 + height ** 2) ** 0.5)
                    score = 1.6 * contain + 1.0 * b_iou + 0.5 * conf - 2.0 * norm_dist
                    candidates.append({
                        **prop, "iou_old": b_iou, "containment": contain,
                        "center_dist_norm": norm_dist, "expansion": area / old_area,
                        "score": score,
                    })
                # Require the independent detector to make a substantive proposal,
                # not merely reproduce a local box.
                if candidates:
                    best = max(candidates, key=lambda x: x["score"])
                    if best["expansion"] >= 1.04 or best["containment"] >= 0.97:
                        selected = best

            if selected is None:
                if old is None:
                    # Keep empty for the manual review queue.
                    (labels_dir / f"{stem}.txt").write_text("", encoding="utf-8")
                    status = "manual_required"
                else:
                    write_yolo(labels_dir / f"{stem}.txt", class_id, old, width, height)
                    status = "kept_old"
            else:
                write_yolo(labels_dir / f"{stem}.txt", class_id, selected["box"], width, height)
                status = "selected"

            overlay = cv2.imread(str(image_path))
            if old is not None:
                draw_box(overlay, old, (0, 255, 255), "old")
            for prop in proposals:
                draw_box(overlay, prop["box"], (0, 220, 0), f'{prop["name"]} {prop["conf"]:.2f}')
            if selected is not None:
                draw_box(overlay, selected["box"], (0, 0, 255), "selected")
            cv2.imwrite(str(overlays_dir / f"{stem}_independent_gt.jpg"), overlay)
            rows.append({
                "file": image_path.name,
                "image": str(image_path),
                "width": width,
                "height": height,
                "old": old,
                "proposals": proposals,
                "selected": selected,
                "status": status,
            })
        all_summary[label] = {
            "count": len(rows),
            "selected": sum(r["status"] == "selected" for r in rows),
            "kept_old": sum(r["status"] == "kept_old" for r in rows),
            "manual_required": sum(r["status"] == "manual_required" for r in rows),
            "rows": rows,
        }
        sheet_rows[label] = rows
        make_sheet(label, rows, args.out / f"{label}_independent_GT总览.jpg")

    (args.out / "summary.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        k: {kk: vv for kk, vv in v.items() if kk != "rows"}
        for k, v in all_summary.items()
    }, ensure_ascii=False, indent=2))
    print(f"out: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
