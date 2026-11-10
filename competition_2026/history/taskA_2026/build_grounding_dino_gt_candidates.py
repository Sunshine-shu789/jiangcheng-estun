#!/usr/bin/env python3
"""Build independent GT candidates using Grounding DINO Tiny.

This is an evaluation-only annotation aid. It does not train or alter the
competition detector. Existing v2 labels are used only as a location prior;
the selected box is the best matching Grounding DINO full-object proposal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

PROMPTS: dict[str, str] = {
    "1-Cheez-it": "a cheez-it box. a cracker box.",
    "2-Starkist_Tuna": "a tuna can. a tin of tuna. a canned fish can.",
    "3-Scissors": "a pair of scissors. scissors.",
    "4-Frenchs_Mustard": "a mustard bottle. a yellow mustard bottle.",
    "5-Tomato_Soup": "a tomato soup can. a red soup can. a tin can.",
    "6-Foam_Brick": "a white foam block. a foam brick. a sponge block.",
    "7-Clamp": "a black and orange clip. a black clip. a spring clamp.",
    "8-Plastic_Banana": "a plastic banana. a yellow banana.",
    "9-Mug": "a coffee mug. a white mug. a ceramic mug.",
    "10-meat_can": "a canned meat can. a meat can. a tin can.",
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
    p.add_argument("--prior-gt-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--classes", default="2-Starkist_Tuna,5-Tomato_Soup,6-Foam_Brick,7-Clamp,9-Mug")
    p.add_argument("--threshold", type=float, default=0.08)
    p.add_argument("--text-threshold", type=float, default=0.08)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cuda")
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


def load_yolo(path: Path, width: int, height: int) -> list[float] | None:
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
    path.write_text(
        f"{class_id} {((x1+x2)/2)/width:.8f} {((y1+y2)/2)/height:.8f} "
        f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}\n",
        encoding="utf-8",
    )


def get_font(size: int) -> ImageFont.ImageFont:
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
    tw, th, cap = 320, 240, 58
    cols = 4
    sheet = Image.new("RGB", (cols * tw, ((len(rows) + cols - 1) // cols) * (th + cap)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    f = get_font(14)
    for i, row in enumerate(rows):
        image = cv2.imread(str(row["image"]))
        if image is None:
            continue
        image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
        if row.get("selected"):
            x1, y1, x2, y2 = [v * tw / row["width"] for v in row["selected"]["box"][:2]] + [v * tw / row["width"] for v in row["selected"]["box"][2:]]
            y1 *= th / row["height"]; y2 *= th / row["height"]
            cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 0, 255), 2)
        x = (i % cols) * tw
        y = (i // cols) * (th + cap)
        sheet.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (x, y))
        status = row["status"]
        color = {"selected": (80, 210, 80), "manual_required": (240, 170, 50), "kept_prior": (220, 220, 80)}[status]
        d.rectangle((x, y + th, x + tw - 1, y + th + cap - 1), fill=color)
        sel = row.get("selected")
        detail = "no candidate"
        if sel:
            detail = f'{sel["label"]} {sel["conf"]:.3f} iou={sel["iou_prior"]:.2f} cov={sel["containment_prior"]:.2f}'
        d.text((x + 5, y + th + 4), f"{status} | {detail}", fill=(5, 5, 5), font=f)
        d.text((x + 5, y + th + 30), row["file"][:46], fill=(5, 5, 5), font=f)
    sheet.save(out, quality=93)


def main() -> int:
    args = parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output dir is not empty: {args.out}")
    labels_dir = args.out / "labels"
    overlays_dir = args.out / "overlays"
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(args.model), local_files_only=True
    ).to(device).eval()

    all_summary: dict[str, Any] = {}
    for label in args.classes.split(","):
        if label not in PROMPTS:
            raise SystemExit(f"unknown class: {label}")
        class_id = CLASS_IDS[label]
        image_paths = sorted((args.capture_root / label).glob("*.jpg"))
        rows: list[dict[str, Any]] = []
        prompt = PROMPTS[label]
        for start in range(0, len(image_paths), args.batch):
            batch_paths = image_paths[start:start + args.batch]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
            sizes = [im.size[::-1] for im in images]
            processed = processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=args.threshold,
                text_threshold=args.text_threshold, target_sizes=sizes,
            )
            for image_path, image, result in zip(batch_paths, images, processed):
                width, height = image.size
                prior_path = args.prior_gt_root / label / "labels" / f"{image_path.stem}.txt"
                if not prior_path.exists():
                    prior_path = args.prior_gt_root / "labels" / f"{image_path.stem}.txt"
                prior = load_yolo(prior_path, width, height)
                proposals: list[dict[str, Any]] = []
                for score, box_t, label_t in zip(result["scores"], result["boxes"], result.get("text_labels", result["labels"])):
                    box = [float(v) for v in box_t.tolist()]
                    proposals.append({"box": box, "conf": float(score), "label": str(label_t)})
                unique: list[dict[str, Any]] = []
                for prop in sorted(proposals, key=lambda x: -x["conf"]):
                    if all(iou(prop["box"], prev["box"]) < 0.82 for prev in unique):
                        unique.append(prop)

                scored: list[dict[str, Any]] = []
                if prior is not None:
                    pa = max(0.0, prior[2] - prior[0]) * max(0.0, prior[3] - prior[1])
                    pcx, pcy = center(prior)
                    diag = (width ** 2 + height ** 2) ** 0.5
                    for prop in unique:
                        box = prop["box"]
                        b_iou = iou(prior, box)
                        cov = containment(prior, box)
                        bx = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                        expansion = bx / pa if pa > 0 else 0.0
                        bcx, bcy = center(box)
                        norm_dist = (((pcx - bcx) ** 2 + (pcy - bcy) ** 2) ** 0.5) / diag
                        # Reject only boxes that clearly miss most of the prior object.
                        if cov < 0.65 and b_iou < 0.30:
                            continue
                        if expansion > 4.5 or bx > width * height * 0.85:
                            continue
                        prop = {**prop, "iou_prior": b_iou, "containment_prior": cov,
                                "expansion": expansion, "center_dist_norm": norm_dist}
                        prop["score"] = 2.4 * cov + 0.9 * b_iou + 0.7 * prop["conf"] - 1.5 * norm_dist
                        scored.append(prop)

                selected = max(scored, key=lambda x: x["score"]) if scored else None
                if selected is None and prior is not None:
                    write_yolo(labels_dir / f"{image_path.stem}.txt", class_id, prior, width, height)
                    status = "kept_prior"
                elif selected is None:
                    (labels_dir / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
                    status = "manual_required"
                else:
                    write_yolo(labels_dir / f"{image_path.stem}.txt", class_id, selected["box"], width, height)
                    status = "selected"

                overlay = cv2.imread(str(image_path))
                if prior is not None:
                    draw_box(overlay, prior, (0, 255, 255), "prior")
                for prop in scored:
                    draw_box(overlay, prop["box"], (0, 220, 0), f'{prop["label"]} {prop["conf"]:.2f}')
                if selected is not None:
                    draw_box(overlay, selected["box"], (0, 0, 255), "selected")
                cv2.imwrite(str(overlays_dir / image_path.name), overlay)

                rows.append({
                    "file": image_path.name,
                    "image": str(image_path),
                    "width": width,
                    "height": height,
                    "prior": prior,
                    "proposals": unique,
                    "scored": scored,
                    "selected": selected,
                    "status": status,
                })
        all_summary[label] = {
            "prompt": prompt,
            "count": len(rows),
            "selected": sum(r["status"] == "selected" for r in rows),
            "kept_prior": sum(r["status"] == "kept_prior" for r in rows),
            "manual_required": sum(r["status"] == "manual_required" for r in rows),
            "rows": rows,
        }
        make_sheet(label, rows, args.out / f"{label}_grounding_dino_GT总览.jpg")

    (args.out / "summary.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: {x: v[x] for x in ("count", "selected", "kept_prior", "manual_required")}
                      for k, v in all_summary.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
