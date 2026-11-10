#!/usr/bin/env python3
"""Evaluate detector + MobileSAM refinement for flat open scissors.

This is an offline feasibility test. It never uses GT to choose a prompt or a
class. GT is used only after inference to score boxes and class decisions.

Pipeline under test:
  1. Run YOLO at a detector size and keep low-confidence detector boxes from
     Scissors plus known confusers (Banana, meat_can, Tomato_Soup).
  2. Turn each detector box into bbox prompts (optionally extending left).
  3. Run MobileSAM and take the largest connected mask component.
  4. Crop the SAM bbox with a small context margin and ask YOLO for the class.
  5. Accept as Scissors only when the crop classifier's Scissors score beats
     other classes under the configured threshold/margin.

Outputs candidate-level CSV so class recovery, box quality, FP risk and latency
can be reported independently.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import SAM, YOLO

CLASS_NAMES = (
    "1-Cheez-it", "2-Starkist_Tuna", "3-Scissors", "4-Frenchs_Mustard",
    "5-Tomato_Soup", "6-Foam_Brick", "7-Clamp", "8-Plastic_Banana",
    "9-Mug", "10-meat_can",
)
SCISSORS_ID = 2
CONFUSER_IDS = (SCISSORS_ID, 4, 7, 9)  # scissors, soup, banana, meat can


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    bb = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def read_gt(label_path: Path, width: int, height: int):
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        rows.append((cls, [
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ]))
    return rows


def clip_box(box, width: int, height: int):
    return [
        min(max(float(box[0]), 0.0), width - 1.0),
        min(max(float(box[1]), 0.0), height - 1.0),
        min(max(float(box[2]), 0.0), width - 1.0),
        min(max(float(box[3]), 0.0), height - 1.0),
    ]


def expand_left(box, factor: float, width: int, height: int):
    x1, y1, x2, y2 = (float(v) for v in box)
    return clip_box([x1 - factor * max(0.0, x2 - x1), y1, x2, y2], width, height)


def pad_box(box, ratio: float, width: int, height: int):
    x1, y1, x2, y2 = (float(v) for v in box)
    dx = (x2 - x1) * ratio
    dy = (y2 - y1) * ratio
    return clip_box([x1 - dx, y1 - dy, x2 + dx, y2 + dy], width, height)


def largest_mask_component(mask: np.ndarray):
    binary = (mask > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == index).astype(np.uint8)


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0]


def mask_features(frame: np.ndarray, mask: np.ndarray):
    """Class-agnostic shape/color features of the SAM mask."""
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    area = int(mask.sum())
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    pixels = mask > 0

    metal = pixels & (sat < 85) & (val > 65) & (val < 250)
    yellow = pixels & (hue >= 18) & (hue <= 40) & (sat > 100) & (val > 90)
    orange = pixels & ((hue <= 18) | (hue >= 170)) & (sat > 100) & (val > 90)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
    points = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(points)
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    solidity = area / hull_area if hull_area > 0 else 0.0

    return {
        "aspect_ratio": float(width / height),
        "bbox_fill": float(area / (width * height)),
        "metal_frac": float(metal.sum() / area),
        "yellow_frac": float(yellow.sum() / area),
        "orange_frac": float(orange.sum() / area),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "mask_area": area,
        "bbox_width": width,
        "bbox_height": height,
    }


def scissors_feature_rule(features):
    """A fixed, explainable verifier for the competition's orange-handled scissors.

    It is deliberately strict. It is intended to recover only cases where SAM
    has already segmented the entire elongated object. Native YOLO Scissors
    detections remain untouched by this verifier.
    """
    if not features:
        return False
    return (
        features["aspect_ratio"] >= 3.0
        and features["bbox_fill"] >= 0.20
        and features["yellow_frac"] <= 0.30
        and features["metal_frac"] >= 0.20
        and features["orange_frac"] >= 0.03
        and features["circularity"] <= 0.40
        and features["solidity"] <= 0.85
    )


def dedupe_boxes(items):
    """Items are dicts with box + rank; remove near-duplicate boxes."""
    ordered = sorted(items, key=lambda item: item.get("rank", 0.0), reverse=True)
    kept = []
    for item in ordered:
        if all(iou_xyxy(item["box"], other["box"]) < 0.80 for other in kept):
            kept.append(item)
    return kept


def run_yolo_boxes(model: YOLO, frame: np.ndarray, imgsz: int, conf: float):
    started = time.perf_counter()
    result = model.predict(
        source=frame, imgsz=imgsz, conf=conf, iou=0.7, max_det=300, verbose=False
    )[0]
    elapsed = time.perf_counter() - started
    boxes = []
    if result.boxes is not None:
        for box, cls, score in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            boxes.append({
                "class_id": int(cls),
                "class_name": CLASS_NAMES[int(cls)],
                "conf": float(score),
                "box": [float(v) for v in box],
            })
    return boxes, elapsed


def run_sam(sam: SAM, frame: np.ndarray, prompt, width: int, height: int):
    started = time.perf_counter()
    result = sam(frame, bboxes=[prompt], verbose=False)[0]
    elapsed = time.perf_counter() - started
    if result.masks is None or len(result.masks.data) == 0:
        return None, elapsed
    # SAM can return multiple masks. Keep the component with the largest bbox
    # among its masks; this is class-agnostic and does not use GT.
    options = []
    for raw in result.masks.data.cpu().numpy():
        mask = raw
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = largest_mask_component(mask)
        box = mask_bbox(mask)
        if box is not None:
            options.append((box, mask))
    if not options:
        return None, elapsed
    box, mask = max(options, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))
    return {"box": box, "mask": mask}, elapsed


def classify_crop(model: YOLO, frame: np.ndarray, box, pad_ratio: float, imgsz: int):
    height, width = frame.shape[:2]
    crop_box = pad_box(box, pad_ratio, width, height)
    x1, y1, x2, y2 = (int(round(v)) for v in crop_box)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
    crop = frame[y1:min(y2, height), x1:min(x2, width)]
    if crop.size == 0:
        return {
            "top_class": "none", "top_conf": 0.0, "scissors_conf": 0.0,
            "banana_conf": 0.0, "meat_can_conf": 0.0, "soup_conf": 0.0,
            "clamp_conf": 0.0, "crop_box": crop_box, "elapsed": 0.0,
            "all": {},
        }
    boxes, elapsed = run_yolo_boxes(model, crop, imgsz=imgsz, conf=0.01)
    best = {}
    for item in boxes:
        best[item["class_id"]] = max(best.get(item["class_id"], 0.0), item["conf"])
    top = max(boxes, key=lambda item: item["conf"]) if boxes else None
    return {
        "top_class": top["class_name"] if top else "none",
        "top_conf": top["conf"] if top else 0.0,
        "scissors_conf": best.get(SCISSORS_ID, 0.0),
        "banana_conf": best.get(7, 0.0),
        "meat_can_conf": best.get(9, 0.0),
        "soup_conf": best.get(4, 0.0),
        "clamp_conf": best.get(6, 0.0),
        "crop_box": crop_box,
        "elapsed": elapsed,
        "all": {CLASS_NAMES[k]: v for k, v in sorted(best.items())},
    }


def dedupe_predictions(candidates, threshold: float, margin: float):
    accepted = []
    for item in candidates:
        other = max(
            item["banana_conf"], item["meat_can_conf"], item["soup_conf"],
            item["clamp_conf"],
        )
        top_other = max(
            (value for cls, value in item["crop_all"].items() if cls != "3-Scissors"),
            default=0.0,
        )
        if item["scissors_conf"] >= threshold and item["scissors_conf"] >= margin * top_other:
            accepted.append(item)
    accepted.sort(key=lambda item: item["scissors_conf"], reverse=True)
    deduped = []
    for item in accepted:
        if all(iou_xyxy(item["sam_box_list"], kept["sam_box_list"]) < 0.80 for kept in deduped):
            deduped.append(item)
    return deduped


def greedy_match(gt_boxes, predictions):
    """Return matches and unmatched predictions. GT/pred are (index, box)."""
    pairs = []
    for gi, gb in gt_boxes:
        for pi, pb in predictions:
            overlap = iou_xyxy(gb, pb)
            if overlap >= 0.75:
                pairs.append((overlap, gi, pi))
    pairs.sort(reverse=True)
    used_gt, used_pred, matches = set(), set(), []
    for overlap, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, overlap))
    return matches, [pi for pi, _ in predictions if pi not in used_pred]


def draw_overlay(image_path: Path, gt_rows, baseline, refined, out_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for cls, box in gt_rows:
        color = (0, 180, 0) if cls == SCISSORS_ID else (130, 210, 130)
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 3, max(0, box[1] - 14)), f"GT {CLASS_NAMES[cls]}", fill=color)
    for item in baseline:
        if item["class_id"] != SCISSORS_ID or item["conf"] < 0.25:
            continue
        box = item["box"]
        draw.rectangle(box, outline=(30, 110, 255), width=2)
        draw.text((box[0] + 3, box[1] + 3), f"base {item['conf']:.2f}", fill=(30, 110, 255))
    for item in refined:
        box = item["sam_box_list"]
        draw.rectangle(box, outline=(220, 0, 180), width=3)
        draw.text((box[0] + 3, box[1] + 16), f"SAM {item['scissors_conf']:.2f}", fill=(220, 0, 180))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="+")
    parser.add_argument("--data", default="/media/sunshine/新加卷/taskA/datasets/real_ft_v25_hist_holdout_20261016")
    parser.add_argument("--sam", default="/home/sunshine/codex/easton/taskA/mobile_sam.pt")
    parser.add_argument("--out", default="/media/sunshine/新加卷/taskA/reports/scissors_sam_refine_20261016")
    parser.add_argument("--det-imgsz", type=int, default=960)
    parser.add_argument("--crop-imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.05)
    parser.add_argument("--candidate-conf", type=float, default=0.05)
    parser.add_argument("--crop-pad", type=float, default=0.08)
    parser.add_argument("--thresholds", default="0.15,0.25,0.35,0.50")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--limit-nonscissors", type=int, default=0,
                        help="0 means evaluate all holdout images; >0 samples this many non-scissors images per class.")
    args = parser.parse_args()

    root = Path(args.data).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images under {image_dir}")

    manifest = {}
    manifest_path = root / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as fh:
            manifest = {row["image"]: row for row in csv.DictReader(fh)}

    thresholds = [float(v) for v in args.thresholds.split(",") if v]
    sam = SAM(args.sam)
    all_candidates = []
    baseline_rows = []
    image_rows = []
    timing = defaultdict(list)

    for weight_arg in args.weights:
        weight = Path(weight_arg).expanduser().resolve()
        model_name = weight.parent.parent.name if weight.name == "best.pt" else weight.stem
        model = YOLO(str(weight))
        selected_images = []
        class_seen = Counter()
        for image_path in images:
            gt = read_gt(label_dir / f"{image_path.stem}.txt", *cv2.imread(str(image_path)).shape[1::-1])
            classes = {cls for cls, _ in gt}
            if SCISSORS_ID in classes:
                selected_images.append(image_path)
            elif args.limit_nonscissors <= 0:
                selected_images.append(image_path)
            else:
                primary = next((CLASS_NAMES[cls] for cls, _ in gt), "unknown")
                if class_seen[primary] < args.limit_nonscissors:
                    selected_images.append(image_path)
                    class_seen[primary] += 1

        for image_index, image_path in enumerate(selected_images, 1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            height, width = frame.shape[:2]
            gt = read_gt(label_dir / f"{image_path.stem}.txt", width, height)
            gt_scissors = [(i, box) for i, (cls, box) in enumerate(gt) if cls == SCISSORS_ID]
            detections, det_time = run_yolo_boxes(model, frame, args.det_imgsz, args.det_conf)
            timing[f"{model_name}_detection"].append(det_time)

            candidates = []
            for detection in detections:
                if detection["class_id"] not in CONFUSER_IDS:
                    continue
                if detection["conf"] < args.candidate_conf:
                    continue
                prompts = []
                if detection["class_id"] == SCISSORS_ID:
                    prompts.append(("exact", list(detection["box"])))
                    prompts.append(("left0.5", expand_left(detection["box"], 0.5, width, height)))
                else:
                    prompts.append(("left0.5", expand_left(detection["box"], 0.5, width, height)))
                    prompts.append(("left0.8", expand_left(detection["box"], 0.8, width, height)))
                    prompts.append(("left1.0", expand_left(detection["box"], 1.0, width, height)))
                seen_prompts = set()
                for prompt_name, prompt in prompts:
                    prompt = clip_box(prompt, width, height)
                    key = tuple(round(v, 1) for v in prompt)
                    if key in seen_prompts:
                        continue
                    seen_prompts.add(key)
                    sam_result, sam_time = run_sam(sam, frame, prompt, width, height)
                    timing[f"{model_name}_sam"].append(sam_time)
                    if sam_result is None:
                        continue
                    crop_info = classify_crop(model, frame, sam_result["box"], args.crop_pad, args.crop_imgsz)
                    timing[f"{model_name}_crop_classifier"].append(crop_info["elapsed"])
                    features = mask_features(frame, sam_result["mask"])
                    feature_accept = scissors_feature_rule(features)
                    best_gt_iou = max((iou_xyxy(sam_result["box"], box) for _, box in gt_scissors), default=None)
                    row = {
                        "model": model_name,
                        "det_imgsz": args.det_imgsz,
                        "crop_imgsz": args.crop_imgsz,
                        "image": image_path.name,
                        "physical_group": manifest.get(image_path.name, {}).get("physical_group", ""),
                        "source_class": detection["class_name"],
                        "source_conf": round(detection["conf"], 6),
                        "prompt_variant": prompt_name,
                        "prompt": json.dumps([round(v, 2) for v in prompt]),
                        "sam_box": json.dumps([round(v, 2) for v in sam_result["box"]]),
                        "sam_area": int(sam_result["mask"].sum()),
                        "iou_best_scissors_gt": None if best_gt_iou is None else round(best_gt_iou, 6),
                        "crop_top_class": crop_info["top_class"],
                        "crop_top_conf": round(crop_info["top_conf"], 6),
                        "scissors_conf": round(crop_info["scissors_conf"], 6),
                        "banana_conf": round(crop_info["banana_conf"], 6),
                        "meat_can_conf": round(crop_info["meat_can_conf"], 6),
                        "soup_conf": round(crop_info["soup_conf"], 6),
                        "clamp_conf": round(crop_info["clamp_conf"], 6),
                        "crop_all": json.dumps({k: round(v, 6) for k, v in crop_info["all"].items()}, ensure_ascii=False),
                        "feature_accept": int(feature_accept),
                        "feature_aspect_ratio": round(features["aspect_ratio"], 6) if features else 0.0,
                        "feature_bbox_fill": round(features["bbox_fill"], 6) if features else 0.0,
                        "feature_metal_frac": round(features["metal_frac"], 6) if features else 0.0,
                        "feature_yellow_frac": round(features["yellow_frac"], 6) if features else 0.0,
                        "feature_orange_frac": round(features["orange_frac"], 6) if features else 0.0,
                        "feature_circularity": round(features["circularity"], 6) if features else 0.0,
                        "feature_solidity": round(features["solidity"], 6) if features else 0.0,
                    }
                    candidates.append({**row, "sam_box_list": sam_result["box"], "crop_all": crop_info["all"]})
            all_candidates.extend(candidates)

            baseline = [item for item in detections if item["class_id"] == SCISSORS_ID and item["conf"] >= 0.25]
            baseline_matches, _ = greedy_match(
                gt_scissors,
                [(i, item["box"]) for i, item in enumerate(baseline)],
            )
            for gi, gt_box in gt_scissors:
                best_base = max((iou_xyxy(item["box"], gt_box) for item in baseline), default=0.0)
                baseline_rows.append({
                    "model": model_name,
                    "image": image_path.name,
                    "physical_group": manifest.get(image_path.name, {}).get("physical_group", ""),
                    "gt": json.dumps([round(v, 2) for v in gt_box]),
                    "baseline_best_iou": round(best_base, 6),
                    "baseline_pass": int(best_base >= 0.75),
                })

            feature_items = [
                {
                    "box": item["sam_box_list"],
                    "rank": float(item["feature_aspect_ratio"]) + float(item["feature_metal_frac"])
                            - float(item["feature_yellow_frac"]),
                }
                for item in candidates if int(item["feature_accept"]) == 1
            ]
            feature_final = dedupe_boxes(feature_items)
            baseline_items = [
                {"box": item["box"], "rank": float(item["conf"])} for item in baseline
            ]
            hybrid_final = dedupe_boxes(baseline_items + feature_items)

            image_record = {
                "model": model_name,
                "image": image_path.name,
                "physical_group": manifest.get(image_path.name, {}).get("physical_group", ""),
                "has_scissors_gt": int(bool(gt_scissors)),
                "gt_scissors": len(gt_scissors),
                "detections": len(detections),
                "candidates": len(candidates),
                "oracle_best_iou": round(max(
                    (item["iou_best_scissors_gt"] for item in candidates if item["iou_best_scissors_gt"] is not None),
                    default=0.0,
                ), 6),
                "feature_accepted": len(feature_final),
                "baseline_accepted": len(baseline_items),
                "hybrid_accepted": len(hybrid_final),
            }
            for label, predictions in (("feature", feature_final), ("hybrid", hybrid_final)):
                boxes = [(i, item["box"]) for i, item in enumerate(predictions)]
                matches, unmatched = greedy_match(gt_scissors, boxes)
                image_record[f"{label}_tp"] = len(matches)
                image_record[f"{label}_fp"] = len(unmatched) if gt_scissors else len(predictions)
                image_record[f"{label}_boxes"] = json.dumps([
                    [round(v, 2) for v in item["box"]] for item in predictions
                ])
            for threshold in thresholds:
                final = dedupe_predictions(candidates, threshold, args.margin)
                key = f"t{threshold:.2f}"
                boxes = [(i, item["sam_box_list"]) for i, item in enumerate(final)]
                matches, unmatched = greedy_match(gt_scissors, boxes)
                image_record[f"{key}_accepted"] = len(final)
                image_record[f"{key}_accepted_boxes"] = json.dumps([
                    [round(v, 2) for v in item["sam_box_list"]] for item in final
                ])
                image_record[f"{key}_tp"] = len(matches)
                image_record[f"{key}_fp"] = len(unmatched) if gt_scissors else len(final)
            image_rows.append(image_record)

            # Save overlays for the middle threshold at the first evaluation size.
            overlay_threshold = thresholds[min(1, len(thresholds) - 1)] if thresholds else 0.25
            refined = dedupe_predictions(candidates, overlay_threshold, args.margin)
            overlay_path = out / "overlays" / model_name / image_path.name
            draw_overlay(image_path, gt, detections, refined, overlay_path)

            if image_index % 25 == 0:
                print(f"[{model_name}] {image_index}/{len(selected_images)} images")
        del model

    candidate_fields = list(all_candidates[0]) if all_candidates else []
    if candidate_fields:
        with (out / "candidates.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=candidate_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_candidates)

    baseline_fields = ["model", "image", "physical_group", "gt", "baseline_best_iou", "baseline_pass"]
    with (out / "baseline_instances.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=baseline_fields)
        writer.writeheader()
        writer.writerows(baseline_rows)

    image_fields = []
    if image_rows:
        for key in image_rows[0]:
            if key not in image_fields:
                image_fields.append(key)
        for row in image_rows:
            for key in row:
                if key not in image_fields:
                    image_fields.append(key)
    with (out / "images.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=image_fields)
        writer.writeheader()
        writer.writerows(image_rows)

    # Summarize fixed rules. The feature rule is intentionally explicit rather
    # than fitted on this evaluation set; both feature-only and hybrid results
    # are reported without selecting a new threshold from the holdout.
    def batch_tag(name):
        if name.startswith("sim_") or "_sim_" in name:
            return "sim"
        if "20260926" in name:
            return "real_v5_0926"
        if "20260927" in name:
            return "real_v9_0927"
        return "other"

    summary = {
        "dataset": str(root),
        "evaluation_only": True,
        "det_imgsz": args.det_imgsz,
        "crop_imgsz": args.crop_imgsz,
        "det_conf": args.det_conf,
        "candidate_conf": args.candidate_conf,
        "crop_pad": args.crop_pad,
        "margin": args.margin,
        "thresholds": thresholds,
        "feature_rule": {
            "aspect_ratio_min": 3.0,
            "bbox_fill_min": 0.20,
            "yellow_frac_max": 0.30,
            "metal_frac_min": 0.20,
            "orange_frac_min": 0.03,
            "circularity_max": 0.40,
            "solidity_max": 0.85,
        },
        "models": {},
        "timing_ms_median": {name: round(float(np.median(values)) * 1000.0, 3) for name, values in timing.items()},
        "timing_ms_mean": {name: round(float(np.mean(values)) * 1000.0, 3) for name, values in timing.items()},
    }
    lines = [
        "# 剪刀：检测器候选 + MobileSAM 校正可行性报告", "",
        f"- 数据：`{root}`（仅评估，不训练、不采集）",
        f"- 检测尺寸：{args.det_imgsz}，低置信入口：{args.det_conf}",
        f"- 候选尺寸：{args.crop_imgsz}，裁剪上下文：{args.crop_pad:.2f}",
        "- 主判定：MobileSAM 分割后的形状/颜色规则；同时保留 YOLO 原生 Scissors 检测。",
        "- GT 只用于推理后的评分，不参与候选选择、SAM 提示词或类别判定。", "",
        "## 指标定义", "",
        "- `baseline`：YOLO 原生 Scissors 框，入口 conf>=0.25 且 IoU>=0.75。",
        "- `oracle_box`：只问“SAM 能否把扩左后的候选框恢复成 IoU>=0.75”，不考虑类别判定。",
        "- `feature_only`：只看 SAM mask 的形状/颜色规则。",
        "- `hybrid`：原生 Scissors 框 + 形状/颜色规则校正框，按 0.80 IoU 去重。",
        "- `FP`：在非剪刀图上输出 Scissors，或在剪刀图上多输出未匹配框。", "",
        "## 总体", "",
        "| 模型 | 剪刀图 | GT | baseline TP | oracle 框 IoU75 | feature TP | feature FP | feature recall | hybrid TP | hybrid FP | hybrid recall | hybrid precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in sorted({row["model"] for row in image_rows}):
        model_images = [row for row in image_rows if row["model"] == model_name]
        scissor_images = [row for row in model_images if row["has_scissors_gt"]]
        gts = sum(row["gt_scissors"] for row in scissor_images)
        base_tp = sum(row["baseline_pass"] for row in baseline_rows if row["model"] == model_name)
        oracle = sum(row["oracle_best_iou"] >= 0.75 for row in scissor_images)
        feature_tp = sum(row["feature_tp"] for row in model_images)
        feature_fp = sum(row["feature_fp"] for row in model_images)
        hybrid_tp = sum(row["hybrid_tp"] for row in model_images)
        hybrid_fp = sum(row["hybrid_fp"] for row in model_images)
        values = {
            "scissors_images": len(scissor_images),
            "gt": gts,
            "baseline_tp": base_tp,
            "oracle_iou75_images": oracle,
            "feature_tp": feature_tp,
            "feature_fp": feature_fp,
            "hybrid_tp": hybrid_tp,
            "hybrid_fp": hybrid_fp,
            "feature_recall": feature_tp / gts if gts else 0.0,
            "hybrid_recall": hybrid_tp / gts if gts else 0.0,
            "hybrid_precision": hybrid_tp / (hybrid_tp + hybrid_fp) if hybrid_tp + hybrid_fp else 0.0,
        }
        summary["models"][model_name] = values
        lines.append(
            f"| {model_name} | {len(scissor_images)} | {gts} | {base_tp} | {oracle} | "
            f"{feature_tp} | {feature_fp} | {values['feature_recall']:.3f} | "
            f"{hybrid_tp} | {hybrid_fp} | {values['hybrid_recall']:.3f} | {values['hybrid_precision']:.3f} |"
        )

    lines.extend(["", "## 分批次（每模型）", "", "| 模型 | 批次 | 剪刀图 | GT | baseline TP | oracle 框 IoU75 | feature TP | hybrid TP | hybrid FP |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for model_name in sorted({row["model"] for row in image_rows}):
        model_images = [row for row in image_rows if row["model"] == model_name]
        for batch in ("real_v5_0926", "real_v9_0927", "sim", "other"):
            batch_images = [row for row in model_images if row["has_scissors_gt"] and batch_tag(row["image"]) == batch]
            if not batch_images:
                continue
            gts = sum(row["gt_scissors"] for row in batch_images)
            base_tp = sum(row["baseline_pass"] for row in baseline_rows if row["model"] == model_name and batch_tag(row["image"]) == batch)
            oracle = sum(row["oracle_best_iou"] >= 0.75 for row in batch_images)
            feature_tp = sum(row["feature_tp"] for row in batch_images)
            hybrid_tp = sum(row["hybrid_tp"] for row in batch_images)
            hybrid_fp = sum(row["hybrid_fp"] for row in batch_images)
            lines.append(f"| {model_name} | {batch} | {len(batch_images)} | {gts} | {base_tp} | {oracle} | {feature_tp} | {hybrid_tp} | {hybrid_fp} |")

    lines.extend(["", "## 非剪刀图误报审计", "", "| 模型 | 非剪刀图 | baseline 输出框 | feature 输出框 | hybrid 输出框 | feature 误报图 | hybrid 误报图 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for model_name in sorted({row["model"] for row in image_rows}):
        non = [row for row in image_rows if row["model"] == model_name and not row["has_scissors_gt"]]
        base_boxes = sum(row["baseline_accepted"] for row in non)
        feature_boxes = sum(row["feature_accepted"] for row in non)
        hybrid_boxes = sum(row["hybrid_accepted"] for row in non)
        feature_images = sum(row["feature_accepted"] > 0 for row in non)
        hybrid_images = sum(row["hybrid_accepted"] > 0 for row in non)
        lines.append(f"| {model_name} | {len(non)} | {base_boxes} | {feature_boxes} | {hybrid_boxes} | {feature_images} | {hybrid_images} |")

    lines.extend(["", "## 形状/颜色验证规则", "",
                  "- `aspect_ratio >= 3.0`：完整平放剪刀应明显细长。",
                  "- `bbox_fill >= 0.20`：排除几乎空白的 SAM mask。",
                  "- `yellow_frac <= 0.30`：拒绝塑料香蕉的黄色主体。",
                  "- `metal_frac >= 0.20`：要求 SAM mask 内存在足够的灰色金属刀刃。",
                  "- `orange_frac >= 0.03`：要求存在橙色手柄。",
                  "- `circularity <= 0.40`、`solidity <= 0.85`：拒绝规则圆润/实心的罐头、盒子等。", "",
                  "## 提示词统计", "",
                  "| 模型 | 提示词 | 候选数 | SAM 框 IoU>=0.75 | feature 接受 |",
                  "|---|---|---:|---:|---:|"])
    for model_name in sorted({row["model"] for row in all_candidates}):
        candidates = [row for row in all_candidates if row["model"] == model_name]
        for variant in sorted({row["prompt_variant"] for row in candidates}):
            group = [row for row in candidates if row["prompt_variant"] == variant]
            good = sum(row["iou_best_scissors_gt"] is not None and row["iou_best_scissors_gt"] >= 0.75 for row in group)
            accepted = sum(int(row["feature_accept"]) for row in group)
            lines.append(f"| {model_name} | {variant} | {len(group)} | {good} | {accepted} |")

    lines.extend(["", "## 耗时", "", "| 阶段 | 中位 ms | 均值 ms |", "|---|---:|---:|"])
    for stage in sorted(timing):
        values = timing[stage]
        lines.append(f"| {stage} | {np.median(values) * 1000:.2f} | {np.mean(values) * 1000:.2f} |")
    lines.extend([
        "", "## 结论边界", "",
        "- 该结果是在历史未见 holdout 上的一次可行性验证；规则本身尚未作为生产阈值冻结。",
        "- 真实平放剪刀主要走 MobileSAM 恢复，竖直/闭合剪刀仍优先保留原生 YOLO Scissors 框。",
        "- 非剪刀图 FP 是部署门禁；上线前仍需在更多真实空桌、香蕉、肉罐头和橙柄物体上复核。",
        "- 当前未训练、未替换 `camera_recognition.py` 权重、未控制机械臂/机械爪。",
    ])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out / 'summary.md'}")
    print(f"[saved] {out / 'candidates.csv'}")
    print(f"[saved] {out / 'images.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
