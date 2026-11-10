#!/usr/bin/env python3
"""Read-only per-frame scissors audit for the Track-1 v3 dataset.

Runs the v2 YOLO checkpoint on all 21 self-collected scissors frames, writes a
machine-readable report and a contact sheet. It never edits the v2 dataset and
never opens a camera or ROS device.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序")
RAW_ROOT = PROJECT_ROOT / "datasets/track1_front10_selfcollected_v1/rgb"
WEIGHTS = PROJECT_ROOT / "runs/track1_train_40class_v2/track1_40class_20261029_221534/weights/best.pt"
OUT = PROJECT_ROOT / "reports/track1_scissors_v3_allframes"

# Keep these values aligned with the YOLO checkpoint. Index 2 is class 3-Scissors.
SCISSORS_INDEX = 2
NAMES = [
    "1-Cheez-it", "2-Starkist_Tuna", "3-Scissors", "4-Frenchs_Mustard",
    "5-Tomato_Soup", "6-Foam_Brick", "7-Clamp", "8-Plastic_Banana", "9-Mug",
    "10-meat_can", "11-Padlock", "12-Baseball", "13-Bowl", "14-Sleeve",
    "15-Coca_cola_half", "16-ICRA_Duckie", "17-Eggs_Plush", "18-Dice",
    "19-Elmers_Glue", "20-Scotch_Tape", "21-Haribo_Golden_Bear",
    "22-E27_Adapter", "23-GU24_Adapter", "24-Spoon", "25-Flat_Screwdriver",
    "26-Plastic_Strawberry", "27-Coca_cola_full", "28-Kong_Duck_Dog_Toy",
    "29-Paper_Towels", "30-Stabilo_OHPen", "31-Plastic_White_Cup",
    "32-Wine_Glass", "33-Key", "34-Nail", "35-Book", "36-Adjustable_Wrench",
    "37-T-shirt", "38-Jumbo_Pencil_Cup", "39-Glove", "40-Pringles",
]

GROUPS = {
    "G1": {"frames": [1334, 1335, 1336], "split": "train", "status": "v2_reviewed_train"},
    "G2": {"frames": [1337, 1338, 1339], "split": "train", "status": "v3_needs_review"},
    "G3": {"frames": [1340, 1341, 1342], "split": "train", "status": "v3_needs_review"},
    "G4": {"frames": [1343, 1344, 1345], "split": "train", "status": "v3_needs_review"},
    "G5": {"frames": [1346, 1347, 1348], "split": "train", "status": "v3_needs_review"},
    "G6": {"frames": [1349, 1350, 1351], "split": "val", "status": "v2_reviewed_val"},
    "G7": {"frames": [1352, 1353, 1354], "split": "train", "status": "v3_needs_review"},
}


def image_for(frame: int) -> Path:
    matches = sorted(RAW_ROOT.glob(f"{frame:06d}_3-Scissors.*"))
    if len(matches) != 1:
        raise RuntimeError(f"frame {frame}: expected one image, found {matches}")
    return matches[0]


def main() -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    cache_root = PROJECT_ROOT / "cache/track1_training"
    for key in ("tmp", "matplotlib", "numba", "cuda", "torch", "xdg"):
        (cache_root / key).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(cache_root / "tmp"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(cache_root / "cuda"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "config/ultralytics"))
    os.environ.setdefault("WANDB_MODE", "disabled")

    from PIL import Image, ImageDraw, ImageFont
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    records = []
    thumbs = []
    for group, meta in GROUPS.items():
        for frame in meta["frames"]:
            image_path = image_for(frame)
            result = model.predict(
                source=str(image_path), conf=0.05, iou=0.60, imgsz=640,
                device=0, verbose=False, max_det=100,
            )[0]
            detections = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.detach().cpu().tolist()
                confs = result.boxes.conf.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                for box, conf, cls_id in zip(xyxy, confs, classes):
                    cls_id = int(cls_id)
                    detections.append({
                        "class_id": cls_id,
                        "class_name": NAMES[cls_id] if 0 <= cls_id < len(NAMES) else str(cls_id),
                        "confidence": round(float(conf), 6),
                        "xyxy": [round(float(v), 3) for v in box],
                    })
            detections.sort(key=lambda row: row["confidence"], reverse=True)
            scissors = [row for row in detections if row["class_id"] == SCISSORS_INDEX]
            best_scissors = scissors[0] if scissors else None
            top = detections[0] if detections else None
            record = {
                "group": group,
                "split": meta["split"],
                "status": meta["status"],
                "frame": frame,
                "image": str(image_path),
                "best_scissors": best_scissors,
                "top_detection": top,
                "detections_top10": detections[:10],
            }
            records.append(record)

            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            if best_scissors is not None:
                x1, y1, x2, y2 = best_scissors["xyxy"]
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=4)
                text = f"P scissors {best_scissors['confidence']:.3f}"
                color = (0, 255, 0)
            else:
                text = "NO scissors prediction"
                color = (255, 40, 40)
            if top is not None:
                extra = f" | top {top['class_name']} {top['confidence']:.3f}"
                if len(text) + len(extra) < 76:
                    text += extra
            draw.rectangle((0, 0, 639, 24), fill=(0, 0, 0))
            draw.text((5, 5), f"{group} {frame} {text}", fill=color, font=font)
            resized = image.resize((640, 480), Image.Resampling.LANCZOS)
            thumbs.append((group, frame, resized))

    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 640, rows * 480), (32, 32, 32))
    for idx, (_, _, image) in enumerate(thumbs):
        sheet.paste(image, ((idx % cols) * 640, (idx // cols) * 480))
    sheet_path = OUT / "scissors_allframes_contact.jpg"
    sheet.save(sheet_path, quality=94)

    created = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "schema_version": 1,
        "created_at": created,
        "scope": "read-only v2 inference on all 21 self-collected scissors frames",
        "weights": str(WEIGHTS),
        "conf": 0.05,
        "iou": 0.60,
        "imgsz": 640,
        "device": 0,
        "records": records,
    }
    (OUT / "scissors_allframes_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "scissors_allframes_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "split", "frame", "scissors_conf", "scissors_xyxy", "top_class", "top_conf"])
        for row in records:
            sc = row["best_scissors"]
            top = row["top_detection"]
            writer.writerow([
                row["group"], row["split"], row["frame"],
                "" if sc is None else f"{sc['confidence']:.6f}",
                "" if sc is None else " ".join(str(v) for v in sc["xyxy"]),
                "" if top is None else top["class_name"],
                "" if top is None else f"{top['confidence']:.6f}",
            ])
    print(json.dumps({
        "ok": True,
        "contact_sheet": str(sheet_path),
        "report": str(OUT / "scissors_allframes_audit.json"),
        "records": len(records),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
