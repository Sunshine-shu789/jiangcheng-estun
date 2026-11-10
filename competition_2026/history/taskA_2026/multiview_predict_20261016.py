#!/usr/bin/env python3
"""Multi-view offline inference on the fixed 2026-10-16 eval set.

Reproduces the deployment views (plain + the service's crop-zoom onto a gray
canvas) crossed with TTA, and saves every view's detections in ORIGINAL frame
coordinates so a fusion/arbiter stage can be tested without a camera.

Views:
  v1.00        imgsz=640, augment=False
  v1.00_tta    imgsz=640, augment=True
  v0.75        crop-zoom 0.75, augment=False
  v0.75_tta    crop-zoom 0.75, augment=True
  v0.65        crop-zoom 0.65, augment=False
  v0.65_tta    crop-zoom 0.65, augment=True

Output: reports/camera_eval_20261016/multiview_predictions_20261016/<class>/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path("/media/sunshine/新加卷/taskA")
WEIGHTS = ROOT / "runs" / "yolo11s_real_ft_v23_hardneg_20261016" / "weights" / "epoch3.pt"
EVAL = ROOT / "eval_capture_20261016"
OUT = ROOT / "reports" / "camera_eval_20261016" / "multiview_predictions_20261016"

FOLDERS = ["1-Cheez-it", "2-Starkist_Tuna", "3-Scissors", "4-Frenchs_Mustard",
           "5-Tomato_Soup", "6-Foam_Brick", "7-Clamp", "9-Mug", "10-meat_can"]

VIEWS = [
    ("v1.00", 1.0, False),
    ("v1.00_tta", 1.0, True),
    ("v0.75", 0.75, False),
    ("v0.75_tta", 0.75, True),
    ("v0.65", 0.65, False),
    ("v0.65_tta", 0.65, True),
]


def crop_zoom(frame: np.ndarray, scale: float) -> tuple[np.ndarray, int, int]:
    """Same transform as live_camera_preview.prepare_inference_frame."""
    if scale == 1.0:
        return frame, 0, 0
    h, w = frame.shape[:2]
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    xo, yo = (w - sw) // 2, (h - sh) // 2
    canvas[yo:yo + sh, xo:xo + sw] = resized
    return canvas, xo, yo


def map_back(box: list[float], xo: int, yo: int, scale: float) -> list[float]:
    return [round((box[0] - xo) / scale, 1), round((box[1] - yo) / scale, 1),
            round((box[2] - xo) / scale, 1), round((box[3] - yo) / scale, 1)]


def run_view(model: YOLO, images: list[Path], scale: float, tta: bool,
             conf: float, imgsz: int, device: str, chunk: int) -> list[list[dict]]:
    """Return per-image detection lists in original coordinates."""
    per_image: list[list[dict]] = [[] for _ in images]
    for start in range(0, len(images), chunk):
        block = images[start:start + chunk]
        frames, metas = [], []
        for p in block:
            frame = cv2.imread(str(p))
            if frame is None:
                raise FileNotFoundError(p)
            if scale != 1.0:
                f, xo, yo = crop_zoom(frame, scale)
                frames.append(f)
                metas.append((xo, yo, scale))
            else:
                frames.append(frame)
                metas.append((0, 0, 1.0))
        results = model.predict(source=frames, imgsz=imgsz, conf=conf, iou=0.7,
                                device=device, augment=tta, verbose=False, max_det=100)
        for i, r in enumerate(results):
            xo, yo, sc = metas[i]
            dets = []
            if r.boxes is not None:
                for box, c, cid in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    dets.append({"class_id": int(cid), "conf": round(float(c), 4),
                                 "xyxy": map_back([x1, y1, x2, y2], xo, yo, sc)})
            per_image[start + i] = dets
    return per_image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--folders", default="")
    args = ap.parse_args()

    folders = [f for f in args.folders.split(",") if f] or FOLDERS
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    for folder in folders:
        images = sorted((EVAL / folder).glob("*.jpg"))
        if not images:
            print(f"[skip] {folder}: no images")
            continue
        rows = {p.name: {"file": p.name} for p in images}
        for name, scale, tta in VIEWS:
            per_image = run_view(model, images, scale, tta, args.conf, args.imgsz,
                                 args.device, args.chunk)
            for p, dets in zip(images, per_image):
                rows[p.name][name] = dets
            print(f"[{folder}] {name}: done")
        d = OUT / folder
        d.mkdir(parents=True, exist_ok=True)
        payload = {"weights": str(WEIGHTS), "conf": args.conf, "imgsz": args.imgsz,
                   "views": [v[0] for v in VIEWS], "rows": list(rows.values())}
        (d / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
        print(f"[{folder}] wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
