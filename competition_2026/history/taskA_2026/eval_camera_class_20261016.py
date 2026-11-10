#!/usr/bin/env python3
"""One-class real-camera evaluation for the Easton task-A camera set.

Outputs per-frame base/TTA results, annotated base predictions, a contact sheet,
and a failure sheet. Inference receives image paths directly so Ultralytics
loads them through its normal BGR pipeline.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


THRESHOLD = 0.25
IMGSZ = 640
WEIGHTS = Path(
    "/media/sunshine/新加卷/taskA/runs/yolo11s_real_ft_v23_hardneg_20261016/weights/epoch3.pt"
)
CAPTURE_ROOT = Path("/media/sunshine/新加卷/taskA/eval_capture_20261016")
REPORT_ROOT = Path("/media/sunshine/新加卷/taskA/reports/camera_eval_20261016")


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def detections(result) -> list[dict]:
    out: list[dict] = []
    boxes = result.boxes
    if boxes is None:
        return out
    names = result.names
    for box in boxes:
        cls_id = int(box.cls.item())
        xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        out.append(
            {
                "class_id": cls_id,
                "name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id],
                "conf": round(float(box.conf.item()), 4),
                "xyxy": xyxy,
            }
        )
    return out


def target_hit(dets: list[dict], class_id: int) -> bool:
    return any(d["class_id"] == class_id for d in dets)


def write_label_image(src: Path, dst: Path, dets: list[dict], status: str) -> None:
    image = cv2.imread(str(src))
    if image is None:
        raise RuntimeError(f"cannot read image: {src}")
    palette = [
        (64, 220, 64), (30, 160, 255), (255, 100, 100), (80, 220, 220),
        (220, 80, 220), (80, 140, 255), (255, 190, 40), (120, 255, 120),
        (220, 220, 60), (160, 80, 255),
    ]
    thickness = 2
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        color = palette[d["class_id"] % len(palette)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        text = f'{d["name"]} {d["conf"]:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        ty = max(y1, th + 8)
        cv2.rectangle(image, (x1, ty - th - 7), (x1 + tw + 7, ty), color, -1)
        cv2.putText(
            image, text, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (15, 15, 15), 1, cv2.LINE_AA,
        )
    cv2.putText(
        image, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
        (0, 255, 255), 2, cv2.LINE_AA,
    )
    if not cv2.imwrite(str(dst), image):
        raise RuntimeError(f"cannot write image: {dst}")


def fit_thumb(image: Image.Image, width: int = 480, height: int = 360) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (28, 28, 28))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def make_contact_sheet(rows: list[dict], pred_dir: Path, dst: Path, title: str) -> None:
    cols, thumb_w, thumb_h, caption_h = 4, 480, 360, 42
    n_rows = (len(rows) + cols - 1) // cols
    header_h = 72
    sheet = Image.new("RGB", (cols * thumb_w, header_h + n_rows * (thumb_h + caption_h)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 12), title, fill=(255, 255, 255), font=font(34))
    for i, row in enumerate(rows):
        col, line = i % cols, i // cols
        x, y = col * thumb_w, header_h + line * (thumb_h + caption_h)
        base_ok = row["base_hit"]
        tta_ok = row["tta_hit"]
        status = "BASE OK" if base_ok else ("TTA OK" if tta_ok else "MISS")
        thumb = fit_thumb(Image.open(pred_dir / f'{Path(row["file"]).stem}_pred.jpg'), thumb_w, thumb_h)
        sheet.paste(thumb, (x, y))
        color = (80, 230, 80) if base_ok else ((70, 190, 255) if tta_ok else (255, 80, 80))
        draw.rectangle((x, y + thumb_h, x + thumb_w - 1, y + thumb_h + caption_h - 1), fill=color)
        draw.text((x + 8, y + thumb_h + 4), status, fill=(10, 10, 10), font=font(20))
        draw.text((x + 118, y + thumb_h + 6), Path(row["file"]).name[:38], fill=(10, 10, 10), font=font(15))
    sheet.save(dst, quality=92)


def make_failure_sheet(rows: list[dict], pred_dir: Path, dst: Path, title: str) -> None:
    failed = [r for r in rows if not (r["base_hit"] or r["tta_hit"])]
    if not failed:
        return
    cols, thumb_w, thumb_h, caption_h = 3, 640, 480, 50
    n_rows = (len(failed) + cols - 1) // cols
    header_h = 78
    sheet = Image.new("RGB", (cols * thumb_w, header_h + n_rows * (thumb_h + caption_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 12), title, fill=(255, 255, 255), font=font(36))
    for i, row in enumerate(failed):
        col, line = i % cols, i // cols
        x, y = col * thumb_w, header_h + line * (thumb_h + caption_h)
        thumb = fit_thumb(Image.open(pred_dir / f'{Path(row["file"]).stem}_pred.jpg'), thumb_w, thumb_h)
        sheet.paste(thumb, (x, y))
        draw.rectangle((x, y + thumb_h, x + thumb_w - 1, y + thumb_h + caption_h - 1), fill=(255, 90, 90))
        draw.text((x + 8, y + thumb_h + 5), Path(row["file"]).name[:48], fill=(10, 10, 10), font=font(17))
        draw.text((x + 8, y + thumb_h + 27), "BASE MISS / TTA MISS", fill=(10, 10, 10), font=font(17))
    sheet.save(dst, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--label", required=True, help="capture/report directory label, e.g. 6-Foam_Brick")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--conf", type=float, default=THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_dir = args.image_dir or (CAPTURE_ROOT / args.label)
    out_dir = REPORT_ROOT / f"{args.label}_predictions_640"
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg")) + sorted(image_dir.glob("*.png"))
    # Avoid accidentally picking up previously generated predictions if a caller passes report dir.
    images = [p for p in images if not p.stem.endswith("_pred")]
    if not images:
        raise SystemExit(f"no source images found in {image_dir}")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    source = [str(p) for p in images]
    common = dict(
        imgsz=args.imgsz, conf=args.conf, iou=0.7, device=args.device,
        verbose=False, max_det=100, save=False,
    )
    base_results = model.predict(source=source, augment=False, **common)
    tta_results = model.predict(source=source, augment=True, **common)

    rows: list[dict] = []
    for src, base_res, tta_res in zip(images, base_results, tta_results):
        base_dets = detections(base_res)
        tta_dets = detections(tta_res)
        row = {
            "file": src.name,
            "base_hit": target_hit(base_dets, args.class_id),
            "tta_hit": target_hit(tta_dets, args.class_id),
            "detections": base_dets,
            "tta_detections": tta_dets,
        }
        # Keep the historical convenience key used by earlier reports.
        row["hit"] = row["base_hit"]
        rows.append(row)
        status = "BASE OK" if row["base_hit"] else ("TTA ONLY" if row["tta_hit"] else "MISS")
        write_label_image(src, out_dir / f"{src.stem}_pred.jpg", base_dets, status)

    base_hits = sum(r["base_hit"] for r in rows)
    tta_hits = sum(r["tta_hit"] for r in rows)
    union_hits = sum(r["base_hit"] or r["tta_hit"] for r in rows)
    summary = {
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "class_id": args.class_id,
        "label": args.label,
        "base_hits": base_hits,
        "tta_hits": tta_hits,
        "union_hits": union_hits,
        "total": len(rows),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        f"# {args.label} 相机评估（v23 epoch3）",
        "",
        f"- 原图：{image_dir}",
        f"- 权重：{args.weights}",
        f"- 设置：imgsz={args.imgsz}, conf={args.conf}（与 9100 服务一致）",
        f"- 类别级检出（640 单帧）：{base_hits}/{len(rows)}",
        f"- imgsz=640 + TTA：{tta_hits}/{len(rows)}",
        f"- 单帧与 TTA 并集：{union_hits}/{len(rows)}",
        "",
    ]
    for r in rows:
        if r["base_hit"]:
            continue
        base_text = "; ".join(f'{d["name"]} {d["conf"]:.3f}' for d in r["detections"]) or "无检测"
        tta_text = "; ".join(f'{d["name"]} {d["conf"]:.3f}' for d in r["tta_detections"]) or "无检测"
        md.append(
            f'- 基准漏检：{r["file"]} -> 单帧[{base_text}]；TTA[{tta_text}]；TTA命中={r["tta_hit"]}'
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    title = f"{args.label} | v23 epoch3 | {base_hits}/{len(rows)} base, {tta_hits}/{len(rows)} TTA, {union_hits}/{len(rows)} union"
    contact = REPORT_ROOT / f"{args.label}_{len(rows)}张总览_服务分辨率640.jpg"
    failures = REPORT_ROOT / f"{args.label}_服务640失败{len(rows) - union_hits}张.jpg"
    make_contact_sheet(rows, out_dir, contact, title)
    make_failure_sheet(rows, out_dir, failures, f"{args.label} union failures | {len(rows) - union_hits}")
    print(json.dumps({
        "label": args.label,
        "total": len(rows),
        "base_hits": base_hits,
        "tta_hits": tta_hits,
        "union_hits": union_hits,
        "out_dir": str(out_dir),
        "contact": str(contact),
        "failures": str(failures) if (len(rows) - union_hits) else None,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
