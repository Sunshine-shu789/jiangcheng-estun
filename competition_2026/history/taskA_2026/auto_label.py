#!/usr/bin/env python3
"""真机图预标注：用当前模型自动打框，产出 YOLO 标签 + 人工复核清单。

关键约定：
- 如果部署时要开 ROI 裁剪，这里必须用**同样的裁剪参数**（--crop），
  因为微调数据集必须等于"部署时网络实际看到的图"。裁剪后标签按裁剪图归一化。
- `review.csv` 按平均置信度从低到高排序：人工从最可疑的图开始改，效率最高。

用法：
    python auto_label.py --weights runs/yolo11n_v2/weights/best.pt \
        --images real_raw --out real_labeled --tta
    python auto_label.py ... --crop 320 200 640 640     # 开 ROI 时
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cepb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="auto", help="auto=有 GPU 用 0，否则 cpu")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
        help="ROI 裁剪框（像素）；部署开 ROI 时这里必须一致",
    )
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    device = cepb.resolve_device(args.device)
    print(f"预标注设备：{device}")
    src = Path(args.images).resolve()
    out = Path(args.out).resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in src.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not files:
        raise SystemExit(f"{src} 下没有图片")

    model = YOLO(args.weights)
    rows = []
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            print(f"跳过无法读取的文件：{path.name}")
            continue
        if args.crop:
            x, y, w, h = args.crop
            img = img[y : y + h, x : x + w]
        H, W = img.shape[:2]
        res = model.predict(
            img, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=device, augment=args.tta, verbose=False,
        )[0]
        boxes = res.boxes
        lines, confs = [], []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {p:.4f}")
                confs.append(float(p))
        (out / "labels" / f"{path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        cv2.imwrite(str(out / "images" / f"{path.stem}.png"), img)
        rows.append(
            {
                "image": f"{path.stem}.png",
                "boxes": len(lines),
                "mean_conf": round(sum(confs) / len(confs), 4) if confs else 0.0,
                "min_conf": round(min(confs), 4) if confs else 0.0,
            }
        )

    rows.sort(key=lambda r: (r["mean_conf"], r["boxes"]))
    with (out / "review.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "boxes", "mean_conf", "min_conf"])
        writer.writeheader()
        writer.writerows(rows)

    names = model.names
    # 直接沿用模型里的类别名（形如 "1-Cheez-it"），不要再拼索引，否则会和仿真 classes.txt 冲突
    (out / "classes.txt").write_text(
        "\n".join(str(names[i]) for i in sorted(names)) + "\n", encoding="utf-8"
    )
    print(f"预标注完成：{len(rows)} 张 -> {out}")
    print("人工复核：从 review.csv 的第一行开始（置信度最低的图最可疑）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
