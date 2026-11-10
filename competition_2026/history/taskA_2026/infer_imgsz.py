#!/usr/bin/env python3
"""按指定 imgsz 推理某个 split，只写 YOLO 标签（不存带框图片），供 eval_detection.py 使用。

与 predict_split.py 的区别：显式 save=False，产物只有 labels/，方便放到 /tmp 做实验。

例：
    python infer_imgsz.py --weights runs/yolo11n_v2/weights/best.pt \
        --data dataset_v2 --split val --imgsz 960 --out /tmp/preds_v2_960
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cepb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="数据集目录（含 images/<split>）")
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="auto", help="auto=有 GPU 用 0，否则 cpu")
    parser.add_argument("--tta", action="store_true", help="测试时增强（多尺度+翻转融合）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    device = cepb.resolve_device(args.device)
    print(f"推理设备：{device}")
    out = Path(args.out).resolve()
    source = Path(args.data).resolve() / "images" / args.split
    model = YOLO(args.weights)
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        augment=args.tta,
        save=False,          # 不写带框图片，只要标签
        save_txt=True,
        save_conf=True,
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
        stream=True,
        verbose=False,
    )
    n = sum(1 for _ in results)
    labels = out / "labels"
    count = len(list(labels.glob("*.txt"))) if labels.exists() else 0
    print(f"imgsz={args.imgsz} 推理 {n} 张，写出 {count} 个标签文件 -> {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
