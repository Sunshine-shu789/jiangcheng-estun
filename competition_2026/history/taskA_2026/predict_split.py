#!/usr/bin/env python3
"""用训练好的权重推理某个 split，输出 YOLO 标签目录给 eval_detection.py 用。

例：
    python predict_split.py --weights runs/yolo11n_v1/weights/best.pt \
        --split val --out preds/yolo11n_v1_val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cepb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=str(cepb.WORKDIR / "dataset_v1"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="auto", help="auto=有 GPU 用 0，否则 cpu")
    args = parser.parse_args()

    from ultralytics import YOLO

    device = cepb.resolve_device(args.device)
    print(f"推理设备：{device}")
    # 必须是绝对路径：Ultralytics 会把相对 project 解析到 runs/detect/ 下
    out = Path(args.out).resolve()
    source = Path(args.data) / "images" / args.split
    model = YOLO(args.weights)
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        save_txt=True,
        save_conf=True,
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
        stream=True,
        verbose=False,
    )
    # stream=True 返回生成器，必须消费才会真正推理/写标签
    n_pred = sum(1 for _ in results)
    print(f"已推理 {n_pred} 张图片")

    labels = out / "labels"
    count = len(list(labels.glob("*.txt"))) if labels.exists() else 0
    print(f"预测标签：{labels}（{count} 个 txt）")
    print(
        "评估命令：python eval_detection.py "
        f"--pred-labels {labels} --split {args.split} "
        f"--out reports/eval_{out.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
