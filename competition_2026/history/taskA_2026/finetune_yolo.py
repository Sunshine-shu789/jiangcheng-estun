#!/usr/bin/env python3
"""真机微调：从仿真预训练权重出发，用「真机 + 仿真」混合数据集做小学习率微调。

与 train_yolo.py 的区别：默认小学习率、关闭 mosaic（小样本微调更稳）、可冻结主干。

例：
    python finetune_yolo.py --data dataset_real_ft/data.yaml \
        --model runs/yolo11n_v2/weights/best.pt --epochs 30 --lr0 1e-4 --name yolo11n_real_ft
"""
from __future__ import annotations

import argparse

import cepb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default=str(cepb.WORKDIR / "runs/yolo11n_v2/weights/best.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=1e-4, help="微调学习率，别用 1e-2 那种从头训练的值")
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument(
        "--optimizer",
        default="AdamW",
        help="显式优化器；auto 会忽略 lr0，不适合小样本微调",
    )
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--freeze", type=int, default=0, help="冻结前 N 层（数据极少时可设 10）")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--device",
        default="0",
        help="微调默认走 GPU（0）；auto=有 GPU 用 0 否则 cpu；cpu=显式回退 CPU",
    )
    parser.add_argument("--name", default="yolo11n_real_ft")
    args = parser.parse_args()

    from ultralytics import YOLO

    device = cepb.resolve_device(args.device)
    if device == "cpu":
        print("⚠️  微调设备：cpu（GPU 不可用或显式指定）——本机训练约定是走 GPU")
    else:
        print(f"微调设备：cuda:{device}")
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        mosaic=args.mosaic,
        close_mosaic=0,
        freeze=args.freeze,
        patience=args.patience,
        project=str(cepb.WORKDIR / "runs"),
        name=args.name,
        exist_ok=True,
        seed=0,
        plots=True,
        val=True,
    )
    print(f"微调完成：{cepb.WORKDIR / 'runs' / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
