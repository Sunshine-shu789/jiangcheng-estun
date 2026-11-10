#!/usr/bin/env python3
"""在 dataset_v1 上训练 YOLO 检测器（CPU 版）。

例：
    python train_yolo.py --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16
"""
from __future__ import annotations

import argparse

import cepb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(cepb.WORKDIR / "dataset_v1" / "data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cache",
        choices=("false", "ram", "disk"),
        default="false",
        help="图像缓存：ram=解一次常驻内存（约 4-5 GB/3600 张@640），disk=存 .npy 到数据集目录",
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--device",
        default="0",
        help="训练默认走 GPU（0）；auto=有 GPU 用 0 否则 cpu；cpu=显式回退 CPU",
    )
    parser.add_argument("--name", default="yolo11n_v1")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 --model 指向的 last.pt 断点续训（自动恢复 epoch/优化器状态）",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    device = cepb.resolve_device(args.device)
    if device == "cpu":
        print("⚠️  训练设备：cpu（GPU 不可用或显式指定）——本机训练约定是走 GPU，检查是否漏了沙箱外执行")
    else:
        print(f"训练设备：cuda:{device}")
    model = YOLO(args.model)
    if args.resume:
        # 从 last.pt 恢复：epoch、优化器、学习率都按保存时继续
        model.train(resume=True)
        print("续训完成")
        return 0
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=str(cepb.WORKDIR / "runs"),
        name=args.name,
        exist_ok=True,
        cache=False if args.cache == "false" else args.cache,
        seed=0,
        plots=True,
        val=True,
    )
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=device)
    print(f"mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
