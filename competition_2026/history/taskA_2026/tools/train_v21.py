#!/usr/bin/env python3
"""v21 正式训练：v20 best.pt → v21 数据集（新砖 31 张 + 空背景负样本，val 首次含砖）。"""
from ultralytics import YOLO

T = "/media/sunshine/新加卷/taskA"
m = YOLO(f"{T}/runs/yolo11s_real_ft_v20_20261015/weights/best.pt")
m.train(
    data=f"{T}/datasets/real_ft_v21_20261016/data.yaml",
    epochs=80, batch=8, imgsz=640, freeze=10,
    mosaic=0.0, close_mosaic=0,
    lr0=0.001, lrf=0.1,
    device=0, workers=8, seed=0, deterministic=True, patience=50,
    project=f"{T}/runs", name="yolo11s_real_ft_v21_20261016", exist_ok=True,
)
