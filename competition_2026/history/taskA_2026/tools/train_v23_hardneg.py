#!/usr/bin/env python3
"""v23 硬负样本修正：从 v22 best 低学习率续训，逐轮保存 checkpoint。

目的：学会把手机/白色杂物判为背景，同时尽量不伤害汤罐、芥末酱、香蕉等旧类。
训练后按 checkpoint 做专项回归，选择手机误报=0 且旧类损失最小的模型。
"""
from pathlib import Path
from ultralytics import YOLO

T = Path("/media/sunshine/新加卷/taskA")
BASE = T / "runs/yolo11s_real_ft_v22_20261016/weights/best.pt"
DATA = T / "datasets/real_ft_v22_20261016/data.yaml"
RUN = T / "runs/yolo11s_real_ft_v23_hardneg_20261016"
if RUN.exists():
    raise SystemExit(f"refuse to overwrite existing run: {RUN}")
if not BASE.is_file() or not DATA.is_file():
    raise SystemExit("missing base weights or data.yaml")

m = YOLO(str(BASE))
m.train(
    data=str(DATA),
    epochs=10,
    batch=8,
    imgsz=640,
    freeze=10,
    mosaic=0.0,
    close_mosaic=0,
    lr0=0.0002,
    lrf=0.1,
    device=0,
    workers=8,
    seed=23,
    deterministic=True,
    patience=10,
    project=str(T / "runs"),
    name=RUN.name,
    exist_ok=False,
    save_period=1,
)
