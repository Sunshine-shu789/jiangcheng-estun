#!/usr/bin/env python3
"""v22 正式训练：v21 best.pt -> 新汤罐域 + 杂物负样本数据集。"""
from pathlib import Path
from ultralytics import YOLO

T = Path("/media/sunshine/新加卷/taskA")
BASE = T / "runs/yolo11s_real_ft_v21_20261016/weights/best.pt"
DATA = T / "datasets/real_ft_v22_20261016/data.yaml"
RUN = T / "runs/yolo11s_real_ft_v22_20261016"
if RUN.exists():
    raise SystemExit(f"refuse to overwrite existing run: {RUN}")
if not BASE.is_file() or not DATA.is_file():
    raise SystemExit("missing base weights or data.yaml")

m = YOLO(str(BASE))
m.train(
    data=str(DATA),
    epochs=80,
    batch=8,
    imgsz=640,
    freeze=10,
    mosaic=0.0,
    close_mosaic=0,
    lr0=0.001,
    lrf=0.1,
    device=0,
    workers=8,
    seed=22,
    deterministic=True,
    patience=50,
    project=str(T / "runs"),
    name=RUN.name,
    exist_ok=False,
)
