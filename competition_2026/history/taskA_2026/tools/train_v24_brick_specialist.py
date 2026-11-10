#!/usr/bin/env python3
"""训练只负责 Foam_Brick(class5) 的专家模型。

旧 9 类权重不参与训练，也不会被覆盖。专家模型仅用于实时管线中的 class5，
从而彻底隔离补泡沫砖对其它物品识别的影响。
"""
from pathlib import Path
from ultralytics import YOLO

T = Path("/media/sunshine/新加卷/taskA")
BASE = T / "runs/yolo11s_real_ft_v23_hardneg_20261016/weights/epoch3.pt"
DATA = T / "datasets/real_ft_v24_brick_specialist_20261016/data.yaml"
RUN = T / "runs/yolo11s_real_ft_v24_brick_specialist_20261016"

if RUN.exists():
    raise SystemExit(f"refuse to overwrite existing run: {RUN}")
if not BASE.is_file() or not DATA.is_file():
    raise SystemExit("missing base weights or data.yaml")

model = YOLO(str(BASE))
model.train(
    data=str(DATA),
    epochs=12,
    batch=8,
    imgsz=640,
    freeze=10,
    mosaic=0.0,
    close_mosaic=0,
    lr0=0.00012,
    lrf=0.05,
    cos_lr=True,
    degrees=4.0,
    translate=0.10,
    scale=0.35,
    fliplr=0.5,
    device=0,
    workers=8,
    seed=24,
    deterministic=True,
    patience=12,
    project=str(T / "runs"),
    name=RUN.name,
    exist_ok=False,
    save_period=1,
    plots=True,
)
