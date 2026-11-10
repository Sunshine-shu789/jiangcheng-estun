#!/usr/bin/env bash
# 2026-09-22 第二轮续训 yolo11n_v2_960：数据管线换成 dataset_v3（1280×960 JPEG）
#
# 起因（2026-09-22 21:05 实测）：
#   workers=16 + batch=16 + imgsz=960 → dataloader 预取 16×4 个 batch，
#   每个 batch 是 176 MB 的 float32 张量（16×3×960×960×4B），
#   共享内存峰值 11.5 GB → swap 8.5 GB，内存可用只剩 100 MB，
#   epoch 反而从 4.4 min 涨到 5.3 min。所以把 workers 收回 8（预取 8×4=5.6 GB）。
#
#   同时把训练图像换成 dataset_v3（预缩到 1280×960、JPEG q92）：
#   NTFS 挂载盘读 2048×1536 PNG 只有 ~40 MB/s，mosaic 又让每张图每 epoch 被读约 4 次
#   （≈15.7 GB/epoch），换成小图后 ≈3.6 GB/epoch。letterbox 几何不变。
#   最终推理/评估仍用 dataset_v2 原图，保证和 640 基线可比。
#
# 不变项：imgsz / batch / epochs / LR / 优化器状态（只改数据来源与 workers，不改优化过程）
set -euo pipefail

ROOT="/media/sunshine/新加卷/taskA"
cd "$ROOT"

NAME="yolo11n_v2_960"
W="$ROOT/runs/$NAME/weights"
PY="$ROOT/venv/bin/python"

[ -f "$ROOT/dataset_v3/data.yaml" ] || { echo "dataset_v3 还不存在，先跑 build_dataset_v3.py"; exit 1; }

"$PY" - <<'PYCODE'
import torch

w = "runs/yolo11n_v2_960/weights"
ck = torch.load(f"{w}/last.pt", map_location="cpu", weights_only=False)
a = ck["train_args"]
print(
    "断点：epoch={} best_fitness={} imgsz={} batch={} workers={} cache={} data={}".format(
        ck["epoch"], ck.get("best_fitness"), a["imgsz"], a["batch"], a["workers"], a["cache"], a["data"]
    )
)
a["workers"] = 8
a["cache"] = False
a["data"] = "/media/sunshine/新加卷/taskA/dataset_v3/data.yaml"
torch.save(ck, f"{w}/last.pt")
print("已改写断点：workers=8, cache=False, data=dataset_v3/data.yaml")
PYCODE

export YOLO_CONFIG_DIR="$ROOT/ultralytics-config"
setsid nohup env NAME="$NAME" IMGSZ=960 BATCH=16 WORKERS=8 CACHE=false EPOCHS=100 \
    DATA="$ROOT/dataset_v2" bash "$ROOT/run_gpu.sh" >>"$ROOT/logs/run_gpu_960.log" 2>&1 &
echo "训练已启动 pid=$!"

OUT="$ROOT/logs/monitor_$(date +%F).csv" INTERVAL=15 \
    setsid nohup bash "$ROOT/monitor_resources.sh" >>"$ROOT/logs/monitor_nohup.log" 2>&1 &
echo "资源采样已启动 pid=$!"
