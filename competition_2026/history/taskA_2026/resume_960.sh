#!/usr/bin/env bash
# 2026-09-22 断点续训 yolo11n_v2_960（dataset_v2, imgsz=960, batch=16）
#
# 背景：20:15 训练进程被 SIGTERM 终止，终止时机正好落在保存 last.pt 上，
#       导致 runs/yolo11n_v2_960/weights/last.pt 只有 380 KB、zip 结构不完整、无法加载。
#       best.pt（epoch 7，含 optimizer/ema，21 MB）完好，可当作断点用。
#
# 变更：只把数据管线 workers 8 → 16（解码/mosaic 是瓶颈，CPU 32 线程够用）；
#       batch / imgsz / epochs / LR 全部不变，优化过程与中断前一致，
#       因此续训不会改变学习率调度和 batch 语义。
#
# 用法（需在沙箱外跑，GPU 才可见）：bash resume_960.sh
set -euo pipefail

ROOT="/media/sunshine/新加卷/taskA"
cd "$ROOT"

NAME="yolo11n_v2_960"
W="$ROOT/runs/$NAME/weights"
PY="$ROOT/venv/bin/python"
STAMP="$(date +%Y%m%d-%H%M%S)"

# 1) 损坏的 last.pt 改名留证（0.4 MB，确认后可删）
if [ -f "$W/last.pt" ]; then
    mv -v "$W/last.pt" "$W/last.pt.corrupt-$STAMP"
fi

# 2) 用 best.pt 生成可续训断点 last.pt（train_args.workers=16）
"$PY" - <<'PYCODE'
import torch

w = "runs/yolo11n_v2_960/weights"
ck = torch.load(f"{w}/best.pt", map_location="cpu", weights_only=False)
a = ck["train_args"]
print(
    "断点来源 best.pt：epoch={} imgsz={} batch={} workers={} cache={} epochs={} lr0={}".format(
        ck["epoch"], a["imgsz"], a["batch"], a["workers"], a["cache"], a["epochs"], a["lr0"]
    )
)
a["workers"] = 16
a["cache"] = False
torch.save(ck, f"{w}/last.pt")
print(f"已写出可续训断点 {w}/last.pt（workers=16，其余参数不变）")
PYCODE

# 3) 启动训练流水线（setsid 脱离当前会话，父 shell 退出不影响）
export YOLO_CONFIG_DIR="$ROOT/ultralytics-config"
setsid nohup env NAME="$NAME" IMGSZ=960 BATCH=16 WORKERS=16 CACHE=false EPOCHS=100 \
    bash "$ROOT/run_gpu.sh" >>"$ROOT/logs/run_gpu_960.log" 2>&1 &
echo "训练已启动 pid=$!"

# 4) 资源采样（内存/swap/PSI/GPU → logs/monitor_YYYY-MM-DD.csv）
OUT="$ROOT/logs/monitor_$(date +%F).csv" INTERVAL=15 \
    setsid nohup bash "$ROOT/monitor_resources.sh" >>"$ROOT/logs/monitor_nohup.log" 2>&1 &
echo "资源采样已启动 pid=$!"
