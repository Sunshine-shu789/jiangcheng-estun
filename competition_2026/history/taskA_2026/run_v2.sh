#!/usr/bin/env bash
# v2 流水线：导出三视角数据集（left/middle/right × 3 光照 = 4500 张）
#     → 训练 yolo11n @640 → 自动推理 + IoU 评估
# 用法：nohup setsid ./run_v2.sh >> train_yolo11n_v2.log 2>&1 &
set -uo pipefail

ROOT="/media/sunshine/新加卷/taskA"
cd "$ROOT"
export YOLO_CONFIG_DIR="$ROOT/ultralytics-config"
PY="$ROOT/venv/bin/python"

NAME="yolo11n_v2"
DATA="$ROOT/dataset_v2"

echo "===== $(date '+%F %T') v2 流水线开始 ====="

# 1) 导出多视角数据集（已有则跳过）
if [ -f "$DATA/data.yaml" ]; then
    echo "== dataset_v2 已存在，跳过导出 =="
else
    echo "== 导出 dataset_v2（500 场景 × 3 相机 × 3 光照）=="
    "$PY" export_dataset.py --scenes 500 --camera all --out "$DATA" || exit 1
fi

# 2) 训练（最多重试 5 次，有断点自动续训）
n=0
while true; do
    if [ -f "runs/$NAME/weights/last.pt" ]; then
        echo "== 续训 runs/$NAME/weights/last.pt =="
        "$PY" train_yolo.py --resume --model "runs/$NAME/weights/last.pt" && break
    else
        echo "== 从头训练 $NAME =="
        "$PY" train_yolo.py \
            --data "$DATA/data.yaml" \
            --model yolo11n.pt \
            --epochs 100 --imgsz 640 --batch 16 --workers 8 \
            --device 0 \
            --name "$NAME" && break
    fi
    n=$((n + 1))
    if [ "$n" -ge 5 ]; then
        echo "训练连续失败 $n 次，放弃"
        exit 1
    fi
    echo "训练中断，60 秒后续训（第 $n 次）"
    sleep 60
done

# 3) 推理 + 评估
./run_after_train.sh "runs/$NAME/weights/best.pt" "$DATA"

echo "===== $(date '+%F %T') v2 流水线完成 ====="
