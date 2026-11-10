#!/usr/bin/env bash
# GPU 训练流水线（2026-09-22 起）：dataset_v2 + yolo11n，指定 imgsz，
# 训练完自动按同一 imgsz 推理 + 出 IoU 报告（基线 + TTA 两份）。
#
# 用法（在数据盘执行，命令要在沙箱外跑）：
#   NAME=yolo11n_v2_960 IMGSZ=960 BATCH=16 bash run_gpu.sh
#   nohup setsid bash run_gpu.sh >> logs/run_gpu_960.log 2>&1 &
set -uo pipefail

ROOT="/media/sunshine/新加卷/taskA"
cd "$ROOT" || exit 1
export YOLO_CONFIG_DIR="$ROOT/ultralytics-config"
PY="$ROOT/venv/bin/python"

NAME="${NAME:-yolo11n_v2_960}"
DATA="${DATA:-$ROOT/dataset_v2}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"
CACHE="${CACHE:-false}"
CONF="${CONF:-0.25}"

echo "===== $(date '+%F %T') GPU 流水线开始：$NAME imgsz=$IMGSZ batch=$BATCH workers=$WORKERS cache=$CACHE ====="

# 1) 训练（最多重试 5 次，有断点自动续训）
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
            --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" \
            --cache "$CACHE" --device 0 \
            --name "$NAME" && break
    fi
    n=$((n + 1))
    if [ "$n" -ge 5 ]; then
        echo "训练连续失败 $n 次，放弃"
        exit 1
    fi
    # 没有断点（多半是显存不足/启动即崩）时，下一轮把 batch 减半再试
    if [ ! -f "runs/$NAME/weights/last.pt" ] && [ "$BATCH" -gt 4 ]; then
        BATCH=$((BATCH / 2))
        echo "未产生断点，batch 降到 $BATCH 重试"
    fi
    echo "训练中断，60 秒后续训（第 $n 次）"
    sleep 60
done

# 2) 基线推理 + 评估（imgsz 与训练一致，避免尺度错配）
echo "== 推理 val @ $IMGSZ（基线）=="
"$PY" infer_imgsz.py --weights "runs/$NAME/weights/best.pt" --data "$DATA" --split val \
    --imgsz "$IMGSZ" --conf "$CONF" --device 0 --out "preds/${NAME}_val${IMGSZ}"
echo "== 评估（基线）=="
"$PY" eval_detection.py --data "$DATA" --pred-labels "preds/${NAME}_val${IMGSZ}/labels" \
    --split val --out "reports/eval_$NAME"

# 3) TTA 推理 + 评估
echo "== 推理 val @ $IMGSZ（TTA）=="
"$PY" infer_imgsz.py --weights "runs/$NAME/weights/best.pt" --data "$DATA" --split val \
    --imgsz "$IMGSZ" --conf "$CONF" --tta --device 0 --out "preds/${NAME}_val${IMGSZ}_tta"
echo "== 评估（TTA）=="
"$PY" eval_detection.py --data "$DATA" --pred-labels "preds/${NAME}_val${IMGSZ}_tta/labels" \
    --split val --out "reports/eval_${NAME}_tta"

echo "===== $(date '+%F %T') GPU 流水线完成：$NAME ====="
echo "报告：$ROOT/reports/eval_$NAME.md、$ROOT/reports/eval_${NAME}_tta.md"
