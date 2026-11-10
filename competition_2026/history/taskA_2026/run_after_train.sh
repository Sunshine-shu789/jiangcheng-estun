#!/usr/bin/env bash
# 训练结束后：在 val 上推理并出 IoU 报告。
# 用法：./run_after_train.sh [weights] [data_dir]
set -euo pipefail

ROOT="/media/sunshine/新加卷/taskA"
cd "$ROOT"
export YOLO_CONFIG_DIR="$ROOT/ultralytics-config"

WEIGHTS="${1:-runs/yolo11n_v1/weights/best.pt}"
DATA="${2:-$ROOT/dataset_v1}"
PY="$ROOT/venv/bin/python"

# runs/<name>/weights/best.pt -> <name>
NAME="$(basename "$(dirname "$(dirname "$WEIGHTS")")")"
TAG="${NAME}_val"

echo "== 推理 val（$NAME）=="
"$PY" predict_split.py --weights "$WEIGHTS" --data "$DATA" --split val --out "preds/$TAG"

echo "== 评估 IoU =="
"$PY" eval_detection.py \
    --data "$DATA" \
    --pred-labels "preds/$TAG/labels" \
    --split val \
    --out "reports/eval_$NAME"

echo "完成：$ROOT/reports/eval_$NAME.md"
