#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TASK_A_ROOT:-/media/sunshine/新加卷/taskA}"
PY="$ROOT/venv/bin/python"

DATA="${DATA:-$ROOT/dataset_real_ft_v9_grouped_20260927}"
BASE_WEIGHTS="${BASE_WEIGHTS:-$ROOT/runs/yolo11s_random_real_v8_20260927/weights/best.pt}"
RUN_NAME="${RUN_NAME:-yolo11s_real_ft_v9_joint_20260927}"
EPOCHS="${EPOCHS:-50}"
LR0="${LR0:-5e-5}"
FREEZE="${FREEZE:-10}"
PATIENCE="${PATIENCE:-20}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/evaluations/$RUN_NAME}"

for required in \
  "$DATA/data.yaml" \
  "$DATA/images/train" \
  "$DATA/images/val" \
  "$DATA/images/real_val" \
  "$DATA/images/sim_val" \
  "$BASE_WEIGHTS"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 2
  fi
done

RUN_DIR="$ROOT/runs/$RUN_NAME"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run: $RUN_DIR" >&2
  exit 2
fi

echo "Joint dataset: $DATA"
echo "Base weights: $BASE_WEIGHTS"
echo "New run: $RUN_DIR"
echo "Epochs: $EPOCHS, lr0: $LR0, freeze: $FREEZE, patience: $PATIENCE"

"$PY" "$SCRIPT_DIR/finetune_yolo.py" \
  --data "$DATA/data.yaml" \
  --model "$BASE_WEIGHTS" \
  --name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --imgsz 640 \
  --batch 8 \
  --optimizer AdamW \
  --lr0 "$LR0" \
  --lrf 0.1 \
  --mosaic 0.0 \
  --freeze "$FREEZE" \
  --workers 8 \
  --patience "$PATIENCE" \
  --device 0

NEW_WEIGHTS="$RUN_DIR/weights/best.pt"
if [[ ! -f "$NEW_WEIGHTS" ]]; then
  echo "Training did not produce best.pt: $NEW_WEIGHTS" >&2
  exit 3
fi

mkdir -p "$EVAL_ROOT/preds" "$EVAL_ROOT/reports"
for pair in "before:$BASE_WEIGHTS" "after:$NEW_WEIGHTS"; do
  tag="${pair%%:*}"
  weights="${pair#*:}"
  for split in real_val sim_val; do
    "$PY" "$SCRIPT_DIR/infer_imgsz.py" \
      --weights "$weights" \
      --data "$DATA" \
      --split "$split" \
      --imgsz 640 \
      --conf 0.25 \
      --iou 0.7 \
      --device 0 \
      --out "$EVAL_ROOT/preds/${tag}_${split}"
    "$PY" "$SCRIPT_DIR/eval_detection.py" \
      --data "$DATA" \
      --split "$split" \
      --pred-labels "$EVAL_ROOT/preds/${tag}_${split}/labels" \
      --iou-threshold 0.75 \
      --match-iou 0.5 \
      --overlays 12 \
      --out "$EVAL_ROOT/reports/${tag}_${split}"
  done
done

echo "Training weights: $NEW_WEIGHTS"
echo "Comparison reports: $EVAL_ROOT/reports"
