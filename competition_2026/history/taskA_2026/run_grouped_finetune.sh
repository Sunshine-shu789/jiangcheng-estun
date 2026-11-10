#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TASK_A_ROOT:-/media/sunshine/新加卷/taskA}"
PY="$ROOT/venv/bin/python"

BASE_WEIGHTS="${BASE_WEIGHTS:-$ROOT/runs/yolo11s_real_ft_v3_staged_20260926/weights/best.pt}"
REAL_DATA="${REAL_DATA:-$ROOT/real_labeled_staged_20260926}"
SIM_DATA="${SIM_DATA:-$ROOT/dataset_v4_640}"
FT_DATA="${FT_DATA:-$ROOT/dataset_real_ft_v4_grouped_20260926}"
RUN_NAME="${RUN_NAME:-yolo11s_real_ft_v4_grouped_20260926}"
EPOCHS="${EPOCHS:-50}"

if [[ -e "$FT_DATA" ]]; then
  echo "Dataset already exists: $FT_DATA" >&2
  exit 2
fi

echo "== Build grouped real/simulation dataset =="
"$PY" "$SCRIPT_DIR/build_grouped_finetune_dataset.py" \
  --real-data "$REAL_DATA" \
  --sim-data "$SIM_DATA" \
  --out "$FT_DATA" \
  --ratio 3.0 \
  --group-gap 2.0 \
  --group-val-count 1 \
  --holdout-source banana_closeup \
  --tail-val-fraction 0.33 \
  --sim-val 60

echo "== Fine-tune from v3 staged best.pt =="
"$PY" "$SCRIPT_DIR/finetune_yolo.py" \
  --data "$FT_DATA/data.yaml" \
  --model "$BASE_WEIGHTS" \
  --name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --imgsz 640 \
  --batch 8 \
  --optimizer AdamW \
  --lr0 5e-5 \
  --lrf 0.1 \
  --mosaic 0.0 \
  --freeze 10 \
  --workers 8 \
  --patience 50 \
  --device 0

echo "Fine-tuned weights: $ROOT/runs/$RUN_NAME/weights/best.pt"

NEW_WEIGHTS="$ROOT/runs/$RUN_NAME/weights/best.pt"
echo "== Compare baseline and fine-tuned model on grouped real validation =="
for pair in "before:$BASE_WEIGHTS" "after:$NEW_WEIGHTS"; do
  tag="${pair%%:*}"
  weights="${pair#*:}"
  for split in real_val sim_val; do
    "$PY" "$SCRIPT_DIR/infer_imgsz.py" \
      --weights "$weights" \
      --data "$FT_DATA" \
      --split "$split" \
      --imgsz 640 \
      --conf 0.25 \
      --device 0 \
      --out "$ROOT/preds/grouped_${tag}_${split}"
    "$PY" "$SCRIPT_DIR/eval_detection.py" \
      --data "$FT_DATA" \
      --pred-labels "$ROOT/preds/grouped_${tag}_${split}/labels" \
      --split "$split" \
      --out "$ROOT/reports/grouped_${tag}_${split}"
  done
done

echo "Baseline real validation: $ROOT/reports/grouped_before_real_val.md"
echo "Fine-tuned real validation: $ROOT/reports/grouped_after_real_val.md"
