#!/usr/bin/env bash
# 真机数据流水线：采图 → 预标注 → 人工复核 → 混合数据集 → 微调 → 前后对比评估
#
# 用法：
#   ./finetune_real.sh label                     # 预标注（模型自动打框 + 生成复核清单）
#   ./finetune_real.sh train                     # 用修正后的标签微调并出对比报告
#   ./finetune_real.sh train --epochs 40         # 透传额外参数
#
# 前置约定：raw 目录里是真机图（real_capture.py 或相机自带工具抓的都行）
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TASK_A_ROOT:-/media/sunshine/新加卷/taskA}"
PY="$ROOT/venv/bin/python"
export YOLO_CONFIG_DIR="$ROOT/ultralytics-config"

RAW="${REAL_RAW:-$ROOT/real_raw}"
LABELED="${REAL_LABELED:-$ROOT/real_labeled}"
FT_DATA="${REAL_FT_DATA:-$ROOT/dataset_real_ft}"
BASE_WEIGHTS="${BASE_WEIGHTS:-$ROOT/runs/yolo11s_v4/weights/best.pt}"
SIM_DATA="${SIM_DATA:-$ROOT/dataset_v4_640}"
RATIO="${RATIO:-3.0}"
TTA="${TTA:---tta}"
CONF="${CONF:---conf 0.35}"   # 预标注用较高阈值：少留误检框，人工更好改
CROP="${CROP:-}"          # 例如 CROP="320 200 640 640"
RUN_NAME="${RUN_NAME:-yolo11s_real_ft_v1}"

PHASE="${1:-}"; shift || true

case "$PHASE" in
  label)
    echo "== 1/2 预标注（$RAW）=="
    # shellcheck disable=SC2086
    "$PY" "$SCRIPT_DIR/auto_label.py" \
        --weights "$BASE_WEIGHTS" --images "$RAW" --out "$LABELED" \
        $TTA $CONF ${CROP:+--crop $CROP} "$@"
    echo
    echo "下一步（人工）："
    echo "  1) 打开 $LABELED/review.csv，从第一行开始（置信度最低的最可疑）"
    echo "  2) 用任意 YOLO 标注工具修正 $LABELED/images 里的框，覆盖 $LABELED/labels 里的 txt"
    echo "  3) 改完运行：./finetune_real.sh train"
    ;;
  label-known)
    if [ -z "${TARGET_CLASS_ID:-}" ]; then
        echo "缺少 TARGET_CLASS_ID。示例：TARGET_CLASS_ID=3 SOURCE_CLASS=bottle $0 label-known"
        exit 2
    fi
    echo "== 已知单类别自动标注（$RAW）=="
    "$PY" "$SCRIPT_DIR/label_known_real.py" \
        --images "$RAW" --out "$LABELED" \
        --model "$ROOT/yolo11s.pt" \
        --target-class-id "$TARGET_CLASS_ID" \
        --source-class "${SOURCE_CLASS:-bottle}" \
        --device "${DEVICE:-0}" "$@"
    echo "复核总览：$LABELED/review_sheet.jpg"
    echo "确认后运行：./finetune_real.sh train"
    ;;
  train)
    echo "== 2/2 构建混合数据集 =="
    "$PY" "$SCRIPT_DIR/build_finetune_dataset.py" \
        --real-images "$LABELED/images" --real-labels "$LABELED/labels" \
        --sim-data "$SIM_DATA" --ratio "$RATIO" --out "$FT_DATA"

    echo "== 微调 =="
    "$PY" "$SCRIPT_DIR/finetune_yolo.py" --data "$FT_DATA/data.yaml" \
        --model "$BASE_WEIGHTS" --name "$RUN_NAME" "$@"
    NEW="$ROOT/runs/$RUN_NAME/weights/best.pt"

    echo "== 对比评估（同一验证集）=="
    for pair in "before:$BASE_WEIGHTS" "after:$NEW"; do
        tag="${pair%%:*}"; w="${pair#*:}"
        "$PY" "$SCRIPT_DIR/infer_imgsz.py" --weights "$w" --data "$FT_DATA" \
            --split val --imgsz 640 $TTA --out "$ROOT/preds/realft_$tag"
        "$PY" "$SCRIPT_DIR/eval_detection.py" --data "$FT_DATA" \
            --pred-labels "$ROOT/preds/realft_$tag/labels" --split val \
            --out "$ROOT/reports/eval_realft_$tag"
        echo "  $tag -> $ROOT/reports/eval_realft_$tag.md"
    done
    echo "完成。对比 $ROOT/reports/eval_realft_before.md 与 eval_realft_after.md"
    ;;
  *)
    echo "用法：$0 label | label-known | train"
    exit 2
    ;;
esac
