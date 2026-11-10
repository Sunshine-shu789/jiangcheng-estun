#!/usr/bin/env bash
# A 方案流水线（2026-09-22 起）：1500 场景 + yolo11s @640
#
# 与 run_gpu.sh 的区别：
#   1) 训练数据 = dataset_v4_640（1500 场景抽样的轻量版，train 已剔除评测场景，防泄漏）
#   2) 评测数据 = dataset_v2（900 张旧 val），和 39.4% / 41.6% 基线同一把尺子
#   3) 模型换 yolo11s，imgsz=640（尺度敏感度实验证明 640 是最稳的工作点）
#
# 用法（沙箱外执行）：
#   nohup setsid bash run_v4.sh >> logs/run_v4.log 2>&1 &
set -uo pipefail

ROOT="/media/sunshine/新加卷/taskA"
cd "$ROOT" || exit 1
export YOLO_CONFIG_DIR="$ROOT/ultralytics-config"
PY="$ROOT/venv/bin/python"

NAME="${NAME:-yolo11s_v4}"
SCENES="${SCENES:-1500}"
MODEL="${MODEL:-yolo11s.pt}"
FULL="$ROOT/dataset_v4"          # 原图（2048×1536 PNG），留给以后换分辨率/真机微调
LIGHT="$ROOT/dataset_v4_640"     # 训练用轻量集（长边 640 无损 PNG）
EVAL="$ROOT/dataset_v2"          # 评测用旧 val（900 张 / 9000 实例）
LIGHT="${LIGHT_OVERRIDE:-$LIGHT}"  # 可用 LIGHT_OVERRIDE 指到本地盘（NVMe）副本，绕开 NTFS 读瓶颈
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"
CONF="${CONF:-0.25}"
# PATIENCE=0 → 关掉早停，跑满 EPOCHS（ultralytics 内部把 0 当 inf）
PATIENCE="${PATIENCE:-30}"

echo "===== $(date '+%F %T') v4 流水线开始：$NAME model=$MODEL scenes=$SCENES imgsz=$IMGSZ batch=$BATCH workers=$WORKERS epochs=$EPOCHS patience=$PATIENCE ====="

# 1) 导出原图数据集（1500 场景）
if [ -f "$FULL/data.yaml" ]; then
    echo "== dataset_v4 已存在，跳过导出 =="
else
    echo "== 导出 dataset_v4：$SCENES 场景 =="
    "$PY" export_dataset.py --scenes "$SCENES" --camera all --out "$FULL" || exit 1
fi

# 2) 生成训练用轻量集（长边 640，train 剔除评测场景，val = dataset_v2 的 val）
if [ -f "$LIGHT/data.yaml" ]; then
    echo "== dataset_v4_640 已存在，跳过 =="
else
    echo "== 生成 dataset_v4_640（轻量 + 防泄漏）=="
    "$PY" build_light_dataset.py --src "$FULL" --val-src "$EVAL" --dst "$LIGHT" \
        --long "$IMGSZ" --jobs 16 || exit 1
fi

# 2.5) 只准备数据（PREP_ONLY=1 时到此为止，训练另外起）
if [ "${PREP_ONLY:-0}" = "1" ]; then
    echo "== PREP_ONLY=1：数据准备完成，退出（$(date '+%F %T')）=="
    exit 0
fi

# 3) 训练（最多重试 5 次，有断点自动续训）
n=0
while true; do
    if [ -f "runs/$NAME/weights/last.pt" ]; then
        echo "== 续训 runs/$NAME/weights/last.pt =="
        "$PY" train_yolo.py --resume --model "runs/$NAME/weights/last.pt" && break
    else
        echo "== 从头训练 $NAME =="
        "$PY" train_yolo.py \
            --data "$LIGHT/data.yaml" \
            --model "$MODEL" \
            --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" \
            --patience "$PATIENCE" \
            --cache false --device 0 \
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

# 4) 评测：dataset_v2 原图 val @640（与 39.4% / 41.6% 基线可比）
echo "== 推理 val @ $IMGSZ（基线）=="
"$PY" infer_imgsz.py --weights "runs/$NAME/weights/best.pt" --data "$EVAL" --split val \
    --imgsz "$IMGSZ" --conf "$CONF" --device 0 --out "preds/${NAME}_val${IMGSZ}"
echo "== 评估（基线）=="
"$PY" eval_detection.py --data "$EVAL" --pred-labels "preds/${NAME}_val${IMGSZ}/labels" \
    --split val --out "reports/eval_$NAME"

echo "== 推理 val @ $IMGSZ（TTA）=="
"$PY" infer_imgsz.py --weights "runs/$NAME/weights/best.pt" --data "$EVAL" --split val \
    --imgsz "$IMGSZ" --conf "$CONF" --tta --device 0 --out "preds/${NAME}_val${IMGSZ}_tta"
echo "== 评估（TTA）=="
"$PY" eval_detection.py --data "$EVAL" --pred-labels "preds/${NAME}_val${IMGSZ}_tta/labels" \
    --split val --out "reports/eval_${NAME}_tta"

echo "===== $(date '+%F %T') v4 流水线完成：$NAME ====="
echo "报告：$ROOT/reports/eval_$NAME.md、$ROOT/reports/eval_${NAME}_tta.md"
