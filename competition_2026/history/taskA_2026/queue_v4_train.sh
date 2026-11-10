#!/usr/bin/env bash
# 排队器：等 dataset_v4_640 准备完成 → 自动开始全量训练（yolo11s@640）→ 训练完自动评测
#
# 为什么要排队：数据准备（导出 60000 个文件 + 45000 张转 640）要跑 1 个多小时，
# 训练只能等它结束；本脚本把这个等待自动化，人可以走开。
#
# 用法（沙箱外）：setsid nohup bash queue_v4_train.sh >> logs/queue_v4.log 2>&1 &
set -uo pipefail

ROOT="/media/sunshine/新加卷/taskA"
cd "$ROOT" || exit 1
PREP_LOG="$ROOT/logs/run_v4_prep.log"

echo "== $(date '+%F %T') 排队：等数据准备完成 =="
for _ in $(seq 1 300); do   # 最多等 5 小时
    if grep -q "数据准备完成" "$PREP_LOG" 2>/dev/null; then
        break
    fi
    # 只看出窗（最近 40 行），否则会把历史失败记录当成新错误
    if tail -40 "$PREP_LOG" 2>/dev/null | grep -q "Traceback"; then
        echo "== 数据准备阶段报错，排队退出 =="
        tail -20 "$PREP_LOG"
        exit 1
    fi
    sleep 60
done

if ! grep -q "数据准备完成" "$PREP_LOG" 2>/dev/null; then
    echo "== 等待超时（5 小时） =="
    exit 1
fi

echo "== $(date '+%F %T') 数据就绪，开始训练 =="
tail -3 "$PREP_LOG"

# batch 32 先试（显存不够时 run_v4.sh 会自动减半重试）
NAME=yolo11s_v4 SCENES=5000 MODEL=yolo11s.pt IMGSZ=640 BATCH=32 WORKERS=8 EPOCHS=100 \
    bash "$ROOT/run_v4.sh"

echo "== $(date '+%F %T') v4 全流程结束 =="
