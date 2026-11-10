#!/bin/bash
# 评估专用采集服务（只采图 / 评估，绝不进训练集）
#
#   eval_capture.sh start    启动评估采集服务（旧 legacy 档，占用 9100 端口）
#   eval_capture.sh vote3    启动评估采集服务 + vote3 推荐融合档
#   eval_capture.sh stop     停止评估采集服务
#   eval_capture.sh restore  恢复日常预览（等价于 preview_switch.sh v23）
#
# 采出的原图 -> $EVAL_ROOT/<类别名>/<类别名>_<时间戳>.jpg
# 冻结标注 -> $SCENE_ROOT
# 本脚本不连接、不控制任何机械臂/机械爪。
set -euo pipefail

T=/media/sunshine/新加卷/taskA
RUNS=$T/runs
V23=$RUNS/yolo11s_real_ft_v23_hardneg_20261016/weights/epoch3.pt
CAM=/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_405_Intel_R__RealSense_TM__Depth_Camera_405_260423072083-video-index4
EVAL_ROOT=$T/eval_capture_20261016
SCENE_ROOT=$T/eval_scenes_20261016
CAL_ROOT=$T/reports/camera_eval_20261016/box_calibration_v3_20261016
STATE=/home/sunshine/.local/state/task-a-camera-recognition
LOG=$STATE/eval_capture.log
PY=/usr/bin/python3
APP=/home/sunshine/codex/easton/taskA/live_camera_preview.py
PORT=9100
VENV_SP=$T/venv/lib/python3.10/site-packages

export PYTHONPATH="$VENV_SP${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

stop_9100() {
  local old
  old=$(ss -ltnp 2>/dev/null | awk '/:9100 /{print $NF}' | grep -oP 'pid=\K[0-9]+' | head -1 || true)
  if [ -n "${old:-}" ]; then
    echo "停止旧服务 PID $old"
    kill "$old" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$old" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$old" 2>/dev/null || true
  fi
}

MODE=${1:-}
case "$MODE" in
  start|vote3)
    [ -e "$V23" ] || { echo "缺少权重：$V23"; exit 1; }
    [ -e "$CAM" ] || { echo "相机未连接：$CAM"; echo "请先把 RealSense D405 插到 USB3 口。"; exit 1; }
    stop_9100
    mkdir -p "$STATE" "$EVAL_ROOT" "$SCENE_ROOT"
    extra_args=()
    if [ "$MODE" = "vote3" ]; then
      [ -e "$CAL_ROOT/base.json" ] || { echo "缺少框校准：$CAL_ROOT/base.json"; exit 1; }
      [ -e "$CAL_ROOT/tta.json" ] || { echo "缺少框校准：$CAL_ROOT/tta.json"; exit 1; }
      detection_scale="1.0,0.75,0.65"
      extra_args=(--fusion vote3 --fusion-min-views 3 --fusion-tta --fusion-arbitration --fusion-calibration-dir "$CAL_ROOT")
    else
      detection_scale="0.65,0.75"
    fi
    cd /home/sunshine/codex/easton/taskA
    setsid nohup "$PY" "$APP" \
      --weights "$V23" \
      --primary-classes 0,1,2,3,4,5,6,7,8,9 \
      --fallback-weights "$V23" \
      --fallback-conf 0.25 --fallback-suppress-iou 0.0 \
      --camera "$CAM" \
      --conf 0.25 --device 0 --imgsz 640 --stable-confidence-alpha 0.1 \
      --rotate 180 --detection-scale "$detection_scale" "${extra_args[@]}" \
      --capture-root "$EVAL_ROOT" \
      --scene-root "$SCENE_ROOT" \
      --rgbd --host 127.0.0.1 --port "$PORT" \
      >> "$LOG" 2>&1 < /dev/null &
    NEW=$!
    echo "$NEW" > "$STATE/eval_capture.pid"
    echo "评估采集服务已启动（$MODE），PID $NEW"
    echo "  地址   ：http://127.0.0.1:$PORT/"
    echo "  原图   ：$EVAL_ROOT"
    echo "  日志   ：$LOG"
    for i in $(seq 1 60); do
      sleep 1
      if curl -s -m 2 "http://127.0.0.1:$PORT/status.json" > /tmp/_eval_status.json 2>/dev/null; then
        echo "服务就绪（${i}s）"
        exit 0
      fi
    done
    echo "启动超时，请查看日志：$LOG"; exit 1
    ;;
  stop)
    stop_9100
    echo "已停止 9100 上的服务"
    ;;
  restore)
    exec /home/sunshine/codex/easton/taskA/preview_switch.sh v23
    ;;
  *)
    echo "用法: $0 {start|vote3|stop|restore}"; exit 2;;
esac
