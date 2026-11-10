#!/bin/bash
set -eu
T=/media/sunshine/新加卷/taskA
LOG=$T/logs/train_v21_formal_20261016.log
nohup setsid "$T/venv/bin/python" /home/sunshine/codex/easton/taskA/tools/train_v21.py > "$LOG" 2>&1 < /dev/null &
echo $! > "$LOG.pid"
echo "launched pid=$(cat $LOG.pid) log=$LOG"
