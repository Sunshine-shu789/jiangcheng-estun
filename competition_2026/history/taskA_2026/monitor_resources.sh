#!/usr/bin/env bash
# 训练期间资源采样：内存 / swap / PSI / GPU，CSV 写到数据盘 logs/
# 用法：OUT=/media/sunshine/新加卷/taskA/logs/monitor_$(date +%F).csv INTERVAL=5 ./monitor_resources.sh &
set -uo pipefail

OUT="${OUT:-/media/sunshine/新加卷/taskA/logs/monitor_$(date +%F).csv}"
INTERVAL="${INTERVAL:-5}"

if [ ! -s "$OUT" ]; then
    echo "ts,mem_avail_mb,swap_total_mb,swap_used_mb,swap_pct,psi_mem10,psi_mem60,psi_io60,load1,gpu_util,gpu_mem_mb,top_proc,top_rss_mb" > "$OUT"
fi

while :; do
    ts=$(date '+%F %T')

    read -r ma st su sp < <(awk '
        /^MemAvailable:/ {ma=$2}
        /^SwapTotal:/    {st=$2}
        /^SwapFree:/     {sf=$2}
        END {
            tot = st/1024; used = (st-sf)/1024
            pct = (st > 0) ? 100*used/tot : 0
            printf "%d %d %d %d", ma/1024, tot, used, pct
        }' /proc/meminfo)

    read -r pm10 pm60 < <(awk '/^some/ {split($2,a,"="); split($3,b,"="); print a[2], b[2]}' /proc/pressure/memory)
    io60=$(awk '/^some/ {split($3,b,"="); print b[2]}' /proc/pressure/io)
    load1=$(cut -d' ' -f1 /proc/loadavg)

    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ -n "$gpu" ] || gpu="NA,NA"

    read -r tname trss < <(ps -eo rss=,comm= --sort=-rss | awk 'NR==1 {printf "%s %d", $2, $1/1024}')

    echo "$ts,$ma,$st,$su,$sp,$pm10,$pm60,$io60,$load1,$gpu,$tname,$trss" >> "$OUT"
    sleep "$INTERVAL"
done
