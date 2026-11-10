#!/usr/bin/env bash
# 重建 taskA 的 CUDA venv。
#
# 背景：2026-09-22 下午关机时 NTFS 卷没落盘，venv 里 cu130 那批目录
# （torch / nvidia / triton / cuda / 各 dist-info，共 25 项）读到 I/O 错误，
# 连删除和改名都失败。这里把整个 venv 挪到 venv_broken_cu130，重建一个新的，
# 并把 cu130 全套轮子备份到 wheelhouse/cu130（以后可离线重装）。
#
# 用法：bash rebuild_venv_gpu.sh
set -uo pipefail

ROOT="/media/sunshine/新加卷/taskA"
TORCH_VER="2.14.0+cu130"
TV_VER="0.29.0+cu130"
UL_VER="8.4.155"
IDX="https://download.pytorch.org/whl/cu130"

cd "$ROOT" || exit 1
say() { echo "===== $(date '+%F %T') $*"; }

if [ -d venv_broken_cu130 ]; then
    say "venv_broken_cu130 已存在，跳过挪动"
else
    say "把损坏的 venv 挪到 venv_broken_cu130"
    mv venv venv_broken_cu130 || { say "挪动失败，退出"; exit 1; }
fi

# 本机没装 python3.10-venv（ensurepip 缺失），所以手工搭 venv 骨架 + 从本地
# pip 轮子引导。venv 只认得 pyvenv.cfg 和 bin/python，不依赖 ensurepip。
say "手工创建 venv 骨架"
if [ -d venv ] && [ -n "$(ls -A venv/lib/python3.10/site-packages 2>/dev/null)" ]; then
    mv venv "venv_partial_$(date '+%H%M%S')"
fi
rm -rf venv
mkdir -p venv/bin venv/lib/python3.10/site-packages venv/include
ln -sf /usr/bin/python3.10 venv/bin/python3.10
ln -sf python3.10 venv/bin/python3
ln -sf python3.10 venv/bin/python
printf 'home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.10.12\n' > venv/pyvenv.cfg
PY="$ROOT/venv/bin/python"
"$PY" -V || { say "venv python 不可用"; exit 1; }

say "从本地 pip 轮子引导 pip"
"$PY" -m zipfile -e "$ROOT"/wheelhouse/pip-*.whl venv/lib/python3.10/site-packages/ || exit 1
"$PY" -m pip --version || exit 1
"$PY" -m pip install --no-index --find-links="$ROOT/wheelhouse" pip || say "pip 正式安装失败，继续"

say "下载 torch $TORCH_VER + torchvision $TV_VER 全套轮子（含 nvidia 运行时）"
mkdir -p "$ROOT/wheelhouse/cu130"
"$PY" -m pip download --no-cache-dir \
    --index-url "$IDX" --extra-index-url https://pypi.org/simple \
    "torch==$TORCH_VER" "torchvision==$TV_VER" \
    -d "$ROOT/wheelhouse/cu130" || { say "轮子下载失败"; exit 1; }

say "从 wheelhouse/cu130 安装 torch / torchvision"
"$PY" -m pip install --no-index --find-links="$ROOT/wheelhouse/cu130" \
    "torch==$TORCH_VER" "torchvision==$TV_VER" || { say "torch 安装失败"; exit 1; }

say "安装 ultralytics $UL_VER 及其余依赖（PyPI）"
"$PY" -m pip install --no-cache-dir "ultralytics==$UL_VER" || { say "ultralytics 安装失败"; exit 1; }

say "校验"
"$PY" - <<'PYEOF'
import torch, torchvision, ultralytics
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| ultralytics", ultralytics.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    print("matmul ok", float((x @ x).sum()) != 0.0)
PYEOF

sync
say "完成"
