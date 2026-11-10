#!/usr/bin/env python3
"""把源模型某一类别的分类行移植到基础模型，其他权重保持不变。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def detect_head(ckpt: dict):
    obj = ckpt.get("ema") or ckpt.get("model")
    if obj is None:
        raise ValueError("checkpoint 中无 model/ema")
    # DetectionModel -> .model Sequential -> [-1] Detect
    if hasattr(obj, "model") and not hasattr(obj, "cv3"):
        obj = obj.model[-1]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="保留其他权重的模型")
    ap.add_argument("--source", required=True, help="提供类别分类行的模型")
    ap.add_argument("--class-index", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    base_head = detect_head(base)
    source_head = detect_head(source)
    branches = len(base_head.cv3)
    if branches != len(source_head.cv3):
        raise ValueError(f"分支数不一致：{branches} vs {len(source_head.cv3)}")

    copied = []
    for i in range(branches):
        bconv = base_head.cv3[i][-1]
        sconv = source_head.cv3[i][-1]
        if bconv.weight.shape != sconv.weight.shape or bconv.bias.shape != sconv.bias.shape:
            raise ValueError(f"分支 {i} 形状不一致：{bconv.weight.shape} vs {sconv.weight.shape}")
        with torch.no_grad():
            bconv.weight[args.class_index].copy_(sconv.weight[args.class_index])
            bconv.bias[args.class_index].copy_(sconv.bias[args.class_index])
        copied.append(i)

    # 若 checkpoint 还保留了非 EMA 的 model，也同步修改，避免加载端选错。
    if base.get("model") is not None:
        try:
            base_model_head = detect_head({"model": base["model"]})
            source_model_head = detect_head({"model": source.get("model") or source.get("ema")})
            for i in range(branches):
                with torch.no_grad():
                    base_model_head.cv3[i][-1].weight[args.class_index].copy_(
                        source_model_head.cv3[i][-1].weight[args.class_index]
                    )
                    base_model_head.cv3[i][-1].bias[args.class_index].copy_(
                        source_model_head.cv3[i][-1].bias[args.class_index]
                    )
        except Exception as exc:  # noqa: BLE001
            print(f"提示：同步 model 权重失败（通常 model 为 None，可忽略）：{exc}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    print(f"已移植 class {args.class_index} 分类行，分支 {copied} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
