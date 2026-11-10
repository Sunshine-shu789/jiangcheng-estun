#!/usr/bin/env python3
"""重初始化检测头中某一类的分类行（weight 归零 + bias 设为给定值）。

用途：当某类在历史数据集中长期无样本、被训练压死（cls bias 极负）时，
直接在小学习率微调中无法恢复。此脚本构造一个"新类起点"权重。

例：
    python reinit_cls_row.py --src runs/.../best.pt --dst runs/.../init.pt \
        --class-index 5 --bias -4.6
"""
from __future__ import annotations
import argparse
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--class-index", type=int, required=True)
    ap.add_argument("--bias", type=float, default=-4.6)
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    model = ck["model"]
    n_done = 0
    for attr in ("cv3",):
        head = getattr(model.model[-1], attr, None)
        if head is None:
            continue
        for i, seq in enumerate(head):
            last = seq[-1]
            if last.out_channels < args.class_index + 1:
                continue
            with torch.no_grad():
                last.weight[args.class_index].zero_()
                last.bias[args.class_index] = args.bias
            n_done += 1
            print(f"reinit {attr}.{i}.{len(seq)-1}: weight[{args.class_index}]=0, bias[{args.class_index}]={args.bias}")

    # ema 也同步（若存在）
    if ck.get("ema") is not None:
        ema = ck["ema"]
        for i, seq in enumerate(ema.model[-1].cv3):
            last = seq[-1]
            with torch.no_grad():
                last.weight[args.class_index].zero_()
                last.bias[args.class_index] = args.bias
        print("ema synced")

    torch.save(ck, args.dst)
    print(f"saved: {args.dst} ({n_done} branches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
