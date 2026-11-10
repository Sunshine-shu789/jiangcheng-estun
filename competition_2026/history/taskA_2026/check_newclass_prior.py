#!/usr/bin/env python3
"""补类训练前体检：检查「有样本的类」其分类头 bias 是否已被历史训练压死。

背景（2026-10-15 泡沫砖首训事故）：
某类在历史数据集中长期 0 样本时，训练会持续用负样本把该类 cls head 的
bias 压得极负（实测 class 5 = -22.8，sigmoid≈6e-10）。之后首次补入样本做
小学习率微调时，每步 bias 位移 ≈ lr，光靠微调无法把 logit 拉回可输出区间
（需要 +20，65 epoch 理论上只能 +1 左右），结果该类永远 0 检出。
正确做法：训练前先把该类 cls 行重初始化（weight=0, bias≈-4.6），再微调。

用法：
  # 体检（只报告，不改任何东西）
  python check_newclass_prior.py --weights <起点.pt> --data <data.yaml>

  # 体检 + 生成修复起点（把有样本但被压死的类重置）
  python check_newclass_prior.py --weights <起点.pt> --data <data.yaml> \
      --fix <输出起点.pt> [--fix-bias -4.6]

推荐同时提供 --history（上一代数据集 data.yaml）以精确区分：
  - 「新补类」：history 样本数 = 0，当前 > 0
  - 正常低 bias 类：history 样本数 > 0（虽然 bias 低，但权重行已学会响应）

判据：history 样本数 = 0 且当前样本数 > 0 且 max(三尺度 bias) < DEAD_LOGIT
(默认 -9.2，sigmoid<1e-4) → 判定「压死」。
无 --history 时只做「疑似」提示，执行 --fix 需要 --force-fix 确认。
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import yaml

DEAD_LOGIT = -9.2  # sigmoid(-9.2) ≈ 1e-4


def load_names(data_yaml: Path) -> dict[int, str]:
    d = yaml.safe_load(data_yaml.read_text())
    names = d.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names or [])}


def count_train_samples(data_yaml: Path) -> collections.Counter:
    d = yaml.safe_load(data_yaml.read_text())
    root = Path(str(d.get("path", data_yaml.parent)))
    train = d.get("train")
    paths = train if isinstance(train, (list, tuple)) else [train]
    cnt: collections.Counter = collections.Counter()
    for p in paths:
        p = Path(str(p))
        if not p.is_absolute():
            p = root / p
        parts = list(p.parts)
        for i, seg in enumerate(parts):
            if seg == "images":
                parts[i] = "labels"
                break
        label_dir = Path(*parts)
        if not label_dir.exists():
            continue
        for f in label_dir.glob("*.txt"):
            for line in f.read_text().splitlines():
                line = line.strip()
                if line:
                    cnt[int(line.split()[0])] += 1
    return cnt


def cls_bias_rows(weights: Path):
    import torch

    ck = torch.load(weights, map_location="cpu", weights_only=False)
    model = ck["model"]
    head = model.model[-1]
    rows = []  # (branches, weight tensor refs, bias tensor refs)
    for i, seq in enumerate(head.cv3):
        rows.append((i, seq[-1]))
    return ck, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--history", default=None, help="上一代数据集 data.yaml（强烈建议提供）")
    ap.add_argument("--fix", default=None, help="生成修复起点权重的输出路径")
    ap.add_argument("--fix-bias", type=float, default=-4.6)
    ap.add_argument("--force-fix", action="store_true", help="无 --history 时也允许执行修复")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    weights = Path(args.weights)
    names = load_names(data_yaml)
    cnt = count_train_samples(data_yaml)
    hist = count_train_samples(Path(args.history)) if args.history else None
    ck, rows = cls_bias_rows(weights)

    dead = []
    print(f"起点权重：{weights}")
    print(f"数据集  ：{data_yaml}")
    hdr = f"{'id':>3} {'name':<18} {'hist_n':>7} {'train_n':>8} " if hist is not None else f"{'id':>3} {'name':<18} {'train_n':>8} "
    print(hdr + " ".join(f"{'bias@P'+str(i+3):>9}" for i, _ in rows) + "  sigmax   状态")
    import math
    for cid in sorted(names):
        n = cnt.get(cid, 0)
        hn = hist.get(cid, 0) if hist is not None else None
        biases = [float(conv.bias[cid]) for _, conv in rows]
        sig = max(biases)
        psig = 1 / (1 + math.exp(-sig))
        flags = ""
        if n > 0 and sig < DEAD_LOGIT:
            if hist is not None:
                if hn == 0:
                    flags = "  WARN(新补类且压死)"
                    dead.append(cid)
                else:
                    flags = "  ok(历史有样本)"
            else:
                flags = "  ?疑似压死(需--history确认)"
        elif n == 0:
            flags = "  (无样本)"
        cols = f"{cid:>3} {names[cid]:<18} "
        if hist is not None:
            cols += f"{hn:>7} "
        cols += f"{n:>8} " + " ".join(f"{b:>9.3f}" for b in biases) + f"  {psig:8.1e}{flags}"
        print(cols)

    suspected = [cid for cid in sorted(names) if cnt.get(cid, 0) > 0
                 and max(float(conv.bias[cid]) for _, conv in rows) < DEAD_LOGIT]
    if hist is None and suspected:
        print(f"\n⚠️  （未提供 --history）疑似低 bias 且有样本的类：{suspected}")
        print("   若其中某类historically无样本，请提供 --history 精确判定。")
    if dead:
        print(f"\n⚠️  共 {len(dead)} 个类「新补且分类行已压死」：{dead}")
        print("   直接微调学不出来；请加 --fix 生成修复起点，或从已修复权重继续。")
    else:
        print("\n✅ 未发现「新补且压死」的类。")

    if args.fix:
        if hist is None and suspected and not args.force_fix:
            print("❌ 无 --history 时如需执行修复，请加 --force-fix（避免误改正常低 bias 类）。")
            return 2
        if not dead:
            print("无需修复，仍按要求保存副本。")
        for cid in dead:
            for _, conv in rows:
                import torch
                with torch.no_grad():
                    conv.weight[cid].zero_()
                    conv.bias[cid] = args.fix_bias
        if ck.get("ema") is not None:
            ema_rows = list(enumerate(ck["ema"].model[-1].cv3))
            for cid in dead:
                for _, conv in ema_rows:
                    import torch
                    with torch.no_grad():
                        conv.weight[cid].zero_()
                        conv.bias[cid] = args.fix_bias
        import torch
        torch.save(ck, args.fix)
        print(f"已保存修复起点：{args.fix}（重置类 {dead}，bias→{args.fix_bias}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
