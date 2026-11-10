#!/usr/bin/env python3
"""用已保存的预测标签（含置信度）扫描 conf 阈值，不重新推理。

对每个阈值：过滤 preds 标签 -> 调 eval_detection.py 评估 -> 汇总通过率/误检。
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def filter_labels(src: Path, dst: Path, conf: float) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.txt"):
        keep = []
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 6 and float(parts[5]) >= conf:
                keep.append(line)
        (dst / f.name).write_text("\n".join(keep) + ("\n" if keep else ""))


def parse_report(path: Path) -> dict:
    text = path.read_text()
    out = {}
    m = re.search(r"通过率（IoU≥0\.75） \| ([\d.]+)%", text)
    if m:
        out["pass"] = float(m.group(1))
    m = re.search(r"Recall@IoU0\.5 \| ([\d.]+)%", text)
    if m:
        out["recall"] = float(m.group(1))
    m = re.search(r"误检（FP 框） \| (\d+)", text)
    if m:
        out["fp"] = int(m.group(1))
    m = re.search(r"平均 best IoU \| ([\d.]+)", text)
    if m:
        out["iou"] = float(m.group(1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pred-labels", required=True, help="原始预测标签目录（含 conf）")
    parser.add_argument("--work", default="/tmp/conf_sweep")
    parser.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70")
    args = parser.parse_args()

    task_dir = Path(__file__).resolve().parent
    work = Path(args.work)
    src = Path(args.pred_labels).resolve()
    rows = []

    for t in [float(x) for x in args.thresholds.split(",")]:
        tag = f"c{int(round(t * 100)):03d}"
        labels = work / tag / "labels"
        if labels.exists():
            shutil.rmtree(labels)
        filter_labels(src, labels, t)
        report = work / tag / "eval"
        subprocess.run(
            [
                sys.executable,
                str(task_dir / "eval_detection.py"),
                "--data", str(Path(args.data).resolve()),
                "--pred-labels", str(labels),
                "--split", "val",
                "--out", str(report),
            ],
            check=True,
            capture_output=True,
        )
        stats = parse_report(report.with_suffix(".md"))
        rows.append((t, stats))
        print(
            f"conf>={t:.2f}  通过率={stats.get('pass', float('nan')):.1f}%  "
            f"Recall@0.5={stats.get('recall', float('nan')):.1f}%  "
            f"平均IoU={stats.get('iou', float('nan')):.3f}  误检={stats.get('fp', -1)}",
            flush=True,
        )

    best = max(rows, key=lambda r: r[1].get("pass", 0))
    print(f"\n最佳阈值：conf>={best[0]:.2f}，通过率 {best[1].get('pass'):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
