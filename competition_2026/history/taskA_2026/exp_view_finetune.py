#!/usr/bin/env python3
"""实验：换视角后，需要多少张标注图才能恢复检测精度？

设置（仿真内的代理实验）：
  - 预训练模型：runs/yolo11n_v1/weights/best.pt —— 只见过 left 相机，
    在 middle 视角上通过率仅 0.6%（2026-09-20 实测），模拟"没见过真机视角"。
  - 微调数据：dataset_v2 的 middle 视角训练图，取 N 张（按场景整取，避免同一场景泄漏）。
  - 评估：完整的 middle 视角 val（300 张 / 3000 实例），IoU≥0.75 通过率。

产出：reports/exp_view_finetune.md（通过率 ↔ 标注张数 曲线）。

用法：
    python exp_view_finetune.py --n-list 51,150,399 --epochs 30
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cepb


def scene_of(name: str) -> int:
    m = re.search(r"_scene_(\d+)_", name)
    if not m:
        raise ValueError(f"文件名里没有场景号：{name}")
    return int(m.group(1))


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build_eval_view(data: Path, out: Path) -> None:
    """构建只含 middle 视角 val 的数据集视图（供 eval_detection.py 用）。"""
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)
    images = sorted((data / "images" / "val").glob("middle_camera_*.png"))
    for img in images:
        link(img, out / "images" / "val" / img.name)
        link(data / "labels" / "val" / f"{img.stem}.txt", out / "labels" / "val" / f"{img.stem}.txt")
    keep = {img.name for img in images}
    with (data / "instances.csv").open(encoding="utf-8") as fh, (
        out / "instances.csv"
    ).open("w", newline="", encoding="utf-8") as fw:
        reader = csv.DictReader(fh)
        writer = csv.DictWriter(fw, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["image"] in keep:
                writer.writerow(row)
    shutil.copy(data / "classes.txt", out / "classes.txt")
    print(f"评估视图：{len(images)} 张 middle val 图 -> {out}")


def build_ft_dataset(data: Path, out: Path, n_images: int, val_images: int = 60) -> int:
    """按场景整取 N 张 middle 训练图 + 少量 val（训练期监控用）。"""
    train_pool = sorted(
        (data / "images" / "train").glob("middle_camera_*.png"), key=lambda p: scene_of(p.name)
    )
    scenes: list[int] = []
    for img in train_pool:
        s = scene_of(img.name)
        if s not in scenes:
            scenes.append(s)
    n_scenes = max(1, round(n_images / 3))
    picked = set(scenes[:n_scenes])
    train_imgs = [img for img in train_pool if scene_of(img.name) in picked]

    val_pool = sorted(
        (data / "images" / "val").glob("middle_camera_*.png"), key=lambda p: scene_of(p.name)
    )
    val_imgs = val_pool[:val_images]

    for split, imgs in (("train", train_imgs), ("val", val_imgs)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in imgs:
            link(img, out / "images" / split / img.name)
            link(data / "labels" / "train" / f"{img.stem}.txt"
                 if split == "train" else data / "labels" / "val" / f"{img.stem}.txt",
                 out / "labels" / split / f"{img.stem}.txt")

    names = (data / "classes.txt").read_text(encoding="utf-8").strip().splitlines()
    (out / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                "names:",
                *[f"- {n}" for n in names],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"微调数据集：train {len(train_imgs)} 张（{len(picked)} 场景），val {len(val_imgs)} 张 -> {out}")
    return len(train_imgs)


def parse_pass_rate(report: Path) -> dict:
    text = report.read_text(encoding="utf-8")
    out = {}
    for key, pat in (
        ("pass", r"通过率（IoU≥0\.75） \| ([\d.]+)%"),
        ("iou", r"平均 best IoU \| ([\d.]+)"),
        ("recall", r"Recall@IoU0\.5 \| ([\d.]+)%"),
        ("fp", r"误检（FP 框） \| (\d+)"),
    ):
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1)) if key != "fp" else int(m.group(1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(cepb.WORKDIR / "dataset_v2"))
    parser.add_argument("--base-weights", default=str(cepb.WORKDIR / "runs/yolo11n_v1/weights/best.pt"))
    parser.add_argument("--work", default=str(cepb.WORKDIR / "exp_view_finetune"))
    parser.add_argument("--n-list", default="51,150,399")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto=有 GPU 用 0，否则 cpu")
    args = parser.parse_args()

    task_dir = Path(__file__).resolve().parent
    device = cepb.resolve_device(args.device)
    data = Path(args.data)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    eval_view = work / "ds_middle_eval"
    if not (eval_view / "instances.csv").exists():
        build_eval_view(data, eval_view)

    from ultralytics import YOLO

    rows = []
    for n in [int(x) for x in args.n_list.split(",")]:
        tag = f"n{n:04d}"
        ds = work / "datasets" / tag
        if ds.exists():
            shutil.rmtree(ds)
        n_real = build_ft_dataset(data, ds, n)

        run_dir = work / "runs" / tag
        model = YOLO(args.base_weights)
        model.train(
            data=str(ds / "data.yaml"),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
                device=device,
            workers=args.workers,
            lr0=args.lr0,
            lrf=0.1,
            mosaic=0.0,       # 小样本微调关掉 mosaic
            close_mosaic=0,
            fliplr=0.5,
            scale=0.5,
            degrees=0.0,
            plots=False,
            val=True,
            seed=0,
            patience=100,     # 实验里不早停，保证各档轮数一致
            project=str(work / "runs"),
            name=tag,
            exist_ok=True,
        )

        weights = run_dir / "weights" / "best.pt"
        preds = work / "preds" / tag
        subprocess.run(
            [
                sys.executable, str(task_dir / "infer_imgsz.py"),
                "--weights", str(weights), "--data", str(eval_view),
                "--split", "val", "--imgsz", str(args.imgsz), "--out", str(preds),
            ],
            check=True,
        )
        report = work / "reports" / tag
        subprocess.run(
            [
                sys.executable, str(task_dir / "eval_detection.py"),
                "--data", str(eval_view), "--pred-labels", str(preds / "labels"),
                "--split", "val", "--out", str(report),
            ],
            check=True,
        )
        stats = parse_pass_rate(report.with_suffix(".md"))
        rows.append((n_real, stats))
        print(f"[结果] 训练图 {n_real} 张 -> 通过率 {stats.get('pass')}%  IoU {stats.get('iou')}", flush=True)

    baseline = {"pass": 0.6, "iou": 0.222, "recall": 0.0, "fp": 4564}
    full = {"pass": 36.8, "iou": 0.485, "recall": 0.0, "fp": 0}
    lines = [
        "# 实验：换视角后需要多少张标注图（仿真代理）",
        "",
        "- 预训练：`runs/yolo11n_v1/weights/best.pt`（只见过 left 相机）",
        "- 微调数据：dataset_v2 的 middle 视角训练图，按场景整取；val 为完整 middle val（300 张 / 3000 实例）",
        f"- 超参：epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, lr0={args.lr0}, mosaic=0",
        "",
        "| 微调标注张数 | 通过率 IoU≥0.75 | 平均 best IoU | Recall@0.5 | 误检 |",
        "| --- | --- | --- | --- | --- |",
        f"| 0（基线，只训 left） | {baseline['pass']}% | {baseline['iou']} | — | {baseline['fp']} |",
    ]
    for n_real, s in rows:
        lines.append(
            f"| {n_real} | {s.get('pass')}% | {s.get('iou')} | {s.get('recall')}% | {s.get('fp')} |"
        )
    lines += [
        f"| 参照：v2 三视角全量训练 | {full['pass']}% | {full['iou']} | — | — |",
        "",
        "> 说明：这是**仿真内换视角**的代理实验，只包含视角变化，不含真机的光照/材质/镜头差异，",
        "> 因此结论应视为真机采集量的**下界**。",
        "",
    ]
    summary = work / "summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    reports_dir = cepb.WORKDIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "exp_view_finetune.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"汇总：{summary} 与 {reports_dir / 'exp_view_finetune.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
