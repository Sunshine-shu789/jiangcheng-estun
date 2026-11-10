#!/usr/bin/env python3
"""把"真机标注图"和"仿真图"按比例混合，生成微调数据集。

默认 real:sim = 1:1（仿真数据防止灾难性遗忘），val 全部用真机图（按 --val-every 抽），
另外抽少量仿真 val 用于观察是否把仿真能力练丢。

用法：
    python build_finetune_dataset.py \
        --real-images real_labeled/images --real-labels real_labeled/labels \
        --sim-data /media/sunshine/新加卷/taskA/dataset_v2 \
        --out /media/sunshine/新加卷/taskA/dataset_real_ft
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-images", required=True)
    parser.add_argument("--real-labels", required=True)
    parser.add_argument("--sim-data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratio", type=float, default=1.0, help="仿真图 : 真机图 的比例")
    parser.add_argument("--val-every", type=int, default=5, help="真机图每 N 张抽 1 张做 val")
    parser.add_argument("--sim-val", type=int, default=60)
    args = parser.parse_args()

    real_img = Path(args.real_images).resolve()
    real_lbl = Path(args.real_labels).resolve()
    sim = Path(args.sim_data).resolve()
    out = Path(args.out).resolve()

    files = image_files(real_img)
    if not files:
        raise SystemExit(f"{real_img} 下没有真机图")
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    real_train, real_val = [], []
    for i, p in enumerate(files):
        (real_val if i % args.val_every == 0 else real_train).append(p)
    if not real_val:                     # 图太少时至少留 1 张验证
        real_val, real_train = real_train[:1], real_train[1:]
    for p in real_train + real_val:
        split = "val" if p in real_val else "train"
        link(p, out / "images" / split / f"real_{p.stem}{p.suffix.lower()}")
        link(real_lbl / f"{p.stem}.txt", out / "labels" / split / f"real_{p.stem}.txt")

    sim_train_pool = image_files(sim / "images" / "train")
    n_sim = min(len(sim_train_pool), int(round(args.ratio * len(real_train))))
    step = max(1, len(sim_train_pool) // max(1, n_sim))
    sim_train = sim_train_pool[::step][:n_sim]
    for p in sim_train:
        link(p, out / "images" / "train" / f"sim_{p.name}")
        link(sim / "labels" / "train" / f"{p.stem}.txt",
             out / "labels" / "train" / f"sim_{p.stem}.txt")

    sim_val_pool = image_files(sim / "images" / "val")
    sim_val = sim_val_pool[:: max(1, len(sim_val_pool) // args.sim_val)][: args.sim_val]
    for p in sim_val:
        link(p, out / "images" / "val" / f"sim_{p.name}")
        link(sim / "labels" / "val" / f"{p.stem}.txt",
             out / "labels" / "val" / f"sim_{p.stem}.txt")

    names_file = real_lbl.parent / "classes.txt"
    names = (
        names_file.read_text(encoding="utf-8").strip().splitlines()
        if names_file.exists()
        else (sim / "classes.txt").read_text(encoding="utf-8").strip().splitlines()
    )
    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
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
    print(
        f"微调数据集 -> {out}\n"
        f"  train：真机 {len(real_train)} 张 + 仿真 {len(sim_train)} 张\n"
        f"  val  ：真机 {len(real_val)} 张 + 仿真 {len(sim_val)} 张（观察是否遗忘仿真能力）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
