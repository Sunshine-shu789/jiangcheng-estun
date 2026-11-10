#!/usr/bin/env python3
"""生成训练用「轻量数据集」：原图预缩到长边 640 的无损 PNG，并做防泄漏切分。

为什么：
  原图 2048×1536 PNG 放在 NTFS 盘上读取只有 ~40 MB/s，而 mosaic=1.0 会让每张图
  每 epoch 被读约 4 次 → 光是读图就压死训练（实测 GPU 利用率只有 28%）。
  训练 imgsz=640 时，dataloader 本来就会把每张图 letterbox 到 640×640，
  所以提前缩到 640×480（4:3）不丢任何有效信息，反而少一次重采样（INTER_AREA 更干净）。

防泄漏：
  dataset_v4（1500 场景）是按等差数列抽的场景，会把 dataset_v2 的 val 场景
  （1, 51, 101, …, 4951）也抽进去。本脚本把 --val-src 的 val 场景从 train 里整份剔除，
  并用 --val-src 的 val 作为轻量集的 val —— 这样早停曲线和最终评测（dataset_v2 val 900 张）
  是同一个分布，和 39.4% / 41.6% 的基线可直接比较。

用法：
  ./venv/bin/python build_light_dataset.py \
      --src dataset_v4 --val-src dataset_v2 --dst dataset_v4_640 \
      --long 640 --jobs 16
"""
from __future__ import annotations

import argparse
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
SCENE_RE = re.compile(r"_scene_(\d+)_")


def scene_of(name: str) -> int | None:
    m = SCENE_RE.search(name)
    return int(m.group(1)) if m else None


def convert_one(task: tuple[str, str, int]) -> None:
    src, dst, long_side = task
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"读图失败: {src}")
    h, w = im.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1:
        im = cv2.resize(im, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(dst, im, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"写图失败: {dst}")


def collect(img_dir: Path, skip_scenes: set[int] | None = None) -> list[Path]:
    out = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        s = scene_of(p.name)
        if skip_scenes and s is not None and s in skip_scenes:
            continue
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原图数据集（train 从这里取）")
    ap.add_argument("--val-src", required=True, help="val 来源数据集（其 val 场景也会从 train 剔除）")
    ap.add_argument("--dst", required=True, help="输出的轻量数据集目录")
    ap.add_argument("--long", type=int, default=640, help="长边目标像素")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    val_src = Path(args.val_src).resolve()
    dst = Path(args.dst).resolve()
    t0 = time.time()

    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    # val 场景（= 最终评测用的场景），从 train 里整份剔除
    val_imgs = sorted(p for p in (val_src / "images" / "val").iterdir() if p.suffix.lower() in IMG_EXT)
    val_scenes = {s for s in (scene_of(p.name) for p in val_imgs) if s is not None}

    train_all = sorted(p for p in (src / "images" / "train").iterdir() if p.suffix.lower() in IMG_EXT)
    train_imgs = [p for p in train_all if (scene_of(p.name) not in val_scenes)]
    dropped = len(train_all) - len(train_imgs)

    print(f"val 场景 {len(val_scenes)} 个（来自 {val_src.name}）")
    print(f"train：{len(train_all)} → {len(train_imgs)} 张（剔除 {dropped} 张，防泄漏）；val {len(val_imgs)} 张")
    print(f"输出 {dst}，长边 {args.long}，进程 {args.jobs}")

    tasks: list[tuple[str, str, int]] = []
    for p in train_imgs:
        tasks.append((str(p), str(dst / "images" / "train" / (p.stem + ".png")), args.long))
    for p in val_imgs:
        tasks.append((str(p), str(dst / "images" / "val" / (p.stem + ".png")), args.long))

    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(convert_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            done += 1
            if i % 1000 == 0 or i == len(futs):
                print(f"  已转换 {i}/{len(futs)}（{time.time() - t0:.0f}s）", flush=True)

    # 标签原样复制（归一化坐标，与分辨率无关）
    n_lab = 0
    keep_stems = {q.stem for q in train_imgs}
    for p in (src / "labels" / "train").iterdir():
        if p.suffix == ".txt" and p.stem in keep_stems:
            shutil.copy2(p, dst / "labels" / "train" / p.name)
            n_lab += 1
    for p in (val_src / "labels" / "val").iterdir():
        if p.suffix == ".txt":
            shutil.copy2(p, dst / "labels" / "val" / p.name)
            n_lab += 1

    txt = (val_src / "data.yaml").read_text()
    lines = [f"path: {dst}" if line.startswith("path:") else line for line in txt.splitlines()]
    (dst / "data.yaml").write_text("\n".join(lines) + "\n")

    n_img = sum(1 for _ in (dst / "images").rglob("*.png"))
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"完成：图片 {n_img}（train {len(train_imgs)} / val {len(val_imgs)}）、标签 {n_lab}、"
          f"共 {size / 2**30:.2f} GiB，耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
