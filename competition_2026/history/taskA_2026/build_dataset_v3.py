#!/usr/bin/env python3
"""从 dataset_v2 生成「小图版」dataset_v3（长边预缩到 1280，JPEG q92）。

为什么要这个：
  dataset_v2 是 2048×1536 PNG（约 1.09 MB/张），放在 NTFS 挂载盘上单线程读取
  只有 ~40 MB/s；而 mosaic=1.0 会让每张图每 epoch 被读约 4 次 →
  光读图就是 15.7 GB/epoch，IO 成为训练瓶颈（GPU 利用率只有个位数）。
  预缩到 1280×960 后：每 epoch IO ≈ 3.6 GB，解码量降 2.6×，
  而训练侧 imgsz=960 的 letterbox 几何完全不变（1280×960 → 960×720 + padding）。

  标签是归一化坐标，直接复制；文件名（去扩展名）保持一致。
  最终评估仍然用 dataset_v2 原图（reports 里与 640 基线可比）。

用法（沙箱外/数据盘执行）：
  ./venv/bin/python build_dataset_v3.py \
      --src dataset_v2 --dst dataset_v3 --long 1280 --jobs 16 --quality 92
"""
from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def convert_one(task: tuple[str, str, int, int]) -> tuple[str, int, int]:
    src, dst, long_side, quality = task
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"读图失败: {src}")
    h, w = im.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1:
        im = cv2.resize(im, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if not cv2.imwrite(dst, im, params):
        raise RuntimeError(f"写图失败: {dst}")
    return dst, im.shape[0], im.shape[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset_v2")
    ap.add_argument("--dst", default="dataset_v3")
    ap.add_argument("--long", type=int, default=1280, help="长边目标像素")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    t0 = time.time()

    # 1) 目录骨架
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2) 图像：并行缩放 + 写 jpg
    tasks = []
    for split in ("train", "val"):
        for p in sorted((src / "images" / split).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                tasks.append((str(p), str(dst / "images" / split / (p.stem + ".jpg")), args.long, args.quality))

    print(f"待转换 {len(tasks)} 张（长边 {args.long}，JPEG q{args.quality}，进程 {args.jobs}）")
    done, fail = 0, []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(convert_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
                done += 1
            except Exception as exc:  # noqa: BLE001
                fail.append((futs[fut], repr(exc)))
            if i % 500 == 0 or i == len(futs):
                print(f"  已处理 {i}/{len(futs)}（{time.time() - t0:.0f}s）", flush=True)

    # 3) 标签原样复制（归一化坐标，与分辨率无关）
    n_lab = 0
    for split in ("train", "val"):
        for p in (src / "labels" / split).iterdir():
            if p.suffix == ".txt":
                shutil.copy2(p, dst / "labels" / split / p.name)
                n_lab += 1

    # 4) data.yaml（只改 path）
    txt = (src / "data.yaml").read_text()
    lines = []
    for line in txt.splitlines():
        lines.append(f"path: {dst}" if line.startswith("path:") else line)
    (dst / "data.yaml").write_text("\n".join(lines) + "\n")

    # 5) 统计
    n_img = sum(1 for _ in (dst / "images").rglob("*.jpg"))
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(
        f"完成：图像 {n_img}、标签 {n_lab}、失败 {len(fail)}，"
        f"总计 {size / 2**30:.2f} GiB，耗时 {time.time() - t0:.0f}s"
    )
    for path, err in fail[:10]:
        print("  失败:", path, err)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
