#!/usr/bin/env python3
"""算 models_zip 里每个网格的包围盒，用来核对物品清单里的尺寸。

只读 obj/dae 的顶点，输出每件在 x/y/z 三个方向的尺寸（米，假设网格单位是米）。
"""

import re
import sys
from pathlib import Path


def obj_bounds(path: Path):
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    with path.open("r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("v "):
                continue
            parts = line.split()
            for i in range(3):
                v = float(parts[i + 1])
                lo[i] = min(lo[i], v)
                hi[i] = max(hi[i], v)
    return lo, hi


def dae_bounds(path: Path):
    """粗略解析 dae：把 <float_array> 里 3 个一组当顶点。"""
    text = path.read_text(errors="ignore")
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for m in re.finditer(r"<float_array[^>]*>(.*?)</float_array>", text, re.S):
        nums = [float(x) for x in m.group(1).split()]
        if len(nums) % 3:  # 顶点数组通常是 3 的倍数；别的数组跳过
            continue
        for i in range(0, len(nums), 3):
            for a in range(3):
                lo[a] = min(lo[a], nums[i + a])
                hi[a] = max(hi[a], nums[i + a])
    return lo, hi


def main(root: Path):
    print(f"{'模型':<42} {'x(m)':>7} {'y(m)':>7} {'z(m)':>7}  {'包围盒体积(L)':>10}")
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        meshes = sorted(folder.glob("*.obj")) or sorted(folder.glob("*.dae"))
        if not meshes:
            print(f"{folder.name:<42} {'没有网格':>7}")
            continue
        path = meshes[0]
        lo, hi = (dae_bounds(path) if path.suffix == ".dae" else obj_bounds(path))
        dim = [hi[i] - lo[i] for i in range(3)]
        vol = dim[0] * dim[1] * dim[2] * 1000
        print(
            f"{folder.name:<42} {dim[0]:7.3f} {dim[1]:7.3f} {dim[2]:7.3f}  {vol:10.2f}"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/models_zip/models"))
