#!/usr/bin/env python3
"""把 CEPB models.zip 里的网格包围盒，跟赛道一物品清单逐件对比。

网格里混了三种单位（米 / 厘米 / 毫米），所以每件都拿 {1, 0.01, 0.001}
三个标度跟清单尺寸试一遍，取最接近的那个——清单里那 11 件标着 "ycb近似"
的本来就是从 YCB 数据抄的，能当作标度判定的参照。
"""

import math
import re
import sys
from pathlib import Path

import yaml

MODELS = Path("datasets/models_zip/models")
LIST = Path("ros2_ws/src/s3_60_eco_2024_moveit_config/config/track1_items.yaml")

# 网格文件夹 -> 清单里的物品 id（编号体系不同：网格里 Kong 是 27、可乐两个
# id 共用 "15-28 Bottle" 一个网格，所以按名字手工映射）
FOLDER_TO_IDS = {
    "1-Cheez-it": [1], "2-Starkist_Tuna": [2], "3-Scissors": [3],
    "4-Frenchs_Mustard": [4], "5-Tomato_Soup": [5], "6-Foam_Brick": [6],
    "7-Clamp": [7], "8-Plastic_Banana": [8], "9-Mug": [9],
    "10-meat_can": [10], "11-Padlock": [11], "12-Baseball": [12],
    "13-Bowl": [13], "14-Sleeve": [14], "15-28 Bottle": [15, 27],
    "16-ICRA_Duckie": [16], "17-Kygen_Squeakin_Eggs_Plush_Puppies": [17],
    "18-Dice": [18], "19-Elmers_Glue": [19], "20-Scotch_Duct_Tape": [20],
    "21_Haribo_gold_bear": [21], "22-E27_adapter": [22],
    "23-GU24_adapter": [23], "24-Spoon": [24], "25-Flat_Screwdriver": [25],
    "26-Plastic_Strawberry": [26], "27-Kong_Duck_Dog_Toy": [28],
    "29-Paper_Towels": [29], "30-Stabilo_OHPen": [30],
    "31-Plastic_White_Cup": [31], "32-Wine_Glass": [32], "33-Key": [33],
    "34-Nail": [34], "35-Laugh_Out_Loud_Joke_Book": [35],
    "36-Adjustable_Wrench": [36], "37-T-shirt": [37],
    "38-Rolodex_Jumbo_Pencil_Cup": [38], "39-Glove": [39], "40-Pringles": [40],
}


def obj_dims(path: Path):
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    with path.open("r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("v "):
                continue
            p = line.split()
            for i in range(3):
                v = float(p[i + 1])
                lo[i] = min(lo[i], v)
                hi[i] = max(hi[i], v)
    return sorted((hi[i] - lo[i] for i in range(3)), reverse=True)


def dae_dims(path: Path):
    text = path.read_text(errors="ignore")
    pts = []
    for m in re.finditer(r"<float_array[^>]*>(.*?)</float_array>", text, re.S):
        nums = [float(x) for x in m.group(1).split()]
        if len(nums) % 3 == 0 and len(nums) >= 9:
            pts.extend(nums)
    if not pts:
        return [0.0, 0.0, 0.0]
    dims = []
    for a in range(3):
        col = pts[a::3]
        dims.append(max(col) - min(col))
    return sorted(dims, reverse=True)


def mesh_dims(folder: Path):
    meshes = sorted(folder.glob("*.obj")) or sorted(folder.glob("*.dae"))
    if not meshes:
        return None
    path = meshes[0]
    return dae_dims(path) if path.suffix == ".dae" else obj_dims(path)


def list_dims(item):
    """把清单条目换算成"排序后的三个边长"，方便跟网格包围盒比。"""
    size = [float(v) for v in item["size"]]
    shape = item.get("shape", "box")
    if shape == "cylinder":
        dims = [size[0], size[0], size[1] if len(size) > 1 else size[0]]
    elif shape == "sphere":
        dims = [size[0]] * 3
    else:
        dims = size
    return sorted(dims, reverse=True)


def pick_scale(mesh, listed):
    best = min(
        (1.0, 0.01, 0.001),
        key=lambda s: sum(
            abs(math.log((m * s) / l)) for m, l in zip(mesh, listed) if m > 0 and l > 0
        ),
    )
    return best


def main():
    items = {it["id"]: it for it in yaml.safe_load(LIST.read_text(encoding="utf-8"))["items"]}
    rows = []
    for folder in sorted(MODELS.iterdir(), key=lambda p: p.name):
        if not folder.is_dir() or folder.name not in FOLDER_TO_IDS:
            continue
        mesh = mesh_dims(folder)
        if mesh is None:
            continue
        for item_id in FOLDER_TO_IDS[folder.name]:
            item = items[item_id]
            listed = list_dims(item)
            scale = pick_scale(mesh, listed)
            real = [m * scale for m in mesh]
            delta = max(
                abs(r - l) / l for r, l in zip(real, listed) if l > 0
            )
            rows.append((item_id, folder.name, scale, listed, real, delta, item))
    rows.sort()

    print("| id | 清单名 | src | 清单尺寸(排序, m) | 网格尺寸(排序, m) | 标度 | 最大偏差 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for item_id, folder, scale, listed, real, delta, item in rows:
        l = "×".join(f"{v:.3f}" for v in listed)
        r = "×".join(f"{v:.3f}" for v in real)
        unit = {1.0: "米", 0.01: "厘米", 0.001: "毫米"}[scale]
        flag = "✅" if delta <= 0.1 else ("⚠️" if delta <= 0.3 else "❌")
        print(
            f"| {item_id} | {item['name_en']} | {item.get('source','')} | {l} | {r} "
            f"| {unit} | {flag} {delta*100:.0f}% |"
        )

    bad = [r for r in rows if r[5] > 0.3]
    print(f"\n合计 {len(rows)} 条对比，偏差 >30% 的有 {len(bad)} 条：")
    for _, folder, _, listed, real, delta, item in bad:
        print(f"  - {item['name_en']}（{folder}）：清单 "
              f"{'×'.join(f'{v:.3f}' for v in listed)} → 网格 "
              f"{'×'.join(f'{v:.3f}' for v in real)}")


if __name__ == "__main__":
    sys.exit(main())
