#!/usr/bin/env python3
"""给赛道一物品清单补上网格字段：mesh / mesh_scale / mesh_rpy，并把尺寸换成实测值。

对每件物品：

1. 按名字找到 ``models_zip`` 里对应的网格；
2. 在 {1, 0.01, 0.001} 里挑一个标度（米 / 厘米 / 毫米），挑法和
   ``mesh_vs_list.py`` 一样——跟清单里的估计尺寸最接近的那个；
3. 在 6 种 90° 轴置换里挑一个旋转，让网格的包围盒朝向跟清单里的 x/y/z 最贴合
   （比如瓶子是躺着的网格，这里会把它转成正立）；
4. ``size`` 改成网格实测的包围盒尺寸（米），另记 ``size_source: mesh``。

原来那份纯包围盒清单会备份成 ``track1_items.primitives.yaml``，方便两种模式对照。
"""

import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "ros2_ws/src/s3_60_eco_2024_moveit_config/scripts")
from mesh_vs_list import FOLDER_TO_IDS  # noqa: E402
from track1_meshes import load_geometry  # noqa: E402

REPO = Path("ros2_ws/src/s3_60_eco_2024_moveit_config/config")
CONFIG = REPO / "track1_items.yaml"
BACKUP = REPO / "track1_items.primitives.yaml"
MESH_ROOT = Path("datasets/models_zip").resolve()

# 网格文件夹 -> 清单 id（编号体系不同，按名字手工映射）
ID_TO_FOLDER = {}
for folder, ids in FOLDER_TO_IDS.items():
    for item_id in ids:
        ID_TO_FOLDER[item_id] = folder

PERMUTATIONS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

# 自动判断会摆错的几件，按实物语义手动指定"立起来"的原生轴（0=X, 1=Y, 2=Z）。
# 判断依据：马克杯带把手的整体包围盒最长边是把手方向（X=0.117），得让它躺平、
# 用杯身高度（Z=0.093）当竖直；胶水瓶（Y=0.143）和笔筒（Y=0.135）比估算高一截，
# 按估算匹配会被挑成别的轴。
UP_AXIS_OVERRIDE = {9: 2, 19: 1, 38: 1}


def target_dims(item):
    """清单里的 (x, y, z) 三边；圆柱是 [直径, 高度]，球是 [直径]。"""
    size = [float(v) for v in item["size"]]
    if item["shape"] == "cylinder":
        return [size[0], size[0], size[1] if len(size) > 1 else size[0]]
    if item["shape"] == "sphere":
        return [size[0]] * 3
    return size


def pick_scale(raw_dims, target):
    return min(
        (1.0, 0.01, 0.001),
        key=lambda s: sum(
            abs(math.log((r * s) / t)) for r, t in zip(sorted(raw_dims), sorted(target))
        ),
    )


def pick_rotation(raw_dims, scale, target):
    """挑一个轴置换，使旋转后的包围盒三边最接近清单里的 (x, y, z)。"""
    best = None
    for perm in PERMUTATIONS:
        mapped = [raw_dims[perm[i]] * scale for i in range(3)]
        error = sum(abs(math.log(m / t)) for m, t in zip(mapped, target))
        if best is None or error < best[0]:
            best = (error, perm, mapped)
    _, perm, mapped = best
    matrix = np.zeros((3, 3))
    for i, axis in enumerate(perm):
        matrix[i, axis] = 1.0
    if np.linalg.det(matrix) < 0:  # 保证是旋转不是镜像
        matrix[2, perm[2]] = -1.0
    return matrix, mapped


def pick_axis_up(raw_dims, scale, estimate_height):
    """圆柱类物品：把"最接近估算高度的那个原生轴"立起来当 +Z。

    整体三边匹配对圆柱不可靠——马克杯的估算是个不带把手的圆柱（高度 0.088），
    而网格三边是 0.117/0.093/0.081，整体匹配会把带把手的那条边当成高度，
    杯子就躺倒了。只比高度这一条边就稳得多。
    """
    errors = [abs(raw * scale - estimate_height) for raw in raw_dims]
    up = errors.index(min(errors))
    rest = [axis for axis in range(3) if axis != up]
    return (rest[0], rest[1], up)


def matrix_to_rpy_deg(matrix):
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2, 0])))
    if abs(matrix[2, 0]) < 0.999999:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:  # 万向锁
        roll = 0.0
        yaw = math.atan2(-matrix[0, 1], matrix[1, 1])
    return [round(math.degrees(v), 1) for v in (roll, pitch, yaw)]


def main():
    source = BACKUP if BACKUP.exists() else CONFIG
    text = source.read_text(encoding="utf-8")
    header_end = text.index("frame_id:")
    header = text[:header_end]
    config = yaml.safe_load(text)

    if source is CONFIG:  # 第一次跑：先把纯包围盒版本留一份
        BACKUP.write_text(text, encoding="utf-8")
        print(f"原始（纯包围盒）清单已备份到 {BACKUP}")

    config["mesh_root"] = str(MESH_ROOT)
    print(f"\n{'id':>2} {'物品':<28} {'标度':<4} {'立起来的轴':<10} {'旋转(deg)':<20} 原尺寸 → 网格实测")
    for item in config["items"]:
        folder = ID_TO_FOLDER.get(item["id"])
        if folder is None:
            print(f"{item['id']:>2} {item['name_en']:<28} 没有对应网格")
            continue
        mesh_files = sorted((MESH_ROOT / "models" / folder).glob("*.obj"))
        if not mesh_files:
            mesh_files = sorted((MESH_ROOT / "models" / folder).glob("*.dae"))
        mesh_path = mesh_files[0]
        vertices, _triangles, _hull = load_geometry(mesh_path)
        raw_axes = [float(v) for v in (vertices.max(axis=0) - vertices.min(axis=0))]
        raw = sorted(raw_axes, reverse=True)

        target = target_dims(item)
        scale = pick_scale(raw, target)
        if item["id"] in UP_AXIS_OVERRIDE or item["shape"] == "cylinder":
            if item["id"] in UP_AXIS_OVERRIDE:
                up = UP_AXIS_OVERRIDE[item["id"]]
                rest = [axis for axis in range(3) if axis != up]
                perm = (rest[0], rest[1], up)
            else:
                perm = pick_axis_up(raw_axes, scale, target[2])
            mapped = [raw_axes[perm[i]] * scale for i in range(3)]
            matrix = np.zeros((3, 3))
            for i, axis in enumerate(perm):
                matrix[i, axis] = 1.0
            if np.linalg.det(matrix) < 0:
                matrix[2, perm[2]] = -1.0
            up_note = f"轴{'XYZ'[perm[2]]}={mapped[2]:.3f}"
            if item["id"] in UP_AXIS_OVERRIDE:
                up_note += "(指定)"
        else:
            matrix, mapped = pick_rotation(raw_axes, scale, target)
            up_note = "—"
        rpy = matrix_to_rpy_deg(matrix)

        if item["shape"] == "cylinder":
            new_size = [round(max(mapped[0], mapped[1]), 4), round(mapped[2], 4)]
        elif item["shape"] == "sphere":
            new_size = [round(max(mapped), 4)]
        else:
            new_size = [round(v, 4) for v in mapped]

        old = "×".join(f"{v:.3f}" for v in item["size"])
        new = "×".join(f"{v:.3f}" for v in new_size)
        unit = {1.0: "米", 0.01: "厘米", 0.001: "毫米"}[scale]
        print(f"{item['id']:>2} {item['name_en']:<28} {unit:<4} {up_note:<10} {str(rpy):<20} {old} → {new}")

        item["mesh"] = str(mesh_path.relative_to(MESH_ROOT))
        item["mesh_scale"] = scale
        item["mesh_rpy"] = rpy
        item["size"] = new_size
        item["size_source"] = "mesh"

    CONFIG.write_text(
        header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    print(f"\n已写入 {CONFIG}（mesh_root = {MESH_ROOT}）")


if __name__ == "__main__":
    main()
