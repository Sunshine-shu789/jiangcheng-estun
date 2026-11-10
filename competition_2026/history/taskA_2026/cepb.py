#!/usr/bin/env python3
"""CEPB Subset 1 数据访问、GT 框解析与 IoU 工具。

压缩包用的是 Deflate64（Python 标准库 zipfile 不支持），所以统一走
Info-ZIP 的 unzip 命令读取。

GT yaml 里的关键字段：
    projected_cuboid: 3D 包围盒 8 个角点投影到图像上的 (x, y) 像素坐标。
                       原始 y 以图像底边为原点；GtObject 会将其转换为常见的
                       图像坐标（左上角为原点）。
    visibility:       可见比例（0~1），越小遮挡越严重
我们取 8 个点的凸包作为物体的真值轮廓，凸包的轴对齐外接矩形是 AABB 框。
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import yaml

ARCHIVE = os.environ.get(
    "CEPB_ARCHIVE", "/media/sunshine/新加卷/subset1_rand.zip"
)
# 所有任务产物（样张、报告、后续数据集/训练）都放数据盘
WORKDIR = Path(os.environ.get("TASK_A_ROOT", "/media/sunshine/新加卷/taskA"))
CAMERAS = ("left", "middle", "right")
LIGHTS = ("directional_light", "point_light", "spot_light")


def resolve_device(requested: str = "auto") -> str:
    """解析训练/推理设备：'auto' 有可用 GPU 时用 '0'，否则回落 'cpu'。

    注意：Codex 沙箱内看不到 /dev/nvidia*，torch.cuda.is_available() 为 False，
    因此真正用 GPU 的命令必须在沙箱外执行（或显式 --device 0）。
    """
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:  # 没装 torch 时按 CPU 处理
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


# CEPB 的 Unity 仿真相机分辨率（K: f=3200, cx=1024, cy=768）
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536


def _check_archive() -> None:
    if not os.path.exists(ARCHIVE):
        raise FileNotFoundError(f"找不到数据包：{ARCHIVE}")


@lru_cache(maxsize=1)
def list_entries() -> Tuple[str, ...]:
    """返回压缩包内全部文件名（只读中央目录，很快）。"""
    _check_archive()
    out = subprocess.run(["unzip", "-Z1", ARCHIVE], check=True, capture_output=True)
    names = tuple(
        n for n in out.stdout.decode("utf-8", "replace").splitlines() if n
    )
    if not names:
        raise RuntimeError(f"{ARCHIVE} 里没有条目")
    return names


def read_entry(name: str) -> bytes:
    """读取压缩包内单个文件（Deflate64，只能靠 unzip 解压）。"""
    _check_archive()
    out = subprocess.run(
        ["unzip", "-p", ARCHIVE, name], check=True, capture_output=True
    )
    if not out.stdout:
        raise FileNotFoundError(f"压缩包内没有 {name}")
    return out.stdout


def gt_name(camera: str, scene: int) -> str:
    return f"GT_{camera}_camera_{scene}.yaml"


def rgb_name(camera: str, light: str, scene: int) -> str:
    return f"{camera}_camera_{light}_scene_{scene}_rgb.png"


def depth_name(camera: str, light: str, scene: int) -> str:
    return f"{camera}_camera_{light}_scene_{scene}_depth.png"


def normals_name(camera: str, light: str, scene: int) -> str:
    return f"{camera}_camera_{light}_scene_{scene}_normals.png"


def segmentation_name(camera: str, scene: int) -> str:
    return f"{camera}_camera_scene_{scene}_segmentation.png"


def scene_index(name: str) -> int:
    """从 GT_*_camera_<n>.yaml 里取场景号。"""
    base = os.path.basename(name)
    return int(base.rsplit("_", 1)[1].split(".")[0])


@dataclass(frozen=True)
class GtObject:
    name: str
    position: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]
    cuboid: Tuple[Tuple[float, float, float], ...]
    raw_projected_cuboid: Tuple[Tuple[float, float], ...]
    projected_cuboid: Tuple[Tuple[float, float], ...]
    visibility: float

    @property
    def hull(self) -> np.ndarray:
        """8 个投影点的凸包轮廓，形状 (N, 2)，float32。"""
        pts = np.asarray(self.projected_cuboid, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.convexHull(pts).reshape(-1, 2)

    @property
    def aabb(self) -> Tuple[float, float, float, float]:
        """轴对齐外接框 (x1, y1, x2, y2)。"""
        return aabb_of_polygon(self.projected_cuboid)

    def inside_frame(self, margin: float = 0.0) -> bool:
        """投影点是否落在图像范围内（允许 margin 像素容差）。"""
        for x, y in self.projected_cuboid:
            if not (
                -margin <= x <= IMAGE_WIDTH + margin
                and -margin <= y <= IMAGE_HEIGHT + margin
            ):
                return False
        return True


@dataclass(frozen=True)
class GtFrame:
    camera: str
    scene: int
    objects: Tuple[GtObject, ...]


def _as_float_list(values: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(v) for v in values)


def load_gt(camera: str, scene: int) -> GtFrame:
    return parse_gt_frame(camera, scene, read_entry(gt_name(camera, scene)))


def parse_gt_frame(camera: str, scene: int, raw: bytes) -> GtFrame:
    """从 GT yaml 字节流解析一帧。"""
    data = yaml.safe_load(raw)
    objects: List[GtObject] = []
    for name, item in (data.get("objects") or {}).items():
        visibility = item.get("visibility")
        if isinstance(visibility, (list, tuple)):
            visibility = float(visibility[0]) if visibility else 0.0
        else:
            visibility = float(visibility or 0.0)
        raw_projected = tuple(
            (float(p[0]), float(p[1])) for p in item["projected_cuboid"]
        )
        # CEPB 的 projected_cuboid 使用以图像底边为原点的 y 坐标。
        # 转成 OpenCV/YOLO 使用的左上角原点坐标。
        projected = tuple(
            (float(x), float(IMAGE_HEIGHT - y)) for x, y in raw_projected
        )
        objects.append(
            GtObject(
                name=str(name),
                position=_as_float_list(item["position"]),
                quaternion=_as_float_list(item["quaternion"]),
                cuboid=tuple(_as_float_list(p) for p in item["cuboid"]),
                raw_projected_cuboid=raw_projected,
                projected_cuboid=projected,
                visibility=visibility,
            )
        )
    objects.sort(key=lambda obj: obj.name)
    return GtFrame(camera=camera, scene=scene, objects=tuple(objects))


# ---------------------------------------------------------------- 几何 / IoU


def aabb_of_polygon(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def aabb_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_aabb(a: Sequence[float], b: Sequence[float]) -> float:
    """两个轴对齐框的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aabb_area(a) + aabb_area(b) - inter
    return inter / union if union > 0 else 0.0


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return abs(float(cv2.contourArea(pts)))


def iou_polygon(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """两个凸多边形的 IoU（用 OpenCV 求交）。"""
    pa = np.asarray(a, dtype=np.float32).reshape(-1, 1, 2)
    pb = np.asarray(b, dtype=np.float32).reshape(-1, 1, 2)
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    union = polygon_area(a) + polygon_area(b) - float(inter)
    return float(inter) / union if union > 0 else 0.0


def shift_aabb(box: Sequence[float], dx: float, dy: float) -> Tuple[float, float, float, float]:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def scale_aabb(box: Sequence[float], factor: float) -> Tuple[float, float, float, float]:
    """以框中心为原点缩放。"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half_w, half_h = (box[2] - box[0]) / 2 * factor, (box[3] - box[1]) / 2 * factor
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def class_from_object_name(name: str) -> str:
    """'9-Mug' -> 'Mug'，'21_Haribo_gold_bear' 保持原样。"""
    if "-" in name:
        return name.split("-", 1)[1]
    return name
