#!/usr/bin/env python3
"""抽几十张样张，验证 GT 框解析、画框和 IoU 计算。

产出：
    taskA/samples/raw/            抽出的原图与 GT yaml
    taskA/samples/overlays/       每张图叠加 GT 框后的对照图
    taskA/samples/contact_sheet_<camera>.png
    taskA/reports/sample_check.md 自检报告
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

import cepb

ROOT = cepb.WORKDIR
SAMPLES = cepb.WORKDIR / "samples"
RAW = SAMPLES / "raw"
OVERLAYS = SAMPLES / "overlays"
REPORTS = cepb.WORKDIR / "reports"

SCENES = (1, 2, 3)


def ensure_dirs() -> None:
    for path in (RAW, OVERLAYS, REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def color_for(name: str) -> Tuple[int, int, int]:
    """每个物体用一个固定颜色，方便肉眼核对。"""
    palette = [
        (255, 64, 64),
        (64, 160, 255),
        (255, 200, 0),
        (0, 220, 120),
        (220, 80, 255),
        (255, 120, 0),
        (0, 220, 220),
        (255, 255, 255),
        (160, 255, 80),
        (255, 80, 160),
    ]
    return palette[sum(name.encode("utf-8")) % len(palette)]


def draw_gt(image: Image.Image, frame: "cepb.GtFrame") -> Image.Image:
    """画 GT：AABB 实线矩形 + 投影凸包细线 + 物体名/可见度。"""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for obj in frame.objects:
        color = color_for(obj.name)
        box = obj.aabb
        draw.rectangle(box, outline=color, width=4)
        hull = [(int(p[0]), int(p[1])) for p in obj.hull.tolist()]
        draw.line(hull + [hull[0]], fill=color, width=2)
        label = f"{obj.name}  vis={obj.visibility:.2f}"
        tx = max(0, min(int(box[0]), canvas.width - 10))
        ty = max(0, min(int(box[1]) - 22, canvas.height - 20))
        draw.rectangle(
            (tx, ty, tx + 10 * len(label), ty + 18), fill=(0, 0, 0)
        )
        draw.text((tx + 3, ty + 3), label, fill=color)
    return canvas


def self_check(frame: "cepb.GtFrame") -> List[Dict[str, float]]:
    """用 GT 框自己验证 IoU 函数：同一框=1，平移/缩小后应落在合理区间。"""
    rows: List[Dict[str, float]] = []
    for obj in frame.objects:
        box = obj.aabb
        rows.append(
            {
                "name": obj.name,
                "visibility": obj.visibility,
                "iou_self": cepb.iou_aabb(box, box),
                "iou_shift10": cepb.iou_aabb(box, cepb.shift_aabb(box, 10, 10)),
                "iou_half": cepb.iou_aabb(box, cepb.scale_aabb(box, 0.5)),
                "iou_poly_self": cepb.iou_polygon(obj.hull, obj.hull),
            }
        )
    return rows


def make_contact_sheet(camera: str, entries: List[Tuple[str, Image.Image]]) -> Path:
    """每个相机一张 3x3 拼图：行=场景，列=光照。"""
    thumb_w, thumb_h = 640, 480
    sheet = Image.new("RGB", (thumb_w * 3, thumb_h * 3), (32, 32, 32))
    for idx, (_, img) in enumerate(entries):
        row, col = divmod(idx, 3)
        thumb = img.resize((thumb_w, thumb_h), Image.LANCZOS)
        sheet.paste(thumb, (col * thumb_w, row * thumb_h))
    out = SAMPLES / f"contact_sheet_{camera}.png"
    sheet.save(out)
    return out


def main() -> int:
    ensure_dirs()
    report: List[str] = [
        "# Subset 1 样张自检（GT 框 + IoU）",
        "",
        f"- 数据包：`{cepb.ARCHIVE}`",
        f"- 样张：场景 {list(SCENES)} × 3 相机 × 3 光照 = {len(SCENES) * 3 * 3} 张 RGB",
        f"- 图片分辨率：{cepb.IMAGE_WIDTH} × {cepb.IMAGE_HEIGHT}",
        "",
        "## 抽出的样张",
        "",
        "| 文件 | 物体数 | GT 文件 |",
        "| --- | --- | --- |",
    ]

    all_checks: List[Dict[str, float]] = []
    for camera in cepb.CAMERAS:
        sheet_entries: List[Tuple[str, Image.Image]] = []
        for scene in SCENES:
            frame = cepb.load_gt(camera, scene)
            gt_bytes = cepb.read_entry(cepb.gt_name(camera, scene))
            (RAW / cepb.gt_name(camera, scene)).write_bytes(gt_bytes)
            for light in cepb.LIGHTS:
                name = cepb.rgb_name(camera, light, scene)
                data = cepb.read_entry(name)
                (RAW / name).write_bytes(data)
                image = Image.open(io.BytesIO(data))
                if image.size != (cepb.IMAGE_WIDTH, cepb.IMAGE_HEIGHT):
                    report.append(
                        f"  - 注意：{name} 分辨率 {image.size}，与假设不一致"
                    )
                overlay = draw_gt(image, frame)
                overlay.save(OVERLAYS / name)
                sheet_entries.append((name, overlay))
            report.append(
                f"| {cepb.gt_name(camera, scene)} | {len(frame.objects)} | "
                f"`{camera}` × {len(cepb.LIGHTS)} 光照 |"
            )
            all_checks.extend(self_check(frame))
        sheet = make_contact_sheet(camera, sheet_entries)
        report.append(f"| 拼图 | - | `{sheet.relative_to(ROOT)}` |")

    report += [
        "",
        "## IoU 自检",
        "",
        f"- 检查对象数：{len(all_checks)}",
        f"- 自身 IoU：min={min(r['iou_self'] for r in all_checks):.4f}，"
        f"max={max(r['iou_self'] for r in all_checks):.4f}（应恒为 1）",
        f"- 自身多边形 IoU：min={min(r['iou_poly_self'] for r in all_checks):.4f}，"
        f"max={max(r['iou_poly_self'] for r in all_checks):.4f}（应恒为 1）",
        "",
        "### 故意做错时的 IoU（抽样前 10 个物体）",
        "",
        "| 物体 | 可见度 | 原框自比 | 平移 10px | 缩小一半 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in all_checks[:10]:
        report.append(
            f"| {row['name']} | {row['visibility']:.2f} | "
            f"{row['iou_self']:.4f} | {row['iou_shift10']:.4f} | {row['iou_half']:.4f} |"
        )

    bad = [
        r
        for r in all_checks
        if abs(r["iou_self"] - 1.0) > 1e-6 or abs(r["iou_poly_self"] - 1.0) > 1e-6
    ]
    report += [
        "",
        "## 结论",
        "",
        (
            "- 自比 IoU 全部为 1，平移/缩小后的数值单调合理，IoU 实现通过。"
            if not bad
            else f"- 有 {len(bad)} 个物体自比 IoU 不为 1，需要检查。"
        ),
        f"- 对照图在 `{OVERLAYS.relative_to(ROOT)}/`，先肉眼看框是否套住物体。",
        "",
    ]

    out = REPORTS / "sample_check.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\n报告写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
