#!/usr/bin/env python3
"""把汤罐 14 组姿态的复核框写成 YOLO 标签和审计文件。

所有框都已按 640x480 原图逐组复核。g10_multi 同框还有
French's Mustard(class 3) 与 meat_can(class 9)，一并标注，避免在微调时
被当成背景。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/sunshine/codex/easton/taskA")
GROUPS = ROOT / "scratch/soup_groups.json"
OUT_LABELS = ROOT / "prelabels_soup/labels_final"
OUT_BOXES = ROOT / "prelabels_soup/manual_boxes_soup.json"
OUT_AUDIT = ROOT / "prelabels_soup/label_audit.csv"

# class-id, x1, y1, x2, y2 (原图像素坐标)
GROUP_BOXES: dict[str, list[list[int]]] = {
    "g01_mid": [[4, 194, 68, 365, 362]],
    "g02_near": [[4, 150, 84, 314, 424]],
    "g03_far": [[4, 236, 135, 324, 279]],
    "g04_left": [[4, 118, 75, 256, 300]],
    "g05_right": [[4, 245, 70, 430, 307]],
    "g06_side": [[4, 52, 162, 392, 288]],
    "g07_backlying": [[4, 28, 146, 377, 311]],
    "g08_backup": [[4, 148, 69, 328, 355]],
    "g09_top": [[4, 150, 67, 360, 354]],
    "g10_multi": [
        [3, 130, 88, 235, 315],
        [4, 278, 104, 398, 308],
        [9, 410, 202, 548, 300],
    ],
    "g11_mat_edge_lying": [[4, 250, 95, 482, 291]],
    "g12_white_tilt": [[4, 240, 70, 520, 378]],
    "g13_white_upright_back": [[4, 134, 38, 384, 362]],
    "g14_white_light_close": [[4, 133, 62, 387, 338]],
}
WIDTH, HEIGHT = 640, 480
CLASS_NAMES = {
    3: "Frenchs_Mustard",
    4: "Tomato_Soup",
    9: "meat_can",
}


def yolo_line(box: list[int]) -> str:
    class_id, x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= WIDTH and 0 <= y1 < y2 <= HEIGHT):
        raise ValueError(f"invalid box: {box}")
    cx = (x1 + x2) / 2 / WIDTH
    cy = (y1 + y2) / 2 / HEIGHT
    width = (x2 - x1) / WIDTH
    height = (y2 - y1) / HEIGHT
    return f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def main() -> int:
    groups = json.loads(GROUPS.read_text(encoding="utf-8"))
    if set(groups) != set(GROUP_BOXES):
        raise SystemExit("group definitions do not match soup_groups.json")

    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    boxes_out: dict[str, list[list[int]]] = {}
    audit_rows: list[dict[str, object]] = []

    for group, names in groups.items():
        boxes = GROUP_BOXES[group]
        if len(names) != 15:
            raise SystemExit(f"{group}: expected 15 images, got {len(names)}")
        for name in names:
            stem = Path(name).stem
            (OUT_LABELS / f"{stem}.txt").write_text(
                "\n".join(yolo_line(box) for box in boxes) + "\n",
                encoding="utf-8",
            )
            boxes_out[name] = boxes
            audit_rows.append(
                {
                    "group": group,
                    "image": name,
                    "boxes": ";".join(
                        f"{box[0]}:{box[1]},{box[2]},{box[3]},{box[4]}"
                        for box in boxes
                    ),
                }
            )

    if len(boxes_out) != 210:
        raise SystemExit(f"expected 210 labels, got {len(boxes_out)}")

    OUT_BOXES.write_text(
        json.dumps(boxes_out, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    with OUT_AUDIT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "image", "boxes"])
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"wrote {len(boxes_out)} labels -> {OUT_LABELS}")
    print(f"boxes -> {OUT_BOXES}")
    print(f"audit -> {OUT_AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
