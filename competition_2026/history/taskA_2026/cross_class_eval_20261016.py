#!/usr/bin/env python3
"""Evaluate cross-class false positives on the 2026-10-16 real-camera set."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path("/media/sunshine/新加卷/taskA/runs/yolo11s_real_ft_v23_hardneg_20261016/weights/epoch3.pt")
ROOT = Path("/media/sunshine/新加卷/taskA/eval_capture_20261016")
OUT = Path("/media/sunshine/新加卷/taskA/reports/camera_eval_20261016/cross_class_matrix_20261016")
CONF = 0.25
IMGSZ = 640
LABELS = [
    ("1-Cheez-it", 0),
    ("2-Starkist_Tuna", 1),
    ("3-Scissors", 2),
    ("4-Frenchs_Mustard", 3),
    ("5-Tomato_Soup", 4),
    ("6-Foam_Brick", 5),
    ("7-Clamp", 6),
    ("8-Plastic_Banana", 7),
    ("9-Mug", 8),
    ("10-meat_can", 9),
]


def extract(result) -> list[dict]:
    out = []
    boxes = result.boxes
    if boxes is None:
        return out
    names = result.names
    for box in boxes:
        cid = int(box.cls.item())
        out.append({
            "class_id": cid,
            "name": names.get(cid, str(cid)) if isinstance(names, dict) else names[cid],
            "conf": round(float(box.conf.item()), 4),
            "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    rows = []
    class_summary = {}

    for label, true_cid in LABELS:
        folder = ROOT / label
        images = sorted(folder.glob("*.jpg")) if folder.exists() else []
        if not images:
            class_summary[label] = {"status": "missing", "total": 0}
            continue
        source = [str(p) for p in images]
        common = dict(imgsz=IMGSZ, conf=CONF, iou=0.7, device=0, verbose=False, max_det=100)
        base_results = model.predict(source=source, augment=False, **common)
        tta_results = model.predict(source=source, augment=True, **common)
        base_wrong_frames = 0
        tta_wrong_frames = 0
        union_wrong_frames = 0
        base_correct_frames = 0
        tta_correct_frames = 0
        union_correct_frames = 0
        base_wrong_dets = 0
        tta_wrong_dets = 0
        base_confusions: Counter[int] = Counter()
        tta_confusions: Counter[int] = Counter()
        for image, br, tr in zip(images, base_results, tta_results):
            bd = extract(br)
            td = extract(tr)
            b_correct = any(d["class_id"] == true_cid for d in bd)
            t_correct = any(d["class_id"] == true_cid for d in td)
            b_wrong = [d for d in bd if d["class_id"] != true_cid]
            t_wrong = [d for d in td if d["class_id"] != true_cid]
            base_correct_frames += b_correct
            tta_correct_frames += t_correct
            union_correct_frames += b_correct or t_correct
            base_wrong_frames += bool(b_wrong)
            tta_wrong_frames += bool(t_wrong)
            union_wrong_frames += bool(b_wrong or t_wrong)
            base_wrong_dets += len(b_wrong)
            tta_wrong_dets += len(t_wrong)
            base_confusions.update(d["class_id"] for d in b_wrong)
            tta_confusions.update(d["class_id"] for d in t_wrong)
            rows.append({
                "source_class": label,
                "true_class_id": true_cid,
                "file": image.name,
                "base_correct": b_correct,
                "tta_correct": t_correct,
                "base_wrong": b_wrong,
                "tta_wrong": t_wrong,
                "base_detections": bd,
                "tta_detections": td,
            })
        def conf_text(counter: Counter[int]) -> str:
            if not counter:
                return "无"
            return ", ".join(f"{LABELS[cid][0]}×{n}" for cid, n in counter.most_common())
        class_summary[label] = {
            "status": "ok",
            "true_class_id": true_cid,
            "total": len(images),
            "base_correct_frames": base_correct_frames,
            "tta_correct_frames": tta_correct_frames,
            "union_correct_frames": union_correct_frames,
            "base_wrong_frames": base_wrong_frames,
            "tta_wrong_frames": tta_wrong_frames,
            "union_wrong_frames": union_wrong_frames,
            "base_wrong_detections": base_wrong_dets,
            "tta_wrong_detections": tta_wrong_dets,
            "base_confusions": {str(k): v for k, v in base_confusions.items()},
            "tta_confusions": {str(k): v for k, v in tta_confusions.items()},
            "base_confusions_text": conf_text(base_confusions),
            "tta_confusions_text": conf_text(tta_confusions),
        }
        print(label, class_summary[label], flush=True)

    summary = {
        "weights": str(WEIGHTS),
        "imgsz": IMGSZ,
        "conf": CONF,
        "note": "Each folder is treated as a single-class capture. Any model class other than the folder's true class is counted as a false positive. Banana was not captured.",
        "classes": class_summary,
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# 相机真实集交叉误报矩阵（v23 epoch3）",
        "",
        f"- 权重：{WEIGHTS}",
        f"- 设置：imgsz={IMGSZ}, conf={CONF}",
        "- 口径：每个文件夹按单一目标真实类别统计；任何其他类别检出都算该帧交叉误报。",
        "- 说明：香蕉尚未采集，因此标注为 missing。",
        "",
        "| 真实类别 | 张数 | 单帧正确 | TTA正确 | 单帧错报帧 | TTA错报帧 | 单帧主要错报 | TTA主要错报 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for label, _ in LABELS:
        s = class_summary[label]
        if s["status"] != "ok":
            md.append(f"| {label} | 0 | - | - | - | - | 未采集 | 未采集 |")
            continue
        md.append(
            f"| {label} | {s['total']} | {s['base_correct_frames']}/{s['total']} | "
            f"{s['tta_correct_frames']}/{s['total']} | {s['base_wrong_frames']}/{s['total']} | "
            f"{s['tta_wrong_frames']}/{s['total']} | {s['base_confusions_text']} | {s['tta_confusions_text']} |"
        )
    md += [
        "",
        "## 重点结论",
        "",
    ]
    ok = [(label, s) for label, s in class_summary.items() if s.get("status") == "ok"]
    worst = sorted(ok, key=lambda x: (-x[1]["base_wrong_frames"], x[0]))
    for label, s in worst:
        if s["base_wrong_frames"]:
            md.append(
                f"- {label}：单帧 {s['base_wrong_frames']}/{s['total']} 帧出现交叉错报；"
                f"TTA {s['tta_wrong_frames']}/{s['total']} 帧；单帧主要错报为 {s['base_confusions_text']}。"
            )
    if not any(s["base_wrong_frames"] for _, s in ok):
        md.append("- 没有类别出现跨类误报。")
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    csv_lines = ["source_class,file,base_correct,tta_correct,base_wrong_classes,tta_wrong_classes"]
    for r in rows:
        bw = "|".join(repr(d) for d in r["base_wrong"])
        tw = "|".join(repr(d) for d in r["tta_wrong"])
        csv_lines.append(f'{r["source_class"]},{r["file"]},{int(r["base_correct"])},{int(r["tta_correct"])},'
                         f'"{bw.replace(chr(34), chr(39))}","{tw.replace(chr(34), chr(39))}"')
    (OUT / "summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print("OUT", OUT)


if __name__ == "__main__":
    main()
