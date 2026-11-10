#!/usr/bin/env python3
"""Render residual failures for the recommended pipeline (vote3+wbf+arb*).

GT = green, fused prediction = orange (class name + conf). Only frames where the
recommended pipeline still fails an instance, or still emits a wrong-class box.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from arbiter_experiment_20261016 import (
    CLASSES, HUMAN, REP, EVAL, attractor_suppress, best_iou, calibrate, det,
    load_gt, size, wbf,
)
from multiview_arbiter_20261016 import VIEWS, as_dets, vote_filter

SRC = REP / "multiview_predictions_20261016"
OUT = REP / "recommended_pipeline_failures_20261016"
PASS = 0.75


def font(n: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, n)
    return ImageFont.load_default()


def render(img: Image.Image, gt: list, dets: list, title: str) -> Image.Image:
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    f = font(20)
    for g in gt:
        d.rectangle(g["xyxy"], outline=(0, 230, 0), width=4)
        d.text((g["xyxy"][0] + 4, g["xyxy"][1] + 4), "GT " + HUMAN.get(g["class_id"], str(g["class_id"])),
               fill=(0, 230, 0), font=f)
    for x in dets:
        d.rectangle(x["xyxy"], outline=(255, 140, 0), width=4)
        d.text((x["xyxy"][0] + 4, x["xyxy"][3] - 24),
               f"{HUMAN.get(x['class_id'], x['class_id'])} {x['conf']:.2f}", fill=(255, 140, 0), font=f)
    d.rectangle([0, 0, im.width, 30], fill=(0, 0, 0))
    d.text((6, 5), title, fill=(255, 255, 255), font=f)
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = {}
    for cid, folder, _ in CLASSES:
        summ = SRC / folder / "summary.json"
        if not summ.exists():
            continue
        rows = json.loads(summ.read_text(encoding="utf-8"))["rows"]
        thumbs = []
        for row in rows:
            fname = row["file"]
            img_path = EVAL / folder / fname
            w, h = size(img_path)
            gt = [g for g in load_gt(Path(fname).stem, w, h) if g["class_id"] == cid]
            if not gt:
                continue
            vd = {v: as_dets(row.get(v) or [], v) for v in VIEWS}
            vd["v1.00"] = calibrate(vd["v1.00"], "base", w, h)
            vd["v1.00_tta"] = calibrate(vd["v1.00_tta"], "tta", w, h)
            fused = attractor_suppress(wbf(vote_filter(vd, 3)))
            passed = sum(1 for g in gt if best_iou(g["xyxy"], fused, cid) >= PASS)
            wrong = [x for x in fused if x["class_id"] != cid]
            if passed < len(gt) or wrong:
                img = Image.open(img_path)
                title = (f"{folder} | {fname} | pass {passed}/{len(gt)}"
                         + (f" | wrong: {', '.join(HUMAN.get(x['class_id'], str(x['class_id'])) for x in wrong)}" if wrong else ""))
                thumbs.append(render(img, gt, fused, title))
        if thumbs:
            index[folder] = len(thumbs)
            cols = 3
            rws = (len(thumbs) + cols - 1) // cols
            tw, th = 480, 300
            sheet = Image.new("RGB", (tw * cols, th * rws), (30, 30, 30))
            for i, t in enumerate(thumbs):
                sheet.paste(t.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
            sheet.save(OUT / f"{folder}_residual_failures.jpg", quality=88)
            print(f"[{folder}] {len(thumbs)} residual frames -> {folder}_residual_failures.jpg")
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
