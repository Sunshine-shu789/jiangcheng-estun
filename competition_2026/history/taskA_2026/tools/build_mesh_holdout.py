#!/usr/bin/env python3
"""Build a synthetic pose/scale holdout by rendering official meshes onto real
empty-table plates, then writing a YOLO-format eval dataset.

Purpose: the v25 audit shows there is NO historically-unseen real image for
Mug (0) or Tuna (1, and that one is group-leaked). This synthesises an
evaluation-only holdout over azimuth / elevation / distance so the real-photo
detectors can be probed for pose & scale robustness.

NEVER use this for training. It is an evaluation probe, not real data.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_mesh_scenes import load_obj, normalize_mesh, render  # noqa: E402

CLASSES = [
    ("1-Cheez-it", 0), ("2-Starkist_Tuna", 1), ("3-Scissors", 2),
    ("4-Frenchs_Mustard", 3), ("5-Tomato_Soup", 4), ("6-Foam_Brick", 5),
    ("7-Clamp", 6), ("8-Plastic_Banana", 7), ("9-Mug", 8), ("10-meat_can", 9),
]
CLASS_TO_ID = {n: i for n, i in CLASSES}


def load_tables(bg_dir: Path, size=(640, 480)):
    plates = []
    for p in sorted(bg_dir.glob("*.jpg")):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        # resize keeping aspect to cover target then center-crop
        tw, th = size
        s = max(tw / w, th / h)
        im = cv2.resize(im, (int(round(w * s)), int(round(h * s))))
        h2, w2 = im.shape[:2]
        x0 = (w2 - tw) // 2
        y0 = (h2 - th) // 2
        im = im[y0:y0 + th, x0:x0 + tw]
        plates.append(im)
    return plates


def render_object(obj_path: Path, az, el, dist, tex_cache):
    key = str(obj_path)
    if key not in tex_cache:
        verts, uvs, norms, faces, tex_path = load_obj(obj_path)
        verts = normalize_mesh(verts, faces)
        tex = None
        if tex_path and tex_path.exists():
            bgr = cv2.imread(str(tex_path), cv2.IMREAD_COLOR)
            tex = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tex_cache[key] = (verts, uvs, faces, tex)
    verts, uvs, faces, tex = tex_cache[key]
    size = 512
    focal = size * 1.15
    # camera distance fixed; apparent size handled by composite scale
    cam_dist = 3.0
    rgb, alpha = render(verts, uvs, faces, tex, az, el, cam_dist, size, focal)
    return rgb, alpha


def crop_to_alpha(rgb, alpha, pad=6):
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return rgb, alpha, (0, 0, 0, 0)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgb.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgb.shape[0])
    return rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1], (x0, y0, x1, y1)


def composite(table, rgb, alpha, scale_h, rng, shadow=True):
    H, W = table.shape[:2]
    obj_h = max(24, int(H * scale_h))
    ah, aw = alpha.shape[:2]
    obj_w = max(18, int(round(aw * obj_h / ah)))
    # fit inside frame
    fit = min(1.0, (W - 16) / obj_w, (H - 16) / obj_h)
    if fit < 1.0:
        obj_w = max(18, int(obj_w * fit))
        obj_h = max(24, int(obj_h * fit))
    # render() returns RGB but the plate is BGR -> convert before pasting
    rgb255 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)[..., ::-1].copy()
    rgb_s = cv2.resize(rgb255, (obj_w, obj_h), interpolation=cv2.INTER_AREA)
    a_s = cv2.resize(alpha, (obj_w, obj_h), interpolation=cv2.INTER_AREA)
    # keep object from touching frame edges
    max_x = max(1, W - obj_w - 8)
    max_y = max(1, H - obj_h - 8)
    x = rng.randint(8, max_x) if max_x > 8 else 0
    y = rng.randint(8, max_y) if max_y > 8 else 0
    out = table.astype(np.float32)
    if shadow:
        sh = np.zeros((H, W), np.float32)
        cv2.ellipse(sh, (x + obj_w // 2, min(H - 2, y + int(obj_h * 0.94))),
                    (max(6, int(obj_w * 0.55)), max(4, int(obj_h * 0.12))),
                    0, 0, 360, 0.45, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), max(3, obj_w * 0.05))
        alpha_sh = sh[..., None]
        out = out * (1 - alpha_sh) + (out * 0.55) * alpha_sh
    a = (a_s.astype(np.float32) / 255.0)[..., None]
    region = out[y:y + obj_h, x:x + obj_w]
    region = region * (1 - a) + rgb_s.astype(np.float32) * a
    out[y:y + obj_h, x:x + obj_w] = region
    box = (x, y, x + obj_w, y + obj_h)
    return np.clip(out, 0, 255).astype(np.uint8), box


def yolo_line(cls_id, box, W, H):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path,
                    default=Path("/media/sunshine/新加卷/埃斯顿比赛_2026/easton/datasets/models_zip/models"))
    ap.add_argument("--bg", type=Path,
                    default=Path("/media/sunshine/新加卷/taskA/real_raw/empty_background"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--size", type=str, default="640x480")
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--classes", type=str, default=",".join(n for n, _ in CLASSES))
    args = ap.parse_args()

    W, H = (int(v) for v in args.size.lower().split("x"))
    rng = random.Random(args.seed)
    plates = load_tables(args.bg, (W, H))
    if not plates:
        raise SystemExit(f"no background plates in {args.bg}")
    want = [c.strip() for c in args.classes.split(",") if c.strip()]

    out = args.out.expanduser().resolve()
    img_dir = out / "images" / "val"
    lbl_dir = out / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    tex_cache = {}
    records = []
    az_grid = [0, 45, 90, 135, 180, 225, 270, 315]
    el_grid = [12, 30, 55, 75]
    dist_grid = [0.62, 0.82, 1.05, 1.35]
    scales = {0.62: 0.55, 0.82: 0.42, 1.05: 0.30, 1.35: 0.20}

    for cname in want:
        if cname not in CLASS_TO_ID:
            print(f"[skip] unknown class {cname}")
            continue
        cls_id = CLASS_TO_ID[cname]
        obj_path = args.models / cname / "textured.obj"
        if not obj_path.exists():
            print(f"[skip] no mesh for {cname}")
            continue
        made = 0
        combo_iter = [(az, el, ds) for ds in dist_grid for el in el_grid for az in az_grid]
        rng.shuffle(combo_iter)
        for az, el, ds in combo_iter:
            if made >= args.per_class:
                break
            rgb, alpha = render_object(obj_path, az, el, ds, tex_cache)
            if alpha.max() == 0:
                continue
            rgb_c, a_c, _ = crop_to_alpha(rgb, alpha)
            table = rng.choice(plates).copy()
            img, box = composite(table, rgb_c, a_c, scales[ds], rng)
            name = f"mesh_{cname}_{made:03d}_az{int(az)}_el{int(el)}_d{ds:.2f}.jpg"
            cv2.imwrite(str(img_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lbl_dir / (Path(name).stem + ".txt")).write_text(
                yolo_line(cls_id, box, W, H) + "\n", encoding="utf-8")
            records.append({"image": name, "class": cname, "az": az, "el": el, "dist": ds, "box": box})
            made += 1
        print(f"[{cname}] {made} images")

    names = [n for n, _ in CLASSES]
    (out / "data.yaml").write_text(
        "path: " + str(out) + "\n"
        "val: images/val\n"
        "nc: 10\n"
        "names:\n" + "".join(f"- {n}\n" for n in names),
        encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out}  total={len(records)}")


if __name__ == "__main__":
    raise SystemExit(main())
