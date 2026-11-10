#!/usr/bin/env python3
"""Offline mesh renderer + compositor for the Easton task-A 10-class set.

Why: there is no historically-unseen real image for Mug / Tuna (all existing
images are in v17b..v24 train). This tool renders the official competition
meshes (models_zip) at swept azimuth / elevation / distance and composites the
object onto real empty-table plates, producing a synthesised holdout that can
probe pose & scale robustness of the real-photo-trained detectors.

It never trains and never touches production code.

Usage (core check):
    render_mesh_scenes.py --obj <textured.obj> --out /tmp/mug.png \
        --az 30 --el 25 --size 512
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


# ---------- OBJ / MTL parsing ----------
def load_obj(obj_path: Path):
    verts, uvs, norms, faces = [], [], [], []
    mtl_file = None
    for raw in obj_path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("mtllib"):
            mtl_file = line.split(maxsplit=1)[1].strip()
        elif line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            parts = line.split()
            uvs.append([float(parts[1]), float(parts[2])])
        elif line.startswith("vn "):
            norms.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            toks = line.split()[1:]
            fi = []
            for t in toks:
                comp = t.split("/")
                vi = int(comp[0]) - 1
                ti = int(comp[1]) - 1 if len(comp) > 1 and comp[1] != "" else -1
                ni = int(comp[2]) - 1 if len(comp) > 2 and comp[2] != "" else -1
                fi.append((vi, ti, ni))
            # fan triangulate
            for k in range(1, len(fi) - 1):
                faces.append((fi[0], fi[k], fi[k + 1]))
    verts = np.asarray(verts, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64) if uvs else np.zeros((0, 2))
    norms = np.asarray(norms, dtype=np.float64) if norms else np.zeros((0, 3))
    tex_path = None
    if mtl_file:
        mtl_path = obj_path.parent / mtl_file
        if mtl_path.exists():
            for raw in mtl_path.read_text(errors="ignore").splitlines():
                s = raw.strip()
                if s.lower().startswith("map_kd"):
                    tex_path = obj_path.parent / s.split(maxsplit=1)[1].strip()
    return verts, uvs, norms, faces, tex_path


def normalize_mesh(verts, faces):
    """Center at centroid of bbox and scale so max extent = 1.0 (Y up)."""
    lo, hi = verts.min(0), verts.max(0)
    ctr = (lo + hi) / 2.0
    v = verts - ctr
    scale = 1.0 / max(hi - lo)
    v = v * scale
    return v


def look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = r, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def render(verts, uvs, faces, tex, az_deg, el_deg, dist, size, focal, bg=None,
           light_dir=None, ambient=0.35, obj_rot=None):
    """Return (rgb float array size x size x 3, alpha mask)."""
    # optional extra object rotation (3x3)
    V = verts.copy()
    if obj_rot is not None:
        V = V @ obj_rot.T
    az, el = math.radians(az_deg), math.radians(el_deg)
    eye = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)]) * dist
    up = np.array([0.0, 1.0, 0.0])
    view = look_at(eye, np.zeros(3), up)
    # camera intrinsic
    cx = cy = size / 2.0
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
    # transform verts to camera
    Vh = np.hstack([V, np.ones((V.shape[0], 1))])
    Vc = (view @ Vh.T).T[:, :3]
    # project
    depth = -Vc[:, 2]
    valid = depth > 1e-6
    p = np.zeros((V.shape[0], 2))
    d = np.where(valid, depth, 1.0)
    p[:, 0] = K[0, 0] * (Vc[:, 0] / d) + K[0, 2]
    p[:, 1] = K[1, 2] - K[1, 1] * (Vc[:, 1] / d)
    zz = depth

    if light_dir is None:
        light_dir = np.array([0.35, 0.8, 0.45])
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-9)

    if bg is None:
        bg = np.ones((size, size, 3)) * 1.0
    color = bg.copy()
    zbuf = np.full((size, size), np.inf)

    TH, TW = tex.shape[:2] if tex is not None else (1, 1)
    for tri in faces:
        idx = [t[0] for t in tri]
        if not all(valid[i] for i in idx):
            continue
        P = p[idx]
        Z = zz[idx]
        # normal (world) for shading
        v0, v1, v2 = V[idx[0]], V[idx[1]], V[idx[2]]
        n = np.cross(v1 - v0, v2 - v0)
        nl = np.linalg.norm(n)
        if nl < 1e-12:
            continue
        n = n / nl
        shade = ambient + (1 - ambient) * max(0.0, float(n @ light_dir))
        # uv per vertex
        tis = [t[1] for t in tri]
        UV = np.zeros((3, 2))
        for k, ti in enumerate(tis):
            if ti >= 0 and ti < len(uvs):
                UV[k] = uvs[ti]
        x0 = max(int(np.floor(P[:, 0].min())), 0)
        x1 = min(int(np.ceil(P[:, 0].max())) + 1, size)
        y0 = max(int(np.floor(P[:, 1].min())), 0)
        y1 = min(int(np.ceil(P[:, 1].max())) + 1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        # sub-sample grid (2x) to reduce aliasing / holes
        gx, gy = np.meshgrid(xs + 0.5, ys + 0.5)
        ax, ay = P[0]; bx, by = P[1]; cx2, cy2 = P[2]
        d = (by - cy2) * (ax - cx2) + (cx2 - bx) * (ay - cy2)
        if abs(d) < 1e-9:
            continue
        w0 = ((by - cy2) * (gx - cx2) + (cx2 - bx) * (gy - cy2)) / d
        w1 = ((cy2 - ay) * (gx - cx2) + (ax - cx2) * (gy - cy2)) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        depth = w0 * Z[0] + w1 * Z[1] + w2 * Z[2]
        sub = zbuf[y0:y1, x0:x1]
        mask = inside & (depth < sub)
        if not mask.any():
            continue
        sub[mask] = depth[mask]
        u = w0 * UV[0, 0] + w1 * UV[1, 0] + w2 * UV[2, 0]
        vv = w0 * UV[0, 1] + w1 * UV[1, 1] + w2 * UV[2, 1]
        uu = np.clip((u * (TW - 1)).astype(np.int32), 0, TW - 1)
        vvi = np.clip(((1.0 - vv) * (TH - 1)).astype(np.int32), 0, TH - 1)
        if tex is not None:
            tc = tex[vvi, uu].astype(np.float64) / 255.0
        else:
            tc = np.ones((gx.shape[0], gx.shape[1], 3))
        tc = tc * shade
        tgt = color[y0:y1, x0:x1]
        tgt[mask] = np.clip(tc[mask], 0, 1)
        color[y0:y1, x0:x1] = tgt
    out = np.clip(color, 0, 1)
    alpha = (zbuf < np.inf).astype(np.uint8) * 255
    return out, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--az", type=float, default=30.0)
    ap.add_argument("--el", type=float, default=25.0)
    ap.add_argument("--dist", type=float, default=3.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--focal", type=float, default=None)
    args = ap.parse_args()

    verts, uvs, norms, faces, tex_path = load_obj(args.obj)
    verts = normalize_mesh(verts, faces)
    tex = None
    if tex_path and tex_path.exists():
        bgr = cv2.imread(str(tex_path), cv2.IMREAD_COLOR)
        tex = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    focal = args.focal or args.size * 1.2
    rgb, alpha = render(verts, uvs, faces, tex, args.az, args.el, args.dist,
                        args.size, focal)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.out), bgr)
    # also alpha preview
    ap_path = args.out.with_name(args.out.stem + "_alpha.png")
    cv2.imwrite(str(ap_path), alpha)
    print(f"[saved] {args.out}  faces={len(faces)} verts={len(verts)}")


if __name__ == "__main__":
    raise SystemExit(main())
