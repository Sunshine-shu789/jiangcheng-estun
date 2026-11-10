#!/usr/bin/env python3
"""在 v18 上做 class5 线性探针：只训练 Detect.cv3[i][-1] 的 class5 权重行 + 偏置。

设计要点
--------
- 除 class5 行的 387 个参数外，全部冻结；反向时把非 class5 行的梯度清零。
  ⇒ 旧 9 类的 logits 逐位不变，理论上不可能遗忘旧类（只可能因 class5 抢 argmax 而改变）。
- 预处理复刻服务口径：先按 scale ∈ {0.65,0.75} 缩放贴到原尺寸 114 灰底画布，再 letterbox。
- 正样本：GT 砖框内的 anchor 中心；负样本：其它类 GT 框内的 anchor（硬负样本）+ 背景。
- 损失：BCE + 排序 hinge（要求 class5 logit 压过其余类最大 logit）。
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------- 预处理（与服务一致） ----------
def prepared_canvas(frame: np.ndarray, scale: float) -> np.ndarray:
    h, w = frame.shape[:2]
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    rs = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    xo, yo = (w - sw) // 2, (h - sh) // 2
    canvas[yo : yo + sh, xo : xo + sw] = rs
    return canvas


def letterbox(img: np.ndarray, imgsz: int = 640, stride: int = 32):
    h, w = img.shape[:2]
    r = min(imgsz / w, imgsz / h)
    rw, rh = max(1, round(w * r)), max(1, round(h * r))
    nw = min(imgsz, int(np.ceil(rw / stride) * stride))
    nh = min(imgsz, int(np.ceil(rh / stride) * stride))
    left, top = (nw - rw) // 2, (nh - rh) // 2
    im = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    out = np.full((nh, nw, 3), 114, dtype=np.uint8)
    out[top : top + rh, left : left + rw] = im
    return out, r, left, top


def to_tensor(bgr: np.ndarray, device) -> torch.Tensor:
    rgb = bgr[:, :, ::-1]
    t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
    return (t / 255.0).unsqueeze(0).to(device)


def load_yolo_labels(path: Path, w: int, h: int):
    """返回 [(cls, x1, y1, x2, y2)] 像素坐标。"""
    out = []
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


class DataIndex:
    def __init__(self, root: Path, split: str = "train"):
        self.root = root
        self.img_dir = root / "images" / split
        self.lbl_dir = root / "labels" / split
        self.items = sorted(p for p in self.img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.cache = {}
        for p in self.items:
            im = cv2.imread(str(p))
            h, w = im.shape[:2]
            self.cache[p] = load_yolo_labels(self.lbl_dir / f"{p.stem}.txt", w, h)

    def with_class(self, cls: int):
        return [p for p in self.items if any(b[0] == cls for b in self.cache[p])]

    def without_class(self, cls: int):
        return [p for p in self.items if not any(b[0] == cls for b in self.cache[p])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="v18 基线权重")
    ap.add_argument("--init-row", default="", help="class5 行初值来源（如 graft 权重），留空=零初始化")
    ap.add_argument("--brick-root", required=True)
    ap.add_argument("--replay-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-pos", type=int, default=96, help="每图正 anchor 上限")
    ap.add_argument("--n-neg-obj", type=int, default=96, help="每图其它类物体 anchor 上限")
    ap.add_argument("--n-neg-bg", type=int, default=128, help="每图背景 anchor 上限")
    ap.add_argument("--replay-per-epoch", type=int, default=320)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--cls5", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-lo", type=float, default=0.65)
    ap.add_argument("--scale-hi", type=float, default=0.75)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)

    from ultralytics import YOLO

    net = YOLO(args.base).model.float().eval().to(dev)
    head = net.model[-1]
    rows = [head.cv3[i] for i in range(3)]          # 完整 cls 分支（前两块+最后 1x1）
    convs = [r[-1] for r in rows]                    # 最后 1x1：唯一可训练层
    nc = head.nc
    C5 = args.cls5

    for p in net.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        for ci, c in enumerate(convs):
            c.weight.requires_grad_(True)
            c.bias.requires_grad_(True)
            if args.init_row:
                src = torch.load(args.init_row, map_location="cpu", weights_only=False)
                sm = (src.get("ema") or src.get("model")).model[-1]
                idx = ci
                c.weight[C5].copy_(sm.cv3[idx][-1].weight[C5])
                c.bias[C5].copy_(sm.cv3[idx][-1].bias[C5])
            else:
                c.weight[C5].zero_()
                c.bias[C5].fill_(-4.6)
    n_train = sum(int(c.weight[C5].numel() + c.bias[C5].numel()) for c in convs)

    brick = DataIndex(Path(args.brick_root))
    replay = DataIndex(Path(args.replay_root))
    pos_imgs = brick.with_class(C5)
    neg_pool = replay.without_class(C5)
    if not pos_imgs:
        raise SystemExit(f"{args.brick_root} 无 class {C5} 样本")
    print(f"[data] 砖图 {len(pos_imgs)}，replay 池 {len(neg_pool)}，可训练参数 {n_train}")

    other_cols = [i for i in range(nc) if i != C5]
    # 注意：c.weight[C5] 是视图（非 leaf），不能直接交给优化器；
    # 改为优化整个张量、在 backward 后把非 C5 行的梯度清零（AdamW weight_decay=0 时不会被动到）。
    opt = torch.optim.AdamW([c.weight for c in convs] + [c.bias for c in convs], lr=args.lr, weight_decay=0.0)

    # 几何参数由第一张图确定（所有图同尺寸）
    probe_img = cv2.imread(str(pos_imgs[0]))
    canvas = prepared_canvas(probe_img, 0.75)
    _, r, left, top = letterbox(canvas, args.imgsz)
    with torch.no_grad():
        net(to_tensor(letterbox(canvas, args.imgsz)[0], dev))
    st = head.strides[0] if head.strides.shape[0] == 1 else head.strides[0]
    ax = (head.anchors[0] * st).cpu().numpy()
    ay = (head.anchors[1] * st).cpu().numpy()
    n_anchor = ax.shape[0]
    print(f"[geom] anchors {n_anchor}，letterbox r={r:.3f} pad=({left},{top})")

    def net_boxes(boxes):
        o = np.array(boxes, dtype=np.float32).copy()
        o[:, [0, 2]] = o[:, [0, 2]] * r + left
        o[:, [1, 3]] = o[:, [1, 3]] * r + top
        return o

    def anchors_in(boxes, exclude=()):
        m = np.zeros(n_anchor, dtype=bool)
        for b in boxes:
            m |= (ax >= b[0]) & (ax <= b[2]) & (ay >= b[1]) & (ay <= b[3])
        return m

    log_every = max(1, args.epochs // 10)
    best = None
    for ep in range(1, args.epochs + 1):
        random.shuffle(neg_pool)
        batch_imgs = [(p, True) for p in pos_imgs] + [(p, False) for p in neg_pool[: args.replay_per_epoch]]
        random.shuffle(batch_imgs)
        run_loss, nb = 0.0, 0

        for path, is_pos_img in batch_imgs:
            scale = random.uniform(args.scale_lo, args.scale_hi)
            frame = cv2.imread(str(path))
            canvas = prepared_canvas(frame, scale)
            lb, r_i, lx_i, ty_i = letterbox(canvas, args.imgsz)
            lb = cv2.resize(lb, (640, 480)) if lb.shape[:2] != (480, 640) else lb
            with torch.no_grad():
                _, preds = net(to_tensor(lb, dev))
            feats = [f.detach() for f in preds["feats"]]

            outs = [rows[i](feats[i]) for i in range(3)]
            flat = [o.view(nc, -1) for o in outs]          # 每尺度 (nc, H*W)
            l5 = torch.cat([f[C5] for f in flat])            # (N,)
            lall = torch.stack([torch.cat([f[i] for f in flat]) for i in range(nc)])  # (nc, N)

            gts = brick.cache[path] if is_pos_img else replay.cache[path]
            gts_n = net_boxes([g[1:] for g in gts])
            cls_arr = np.array([g[0] for g in gts], dtype=int)

            brick_mask = anchors_in([b for b, c in zip(gts_n, cls_arr) if c == C5])
            other_mask = anchors_in([b for b, c in zip(gts_n, cls_arr) if c != C5])
            if is_pos_img:
                pos_idx = np.where(brick_mask)[0]
                if pos_idx.size > args.n_pos:
                    pos_idx = np.random.choice(pos_idx, args.n_pos, replace=False)
                neg_obj_idx = np.where(other_mask)[0]
                bg_idx = np.where(~brick_mask & ~other_mask)[0]
            else:
                pos_idx = np.array([], dtype=int)
                neg_obj_idx = np.where(other_mask)[0]
                bg_idx = np.where(~other_mask)[0]
            l5n = l5.detach().float().cpu().numpy()
            if neg_obj_idx.size > args.n_neg_obj:
                order = neg_obj_idx[np.argsort(-l5n[neg_obj_idx])]
                hard = order[: args.n_neg_obj // 2]
                rest = np.setdiff1d(order, hard)
                rand = np.random.choice(rest, min(rest.size, args.n_neg_obj - hard.size), replace=False)
                neg_obj_idx = np.concatenate([hard, rand])
            if bg_idx.size > args.n_neg_bg:
                order = bg_idx[np.argsort(-l5n[bg_idx])]          # 难负样本挖掘
                hard = order[: args.n_neg_bg // 2]
                rest = np.setdiff1d(order, hard)
                rand = np.random.choice(rest, min(rest.size, args.n_neg_bg - hard.size), replace=False)
                bg_idx = np.concatenate([hard, rand])

            losses = []
            wpos = 1.0
            if pos_idx.size:
                t5 = l5[pos_idx]
                tcur = lall[:, pos_idx]
                cur5 = tcur[C5]
                oth = torch.cat([tcur[:C5], tcur[C5 + 1 :]]).max(0).values
                losses.append(F.binary_cross_entropy_with_logits(t5, torch.ones_like(t5)))
                losses.append(wpos * F.relu(args.margin + oth - t5).mean())
            if neg_obj_idx.size:
                t5n = l5[neg_obj_idx]
                cur = lall[:, neg_obj_idx]
                gt_cls = torch.tensor(
                    [
                        next((c for c, b in zip(cls_arr, gts_n) if b[0] <= ax[i] <= b[2] and b[1] <= ay[i] <= b[3]), -1)
                        for i in neg_obj_idx
                    ],
                    device=dev,
                )
                valid = gt_cls >= 0
                if valid.any():
                    t5n_v = t5n[valid]
                    comp = cur[gt_cls[valid], torch.arange(int(valid.sum()), device=dev)]
                    losses.append(F.binary_cross_entropy_with_logits(t5n_v, torch.zeros_like(t5n_v)))
                    losses.append(F.relu(args.margin + t5n_v - comp).mean())
                else:
                    losses.append(F.binary_cross_entropy_with_logits(t5n, torch.zeros_like(t5n)))
            if bg_idx.size:
                t5b = l5[bg_idx]
                losses.append(F.binary_cross_entropy_with_logits(t5b, torch.zeros_like(t5b)))

            if not losses:
                continue
            loss = torch.stack(losses).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                for c in convs:
                    gw, gb = c.weight.grad, c.bias.grad
                    if gw is not None:
                        gw[:C5] = 0
                        gw[C5 + 1 :] = 0
                    if gb is not None:
                        gb[:C5] = 0
                        gb[C5 + 1 :] = 0
            opt.step()
            run_loss += float(loss)
            nb += 1

        if ep % log_every == 0 or ep == 1:
            with torch.no_grad():
                stats = []
                for c in convs:
                    stats.append(f"|w5|={float(c.weight[C5].abs().mean()):.3f}")
                print(
                    f"[ep {ep:3d}] loss={run_loss / max(1, nb):.4f} b5="
                    + ",".join(f"{float(c.bias[C5]):+.2f}" for c in convs)
                    + " "
                    + " ".join(stats),
                    flush=True,
                )

    args.out = str(Path(args.out))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    src = ckpt.get("ema") or ckpt.get("model")
    with torch.no_grad():
        for i, c in enumerate(convs):
            src.model[-1].cv3[i][-1].weight[C5].copy_(c.weight[C5].cpu())
            src.model[-1].cv3[i][-1].bias[C5].copy_(c.bias[C5].cpu())
    torch.save({"model": src, "epoch": -1, "note": "v18 + cls5 linear probe"}, args.out)
    print(f"[save] {args.out}")
    print("[cls5] bias " + ", ".join(f"{float(c.bias[C5]):+.3f}" for c in convs))
    del best
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
