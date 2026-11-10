#!/usr/bin/env python3
"""构建 real_ft_v21_20261016 数据集（泡沫砖补采 + 空背景负样本）。

来源与规则（全部为真实文件拷贝，不用符号链接，保证冻结）:
  底座      : real_ft_v19_20261015  train 3000 / val 214
  旧砖降采样: class5 共 204 张 = 68 唯一场景 ×3 副本；按唯一场景随机保留 23 组(=69 张)
  新增砖    : real_raw/6-Foam_Brick 31 张（废片 015220_297811 不在框表内，自动排除）
              val 10 张（每姿态组 1 张，补上 v19 val 无砖类的空洞）
              train 21 张
  空背景    : real_raw/empty_background 26 张（含 1 张 20260927 旧图）
              train 负样本 14 张（13 新 + 1 旧），留出 12 张只用于 FP 复测
"""
import json, os, shutil, random, collections
from pathlib import Path

T     = Path("/media/sunshine/新加卷/taskA")
V19   = T / "datasets/real_ft_v19_20261015"
OUT   = T / "datasets/real_ft_v21_20261016"
RAW   = T / "real_raw/6-Foam_Brick"
EMPTY = T / "real_raw/empty_background"
LBLS  = Path("/home/sunshine/codex/easton/taskA/prelabels_v21/labels")
BOXJSON = Path("/home/sunshine/codex/easton/taskA/prelabels_v21/manual_boxes_v2.json")
SCRATCH = Path("/home/sunshine/codex/easton/taskA/scratch")
SEED = 21
KEEP_OLD_SCENES = 23

if OUT.exists():
    raise SystemExit(f"目标已存在，拒绝覆盖: {OUT}")
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# ---------- 1) 旧砖 scene 分组（204 = 68×3） ----------
def brick_groups():
    g = collections.defaultdict(list)
    for p in sorted((V19 / "labels/train").glob("*.txt")):
        cls = {int(l.split()[0]) for l in p.read_text().splitlines() if l.strip()}
        if 5 in cls:
            stem = p.stem
            base = stem.split("__k")[0]
            g[base].append(stem)
    return g

groups = brick_groups()
uniq = sorted(groups)
rng = random.Random(SEED)
kept_scenes = set(rng.sample(uniq, KEEP_OLD_SCENES))
drop_stems = {s for b, ss in groups.items() if b not in kept_scenes for s in ss}
assert len(groups) == 68 and sum(len(v) for v in groups.values()) == 204, (len(groups), sum(len(v) for v in groups.values()))
print(f"[旧砖] 唯一场景 {len(groups)} → 保留 {len(kept_scenes)} 组，丢弃图片 {len(drop_stems)} 张")

# ---------- 2) 拷贝 v19 train/val ----------
copied = {"train": 0, "val": 0}
for split in ("train", "val"):
    for img in sorted((V19 / "images" / split).iterdir()):
        stem = img.stem
        if split == "train" and stem in drop_stems:
            continue
        lbl = V19 / "labels" / split / (stem + ".txt")
        if not lbl.exists():
            raise SystemExit(f"缺标签: {lbl}")
        shutil.copy2(img, OUT / "images" / split / img.name)
        shutil.copy2(lbl, OUT / "labels" / split / lbl.name)
        copied[split] += 1
print(f"[底座] 拷贝 train {copied['train']} / val {copied['val']}")

# ---------- 3) 新砖分配 ----------
POSE = {
    "center_v":  ["015555_855884","015556_244603","015556_632185","015557_006862","015557_378440",
                  "015557_758709","015558_141195","015558_518728","015558_891405","015559_273953"],
    "center_h":  ["015651_427267","015651_806864","015652_181086"],
    "top_left":  ["015946_619787","015946_992055"],
    "top_right": ["020108_081864","020108_464841"],
    "bot_left":  ["020157_646343","020158_025628"],
    "bot_right": ["020243_834170","020244_213967"],
    "standing":  ["021651_419155","021651_797070","021652_180906"],
    "leaning":   ["021735_717901","021736_092395","021736_469985"],
    "brick_can": ["022140_850009","022141_233823"],
    "can_occl":  ["022313_628794","022314_004847"],
}
boxes = json.load(open(BOXJSON))
raw_by_tag = {f[:-4][-13:]: f for f in os.listdir(RAW) if f.endswith(".jpg")}
val_tags, train_tags = [], []
for pose, tags in POSE.items():
    assert len(tags) >= 2, pose
    v = tags[-1]              # 每组最后一张进 val
    val_tags.append(v); train_tags += [t for t in tags if t != v]
assert len(val_tags) == 10 and len(train_tags) == 21, (len(val_tags), len(train_tags))
new_written = {"train": [], "val": []}
for split, tags in (("val", val_tags), ("train", train_tags)):
    for tag in tags:
        fn = raw_by_tag[tag]
        assert fn in boxes, fn
        shutil.copy2(RAW / fn, OUT / "images" / split / fn)
        shutil.copy2(LBLS / (fn[:-4] + ".txt"), OUT / "labels" / split / (fn[:-4] + ".txt"))
        new_written[split].append(fn)
print(f"[新砖] train {len(new_written['train'])} / val {len(new_written['val'])}（val 覆盖 {len(POSE)} 种姿态）")

# ---------- 4) 空背景负样本 ----------
new_empty = sorted(p for p in EMPTY.glob("empty_background_20261016_*.jpg"))
old_empty = sorted(EMPTY.glob("empty_background_20260927_*.jpg"))
train_neg, holdout_neg = [], []
for i, p in enumerate(new_empty):
    (train_neg if i % 2 == 0 else holdout_neg).append(p.name)
train_neg += [p.name for p in old_empty]
for name in train_neg:
    shutil.copy2(EMPTY / name, OUT / "images/train" / name)
    (OUT / "labels/train" / (name[:-4] + ".txt")).write_text("")
print(f"[空背景] train 负样本 {len(train_neg)} / 留出复测 {len(holdout_neg)}（不参与训练）")

# ---------- 5) data.yaml ----------
(OUT / "data.yaml").write_text(f"""path: {OUT}
train: images/train
val: images/val
nc: 10
names:
- 1-Cheez-it
- 2-Starkist_Tuna
- 3-Scissors
- 4-Frenchs_Mustard
- 5-Tomato_Soup
- 6-Foam_Brick
- 7-Clamp
- 8-Plastic_Banana
- 9-Mug
- 10-meat_can
""")

# ---------- 6) 统计校验 ----------
def stats(split):
    c, n_empty = collections.Counter(), 0
    for lbl in (OUT / "labels" / split).glob("*.txt"):
        ls = [l for l in lbl.read_text().splitlines() if l.strip()]
        if not ls: n_empty += 1
        for l in ls: c[int(l.split()[0])] += 1
    n_img = len(list((OUT / "images" / split).iterdir()))
    n_lbl = len(list((OUT / "labels" / split).glob("*.txt")))
    return n_img, n_lbl, n_empty, dict(sorted(c.items()))

for split in ("train", "val"):
    ni, nl, ne, c = stats(split)
    print(f"[校验] {split}: images={ni} labels={nl} empty={ne} 实例={c}")
    assert ni == nl
    if split == "val":
        assert c.get(5, 0) == 10, c

manifest = dict(
    out=str(OUT), seed=SEED, keep_old_scenes=KEEP_OLD_SCENES,
    kept_scenes=sorted(kept_scenes), dropped_old_brick=sorted(drop_stems),
    new_brick_train=new_written["train"], new_brick_val=new_written["val"],
    negatives_train=train_neg, negatives_holdout=holdout_neg,
    val_pose=val_tags,
)
manifest_dir = Path("/home/sunshine/codex/easton/taskA/scratch"); manifest_dir.mkdir(exist_ok=True)
(manifest_dir / "v21_build_manifest_20261016.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
(manifest_dir / "v21_empty_holdout_20261016.txt").write_text("\n".join(holdout_neg) + "\n")
print("[完成] 写出", OUT)
