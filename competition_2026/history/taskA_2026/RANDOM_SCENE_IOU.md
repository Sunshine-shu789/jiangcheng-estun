# 随机场景、随机摆放与 IoU 大于 0.75

## 先说结论

“随机生成场景”和“模型预测 IoU 大于 0.75”是两层不同的验收：

1. 合成图片的标注框由最终 alpha 掩膜外接框直接生成，解码回算的标签 IoU
   可以达到 `0.9999+`。这部分可以严格保证。
2. 模型在随机场景中的预测 IoU 不能预先保证，只能在未参与训练的真机随机
   场景上测出来。合成验证集只能证明数据管线正确，不能替代真机验收。

因此，正确做法是“掩膜生成精确标签 -> 随机场景混合训练 -> 独立真机随机
场景验收”。不能拿合成标签的 IoU 当作模型 IoU。

## 数据规模建议

- 仿真切片：从 300 到 500 个 CEPB 源场景提取，三相机、三种光照都要覆盖。
- 真实干净背景：每个部署相机、每种主要光照准备 20 到 30 张，图中不能有
  待检测物品或相似干扰物。训练背景和验收背景必须来自不同采集场次。
- 随机合成训练图：建议先生成 3000 到 5000 张，每张 2 到 7 个物品。
- 真机随机摆放验收：每个相机至少 100 到 200 个实例，位置、朝向、距离和
  光照要覆盖现场工作范围，并按源场景严格隔离。

## 一、提取无背景 RGBA 切片

工作脚本在仓库的 `codex/easton/taskA`，产物统一写到数据盘：

```bash
cd /home/sunshine/codex/easton/taskA
ROOT=/media/sunshine/新加卷/taskA
PY="$ROOT/venv/bin/python"

"$PY" extract_sim_cutouts.py \
  --scenes 500 \
  --camera all \
  --lights all \
  --out "$ROOT/cutouts_cepb_500"
```

提取器只保留高可见度、不贴边、主体连通、分割掩膜与 GT 框一致的样本。
输出中的每个 PNG 都带真实 alpha 通道，这是后续精确标签的关键。

## 二、准备独立背景

目录示例：

```text
$ROOT/random_backgrounds/
  train/   # 训练背景
  val/     # 验收背景，不能与 train 重复
```

如果每个清单里只有一张图，脚本会回退为重复使用，并在报告中明确警告。
这种数据集可以用于训练，但不能作为跨场景泛化证明。

## 三、随机生成场景

```bash
"$PY" generate_random_scenes.py \
  --cutouts "$ROOT/cutouts_cepb_500" \
  --backgrounds "$ROOT/random_backgrounds/train" \
  --val-backgrounds "$ROOT/random_backgrounds/val" \
  --out "$ROOT/dataset_random_v1" \
  --count 5000 \
  --val-fraction 0.20 \
  --objects-min 2 \
  --objects-max 7 \
  --target-long-min 70 \
  --target-long-max 190 \
  --rotation-deg 180 \
  --shadow-prob 0.65 \
  --image-format jpg \
  --jpeg-quality 92 \
  --min-label-iou 0.999 \
  --seed 20260926
```

生成器会随机化背景裁切、物品类别、位置、平面旋转、缩放、亮度、对比度、
模糊、噪声和阴影，同时强制：

- 物品不被画面边缘裁断；
- alpha 掩膜重叠小于 1%；
- 物品框之间的 IoU 小于 2%；
- 每个 YOLO 框都从最终变换后的 alpha 掩膜重新计算；
- 标签编码后再解码，和 alpha 框的 IoU 低于 `0.999` 就停止生成。

训练和验证默认按 CEPB 源场景隔离，并会重试划分，尽量保证每个类别在两边
都出现。若某类确实缺少独立验证源，程序默认直接报错；不要静默忽略这类缺口。

CEPB 清单中的类别 ID 是 `1..10`，写入 YOLO 标签时会自动转换为 `0..9`，
生成的 `classes.txt` 与 `data.yaml` 也统一使用这个顺序。

生成的 `manifest.csv` 里记录了每个实例的来源场景、相机、光照、位置和
`alpha_box_iou`。`generation_report.md` 是这张数据集的自检报告。

## 四、和真机标注混合训练

先确保真机标注的类别顺序与 `classes.txt` 一致，再生成混合数据集：

```bash
"$PY" build_finetune_dataset.py \
  --real-images "$ROOT/real_random_train/images" \
  --real-labels "$ROOT/real_random_train/labels" \
  --sim-data "$ROOT/dataset_random_v1" \
  --ratio 2.0 \
  --out "$ROOT/dataset_random_mix_v1"
```

真机随机摆放样本不能只放在仿真训练里。建议至少按 `真机 : 合成 = 1 : 1`
到 `1 : 3` 混合，避免模型只记住仿真材质和光照。

从一个已经可用的权重开始小学习率微调：

```bash
"$PY" finetune_yolo.py \
  --data "$ROOT/dataset_random_mix_v1/data.yaml" \
  --model "$ROOT/runs/yolo11s_real_ft_v5_grouped_20260926/weights/best.pt" \
  --epochs 50 \
  --imgsz 640 \
  --batch 32 \
  --workers 16 \
  --lr0 1e-4 \
  --mosaic 0.20 \
  --name yolo11s_random_scene_v1
```

训练完成后，用与部署一致的相机、裁剪范围和输入尺寸做真机随机场景验收。

## 五、独立真机随机场景验收

把未参与训练的真机随机摆放数据整理成：

```text
$ROOT/random_real_val/
  images/val/*.jpg
  labels/val/*.txt
```

推理并评估：

```bash
PY="$ROOT/venv/bin/python"
WEIGHTS="$ROOT/runs/yolo11s_random_scene_v1/weights/best.pt"

"$PY" infer_imgsz.py \
  --weights "$WEIGHTS" \
  --data "$ROOT/random_real_val" \
  --split val \
  --imgsz 640 \
  --conf 0.25 \
  --tta \
  --out "$ROOT/preds/random_real_val_v1"

"$PY" eval_detection.py \
  --data "$ROOT/random_real_val" \
  --split val \
  --pred-labels "$ROOT/preds/random_real_val_v1/labels" \
  --iou-threshold 0.75 \
  --out "$ROOT/reports/eval_random_real_val_v1"
```

验收至少同时看这四项：

- `IoU >= 0.75` 的实例通过率；
- 平均 best IoU；
- 每个相机分组的通过率；
- 每个类别分组的通过率和误检数。

只有独立真机随机场景达到要求，才能说“模型在随机场景中 IoU 大于 0.75”。
如果通过率不够，按下面的顺序处理：

1. 先看失败案例的框是漏检、框偏、类别错还是尺度不匹配；
2. 补采集失败相机、失败类别和失败尺度区间的真机随机摆放图；
3. 增加真实干净背景和部署光照下的合成图；
4. 调整 `conf`、输入尺度，并比较是否开启 TTA；
5. 重新训练后使用同一批独立真机验证集复测，不能把验证图加回训练。

## 当前基线

修复类别 ID 后重新生成的 40 张 JPEG 冒烟数据包含 119 个实例，覆盖
`class 0..9`，标签解码后的最小框 IoU 是 `0.999982`。完美预测验收在独立
验证 split 上得到 `IoU >= 0.75` 通过率 `100%`，说明掩膜到标签的生成链路
和评估坐标换算正确。

修复前的 100 张随机数据虽然掩膜框 IoU 也很高，但类别 ID 错写成了
`1..10`，不能用于训练，也不能用它的类别统计评估模型。

旧 v5 在真机验证上的 `IoU >= 0.75` 通过率仍只有约 `2.5%`，因此现阶段
不能宣称随机场景模型 IoU 已经达标。缺口主要是仿真到真机的域差异、真实
场景覆盖不足，以及缺少部分物理类别，而不是随机生成标签不准。

另外，旧的 `dataset_v4` 和 `dataset_v4_640` 使用了修复前的
`projected_cuboid` 纵坐标，不能继续作为基准训练数据。应先用修正后的
`cepb.py` 重新生成，或直接使用本文的随机场景数据集重新训练。
