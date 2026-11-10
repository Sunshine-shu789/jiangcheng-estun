# 赛道一 任务 A 技术方案（智能抓取 · 物体检测）

> 版本：2026-09-20 · 状态：仿真阶段完成，待上真机

## 一、赛题要求与我们的指标

赛题原文："任务 A：在限定时间内系统需对不同空间配置和复杂姿态下的物体进行 2D 或 3D 检测，
确保识别精度达到 **IoU 不低于 0.75**……机器人需在**不同方向下多次测试**，以验证视觉系统的稳定性与鲁棒性。"

我们采用的评测口径（与赛题一致）：

- 判定单位是**实例**：每个真值物体只要有任意一个预测框与之 IoU ≥ 0.75 即算通过；
- 主指标是**通过率**，辅以平均 best IoU、Recall@IoU0.5、误检框数；
- 分组报告：按相机（对应"不同方向"）、按可见度（遮挡）、按光照、按类别。

## 二、数据

| 项 | 内容 |
| --- | --- |
| 来源 | CEPB 仿真数据集（`subset1_rand.zip`，105,000 个文件，官方清单 CRC 抽检通过） |
| 场景 | 1–5000 号场景，每场景含左/中/右三相机 × 三种光照（平行光/点光/聚光） |
| 图像 | 2048×1536，另附深度图（900–1400 mm 映射到 8 位） |
| 真值 | `projected_cuboid` 8 角点投影后取凸包，再取轴对齐外接矩形作为检测框 |
| 类别 | 10 类：Cheez-it、Starkist_Tuna、Scissors、Frenchs_Mustard、Tomato_Soup、Foam_Brick、Clamp、Plastic_Banana、Mug、meat_can |
| 可见度 | 每个实例带 `visibility`（0–1），后续所有遮挡分析都基于它 |

> 现场记录：2026-09-26 误拍了一组 Campbell's Cream of Potato 罐头，已按要求
> 删除，不能进入训练集；类别 4 的真实目标仍为 `5-Tomato_Soup`。

两个数据集版本：

- `dataset_v1`：500 场景 × **左相机** × 3 光照 = 1500 张（train 1200 / val 300）
- `dataset_v2`：500 场景 × **三相机** × 3 光照 = 4500 张（train 3600 / val 900，val 9000 实例）

同一场景的三种光照不会跨 train/val，避免泄漏。

## 三、方法

- 模型：YOLO11n（检测），输入 640，CPU 训练（本阶段无 GPU 可用）
- 优化器：Ultralytics 自动选择（小数据量下为 AdamW），warmup 3 轮 + 余弦衰减
- 增强：mosaic 1.0、`close_mosaic` 10、scale 0.5、translate 0.1、fliplr 0.5、HSV 默认；**degrees/perspective 均为 0**
- v2 训练：100 轮上限，batch 16，第 81 轮触发早停（patience=30），最优权重为**第 51 轮**
- 复现：

```bash
python export_dataset.py --scenes 500 --camera all --out dataset_v2
python train_yolo.py --data dataset_v2/data.yaml --model yolo11n.pt \
    --epochs 100 --imgsz 640 --batch 16 --name yolo11n_v2
./run_after_train.sh runs/yolo11n_v2/weights/best.pt dataset_v2
```

## 四、结果

### 4.1 三视角训练 vs 单视角（val 900 张 / 9000 实例）

| 指标 | v2（三视角） | v1（仅左相机） |
| --- | --- | --- |
| 通过率 IoU≥0.75 | **39.4%** | 33.6% |
| 平均 best IoU | **0.492** | 0.438 |
| Recall@IoU0.5 | **59.6%** | 52.6% |
| 误检（每张图） | 3.47 | 3.84 |

### 4.2 分组结果（v2）

- **按相机**：left 41.3%、middle 36.8%、right 40.1% → 三视角差距 ≤4.5 个点，方向鲁棒性达标
- **按可见度**：>0.75 → 57.5%；0.5–0.75 → 41.3%；0.25–0.5 → 17.5%；<0.25 → 2.8%
- **按光照**：39.4% / 39.2% / 39.6% → 光照不是问题，可以不投入
- **按类别**：Cheez-it 91.8%、Mug 58.7%、meat_can 45.6%、Frenchs_Mustard 43.4%、
  Tomato_Soup 40.9%、Clamp 34.4%、Starkist_Tuna 25.6%、Foam_Brick 22.7%、
  Plastic_Banana 21.7%、Scissors 9.4%

### 4.3 结构性结论（重要）

可见度 <0.5 的实例占 31.4%，却只贡献 7.8% 的通过数；可见度 ≥0.5 的实例通过率为 **52.9%**。
原因是真值框是物体的完整外接矩形，被遮挡的物体即使检测框正确，IoU 也难到 0.75。
**这条指标的天花板由场景遮挡分布决定，靠继续训练突破不了**；要提分只能靠换视角（腕部相机）
或把物体摆得更开。

## 五、关键实验（2026-09-20，全部 CPU、离线）

| # | 实验 | 结论 |
| --- | --- | --- |
| 1 | 推理期提高分辨率（960/1280） | **不可行**：通过率 39.4% → 33.3% → 4.1%，物体被放大到训练尺度外，大物体被拆成多块 |
| 2 | TTA（测试时增强） | **真收益**：41.6%（+2.2），IoU 0.541（+0.049），Recall +6 个点；代价误检 ×2、耗时 ×1.42 |
| 3 | v1 权重跨视角测试 | 左 33.6%、右 34.3%、**中 0.6%** → 模型对俯仰角极度敏感，对小幅平移不敏感；多视角训练是必需的 |
| 4 | 换视角需多少标注图 | 0 张 0.6% → 51 张 11.9% → 150 张 17.1% → 399 张 26.1%（全量训练 36.8%）；头 50 张收益最大，曲线未饱和 |
| 5 | 物体尺度敏感度 | 0.5× 30.7%、0.7× 38.2%、1.0× 39.4%、1.5× 33.3%、2.0× 4.1% → **曲线不对称，宁可偏小不可偏大** |

配套报告：`reports/eval_yolo11n_v2.md`、`reports/eval_yolo11n_v2_tta.md`、
`reports/exp_view_finetune.md`、`reports/exp_scale_sensitivity.md`。

## 六、局限

1. **仿真到真机的域差异**：本版权重只在 CEPB 仿真图上训练，真机的镜头、光照、材质都与仿真不同，
   必须用真机数据微调后才能上场。
2. **遮挡天花板**：见 4.3，遮挡物体通过率结构性偏低。
3. **CPU 训练**：imgsz 960 重训需要约 30 小时，本阶段未做。
4. **规模**：只用 500 个场景，CEPB 全集有 5000 个。

## 七、真机方案（待落地）

- **相机**：三台固定相机做任务 A 的三个方向（现有 Gemini 335L / Gemini 2 XL / Gemini 305），
  Intel D405 留作抓取阶段的近距复核。
- **安装**：俯视 0.8–1.0 m；斜视两台按方位角分开（正面、侧面各偏 30–35°）以覆盖真正的不同方向；
  三台装同一根刚性支架，RGB-only（关红外投射器避免互相干扰）。
- **取图**：按原生分辨率裁 ROI，让物体在网络输入里落在 0.7×–1.2× 训练尺度（80–190 px，中位 138 px）。
- **采集量**：每个视角 **≥150–200 张**，想逼近全量训练水平按 **400+ 张**准备（仿真代理实验的下界）。
- **微调流程**（工具已就绪）：

```bash
./finetune_real.sh label     # 预标注 + review.csv（按可疑度排序）
# 人工按 review.csv 修正 labels/*.txt
./finetune_real.sh train     # 真机:仿真=1:1 混合，lr=1e-4，微调并出前后对比报告
```

- **评测**：开 TTA（conf 0.25 拼精度 / conf 0.35 压误检），按相机分组出报告，
  正好作为"不同方向多次测试"的证据。

## 八、产物清单

| 类型 | 路径 |
| --- | --- |
| 权重 | `runs/yolo11n_v1/weights/best.pt`、`runs/yolo11n_v2/weights/best.pt` |
| 数据集 | `dataset_v1/`、`dataset_v2/` |
| 评估报告 | `reports/eval_yolo11n_v2.md`、`reports/eval_yolo11n_v2_tta.md`、`reports/exp_*.md` |
| 脚本 | `export_dataset.py`、`train_yolo.py`、`predict_split.py`、`infer_imgsz.py`、`eval_detection.py`、`sweep_conf.py` |
| 真机流水线 | `real_capture.py`、`auto_label.py`、`build_finetune_dataset.py`、`finetune_yolo.py`、`finetune_real.sh`、`README_real_pipeline.md` |
| 日志 | `logs/2026-09-18.md`、`logs/2026-09-19.md`、`logs/2026-09-20.md` |
