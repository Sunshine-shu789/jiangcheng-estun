# 真机数据流水线（任务 A）

从"拍图"到"微调权重"的一条龙工具，全部本地运行、不需要外网。

## 一、整体流程

```
拍图               预标注                       人工复核             微调 + 评估
real_capture.py → auto_label.py → review.csv → 标注工具修正 → build_finetune_dataset.py
                                                              → finetune_yolo.py → eval_detection.py
```

## 二、命令

```bash
ROOT=/media/sunshine/新加卷/taskA

# 1) 采图（三台相机；也可用相机自带工具抓，图片放同一目录即可）
$ROOT/venv/bin/python real_capture.py \
    --topic /cam_top/color/image_raw --topic /cam_front/color/image_raw \
    --topic /cam_side/color/image_raw \
    --out $ROOT/real_raw --count 60 --interval 2.0

# 2) 预标注（--tta 建议开；开 ROI 时必须同时给 --crop，且要和部署一致）
./finetune_real.sh label                       # 等价于 auto_label.py，见脚本内参数
# 人工：按 review.csv 顺序修 labels/*.txt

# 3) 混合数据集 + 微调 + 前后对比评估
./finetune_real.sh train
```

## 三、六个必须守住的约定

1. **训练图 = 部署时网络看到的图。** 部署要开 ROI 裁剪，`auto_label.py --crop x y w h` 就必须给同样的裁剪；否则模型学到的是另一种取图方式。
2. **物体在 640 输入里要落在 80–190 px**（仿真训练的中位是 138 px）。太小的物体检测不出，太大（>200 px）会像 960/1280 那次一样直接崩。
3. **微调学习率用 1e-4 量级**，别用从头训练的值；数据特别少（<100 张）时可以 `--freeze 10` 冻住主干。
4. **仿真数据按 1:1 混进来**（`build_finetune_dataset.py --ratio 1.0`），否则会把仿真学到的类别能力练丢。验证集里也留了仿真图专门看这个。
5. **RGB-only**：三台主动双目同时开必须关掉红外投射器，否则深度互相打花；检测只用彩色流。
6. **评估用赛题口径**：`eval_detection.py` 按 IoU≥0.75 算通过率，报告的"按相机"表就是"不同方向多次测试"的证据。

> 预标注默认用 `--conf 0.35`（不是 0.25）：预标注的误检框要人工一个个删，阈值高一点更省事；
> 实测 TTA+conf0.35 的通过率只比 conf0.25 低 1.4 个点，但误检少 30%。

## 四、采集多少张

见 `reports/exp_view_finetune.md`（换视角需要多少标注图的仿真代理实验）。
真机还要叠加光照/材质/镜头的域差异，所以**以该实验结论作为下界**，现场多采一点更保险。

## 五、术语速查

- **ROI**：只把收纳箱那一小块画面送进网络。用途是避免"整帧缩到 640 时把物体又缩小一半"，不是放大细节。
- **RGB**：彩色图，检测的唯一输入。深度图只用来量距离，且在透明物体上不可信。
- **TTA**：测试时增强，同一张图多视角各推一次再合并；实测通过率 +2.2 个点、耗时 1.42 倍。
