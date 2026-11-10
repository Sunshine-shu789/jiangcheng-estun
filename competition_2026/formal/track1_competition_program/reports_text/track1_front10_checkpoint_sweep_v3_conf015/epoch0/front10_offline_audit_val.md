# 赛道一前10类离线验收报告

- 生成时间：2026-10-29T23:20:33+08:00
- 权重：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/runs/track1_train_40class_v3/track1_40class_20261029_230844/weights/epoch0.pt`
- 权重 SHA256：`3cd69f31c3e3a55022901f4e5a706fcffc52fb45a0aa0b7449a997d341ea2c7c`
- 数据集：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/datasets/track1_training_merged_v3/dataset.yaml`
- 划分：`val`
- 参数：conf=0.15, NMS IoU=0.6, match IoU=0.5, imgsz=640, device=0
- 前10类 GT 框：25；匹配 TP：6；FN：19
- 漏检：17；错类：2；定位错误：0；未匹配预测：4
- **严格验收结果：未通过**

## 逐类结果

| 类别 | GT | TP | FP | Recall | Precision | 漏检 | 错类 | 定位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 1-Cheez-it | 2 | 0 | 0 | 0.0 | - | 2 | 0 | 0 |
| 2 2-Starkist_Tuna | 3 | 3 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 3 3-Scissors | 3 | 0 | 0 | 0.0 | - | 3 | 0 | 0 |
| 4 4-Frenchs_Mustard | 2 | 0 | 0 | 0.0 | - | 2 | 0 | 0 |
| 5 5-Tomato_Soup | 4 | 1 | 1 | 0.25 | 0.5 | 3 | 0 | 0 |
| 6 6-Foam_Brick | 2 | 0 | 1 | 0.0 | 0.0 | 2 | 0 | 0 |
| 7 7-Clamp | 2 | 1 | 0 | 0.5 | 1.0 | 1 | 0 | 0 |
| 8 8-Plastic_Banana | 2 | 0 | 0 | 0.0 | - | 2 | 0 | 0 |
| 9 9-Mug | 1 | 1 | 1 | 1.0 | 0.5 | 0 | 0 | 0 |
| 10 10-meat_can | 4 | 0 | 0 | 0.0 | - | 2 | 2 | 0 |

## 错误明细

| 图片 | GT | 原因 | 预测 | 置信度 | IoU |
|---|---|---|---|---:|---:|
| front10__front10-review-unit-0002__001294.png | 1-1-Cheez-it | missing | --- | - | 0.0 |
| front10__front10-review-unit-0005__001303.png | 1-1-Cheez-it | missing | --- | - | 0.0 |
| front10__front10-review-unit-0015__001351.png | 3-3-Scissors | missing | --- | - | 0.0 |
| front10__front10-review-unit-0018__001361.png | 4-4-Frenchs_Mustard | missing | --- | - | 0.0 |
| front10__front10-review-unit-0020__001369.png | 4-4-Frenchs_Mustard | missing | --- | - | 0.0 |
| front10__front10-review-unit-0025__001382.png | 5-5-Tomato_Soup | missing | --- | - | 0.0 |
| front10__front10-review-unit-0028__001396.png | 5-5-Tomato_Soup | missing | --- | - | 0.0 |
| front10__front10-review-unit-0031__001404.png | 5-5-Tomato_Soup | missing | --- | - | 0.0 |
| front10__front10-review-unit-0035__001416.png | 6-6-Foam_Brick | missing | --- | - | 0.0 |
| front10__front10-review-unit-0038__001424.png | 6-6-Foam_Brick | missing | --- | - | 0.0 |
| front10__front10-review-unit-0044__001443.png | 7-7-Clamp | missing | --- | - | 0.0 |
| front10__front10-review-unit-0049__001464.png | 8-8-Plastic_Banana | missing | --- | - | 0.0 |
| front10__front10-review-unit-0050__001475.png | 8-8-Plastic_Banana | missing | --- | - | 0.0 |
| front10__front10-review-unit-0065__001538.png | 10-10-meat_can | missing | --- | - | 0.0 |
| front10__front10-review-unit-0068__001535.png | 10-10-meat_can | missing | --- | - | 0.0 |
| front10__front10-review-unit-0069__001544.png | 10-10-meat_can | wrong_class | 6-6-Foam_Brick | 0.33223 | 0.9146 |
| front10__front10-review-unit-0070__001551.png | 10-10-meat_can | wrong_class | 9-9-Mug | 0.34262 | 0.9028 |
| scissors_v3__g6__001349.png | 3-3-Scissors | missing | --- | - | 0.0 |
| scissors_v3__g6__001350.png | 3-3-Scissors | missing | --- | - | 0.0 |

## 输出文件

- JSON：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_checkpoint_sweep_v3_conf015/epoch0/front10_offline_audit_val.json`
- 逐类 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_checkpoint_sweep_v3_conf015/epoch0/front10_offline_audit_val_classes.csv`
- 错误 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_checkpoint_sweep_v3_conf015/epoch0/front10_offline_audit_val_errors.csv`
- 叠加图目录：`overlays/`。绿色为 GT，黄色为预测，红色为预测类别不在前10类。

> 说明：当前验证集每个类别样本很少，尤其剪刀等类别只有 1 个验证框；本报告是发现问题的严格离线筛查，不把总体 mAP 当作现场可用证明。
