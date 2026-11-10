# 赛道一前10类离线验收报告

- 生成时间：2026-10-29T23:01:07+08:00
- 权重：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/runs/track1_train_40class_v2/track1_40class_20261029_221534/weights/best.pt`
- 权重 SHA256：`953bb205d0641eb131fc7ff183b9368d59d1712ff77c26964521edb97c110d72`
- 数据集：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/datasets/track1_training_merged_v2/dataset.yaml`
- 划分：`val`
- 参数：conf=0.05, NMS IoU=0.6, match IoU=0.5, imgsz=640, device=0
- 前10类 GT 框：23；匹配 TP：22；FN：1
- 漏检：0；错类：1；定位错误：0；未匹配预测：8
- **严格验收结果：未通过**

## 逐类结果

| 类别 | GT | TP | FP | Recall | Precision | 漏检 | 错类 | 定位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 1-Cheez-it | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 2 2-Starkist_Tuna | 3 | 3 | 1 | 1.0 | 0.75 | 0 | 0 | 0 |
| 3 3-Scissors | 1 | 0 | 0 | 0.0 | - | 0 | 1 | 0 |
| 4 4-Frenchs_Mustard | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 5 5-Tomato_Soup | 4 | 4 | 2 | 1.0 | 0.66667 | 0 | 0 | 0 |
| 6 6-Foam_Brick | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 7 7-Clamp | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 8 8-Plastic_Banana | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 9 9-Mug | 1 | 1 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 10 10-meat_can | 4 | 4 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |

## 错误明细

| 图片 | GT | 原因 | 预测 | 置信度 | IoU |
|---|---|---|---|---:|---:|
| front10__front10-review-unit-0015__001351.png | 3-3-Scissors | wrong_class | 14-14-Sleeve | 0.33858 | 0.7442 |

## 输出文件

- JSON：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_offline_audit_v2_val/front10_offline_audit_val.json`
- 逐类 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_offline_audit_v2_val/front10_offline_audit_val_classes.csv`
- 错误 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_offline_audit_v2_val/front10_offline_audit_val_errors.csv`
- 叠加图目录：`overlays/`。绿色为 GT，黄色为预测，红色为预测类别不在前10类。

> 说明：当前验证集每个类别样本很少，尤其剪刀等类别只有 1 个验证框；本报告是发现问题的严格离线筛查，不把总体 mAP 当作现场可用证明。
