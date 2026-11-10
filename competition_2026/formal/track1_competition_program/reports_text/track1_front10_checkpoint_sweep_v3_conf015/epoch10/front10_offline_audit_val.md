# 赛道一前10类离线验收报告

- 生成时间：2026-10-29T23:20:41+08:00
- 权重：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/runs/track1_train_40class_v3/track1_40class_20261029_230844/weights/epoch10.pt`
- 权重 SHA256：`b62c5dc446f655ff83bf02b50ae68bf0fac7c93e7b38917212da596bc66b8fa3`
- 数据集：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/datasets/track1_training_merged_v3/dataset.yaml`
- 划分：`val`
- 参数：conf=0.15, NMS IoU=0.6, match IoU=0.5, imgsz=640, device=0
- 前10类 GT 框：25；匹配 TP：20；FN：5
- 漏检：0；错类：5；定位错误：0；未匹配预测：22
- **严格验收结果：未通过**

## 逐类结果

| 类别 | GT | TP | FP | Recall | Precision | 漏检 | 错类 | 定位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 1-Cheez-it | 2 | 2 | 1 | 1.0 | 0.66667 | 0 | 0 | 0 |
| 2 2-Starkist_Tuna | 3 | 3 | 7 | 1.0 | 0.3 | 0 | 0 | 0 |
| 3 3-Scissors | 3 | 3 | 2 | 1.0 | 0.6 | 0 | 0 | 0 |
| 4 4-Frenchs_Mustard | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 5 5-Tomato_Soup | 4 | 1 | 0 | 0.25 | 1.0 | 0 | 3 | 0 |
| 6 6-Foam_Brick | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 7 7-Clamp | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 8 8-Plastic_Banana | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 9 9-Mug | 1 | 1 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 10 10-meat_can | 4 | 2 | 0 | 0.5 | 1.0 | 0 | 2 | 0 |

## 错误明细

| 图片 | GT | 原因 | 预测 | 置信度 | IoU |
|---|---|---|---|---:|---:|
| front10__front10-review-unit-0025__001382.png | 5-5-Tomato_Soup | wrong_class | 2-2-Starkist_Tuna | 0.23736 | 0.8432 |
| front10__front10-review-unit-0028__001396.png | 5-5-Tomato_Soup | wrong_class | 15-15-Coca_cola_half | 0.18122 | 0.8398 |
| front10__front10-review-unit-0031__001404.png | 5-5-Tomato_Soup | wrong_class | 12-12-Baseball | 0.29423 | 0.863 |
| front10__front10-review-unit-0068__001535.png | 10-10-meat_can | wrong_class | 22-22-E27_Adapter | 0.20627 | 0.9015 |
| front10__front10-review-unit-0070__001551.png | 10-10-meat_can | wrong_class | 23-23-GU24_Adapter | 0.35667 | 0.9365 |

## 输出文件

- JSON：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_checkpoint_sweep_v3_conf015/epoch10/front10_offline_audit_val.json`
- 逐类 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_checkpoint_sweep_v3_conf015/epoch10/front10_offline_audit_val_classes.csv`
- 错误 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_checkpoint_sweep_v3_conf015/epoch10/front10_offline_audit_val_errors.csv`
- 叠加图目录：`overlays/`。绿色为 GT，黄色为预测，红色为预测类别不在前10类。

> 说明：当前验证集每个类别样本很少，尤其剪刀等类别只有 1 个验证框；本报告是发现问题的严格离线筛查，不把总体 mAP 当作现场可用证明。
