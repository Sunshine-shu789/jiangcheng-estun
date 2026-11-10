# 赛道一前10类离线验收报告

- 生成时间：2026-10-29T23:21:08+08:00
- 权重：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/runs/track1_train_40class_v3/track1_40class_20261029_230844/weights/epoch30.pt`
- 权重 SHA256：`997a3571215e6a647f6d13bfc144578d4f03890da07859b0e4752d4827d61c8b`
- 数据集：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/datasets/track1_training_merged_v3/dataset.yaml`
- 划分：`val`
- 参数：conf=0.15, NMS IoU=0.6, match IoU=0.5, imgsz=640, device=0
- 前10类 GT 框：25；匹配 TP：21；FN：4
- 漏检：0；错类：4；定位错误：0；未匹配预测：7
- **严格验收结果：未通过**

## 逐类结果

| 类别 | GT | TP | FP | Recall | Precision | 漏检 | 错类 | 定位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 1-Cheez-it | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 2 2-Starkist_Tuna | 3 | 3 | 1 | 1.0 | 0.75 | 0 | 0 | 0 |
| 3 3-Scissors | 3 | 3 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 4 4-Frenchs_Mustard | 2 | 2 | 1 | 1.0 | 0.66667 | 0 | 0 | 0 |
| 5 5-Tomato_Soup | 4 | 1 | 0 | 0.25 | 1.0 | 0 | 3 | 0 |
| 6 6-Foam_Brick | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 7 7-Clamp | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 8 8-Plastic_Banana | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 9 9-Mug | 1 | 1 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 10 10-meat_can | 4 | 3 | 0 | 0.75 | 1.0 | 0 | 1 | 0 |

## 错误明细

| 图片 | GT | 原因 | 预测 | 置信度 | IoU |
|---|---|---|---|---:|---:|
| front10__front10-review-unit-0025__001382.png | 5-5-Tomato_Soup | wrong_class | 27-27-Coca_cola_full | 0.16481 | 0.8949 |
| front10__front10-review-unit-0028__001396.png | 5-5-Tomato_Soup | wrong_class | 15-15-Coca_cola_half | 0.44627 | 0.9538 |
| front10__front10-review-unit-0031__001404.png | 5-5-Tomato_Soup | wrong_class | 29-29-Paper_Towels | 0.39081 | 0.8961 |
| front10__front10-review-unit-0068__001535.png | 10-10-meat_can | wrong_class | 16-16-ICRA_Duckie | 0.38509 | 0.896 |

## 输出文件

- JSON：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_checkpoint_sweep_v3_conf015/epoch30/front10_offline_audit_val.json`
- 逐类 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_checkpoint_sweep_v3_conf015/epoch30/front10_offline_audit_val_classes.csv`
- 错误 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_front10_checkpoint_sweep_v3_conf015/epoch30/front10_offline_audit_val_errors.csv`
- 叠加图目录：`overlays/`。绿色为 GT，黄色为预测，红色为预测类别不在前10类。

> 说明：当前验证集每个类别样本很少，尤其剪刀等类别只有 1 个验证框；本报告是发现问题的严格离线筛查，不把总体 mAP 当作现场可用证明。
