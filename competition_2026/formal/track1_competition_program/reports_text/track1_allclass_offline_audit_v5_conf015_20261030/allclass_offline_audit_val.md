# 赛道一38个有效类别离线验收报告

- 生成时间：2026-10-30T00:02:13+08:00
- 权重：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/runs/track1_train_40class_v5/track1_40class_v5_20261029_234800/weights/best.pt`
- 权重 SHA256：`cdafa5f39f8e0b22fe1a608a126283eba52a81c2daf7a9f20f4352bb55396099`
- 数据集：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/datasets/track1_training_merged_v5/dataset.yaml`
- 划分：`val`
- 参数：conf=0.15, NMS IoU=0.6, match IoU=0.5, imgsz=640, device=0
- 38个有效类别 GT 框：156；匹配 TP：141；FN：15
- 漏检：5；错类：8；定位错误：2；未匹配预测：18
- **严格验收结果：未通过**

## 逐类结果

| 类别 | GT | TP | FP | Recall | Precision | 漏检 | 错类 | 定位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 1-Cheez-it | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 2 2-Starkist_Tuna | 3 | 3 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 3 3-Scissors | 3 | 3 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 4 4-Frenchs_Mustard | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 5 5-Tomato_Soup | 4 | 4 | 1 | 1.0 | 0.8 | 0 | 0 | 0 |
| 6 6-Foam_Brick | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 7 7-Clamp | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 8 8-Plastic_Banana | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 9 9-Mug | 1 | 1 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 10 10-meat_can | 4 | 4 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 11 11-Padlock | 3 | 2 | 0 | 0.66667 | 1.0 | 0 | 1 | 0 |
| 12 12-Baseball | 3 | 3 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 13 13-Bowl | 3 | 3 | 1 | 1.0 | 0.75 | 0 | 0 | 0 |
| 14 14-Sleeve | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 15 15-Coca_cola_half | 4 | 4 | 1 | 1.0 | 0.8 | 0 | 0 | 0 |
| 16 16-ICRA_Duckie | 4 | 4 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 18 18-Dice | 3 | 2 | 1 | 0.66667 | 0.66667 | 1 | 0 | 0 |
| 19 19-Elmers_Glue | 10 | 10 | 3 | 1.0 | 0.76923 | 0 | 0 | 0 |
| 20 20-Scotch_Tape | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 21 21-Haribo_Golden_Bear | 4 | 4 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 22 22-E27_Adapter | 9 | 8 | 3 | 0.88889 | 0.72727 | 0 | 1 | 0 |
| 23 23-GU24_Adapter | 7 | 5 | 1 | 0.71429 | 0.83333 | 1 | 1 | 0 |
| 24 24-Spoon | 13 | 10 | 0 | 0.76923 | 1.0 | 3 | 0 | 0 |
| 25 25-Flat_Screwdriver | 9 | 7 | 2 | 0.77778 | 0.77778 | 0 | 0 | 2 |
| 27 27-Coca_cola_full | 5 | 5 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 28 28-Kong_Duck_Dog_Toy | 14 | 14 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 29 29-Paper_Towels | 3 | 2 | 0 | 0.66667 | 1.0 | 0 | 1 | 0 |
| 30 30-Stabilo_OHPen | 5 | 4 | 1 | 0.8 | 0.8 | 0 | 1 | 0 |
| 31 31-Plastic_White_Cup | 4 | 2 | 0 | 0.5 | 1.0 | 0 | 2 | 0 |
| 32 32-Wine_Glass | 3 | 3 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 33 33-Key | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 34 34-Nail | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 35 35-Book | 3 | 3 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 36 36-Adjustable_Wrench | 2 | 2 | 1 | 1.0 | 0.66667 | 0 | 0 | 0 |
| 37 37-T-shirt | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 38 38-Jumbo_Pencil_Cup | 4 | 3 | 2 | 0.75 | 0.6 | 0 | 1 | 0 |
| 39 39-Glove | 2 | 2 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 40 40-Pringles | 4 | 4 | 1 | 1.0 | 0.8 | 0 | 0 | 0 |

## 错误明细

| 图片 | GT | 原因 | 预测 | 置信度 | IoU |
|---|---|---|---|---:|---:|
| post28__split-unit-0009__001178.png | 11-11-Padlock | wrong_class | 22-22-E27_Adapter | 0.77924 | 0.9101 |
| post28__split-unit-0052__000371.png | 24-24-Spoon | missing | 25-25-Flat_Screwdriver | 0.91273 | 0.1181 |
| post28__split-unit-0052__000371.png | 25-25-Flat_Screwdriver | localization | 25-25-Flat_Screwdriver | 0.91273 | 0.438 |
| post28__split-unit-0092__000488.png | 18-18-Dice | missing | 24-24-Spoon | 0.89869 | 0.038 |
| post28__split-unit-0214__000778.png | 25-25-Flat_Screwdriver | localization | 25-25-Flat_Screwdriver | 0.8983 | 0.4074 |
| post28__split-unit-0214__000778.png | 24-24-Spoon | missing | 25-25-Flat_Screwdriver | 0.8983 | 0.0291 |
| post28__split-unit-0227__000815.png | 29-29-Paper_Towels | wrong_class | 19-19-Elmers_Glue | 0.76458 | 0.8886 |
| post28__split-unit-0227__000815.png | 24-24-Spoon | missing | 19-19-Elmers_Glue | 0.76458 | 0.1119 |
| post28__split-unit-0253__000895.png | 31-31-Plastic_White_Cup | wrong_class | 19-19-Elmers_Glue | 0.6778 | 0.8464 |
| post28__split-unit-0274__001278.png | 31-31-Plastic_White_Cup | wrong_class | 19-19-Elmers_Glue | 0.65493 | 0.8814 |
| post28__split-unit-0282__000954.png | 22-22-E27_Adapter | wrong_class | 23-23-GU24_Adapter | 0.94375 | 0.6991 |
| post28__split-unit-0282__000954.png | 23-23-GU24_Adapter | wrong_class | 22-22-E27_Adapter | 0.94622 | 0.8319 |
| post28__split-unit-0294__000991.png | 23-23-GU24_Adapter | missing | 22-22-E27_Adapter | 0.9095 | 0.0 |
| post28__split-unit-0320__001067.png | 38-38-Jumbo_Pencil_Cup | wrong_class | 13-13-Bowl | 0.85474 | 0.9328 |
| post28__split-unit-0323__001078.png | 30-30-Stabilo_OHPen | wrong_class_low_iou | 38-38-Jumbo_Pencil_Cup | 0.91091 | 0.2461 |

## 输出文件

- JSON：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_allclass_offline_audit_v5_conf015_20261030/allclass_offline_audit_val.json`
- 逐类 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_allclass_offline_audit_v5_conf015_20261030/allclass_offline_audit_val_classes.csv`
- 错误 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_allclass_offline_audit_v5_conf015_20261030/allclass_offline_audit_val_errors.csv`
- 叠加图目录：`overlays/`。绿色为 GT，黄色为预测，红色为预测类别不在38个有效类别。

> 说明：验证集每类样本较少（多数只有 2–10 个框），本报告用于发现漏检、错类和定位错误，不把总体 mAP 当作现场可用证明。第 17、26 项为空槽，不参与 38 类验收。
