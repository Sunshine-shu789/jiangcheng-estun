# 赛道一 v7 前 10 类 + 第 17/26 类离线定向验收

- 时间：2026-10-31T01:44:37+08:00
- 权重：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/runs/track1_train_40class_v7_rowfix/track1_40class_v7_rowfix_20261031_011500/weights/best.pt`
- 权重 SHA256：`0ddf4da76cd56a13fc6423fff6c8ee2b7229c8b970a1fd5d3db07888815ba9ef`
- 数据：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/datasets/track1_training_merged_v7_oversample_new2/dataset.yaml`，split=`val`，图片数=121
- 参数：conf=0.15, NMS IoU=0.6, match IoU=0.5, overlap warning IoU=0.3, imgsz=640, device=cpu
- 安全属性：只读静态图片；不打开相机、不导入 ROS、不控制机械臂或夹爪。

## 结论

- 前 10 类 GT 级核心检查：`未通过`
- 第 17/26 类 GT 级核心检查：`通过`
- 前 10 类相对 v6 AP50 回退不超过 0.03：`通过`
- 第 17/26 类 AP50 >= 0.30：`通过`
- 零错误严格检查：`未通过`（额外预测会使其失败，但不等于 GT 漏检）
- 可进入现场视觉验收：`否`
- 生产模型：**未替换**。

## 逐类结果

| 类别 | GT | TP | FP | FN | 漏检 | 错类 | 定位 | Recall | Precision | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 1-Cheez-it | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 2 2-Starkist_Tuna | 3 | 3 | 2 | 0 | 0 | 0 | 0 | 1.0 | 0.6 | CAUTION |
| 3 3-Scissors | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 4 4-Frenchs_Mustard | 2 | 2 | 1 | 0 | 0 | 0 | 0 | 1.0 | 0.66667 | CAUTION |
| 5 5-Tomato_Soup | 4 | 4 | 2 | 0 | 0 | 0 | 0 | 1.0 | 0.66667 | CAUTION |
| 6 6-Foam_Brick | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 7 7-Clamp | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 8 8-Plastic_Banana | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 9 9-Mug | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 10 10-meat_can | 4 | 3 | 1 | 1 | 0 | 1 | 0 | 0.75 | 0.75 | FAIL |
| 17 17-Eggs_Plush | 10 | 10 | 1 | 0 | 0 | 0 | 0 | 1.0 | 0.90909 | CAUTION |
| 26 26-Plastic_Strawberry | 8 | 8 | 5 | 0 | 0 | 0 | 0 | 1.0 | 0.61538 | CAUTION |

## 跨类重叠告警

| 图片 | GT 类别 | 重叠预测类别 | 置信度 | IoU |
|---|---|---|---:|---:|
| front10__front10-review-unit-0011__001321__v7x0.png | 2-2-Starkist_Tuna | 5-5-Tomato_Soup | 0.83449 | 0.8821 |
| front10__front10-review-unit-0068__001535__v7x0.png | 10-10-meat_can | 17-17-Eggs_Plush | 0.50158 | 0.9053 |
| front10__front10-review-unit-0068__001535__v7x0.png | 10-10-meat_can | 4-4-Frenchs_Mustard | 0.37807 | 0.8617 |

## 错误明细

| 图片 | 目标类别 | 原因 | 预测类别 | 置信度 | IoU |
|---|---|---|---|---:|---:|
| front10__front10-review-unit-0011__001321__v7x0.png | 5-5-Tomato_Soup | false_positive | 5-5-Tomato_Soup | 0.83449 | - |
| front10__front10-review-unit-0035__001416__v7x0.png | 26-26-Plastic_Strawberry | false_positive | 26-26-Plastic_Strawberry | 0.19069 | - |
| front10__front10-review-unit-0068__001535__v7x0.png | 10-10-meat_can | wrong_class | 17-17-Eggs_Plush | 0.50158 | 0.9053 |
| front10__front10-review-unit-0068__001535__v7x0.png | 17-17-Eggs_Plush | false_positive | 17-17-Eggs_Plush | 0.50158 | - |
| front10__front10-review-unit-0068__001535__v7x0.png | 4-4-Frenchs_Mustard | false_positive | 4-4-Frenchs_Mustard | 0.37807 | - |
| post28__split-unit-0003__001160__v7x0.png | 26-26-Plastic_Strawberry | false_positive | 26-26-Plastic_Strawberry | 0.30063 | - |
| post28__split-unit-0009__001178__v7x0.png | 26-26-Plastic_Strawberry | false_positive | 26-26-Plastic_Strawberry | 0.33162 | - |
| post28__split-unit-0017__000018__v7x0.png | 2-2-Starkist_Tuna | false_positive | 2-2-Starkist_Tuna | 0.33018 | - |
| post28__split-unit-0101__000514__v7x0.png | 10-10-meat_can | false_positive | 10-10-meat_can | 0.28039 | - |
| post28__split-unit-0187__000314__v7x0.png | 5-5-Tomato_Soup | false_positive | 5-5-Tomato_Soup | 0.42725 | - |
| post28__split-unit-0221__000797__v7x0.png | 2-2-Starkist_Tuna | false_positive | 2-2-Starkist_Tuna | 0.31288 | - |
| post28__split-unit-0265__000931__v7x0.png | 26-26-Plastic_Strawberry | false_positive | 26-26-Plastic_Strawberry | 0.5598 | - |
| post28__split-unit-0294__000991__v7x0.png | 26-26-Plastic_Strawberry | false_positive | 26-26-Plastic_Strawberry | 0.95872 | - |

## 输出文件

- JSON：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_v7_offline_acceptance_20261031/v7_targeted_offline_audit_val.json`
- 逐类 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_v7_offline_acceptance_20261031/v7_targeted_offline_audit_val_classes.csv`
- 错误 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_v7_offline_acceptance_20261031/v7_targeted_offline_audit_val_errors.csv`
- 跨类重叠 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_v7_offline_acceptance_20261031/v7_targeted_offline_audit_val_overlap_warnings.csv`
- 叠加图目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_v7_offline_acceptance_20261031/overlays/val`

> 说明：验证集样本量较少。本报告用于离线回归和现场前筛查，不代替真实相机、多物品、反光和遮挡条件下的现场验收。
