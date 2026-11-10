# 赛道一 v7 前 10 类 + 第 17/26 类离线定向验收

- 时间：2026-10-31T09:54:25+08:00
- 权重：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/runs/track1_train_40class_v9_hardneg_from_v5/track1_40class_20261031_024314/weights/best.pt`
- 权重 SHA256：`0cf17201d07931507a0576ff7ad79daf981146a099b65b1fcfa3e65ed84815ee`
- 数据：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/datasets/track1_training_merged_v9_hardneg/dataset.yaml`，split=`val`，图片数=121
- 参数：conf=0.15, NMS IoU=0.6, match IoU=0.5, overlap warning IoU=0.3, imgsz=640, device=0
- 安全属性：只读静态图片；不打开相机、不导入 ROS、不控制机械臂或夹爪。

## 结论

- 前 10 类 GT 级核心检查：`通过`
- 第 17/26 类 GT 级核心检查：`未通过`
- 前 10 类相对 v6 AP50 回退不超过 0.03：`通过`
- 第 17/26 类 AP50 >= 0.30：`通过`
- 零错误严格检查：`未通过`（额外预测会使其失败，但不等于 GT 漏检）
- 可进入现场视觉验收：`否`
- 生产模型：**未替换**。

## 逐类结果

| 类别 | GT | TP | FP | FN | 漏检 | 错类 | 定位 | Recall | Precision | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 1-Cheez-it | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 2 2-Starkist_Tuna | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 3 3-Scissors | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 4 4-Frenchs_Mustard | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 5 5-Tomato_Soup | 4 | 4 | 1 | 0 | 0 | 0 | 0 | 1.0 | 0.8 | CAUTION |
| 6 6-Foam_Brick | 2 | 2 | 1 | 0 | 0 | 0 | 0 | 1.0 | 0.66667 | CAUTION |
| 7 7-Clamp | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 8 8-Plastic_Banana | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 9 9-Mug | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 10 10-meat_can | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | PASS |
| 17 17-Eggs_Plush | 10 | 5 | 8 | 5 | 5 | 0 | 0 | 0.5 | 0.38462 | FAIL |
| 26 26-Plastic_Strawberry | 8 | 8 | 6 | 0 | 0 | 0 | 0 | 1.0 | 0.57143 | CAUTION |

## 跨类重叠告警

| 图片 | GT 类别 | 重叠预测类别 | 置信度 | IoU |
|---|---|---|---:|---:|
| front10__front10-review-unit-0011__001321.png | 2-2-Starkist_Tuna | 5-5-Tomato_Soup | 0.58282 | 0.8857 |
| front10__front10-review-unit-0018__001361.png | 4-4-Frenchs_Mustard | 17-17-Eggs_Plush | 0.38810 | 0.8968 |
| front10__front10-review-unit-0068__001535.png | 10-10-meat_can | 17-17-Eggs_Plush | 0.94076 | 0.8924 |
| front10__front10-review-unit-0070__001551.png | 10-10-meat_can | 17-17-Eggs_Plush | 0.85001 | 0.9276 |
| selfcollected__20261027_session_001__001598_26-Plastic_Strawberry.png | 26-26-Plastic_Strawberry | 18-18-Dice | 0.16534 | 0.7384 |
| selfcollected__20261027_session_001__001599_26-Plastic_Strawberry.png | 26-26-Plastic_Strawberry | 18-18-Dice | 0.19917 | 0.7322 |
| selfcollected__20261027_session_001__001600_26-Plastic_Strawberry.png | 26-26-Plastic_Strawberry | 18-18-Dice | 0.29373 | 0.7317 |
| selfcollected__20261027_session_001__001601_26-Plastic_Strawberry.png | 26-26-Plastic_Strawberry | 18-18-Dice | 0.28346 | 0.7232 |
| selfcollected__20261027_session_001__001634_17-Eggs_Plush.png | 17-17-Eggs_Plush | 13-13-Bowl | 0.19760 | 0.8342 |
| selfcollected__20261027_session_001__001635_17-Eggs_Plush.png | 17-17-Eggs_Plush | 6-6-Foam_Brick | 0.15071 | 0.8363 |
| selfcollected__20261027_session_001__001636_17-Eggs_Plush.png | 17-17-Eggs_Plush | 13-13-Bowl | 0.23849 | 0.8473 |
| selfcollected__20261027_session_001__001637_17-Eggs_Plush.png | 17-17-Eggs_Plush | 13-13-Bowl | 0.22610 | 0.8311 |

## 错误明细

| 图片 | 目标类别 | 原因 | 预测类别 | 置信度 | IoU |
|---|---|---|---|---:|---:|
| front10__front10-review-unit-0011__001321.png | 5-5-Tomato_Soup | false_positive | 5-5-Tomato_Soup | 0.58282 | - |
| front10__front10-review-unit-0018__001361.png | 17-17-Eggs_Plush | false_positive | 17-17-Eggs_Plush | 0.3881 | - |
| front10__front10-review-unit-0068__001535.png | 17-17-Eggs_Plush | false_positive | 17-17-Eggs_Plush | 0.94076 | - |
| front10__front10-review-unit-0070__001551.png | 17-17-Eggs_Plush | false_positive | 17-17-Eggs_Plush | 0.85001 | - |
| post28__split-unit-0086__000469.png | 26-26-Plastic_Strawberry | false_positive | 26-26-Plastic_Strawberry | 0.55591 | - |
| post28__split-unit-0089__000478.png | 26-26-Plastic_Strawberry | false_positive | 26-26-Plastic_Strawberry | 0.65431 | - |
| post28__split-unit-0129__000572.png | 17-17-Eggs_Plush | false_positive | 17-17-Eggs_Plush | 0.19133 | - |
| post28__split-unit-0139__000602.png | 17-17-Eggs_Plush | false_positive | 17-17-Eggs_Plush | 0.24287 | - |
| post28__split-unit-0155__000650.png | 26-26-Plastic_Strawberry | false_positive | 26-26-Plastic_Strawberry | 0.46605 | - |
| post28__split-unit-0181__000734.png | 26-26-Plastic_Strawberry | false_positive | 26-26-Plastic_Strawberry | 0.26451 | - |
| post28__split-unit-0253__000895.png | 17-17-Eggs_Plush | false_positive | 17-17-Eggs_Plush | 0.15182 | - |
| post28__split-unit-0271__001261.png | 26-26-Plastic_Strawberry | false_positive | 26-26-Plastic_Strawberry | 0.15178 | - |
| post28__split-unit-0274__001278.png | 17-17-Eggs_Plush | false_positive | 17-17-Eggs_Plush | 0.84594 | - |
| post28__split-unit-0274__001278.png | 26-26-Plastic_Strawberry | false_positive | 26-26-Plastic_Strawberry | 0.15577 | - |
| post28__split-unit-0309__001034.png | 17-17-Eggs_Plush | false_positive | 17-17-Eggs_Plush | 0.3515 | - |
| selfcollected__20261027_session_001__001603_17-Eggs_Plush.png | 17-17-Eggs_Plush | missing | --- | - | 0.0 |
| selfcollected__20261027_session_001__001604_17-Eggs_Plush.png | 17-17-Eggs_Plush | missing | --- | - | 0.0 |
| selfcollected__20261027_session_001__001605_17-Eggs_Plush.png | 17-17-Eggs_Plush | missing | --- | - | 0.0 |
| selfcollected__20261027_session_001__001606_17-Eggs_Plush.png | 17-17-Eggs_Plush | missing | --- | - | 0.0 |
| selfcollected__20261027_session_001__001607_17-Eggs_Plush.png | 17-17-Eggs_Plush | missing | --- | - | 0.0 |
| selfcollected__20261027_session_001__001635_17-Eggs_Plush.png | 6-6-Foam_Brick | false_positive | 6-6-Foam_Brick | 0.15071 | - |

## 输出文件

- JSON：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_v9_offline_acceptance_20261031/v7_targeted_offline_audit_val.json`
- 逐类 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_v9_offline_acceptance_20261031/v7_targeted_offline_audit_val_classes.csv`
- 错误 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_v9_offline_acceptance_20261031/v7_targeted_offline_audit_val_errors.csv`
- 跨类重叠 CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_v9_offline_acceptance_20261031/v7_targeted_offline_audit_val_overlap_warnings.csv`
- 叠加图目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_v9_offline_acceptance_20261031/overlays/val`

> 说明：验证集样本量较少。本报告用于离线回归和现场前筛查，不代替真实相机、多物品、反光和遮挡条件下的现场验收。
