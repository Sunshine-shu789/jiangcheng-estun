# 赛道一 v8 候选模型拒绝报告

- 生成时间：2026-10-31
- 结论：**REJECTED，不部署**
- 原因：标准训练后 AP 审计虽通过，但独立离线定向验收失败；同时存在肉罐头高置信度误报 Eggs。
- 生产模型：未替换。
- 安全属性：本报告只整理静态图片推理证据；未打开相机、未导入 ROS、未控制机械臂或夹爪。

## 模型与数据

- v8 权重：`runs/track1_train_40class_v8_v5rowfix_from_v5/track1_40class_20261031_021609/weights/best.pt`
- v8 SHA-256：`5f0684c270dd55127c43d4d0e2b00bff1e4ed93e5cb248b8b7964c3466f2e31e`
- 当前生产权重：`models/best.pt`
- 生产权重 SHA-256：`76b71ff7227eec9d26f5da0dec917d83e21dc3de1b92456645e9004dfa80de60`
- 验收数据：`datasets/track1_training_merged_v7_oversample_new2/dataset.yaml`，split=`val`
- 验收参数：`conf=0.15`、`NMS IoU=0.60`、`match IoU=0.50`、`imgsz=640`

## 验收结果

- 前 10 类 GT 核心检查：通过。
- 前 10 类 AP50 回退 <= 0.03：通过。
- Eggs / Strawberry AP50 >= 0.30：通过。
- 新增两类 GT 核心检查：**未通过**。
- 可进入现场视觉验收：**否**。
- 汇总：GT 43，TP 38，FP 15，GT 级错误 5，跨类重叠告警 8。

| 类别 | GT | TP | FN | FP | Recall | 状态 |
|---|---:|---:|---:|---:|---:|---|
| 17-Eggs_Plush | 10 | 5 | 5 | 8 | 0.50 | FAIL |
| 26-Plastic_Strawberry | 8 | 8 | 0 | 5 | 1.00 | CAUTION |

## 阻断性错误

### 1. Eggs 漏检 5 张

漏检图像：

- `selfcollected__20261027_session_001__001603_17-Eggs_Plush__v7x0.png`
- `selfcollected__20261027_session_001__001604_17-Eggs_Plush__v7x0.png`
- `selfcollected__20261027_session_001__001605_17-Eggs_Plush__v7x0.png`
- `selfcollected__20261027_session_001__001606_17-Eggs_Plush__v7x0.png`
- `selfcollected__20261027_session_001__001607_17-Eggs_Plush__v7x0.png`

这 5 张均为绿色正面 Eggs。低阈值候选存在，但在正式验收阈值 `conf=0.15` 下未形成可用检出。

### 2. 肉罐头同时高置信度误报 Eggs

| 图像 | GT | 错误预测 | Eggs 分数 | IoU |
|---|---|---|---:|---:|
| `front10__front10-review-unit-0068__001535__v7x0.png` | 10-meat_can | 17-Eggs_Plush | 0.94720 | 0.8924 |
| `front10__front10-review-unit-0070__001551__v7x0.png` | 10-meat_can | 17-Eggs_Plush | 0.85432 | 0.9276 |

该错误会在同一肉罐头上同时输出 `10-meat_can` 和 `17-Eggs_Plush`，属于现场不可接受的跨类误报。

## 证据文件

- `reports/track1_v8_offline_acceptance_rerun_20261031/v7_targeted_offline_audit_val.json`
- `reports/track1_v8_offline_acceptance_rerun_20261031/v7_targeted_offline_audit_val.md`
- `reports/track1_v8_offline_acceptance_rerun_20261031/v7_targeted_offline_audit_val_errors.csv`
- `reports/track1_v8_offline_acceptance_rerun_20261031/v7_targeted_offline_audit_val_overlap_warnings.csv`
- `reports/track1_v8_offline_acceptance_rerun_20261031/contact_val_10_10-meat_can.jpg`
- `reports/track1_v8_offline_acceptance_rerun_20261031/contact_val_17_17-Eggs_Plush.jpg`

## 后续门槛

v9 或后续候选必须同时满足：

1. 前 10 类 GT 核心检查通过。
2. Eggs 和 Strawberry GT 核心检查通过。
3. 前 10 类 AP50 回退不超过 0.03。
4. Eggs / Strawberry AP50 不低于 0.30。
5. 所有 `10-meat_can` GT 图上不得出现高置信度 `17-Eggs_Plush` 同框错误。
6. 通过真实相机静态现场视觉验收后，才允许替换生产模型。

在上述条件满足前，`models/best.pt` 保持原样。
