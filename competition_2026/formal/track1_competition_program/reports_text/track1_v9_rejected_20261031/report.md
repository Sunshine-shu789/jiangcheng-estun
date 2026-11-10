# 赛道一 v9 候选模型拒绝报告

- 结论：**REJECTED，不部署**
- v9 SHA-256：`0cf17201d07931507a0576ff7ad79daf981146a099b65b1fcfa3e65ed84815ee`
- 生产模型 SHA-256：`76b71ff7227eec9d26f5da0dec917d83e21dc3de1b92456645e9004dfa80de60`（未替换）
- 验收：`conf=0.15`、NMS IoU=0.60、match IoU=0.50、imgsz=640、val=121 图
- 安全属性：只读静态图片；未打开相机、未导入 ROS、未控制机械臂或夹爪。

## 结果

- 前 10 类 GT 核心检查：通过；AP50 回退检查：通过。
- Eggs / Strawberry AP50 >= 0.30：通过。
- 新增两类 GT 核心检查：**未通过**。
- 汇总：GT 43，TP 38，FP 16，GT 级错误 5，跨类重叠告警 12。
- 17-Eggs_Plush：GT 10 / TP 5 / FN 5 / FP 8，Recall 0.50，FAIL。
- 26-Plastic_Strawberry：GT 8 / TP 8 / FN 0 / FP 6，Recall 1.00，CAUTION。

## 阻断项

1. 绿色正面 Eggs 的 `001603` 至 `001607` 五张验证图在正式阈值下全部漏检。
2. 肉罐头 `001535` 同框高置信度误报 Eggs，score=0.94076，IoU=0.8924。
3. 肉罐头 `001551` 同框高置信度误报 Eggs，score=0.85001，IoU=0.9276。

不能通过降低阈值、过滤预测或修改 GT 掩盖这些问题。

## 证据

- `reports/track1_v9_offline_acceptance_20261031/v7_targeted_offline_audit_val.md`
- `reports/track1_v9_offline_acceptance_20261031/v7_targeted_offline_audit_val_errors.csv`
- `reports/track1_v9_offline_acceptance_20261031/v7_targeted_offline_audit_val_overlap_warnings.csv`
- `reports/track1_v9_offline_acceptance_20261031/contact_val_17_17-Eggs_Plush.jpg`
- `reports/track1_v9_offline_acceptance_20261031/contact_val_10_10-meat_can.jpg`

## 后续门槛

必须新增独立的绿色正面 Eggs 训练样本和新的肉罐头困难负样本，重建训练集后重新训练；验证集不得进入训练。后续候选必须通过同一套离线验收，再进行真实相机静态现场验收，全部通过后才允许替换生产模型。
