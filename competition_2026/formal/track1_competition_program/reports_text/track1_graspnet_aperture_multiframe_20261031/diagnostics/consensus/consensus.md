# 离线多帧抓取共识报告

- 状态：`consensus_available_offline_only`
- 批次：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/logs/offline_graspnet_aperture_multiframe_20261031_120309`
- 输入有效：`true`
- 达成共识：`true`
- 门禁：至少 `4` 个独立帧，TCP 平移容差 `25.0 mm`，夹爪对称旋转容差 `15.0°`
- 安全边界：`offline_only=true`、`motion_attempted=false`、`hardware_connected=false`、`motion_authorized=false`

## 输入统计

- 发现/成功读取帧：`20` / `20`
- 安全标记通过帧：`20`
- 计数完整性通过帧：`20`
- NMS 保留候选：`12`
- 实际 IK 评估：`12`
- IK 成功：`8`
- 纳入共识聚类：`8`
- 共识簇 / 通过簇：`4` / `1`

## 聚类结果

| 簇 | 门禁 | 候选数 | 独立帧 | 帧号 | 中位质量 | 开口范围 mm | 最大平移 mm | 最大旋转 ° |
|---|---|---:|---:|---|---:|---:|---:|---:|
| consensus_001 | 通过 | 5 | 5 | 9, 12, 14, 15, 19 | 0.7739 | 34.67–36.40 | 21.39 | 0.00 |
| consensus_002 | 拒绝 | 1 | 1 | 22 | 0.8194 | 33.08–33.08 | 0.00 | 0.00 |
| consensus_003 | 拒绝 | 1 | 1 | 23 | 0.7173 | 34.67–34.67 | 0.00 | 0.00 |
| consensus_004 | 拒绝 | 1 | 1 | 11 | 0.7098 | 38.12–38.12 | 0.00 | 0.00 |

## 最佳共识

- 代表候选：`frame_19/graspnet_00050`
- 独立帧：`5`（9, 12, 14, 15, 19）
- 候选：`5`
- 质心 TCP（基座系，m）：`[-0.047473, 0.510328, 0.027063]`
- 共识开口范围：`34.67–36.40 mm`
- 簇内最大平移偏差：`21.39 mm`
- 簇内最大夹爪对称旋转偏差：`0.00°`
- 最小支撑面净空：`21.152 mm`
- 该门禁只证明离线位姿稳定，不证明实机可抓，也不构成运动授权。

## 成员候选

| 簇 | 帧 | 候选 | 代表 | 质量 | 参考宽 mm | 开口 mm | 平移偏差 mm | 旋转偏差 ° | IK 位置误差 mm | IK 姿态误差 ° |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| consensus_001 | 9 | graspnet_00023 |  | 0.8516 | 59.15 | 36.40 | 9.93 | 0.00 | 0.0019 | 0.0054 |
| consensus_001 | 12 | graspnet_00032 |  | 0.8012 | 49.91 | 35.11 | 8.69 | 0.00 | 0.0014 | 0.0053 |
| consensus_001 | 14 | graspnet_00009 |  | 0.7401 | 54.43 | 35.65 | 5.26 | 0.00 | 0.0024 | 0.0054 |
| consensus_001 | 15 | graspnet_00100 |  | 0.7277 | 57.95 | 35.08 | 11.57 | 0.00 | 0.0023 | 0.0054 |
| consensus_001 | 19 | graspnet_00050 | 是 | 0.7739 | 49.11 | 34.67 | 0.00 | 0.00 | 0.0024 | 0.0054 |
| consensus_002 | 22 | graspnet_00009 | 是 | 0.8194 | 54.32 | 33.08 | 0.00 | 0.00 | 0.0042 | 0.0108 |
| consensus_003 | 23 | graspnet_00005 | 是 | 0.7173 | 55.42 | 34.67 | 0.00 | 0.01 | 0.0044 | 0.0080 |
| consensus_004 | 11 | graspnet_00007 | 是 | 0.7098 | 61.83 | 38.12 | 0.00 | 0.00 | 0.0049 | 0.0053 |

## 安全与限制

- Consensus only uses candidates that already passed final CAD/clearance checks, NMS and the existing strict IK acceptance gate.
- The pose gate is based on TCP base pose with 180-degree parallel-jaw symmetry; it does not prove the same contact patch or joint branch.
- A consensus cluster is an offline stability diagnostic, not a grasp-success probability, motion plan or execution authorization.
- No contact, collision, clearance, workspace, NMS or IK threshold was relaxed.
- No camera, robot, controller or gripper was connected and no motion or gripper command was attempted.
