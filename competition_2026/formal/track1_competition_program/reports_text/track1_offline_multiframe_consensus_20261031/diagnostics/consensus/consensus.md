# 离线多帧抓取共识报告

- 状态：`no_consensus_fail_closed`
- 批次：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/logs/offline_multiframe_20261031`
- 输入有效：`true`
- 达成共识：`false`
- 门禁：至少 `4` 个独立帧，TCP 平移容差 `25.0 mm`，夹爪对称旋转容差 `15.0°`
- 安全边界：`offline_only=true`、`motion_attempted=false`、`hardware_connected=false`、`motion_authorized=false`

## 输入统计

- 发现/成功读取帧：`20` / `20`
- 安全标记通过帧：`20`
- 计数完整性通过帧：`20`
- NMS 保留候选：`0`
- 实际 IK 评估：`0`
- IK 成功：`0`
- 纳入共识聚类：`0`
- 共识簇 / 通过簇：`0` / `0`

## 聚类结果

**没有 NMS+IK 成功候选，未形成任何共识簇。**

## 最佳共识

**未达到多帧一致门禁，已 fail-closed；不得把单帧偶然成功用于抓取。**

## 成员候选

| 簇 | 帧 | 候选 | 代表 | 质量 | 参考宽 mm | 开口 mm | 平移偏差 mm | 旋转偏差 ° | IK 位置误差 mm | IK 姿态误差 ° |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|

## 安全与限制

- Consensus only uses candidates that already passed final CAD/clearance checks, NMS and the existing strict IK acceptance gate.
- The pose gate is based on TCP base pose with 180-degree parallel-jaw symmetry; it does not prove the same contact patch or joint branch.
- A consensus cluster is an offline stability diagnostic, not a grasp-success probability, motion plan or execution authorization.
- No contact, collision, clearance, workspace, NMS or IK threshold was relaxed.
- No camera, robot, controller or gripper was connected and no motion or gripper command was attempted.
