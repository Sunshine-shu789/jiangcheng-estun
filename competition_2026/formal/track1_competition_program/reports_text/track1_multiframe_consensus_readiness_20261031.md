# Task C/D 多帧抓取共识证据清单

- 生成时间：`2026-10-31T13:02:29+08:00`
- 状态：`inventory_complete_requires_per_scene_evidence`
- 已发现多帧批次：`2`
- 达成共识：`1`
- fail-closed：`1`

## 场景结果

| 场景 | 帧 | 合格候选 | 通过簇 | 最大独立帧 | 结论 |
|---|---:|---:|---:|---:|---|
| `aperture_multiframe_20261031` | 20 | 8 | 1 | 5 | 通过离线共识门禁 |
| `offline_multiframe_20261031` | 20 | 0 | 0 | 0 | 拒绝，无可用共识 |

## 安全状态

```text
offline_only=true
motion_attempted=false
hardware_connected=false
motion_authorized=false
quality_validated=false
```

## 证据文件

- `aperture_multiframe_20261031`：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_graspnet_aperture_multiframe_20261031/diagnostics/consensus/consensus.json`
  - SHA-256：`39a58f5bd2d27eae27e3f82929b32608a4a2a4656f0fe222ce4e522958f43db8`
- `offline_multiframe_20261031`：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_offline_multiframe_consensus_20261031/diagnostics/consensus/consensus.json`
  - SHA-256：`e5eb50980047bdb29b010a76a89b6c4cebc032f4f0f21d842bb3e7ade725b92f`

## 当前结论

现有离线批次中只有一个场景形成有效共识；另一个 20 帧批次没有任何 NMS+IK 合格候选，已正确 fail-closed。
这些结果不能代表 38 类 Task C/D 的现场能力，也不能作为实体运动授权。

## 下一步

1. 对每个真实 Task C/D 场景和物体分别采集多帧并生成独立 `consensus.json`。
2. 用对应代表位姿做 dry-run/mock 在线规划绑定，拒绝报告与场景错配。
3. 完成 TCP/flange 语义、手眼外参、工作空间、限位和夹爪阈值现场验证后，才讨论启用。
