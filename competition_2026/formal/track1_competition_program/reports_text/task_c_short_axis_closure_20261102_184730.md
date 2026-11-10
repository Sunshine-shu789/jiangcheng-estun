# Task C 短边闭合轴修复

## 根因

2026-11-02 的 4-Frenchs_Mustard 实机尝试中，抓取点确实是点云几何中心，但夹爪闭合轴被对齐到了 PCA 长轴。平行夹爪沿物体长边张合，无法跨两个短边夹紧物体，因此出现“夹住又松开/没有抓起来”。

## 修复

- 正式短边路径：闭合轴改为对齐 PCA 短轴。
- 几何策略路径：`vertical_short_edge` 的 `closure_axis_camera` 改为 `footprint_minor_axis_camera`。
- 抓取点保持优先使用点云几何中心。
- `4-Frenchs_Mustard` 等带柄物品的端部兜底仍沿 PCA 长轴偏移 30% 长轴长度。
- 现场安装方向保持 `closure_axis_tool_xy: [1.0, 0.0]`。
- 只读诊断工具同步输出短轴及短轴对齐误差。

现场姿态回归：

```text
major_axis_base_xy=(0.422413, -0.905578)
expected_minor_axis_base_xy=(0.905578, 0.422413)
expected_closure_axis = PCA minor axis
```

## 验证

- 专项回归：`98 passed`
- 全量回归：`734 passed`
- 本轮未运行实体机械臂、夹爪或 Task D。

## 备份

```text
/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/backups/task_c_short_axis_closure_20261102_184309
```

## 状态

Task C 代码修复和离线测试完成，尚需一次新的、明确的现场安全确认后，单独运行 Task C 做实机验收。Task C 完整通过前不运行 Task D。
