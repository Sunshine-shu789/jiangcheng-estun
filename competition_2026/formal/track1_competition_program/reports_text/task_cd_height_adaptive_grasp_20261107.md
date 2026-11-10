# Task C/D 高度自适应抓取修改报告

日期：2026-11-07  
范围：Task C / Task D 的 `vertical_short_edge`、`vertical_center` 垂直抓取候选  
状态：代码、离线验证和 Task C 单次实机验收均已完成；Task D 实机验证待执行

## 1. 问题根因

旧配置对 `vertical_short_edge` / `vertical_center` 使用固定夹爪尖 Z：

```yaml
jaw_tip_z_m: 0.000103
```

实机日志 `logs/task_C_20261107_152251.log` 中，同帧几何抓取点 Z 为：

```text
jaw_tip_base_xyz Z = 0.012791516 m
```

随后被固定值改写为：

```text
fixed jaw-tip Z = 0.000103000 m
insertion_delta = -0.012688516 m
```

即额外向下压约 12.7 mm，导致抓取位置过低。

## 2. 修改后的算法

启用：

```yaml
height_adaptive_enabled: true
height_target_insertion_fraction: 0.50
height_min_insertion_fraction: 0.30
height_max_insertion_fraction: 0.75
height_min_support_clearance_m: 0.003
height_top_surface_tolerance_m: 0.008
height_min_visible_height_m: 0.008
```

同帧计算流程：

1. 将同帧支撑面中心点、支撑法向和可见物体高度转换到机器人基坐标系。
2. 由 `support_z + visible_height * up_world_z` 重建物体顶面 Z。
3. 如果点云还提供了顶面中心点，则检查它与重建顶面的差值；超过 `8 mm` 直接拒绝。
4. 夹爪尖 Z 取：

```text
jaw_tip_z = top_z - 0.50 * vertical_height
```

5. 结果不得低于：

```text
support_z + 0.003 m
```

6. 如果高度信息缺失、支撑面无效、顶面与高度矛盾、可见高度小于 `8 mm`，或有效插入比例不在 `[0.30, 0.75]`，该候选在运动前拒绝。

`vertical_rim` 继续使用自身测得的夹爪尖 Z，不使用本文的高度插入逻辑。

## 3. 回退路径封闭

高度模式启用后，旧的 `legacy_short_edge` PCA 回退路径已被禁止。这样在同帧支撑面或高度证据缺失时，不会静默回到固定 Z/视觉 Z 路径，而是直接拒绝候选，不张爪、不运动。

## 4. 修改文件

- `app/track1_grasp/track1_grasp/config.py`
- `app/track1_grasp/track1_grasp/pipeline.py`
- `app/track1_grasp/track1_grasp/planner.py`
- `app/track1_grasp/test/test_task_cd_short_edge_grasp.py`
- `app/track1_grasp/test/test_config.py`
- `config/track1_task_c_v11_official_20261102.yaml`
- `config/track1_task_d_v11_official_20261102.yaml`
- `config/track1_task_cd_40class_v11_official.yaml`

修改前备份：

```text
backups/task_cd_height_adaptive_20261107_155039/
```

## 5. 验证结果

离线全量测试：

```text
1081 passed in 116.56s
```

针对性测试：

- `test_task_cd_short_edge_grasp.py`：70 passed
- `test_config.py`：156 passed
- `compileall`：通过
- `bin/run_task_c --check`：通过
- `bin/run_task_d --check`：通过

测试已覆盖：

- 4 cm 高物体选择 `jaw_tip_z = 0.02 m`
- 8 cm 高物体选择 `jaw_tip_z = 0.04 m`
- 顶面高度与可见高度矛盾时无运动、无夹爪事件
- 高度模式下支撑面缺失时不使用 legacy PCA 回退
- 高度比例、配置互斥和最小支撑间隙非法时配置拒绝加载
- 原有抓取安全路径无回归

## 6. 标定板与只读几何诊断

透明收纳箱底部无法提供稳定的深度支撑面。在箱内放置标定板，并将实际目标放在标定板上后，使用以下只读工具验证了场景几何，未连接 RobotInfo、未启动运动桥接、未控制机械臂或夹爪：

```bash
bin/diagnose_task_c_geometry \
  --frames 15 \
  --require-known \
  --require-geometry-candidate
```

最终诊断日志：

```text
logs/task_c_geometry_diagnostic_20261107_161747
```

结果：

- 15/15 帧采集成功
- 15/15 帧稳定识别 `17-Eggs_Plush`
- 15/15 帧几何策略通过
- 物理稳定簇 1 个，稳定几何簇 1 个
- 全图有效深度比例均值 79.17%
- 中央区域有效深度比例均值 96.61%
- 目标中心基坐标中位值 `(-0.091852, 0.523342, 0.106420) m`
- 目标中心帧间最大抖动 2.54 mm，轴向范围 3.92 mm

标定板仅用于提供可测的哑光平面，不用于重新标定 D405 深度。棋盘格正面曾使 YOLO 产生一次低置信度假阳性；实际目标放到板上后，检测恢复为 15/15 帧稳定通过。现场不得将棋盘格本身作为抓取目标。

## 7. Task C 单次实机验收

用户确认允许实体机械臂和夹爪运动后，执行：

```bash
TRACK1_CONFIRM=1 bin/test_task_c_height_single_live
```

首次启动时控制器状态为 `Idle`，预检要求 `AutoReady`，程序在桥接阶段退出，`tx=0`，没有发送机械臂或夹爪命令。控制器切换到 `AutoReady` 后重新执行。

本次正式日志：

```text
logs/task_C_20261107_161947.log
logs/codroid_bridge_20261107_161947.log
```

进入观察位残差：

```text
residual_deg = 0.002
```

几何候选：

```text
label = 17-Eggs_Plush
geometry_mode = vertical_center
geometry_confidence = 0.724971
```

同帧高度自适应选择：

```text
support_z = 0.007024834 m
visible_height = 0.077797531 m
top_z = 0.083467168 m
target_insertion_fraction = 0.500000
effective_insertion_fraction = 0.500000
minimum_support_clearance = 0.003000000 m
selected_jaw_tip_z = 0.045246001 m
top_source = measured_top_centre
clearance_limited = False
```

夹爪闭合证据：

```text
width = 0.060486 m
travel = 1956.8 deg
required_travel = 662.6 deg
stable_samples = 4
hold_verified = true
```

随后完整执行并验证：

1. 夹爪张开
2. 移动到预抓取位
3. 按高度自适应 Z 下降
4. 夹爪闭合
5. `verify_hold` 确认物体已在夹爪内
6. 抬升
7. 转移到固定放置位
8. 释放
9. 退让
10. 返回固定观察位，残差 `0.010 deg`

最终结果：

```text
successes = 1
completed_tracks = 1
failed_tracks_pending_retry = 0
rejected_tracks_pending_retry = 0
final_status = success
```

单次验收门禁生效：完成 1/1 次完整抓取后退出，没有自动重试，且程序没有声称收纳箱已经清空。

## 8. Task D 状态

Task D 尚未运行。Task D 针对未知物体，现场需要先放入合适的未知目标并确认工作区安全，再单独执行：

```bash
TRACK1_CONFIRM=1 bin/test_task_d_height_single_live
```

不得直接用本次 Task C 已转移的已知物体结果代替 Task D 验收。
