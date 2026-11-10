# Task B：2 号点垂直抓取与 2～5 号点抓取高度修正

日期：2026-10-20  
正式目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序`

## 1. 本次修改

### 2 号点改为从示教观察位正下方抓取

新增按点位配置：

```yaml
point_grasp_xy_from_observation: [false, true, false, false, false]
```

- 仅 `pick_2` 启用。
- 香蕉仍必须先通过实时视觉检测和目标点筛选，视觉安全门没有被绕过。
- 视觉确认后，`pick_2` 的抓取与释放 XY 使用该点示教观察位的 FK 结果，
  不再被视觉 XY 横向拉偏。
- 高位航点 `pregrasp/lift/transfer/retreat` 继续使用示教观察锚点。
- 抓取高度和抓取姿态继续使用 2 号点专用标定值。

2 号点当前观察锚点与抓取参数：

```text
anchor_xyz = (-0.124883909, 0.521708714, 0.241215741) m
grasp_xy   = (-0.124883909, 0.521708714) m
grasp_z    = 0.166534092 m
```

因此 `pregrasp -> grasp` 的水平目标位置保持不变，只沿 Z 方向下降。

### 2～5 号点抓取高度增加 5 mm

| 点位 | 原抓取 Z (m) | 新抓取 Z (m) |
|---|---:|---:|
| pick_2 | 0.161534092 | 0.166534092 |
| pick_3 | 0.167258106 | 0.172258106 |
| pick_4 | 0.164424007 | 0.169424007 |
| pick_5 | 0.161454634 | 0.166454634 |

正式配置：

```text
config/track1_task_b_final_20261020_live.yaml
config/track1_task_b_resume2_final_20261020_live.yaml
```

续跑配置只包含 `pick_2`～`pick_5`，因此其
`point_grasp_xy_from_observation` 为 `[true, false, false, false]`。

## 2. 离线验证

两份正式配置均通过：

```text
Python / ROS / YOLO / 五个任务配置检查通过
```

全量单元与回归测试：

```text
200 passed in 12.16s
```

使用 `track1_task_b_resume2_final_20261020_live.yaml` 对 2～5 号点执行完整
六段路线只读预检，全部 PASS：

| 点位 | min_sigma | 最大关节步长 | 航点数 |
|---|---:|---:|---:|
| pick_2 | 0.017237 | 0.499746° | 224 |
| pick_3 | 0.057409 | 0.363841° | 220 |
| pick_4 | 0.060262 | 0.435090° | 280 |
| pick_5 | 0.049869 | 0.300161° | 156 |

安全阈值未放宽：

```text
speed_mps = 0.06
cartesian_step_m = 0.002
cartesian_joint_step_deg = 0.5
cartesian_joint_acceleration_limit_deg = 0.25
cartesian_min_jacobian_singular_value = 0.01
```

## 3. 修改前备份

```text
backups/task_b_point2_vertical_grasp_20261020_042628/
```

## 4. 动作声明

本轮只修改配置和程序，并完成离线配置检查、全量测试及只读路线预检。
**没有发送任何机械臂运动或夹爪动作命令，没有重启现场 ROS 节点。**

下一步实机验证必须重新获得明确的现场安全确认和运动授权。


## 5. 单点实机验证结果

2026-10-20 04:31:30，在用户明确授权后，使用
`config/track1_task_b_point2_live_once.yaml` 完成一次 2 号点低速垂直原位拾放。

```text
speed_mps = 0.01
requested = 1
attempts = 1
successes = 1
result = 100%
```

测试前后机械臂均为 `AutoReady`、`fault_flag=0`；动作结束后已退回 2 号点观察位，
夹爪完成张开。详细记录：

```text
logs/task_b_point2_vertical_live_once_20261020_043130.md
logs/task_B_20261020_043130.log
logs/task_b_point2_vertical_route_readonly_20261020_043041.json
```
