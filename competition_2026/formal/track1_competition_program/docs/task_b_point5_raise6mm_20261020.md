# Task B 5 号点垂直抓取高度提高 6 mm

日期：2026-10-20  
正式目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序`

## 1. 本次修改

根据现场已成功完成的 5 号点实机拾放结果，将 5 号点抓取高度在原来的
`0.161454634 m` 基础上正上方提高 `6 mm`：

```text
旧抓取 Z：0.161454634 m
新抓取 Z：0.167454634 m
```

抓取列固定为刚才实机测试使用的 XY：

```text
grasp_xy = (0.043300, 0.522000) m
```

点位配置中锁定该 XY 后，`pregrasp -> grasp -> lift -> transfer ->
release -> retreat` 六个航点都使用同一组 XY，只沿 Z 方向垂直运动，不再从
视觉横向偏移位置斜着进入或退出盒子。

修改的配置：

```text
config/track1_task_b_point5_live_once.yaml
config/track1_task_b_point5_single.yaml
config/track1_task_b_final_20261020_live.yaml
config/track1_task_b_resume2_final_20261020_live.yaml
```

修改前备份：

```text
backups/task_b_point5_raise6mm_20261020_054745/
```

## 2. 离线路线验证

使用修改后的配置执行只读六段路线预检，没有发送任何机械臂或夹爪动作命令。

```text
track1_task_b_point5_live_once.yaml:
  route = PASS
  segments = 6
  cartesian waypoints = 393
  min_sigma = 0.027962
  max_joint_step = 0.494291 deg
  six-waypoint XY spread = 0.000000000 mm

track1_task_b_final_20261020_live.yaml:
  route = PASS
  segments = 6
  cartesian waypoints = 392
  min_sigma = 0.017946
  max_joint_step = 0.495450 deg
  six-waypoint XY spread = 0.000000000 mm
```

两份预检结果中的抓取 Z 均为 `0.167454634 m`，高位航点 Z 均为
`0.247454634 m`。全部航点 XY 完全一致，确认最终抓取和放回过程为垂直路径。

## 3. 配置和回归检查

四份配置均通过：

```text
Python / ROS / YOLO / 五个任务配置检查通过
```

完整单元测试和回归测试：

```text
204 passed in 12.44s
```

同时核对了四份配置中的 `pick_5`，实际加载值均为：

```text
grasp_z_m = 0.167454634
grasp_xy_m = (0.043300, 0.522000)
```

## 4. 动作声明

本轮只修改配置、测试断言和记录文档。没有发送 `/RobotMove` 命令，没有发送
夹爪动作，也没有重启现场机械臂或 ROS 节点。

新的 5 号点高度要实机确认时，必须先确认现场安全，并由操作员明确授权一次
低速单点拾放测试；不能直接双击整套任务连续运行来代替单点验证。
