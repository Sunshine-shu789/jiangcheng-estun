# Task B 连续轨迹与逐点回初始位修正

日期：2026-10-20  
正式目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序`  
修改前备份：`backups/continuous_motion_fix_20261020_062110/`

## 1. 现场现象

最近一次五点位实机运行的日志为：

```text
logs/task_B_20261020_060029.log
```

结果是 1 号点完成，2 号点视觉未确认香蕉后，程序在搜索回程预检阶段终止：

```text
requested=5 attempts=2 successes=1 rate=50.00%
joint-step limit cannot be satisfied at the minimum spacing
(0.501 deg > 0.500 deg) at target 1 path waypoint, depth=3
```

这不是碰撞、断线、夹爪故障或控制器失联。它有两个叠加原因：

1. 旧的 Cartesian 执行器把每个约 2 mm 的 IK 航点单独发布成一条
   `JointTrajectory`。控制器在相邻航点之间也要经过
   `AutoRunning -> AutoReady`，因此运动频繁停顿，等待和超时机会也被放大。
2. 视觉搜索全部未命中时，旧代码用 Cartesian `move_to()` 从搜索姿态回初始位。
   此时腕部已接近奇异边界，回程第一个航点的 IK 步长达到 `0.501 deg`，超过
   不允许放宽的 `0.500 deg` 连续性上限，所以程序在发命令前拒绝并终止。

## 2. 本次代码修正

### 2.1 每段一条完整多点轨迹

`app/track1_grasp/track1_grasp/motion.py` 中：

- `execute_prechecked_segment()` 把整段已经通过 IK、工作空间、关节限位、
  步长、加速度、近奇异和起点陈旧检查的关节路径一次性发布。
- 一段只等待一次 `AutoRunning -> AutoReady`，并检查段终点。
- `move_joints()` 也一次发布整条六轴插值轨迹。
- 轨迹超时按航点数量动态延长：
  `max(joint_move_timeout_s, move_timeout_s, 10 + 0.5 * point_count)`。
- UDP 桥仍将全部点序列化在一个 IPv4 UDP 包内，因此增加 65507 字节硬上限，
  防止过长路径在 `sendto()` 时才失败。

### 2.2 搜索未命中改用六轴关节回程

`app/track1_grasp/track1_grasp/pipeline.py` 中，搜索未命中且配置
`search.return_to_initial_on_miss: true` 时，改为：

```python
self.motion.move_joints(
    initial_state.joints,
    name="task_search_return",
)
```

这样不再从条件很差的腕部姿态重新做笛卡尔 IK，也不会触发本次
`0.501 deg > 0.500 deg` 的终止。

## 3. 正式执行顺序

正式配置：

```text
config/track1_task_b_final_20261020_live.yaml
```

桌面入口：

```text
/home/sunshine/Desktop/赛道一比赛程序/任务B_五点位拾放.desktop
```

双击后的完整顺序为：

```text
启动并确认六轴初始位
-> 当前点观察位
-> 新帧视觉确认香蕉
-> pregrasp -> grasp -> 合爪 -> 夹持确认
-> lift -> transfer -> 原地 release -> 张开 -> retreat
-> 六轴回初始位并验证到位
-> 若未完成 5 个点，等待操作员更换香蕉并按 Enter
-> 下一点重新视觉确认后才运动
-> 1 到 5 号点全部完成，且 5 号点放回后已回初始位
-> 程序结束
```

`task_b.operator_pause_after_each_pick: true` 会在每轮回初始位之后暂停；
`task_b.return_to_initial_after_each_pick: true` 保证下一点开始前一定先回到
共同初始位；`task_b.return_to_initial_after_task: false` 用于避免 5 号点
完成后重复回位。

## 4. 离线验证结果

```text
针对性测试：71 passed
其余全量回归：201 passed
rclpy / arm_motion_check：6 passed
合计：207 passed
正式 live 配置五点 mock：1 passed
Python / ROS / YOLO / 五个任务配置检查：通过
```

定向命令：

```bash
cd /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_grasp
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /usr/bin/python3 -m pytest -q test/test_motion_linear.py test/test_task_b.py
```

正式配置 mock 测试确认执行顺序为：

```text
task_b_initial_pose
task_b_observation_pick_1
task_b_return_initial_pick_1
...
task_b_observation_pick_5
task_b_return_initial_pick_5
```

每轮 6 段 Cartesian 路径全部使用 `0.06 m/s`，5 个点共 30 段；每段现在是
一条连续多点关节轨迹。

## 5. 安全状态与下一步

本轮只完成代码修改、正式环境检查和离线回归，**没有发送机械臂或夹爪动作，
没有启动正式任务**。

连续多点轨迹属于运动协议变化，不能直接把本次离线通过当作五点位实机通过。
下一步必须先做一次低速单点实机验证，确认：

- 段内不再频繁停顿；
- 控制器不会因零时间戳多点轨迹拒绝命令；
- 段终点误差正常；
- 六轴回初始位正常；
- 夹爪动作时序不变。

单点通过后，才可申请五点位整体实机运行。
