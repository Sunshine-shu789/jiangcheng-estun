# Task B 规划阶段停顿优化（2026-10-20）

## 问题

Task B 每个点位会在规划阶段重复求解同一抓取列的以下路径：

```text
高位 -> 低位 -> 高位 -> 高位 -> 低位 -> 高位
```

2、5 号点姿态较难，每次低位/高位 IK 连续路径约需十余秒，因此重复求解造成明显停顿。

## 修改

`RobotMoveMotionBackend.precheck_linear_targets()` 增加路线内路径复用：

- 只有起点、终点、起点关节角和终点关节角全部严格一致时才复用。
- 反向路线直接倒序复用已通过完整预检的 Cartesian 与关节路径。
- 复用路线保留原路径的关节步长、加速度、Jacobian 奇异值和关节限位检查结果。
- 不同端点或不同关节姿态不会复用。
- 不修改抓取点、法兰 XY 偏移、观察姿态、速度、夹爪参数或最终运动轨迹。

复用日志标记：

```text
[motion:cartesian-precheck-reuse]
```

## 验证

- `test/test_motion_linear.py`：32 项通过
- `test/test_task_b.py`：56 项通过
- 相关 Task A/流水线回归：16 项通过
- 完整单元测试（加载 ROS 2 运行环境）：225 项通过
- `./bin/track1_task_runner check`：通过

本轮只修改离线规划逻辑并运行测试，没有启动任务，没有发送机械臂运动或夹爪动作命令。

## 备份

```text
backups/task_b_precheck_reuse_20261020_215351/
```
