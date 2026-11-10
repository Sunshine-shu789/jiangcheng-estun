# Task B Point 2 单次实体试抓记录（2026-10-27）

## 授权与范围

- 用户明确授权：“现场检查通过，允许对 Point 2 进行单次实体试抓”。
- 仅执行 Point 2、单轮、单次抓取。
- 专用配置：`config/track1_task_b_point2_single_live_20261027.yaml`
- 该配置由安全配置复制，语义差异仅为：
  - `motion.dry_run: true -> false`
  - `gripper.backend: mock -> ms42dc`
- 默认 `bin/run_task_b` 仍指向 `config/track1_task_b_point2_safe_20261027.yaml`，未改为默认实机运行。

## 执行前检查

- `bin/track1_task_runner check`：Python / ROS / YOLO / 动态任务B配置检查通过。
- `/dev/motor_serial -> ttyACM0`，当前用户可读写。
- 无残留 `track1_grasp`、`udp_bridge`、`run_task_b` 或 `motor_node` 任务进程。
- 安全断言通过：单次/单轮、仅 `pick_2`、固定教学 Z、硬 Z 下限、视觉 Z 禁用、学习夹爪偏移必需、夹爪反馈必需、搜索关闭。
- 启动时 `/RobotInfo`：`state=AutoReady fault_flag=0`。

## 实际执行

- 任务日志：`logs/task_B_20261027_015603.log`
- 桥接日志：`logs/codroid_bridge_20261027_015603.log`
- 启动后按配置先移动到初始姿态，再移动到 Point 2 教学观察姿态：
  - 初始姿态残差：`0.012 deg`
  - 观察姿态残差：`0.013 deg`
- 实时视觉确认 `8-Plastic_Banana`，连续确认，置信度 `0.94`。
- `0 deg` pitch 候选在预抓取端点因 IK 容差门禁被拒绝，发生在开爪或任何运动之前；随后选择已验证的 `-5 deg` 分支。
- 完整六段路线预检通过：`min_sigma=0.017260`，关节行程 `245.34 deg`。
- 固定抓取法兰 Z：`0.166534092 m`。
- 垂直下降距离：`0.0750 m`，横向偏移：`0.0000 m`。
- 夹爪闭合反馈：`arrived, angle=2158.9 deg`。
- 保持校验：`[verify_hold] jaws settled at 2158.9 deg`。
- 完成抬升、2 秒保持、原位释放、退避，并返回 Point 2 观察姿态；返回残差 `0.016 deg`。
- 状态机结果：`requested=1 attempts=1 successes=1 rate=100.00%`。
- `[done] pick_2: object returned at its pick point and arm retreated; all expected points completed`。

## 桥接证据

- 实体桥接日志出现 `tx=1` 到 `tx=17`，最终 `tx=17`。
- RobotInfo 接收计数最终 `rx=3101`。
- 未发现 `controller fault`、非零 `fault_flag` 或急停记录。
- 任务进程、桥接进程、夹爪节点和 PID 文件均已清理；任务锁为空闲。
- 桥接退出时出现一次 rclpy 重复 shutdown traceback，发生在任务完成后的清理阶段，不影响本次实体动作和结果；后续可单独修复清理逻辑，不应与运动逻辑混改。

## 结论

本次授权的 Point 2 单次实体试抓已完成，实体运动命令、夹爪闭合反馈、保持校验、抬升、释放、退避和观察姿态返回均有日志证据。状态机报告成功；实际抓持的最终物理效果仍应以现场目视检查为准。
