# Task B Point 2 第二次单次实体试抓记录（2026-10-27）

## 授权与范围

- 用户明确授权：“再一次进行 Point 2 实体抓取”。
- 范围：Point 2、单轮、单次实体抓取。
- 使用配置：`config/track1_task_b_point2_single_live_20261027.yaml`
- 配置仍只相对安全配置改动：
  - `motion.dry_run: false`
  - `gripper.backend: ms42dc`

## 执行前检查

- `bin/track1_task_runner check`：Python / ROS / YOLO / 动态任务B配置检查通过。
- `/dev/motor_serial -> ttyACM0`，当前用户可读写。
- 无残留任务、桥接或夹爪节点进程；PID 文件不存在；任务锁空闲。
- 单点位、单轮、固定 Z、学习偏移必需、夹爪反馈必需、搜索关闭等安全断言全部通过。
- 启动时 `/RobotInfo`：`state=AutoReady fault_flag=0`。

## 实际执行

- 任务日志：`logs/task_B_20261027_020333.log`
- 桥接日志：`logs/codroid_bridge_20261027_020333.log`
- 初始姿态残差：`0.014 deg`
- Point 2 观察姿态残差：`0.015 deg`
- 实时视觉连续确认 `8-Plastic_Banana`，置信度 `0.95`
- 目标中心：`(-0.201223, 0.575837, -0.034252) m`
- 检测主轴：`(0.917180, -0.398473)`
- 学习夹爪偏移：`[0.004289, -0.023589] m`
- `0°` pitch 候选在预抓取端点因 IK 容差门禁被拒绝，发生在开爪或任何运动之前；随后选择 `-5°` 分支。
- 完整六段路线预检通过：`min_sigma=0.017253`，关节行程 `265.20 deg`。
- 固定抓取法兰 Z：`0.166534092 m`
- 垂直下降距离：`0.0750 m`，横向偏移：`0.0000 m`
- 夹爪闭合反馈：`arrived, angle=2159.0 deg`
- 保持校验：`[verify_hold] jaws settled at 2159.0 deg`
- 完成抬升、2 秒保持、原位释放、退避，并返回 Point 2 观察姿态。
- 返回观察姿态残差：`0.017 deg`
- 状态机结果：`requested=1 attempts=1 successes=1 rate=100.00%`
- 完成记录：`[done] pick_2: object returned at its pick point and arm retreated; all expected points completed`

## 桥接证据

- 实体桥接发送 `16` 条 RobotMove 日志。
- 最终接收计数：`rx=3000`。
- 最终发送计数：`tx=16`。
- 未发现 `controller fault`、非零 `fault_flag` 或急停记录。
- 任务进程、桥接、夹爪节点和 PID 文件均已清理；任务锁为空闲。
- 桥接退出时仍有 rclpy 重复 shutdown 清理 traceback，发生在任务完成后的退出阶段，不影响本次动作。

## 结论

第二次授权的 Point 2 单次实体试抓已完成。日志证据覆盖视觉确认、逆解与完整路线预检、实体运动、夹爪闭合反馈、保持校验、抬升、释放、退避和观察姿态返回；实际夹持效果仍需现场目视确认。
