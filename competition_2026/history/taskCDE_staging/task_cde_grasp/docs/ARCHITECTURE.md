# 架构说明

## 组件边界

```text
                     +----------------------+
                     |   VisionSource       |
                     | D405 / replay / new  |
                     +----------+-----------+
                                | Frame[Detection]
                                v
+------------------+   +----------------------+   +----------------------+
| Task C/D/E       |<--| PickPipeline         |-->| MotionBackend        |
| TargetPolicy     |   | state machine        |   | mock / RobotMove     |
+------------------+   +----+------------+----+   +----------------------+
                            |            |
                            v            v
                    +-------------+  +------------------+
                    | PosePlanner |  | GripperBackend   |
                    | camera->base|  | mock / taskA     |
                    +-------------+  +------------------+
                            |            |
                            +-----+------+
                                  v
                           +---------------+
                           | HoldVerifier  |
                           +---------------+
```

`PickPipeline` 不导入 ROS，也不导入 RealSense/YOLO。它可以被纯 Python 测试，也可以被 ROS 2 节点调用。

## 为什么不把识别和运动写在一个节点里

任务 C/D/E 的差异主要在目标选择，不在机械动作：

- C 每次从合法候选里挑任意一个。
- D 接受已知类别和满足可抓性阈值的未见目标。
- E 按清单顺序锁定下一个身份。

因此 `TargetPolicy` 是策略层，`PickPipeline` 是执行层。后续把 YOLO 换成开放词表检测器时，抓取状态机不需要改。

## 坐标约定

- 所有 `Pose.xyz` 单位是米，`rpy` 单位是弧度。
- `Pose` 默认在机器人 base 坐标系。
- 固定相机：`extrinsics_file` 保存 `T_base_camera`。
- 腕部相机：`camera_frame: flange`，`extrinsics_file` 保存 `T_flange_camera`。
- `Detection.base_xyz` 只用于 mock 或上游已经完成坐标转换的系统。真实相机优先使用 `camera_xyz + 标定矩阵`。

## 失败处理

- 相机无帧：回到扫描，连续无帧由上层停止。
- 检测不合法：不动作，继续扫描。
- 规划越界：不动作，拒绝目标。
- 开夹爪失败：不进入接近路径。
- 运动失败/超时：中止本次抓取。
- 夹持确认失败：开爪，退回 pregrasp，不执行 lift。
- Task E 单目标失败次数达到 `max_attempts_per_target` 后才移动清单游标。
