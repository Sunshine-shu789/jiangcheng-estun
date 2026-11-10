# Task C/D RobotInfo 抓图后误报过期修复

日期：2026-11-05  
执行范围：离线代码修复与 pytest；未启动机械臂、夹爪或 Task C 真机任务。

## 故障现象

2026-11-05 12:27:45 启动的 Task C 在首帧检测时中止：

```text
[detect] C/D clear observation fixed view round 1/2: capturing one frame
[yolo] loading .../track1_40class_v11_official_f43234a9.pt
[vision:realsense_yolo] ready ...
[abort] ... frame 'realsense-000001' after state is not ready: no fresh RobotInfo
```

控制器网络、UDP 桥接、D405 相机和 GPU 推理均已确认正常。桥接日志在同一时段持续接收并解码 RobotInfo，未发送运动命令。

## 根因

`_task_cd_capture_frame()` 在 `vision.capture()` 返回后立即读取 `RobotInfo`。首次采集包含相机启动、YOLO 模型加载和首次 CUDA 推理，本次阻塞约 7.7 秒。在该 Python 调用期间，ROS executor 的 RobotInfo 回调得不到及时执行；抓图返回时最后样本已超过 `robot_info_timeout_s: 2.0`，安全检查按设计 fail-closed 中止。

这不是通过放宽新鲜度阈值解决的问题，否则会弱化实体机械臂安全检查。

## 修复

- `RobotMoveMotionBackend` 为每个物理 RobotInfo 维护递增序号。
- 新增 `get_feedback_sequence()` 和 `wait_for_feedback_state()`。
- `wait_for_feedback_state()` 始终读取物理反馈，不使用 dry-run 虚拟姿态。
- C/D 抓图前记录反馈序号；抓图后仅等待一个“抓图开始之后收到”的物理反馈，等待上限仍使用 `robot_info_timeout_s`。
- 收到新反馈后继续执行原有关节漂移、位姿漂移、故障字和 AutoReady 检查。
- 若抓图后没有新反馈，仍在有限时间内失败，不会用旧样本冒充有效反馈。

## 回归测试

新增：

- 旧抓图前样本不能直接通过，必须等待新反馈。
- 抓图/首次推理阻塞超过反馈超时后，只要新 RobotInfo 到达即可恢复。
- 抓图后没有新 RobotInfo 时仍必须 fail-closed。
- dry-run 开启时等待接口仍只返回物理反馈。

结果：

```text
针对性测试：3 passed
test_motion_linear.py：54 passed
test_task_cd_fixed_observation_clear.py：84 passed
全量离线测试：911 passed in 110.23s
```

## 备份

```text
backups/robotinfo_capture_wait_20261105_123321/
```

## 现场下一步

现场环境确认安全后，重新启动 Task C。首次检测仍可能因模型加载耗时较长，但抓图后会等待抓图后的物理反馈，再执行安全检查。若 2 秒内仍收不到新 RobotInfo，任务会继续安全中止并保留明确日志。
