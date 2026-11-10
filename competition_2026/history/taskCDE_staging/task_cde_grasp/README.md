# Task C/D/E 视觉-机械臂-夹爪框架

这个包先解决“系统怎么组织”，而不是立即绑定某一款相机或夹爪。当前可以在**没有相机、没有夹爪**的情况下完整跑通：

```text
假相机帧 -> 目标策略(C/D/E) -> 相机/基座坐标变换 -> 抓取路径
         -> 开夹爪 -> 接近 -> 下探 -> 合夹爪 -> 夹持确认
         -> 抬起 -> 转移 -> 放置 -> 退回
```

机械臂、相机和夹爪分别放在独立接口后面，后续装好硬件时只需替换适配器，不改任务状态机。

## 当前实现

- `mock` 相机：从 YAML 读取固定检测帧，验证遮挡、类别和坐标流程。
- `taska_realsense` 适配器：预留接入现有 D405 + YOLO 代码。
- `mock` 机械臂：记录每个 Cartesian 路径点并模拟执行。
- `robotmove` 机械臂：通过 `/RobotMove` 发布 `JointTrajectory`，订阅 `/RobotInfo`。
- `mock` 夹爪：记录 open/close，并模拟夹持确认。
- `taska` 夹爪：预留接入现有夹爪后端。
- 任务策略：
  - C：`any`，抓任意合法目标。
  - D：`known_and_unseen`，已知类别或带 `object/unknown` 标签且 `graspability` 达标的目标。
  - E：`ordered`，严格按 `target.order` 或 `order_file` 的顺序。
- 安全门：工作空间边界、最低置信度、深度要求、机器人 `AutoReady`、移动超时、夹持确认失败后开爪退回。
- 默认 `dry_run: true`。只有显式 `--execute` 才关闭运动 dry-run。

## 目录

```text
task_cde_grasp/
  task_cde_grasp/
    domain.py       # Pose/Detection/状态等纯数据
    interfaces.py   # Vision/Motion/Gripper/HoldVerifier 协议
    policy.py       # C/D/E 目标选择
    planner.py      # 相机到基座变换 + 抓取路径点
    pipeline.py     # 可观测状态机
    vision.py       # mock + taskA RealSense/YOLO 适配器
    motion.py       # mock + RobotMove 适配器
    gripper.py      # mock + taskA 适配器
    factory.py      # 配置装配
    nodes/          # 离线 demo 和 ROS 2 入口
  config/
    cde_mock.yaml
    cde_real_template.yaml
    camera_to_base.example.yaml
    target_order.example.yaml
  scenarios/mock_cde_scene.yaml
  test/
  launch/
```

## 不接硬件先跑

在当前源码目录：

```bash
PYTHONPATH=. python3 -m task_cde_grasp.nodes.mock_demo \
  --config config/cde_mock.yaml --mode C
```

切换任务：

```bash
PYTHONPATH=. python3 -m task_cde_grasp.nodes.mock_demo \
  --config config/cde_mock.yaml --mode D

PYTHONPATH=. python3 -m task_cde_grasp.nodes.mock_demo \
  --config config/cde_mock.yaml --mode E
```

运行测试：

```bash
PYTHONPATH=. python3 -m pytest -q
```

## ROS 2 构建与运行

建议把它放进新加卷中的工作空间：

```bash
cd /media/sunshine/新加卷/taskCDE/ros2_ws
source /opt/ros/humble/setup.bash
# 新加卷是 NTFS 挂载，普通构建（不用 --symlink-install）更稳妥
colcon build --packages-select task_cde_grasp
source install/setup.bash
ros2 run task_cde_grasp mock_demo --config src/task_cde_grasp/config/cde_mock.yaml
```

也可使用 launch：

```bash
ros2 launch task_cde_grasp cde_mock.launch.py mode:=E max_picks:=4
```

## 状态机

核心阶段由 `PickPhase` 枚举，每次抓取都会返回事件序列：

```text
idle
  -> scan -> detect -> select -> plan
  -> open_gripper
  -> approach -> grasp
  -> close -> verify_hold
  -> lift -> transfer -> release -> retreat
  -> done / abort
```

顺序要点：

1. 先识别并筛选目标。
2. 确认是合法目标后，先开夹爪。
3. 机械臂走 `pregrasp -> grasp`。
4. 到抓取点后才合夹爪。
5. 夹持确认通过后才抬起，否则开爪并退回。
6. 有 `place_pose` 时转移并释放，否则抬起后保持夹持。

## 接真机时需要替换什么

1. **相机机械安装**：固定相机或腕部相机都可以，配置 `pose.camera_frame`。
2. **手眼标定**：把结果写入 `pose.extrinsics_file`。
3. **检测器**：
   - C：可先用类别无关的深度分割/通用目标检测。
   - D：增加 `object/unknown` 输出和 `graspability` 分数。
   - E：检测标签必须能匹配 `target_order`。
4. **夹爪协议**：实现 `GripperBackend.open/close/state`，或在 `taska` 适配器后面接现有 RS485/DO 后端。
5. **夹持确认**：用夹爪位置/电流、腕部相机位移或“抬起 2 cm 后复查”替换 `MockHoldVerifier`。
6. **碰撞规划**：当前路径是 Cartesian 直线点。真机阶段建议让 `MotionBackend` 调用 MoveIt，或由控制器负责碰撞检查。
7. **现场标定与安全**：先用 `dry_run: true` 核对全部路径点，再逐项 `--execute`。

详细说明见 `docs/ARCHITECTURE.md` 和 `docs/INTEGRATION_CHECKLIST.md`。

## 当前边界

- 尚未安装相机和夹爪，因此真实视觉、夹爪反馈和夹持检测都没有进入闭环验证。
- Task D 的“未见物体”需要检测器输出通用 `object` 框及可抓性分数；本仓库目前提供接口和门控，不包含新的开放词表模型。
- 当前规划器只做笛卡尔路径点和工作空间检查，不负责机械臂碰撞规划。
- `robotmove` 适配器复用现有 Codroid UDP 桥的思路，实际运行前仍需确认 ROS 话题、控制器状态字和末端位姿单位。
