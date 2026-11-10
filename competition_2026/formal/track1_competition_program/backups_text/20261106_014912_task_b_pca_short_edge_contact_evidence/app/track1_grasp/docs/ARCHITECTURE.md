# 架构说明（赛道一 A/B/C/D/E）

## 1. 总体原则：五个独立入口，一套底层核心

五个任务不复制五套业务代码，但正式入口、配置、启动和验收必须彼此独立：

```text
launch/task_a.launch.py ─┐
launch/task_b.launch.py ─┤
launch/task_c.launch.py ─┼─> launch_helpers.build_task_launch()
launch/task_d.launch.py ─┤          │
launch/task_e.launch.py ─┘          ▼
                         Track1Config + 共享 factory / 状态机 / 硬件适配器
```

任务 C/D 的现场桌面入口分别是：

```text
bin/run_task_c -> config/track1_task_c_v11_official_20261102.yaml
bin/run_task_d -> config/track1_task_d_v11_official_20261102.yaml
```

共享底层库不等于共享任务入口。C、D 必须分别启动、分别验收，并遵守 C 完成后再
做 D 的顺序。

## 2. 分层

```text
nodes/ (ROS 2 / 离线入口)
  track1_node
  search_approach_node      当前现场识别、搜索、靠近测试
  mock_demo
  annotate_node
             │
             ▼
factory.build_pipeline(config)
  A       -> VisionDemoPipeline
  B/C/D/E -> PickPipeline
             │
     ┌───────┼─────────────────┐
     ▼       ▼                 ▼
 VisionSource MotionBackend GripperBackend
```

核心接口：

| 接口 | 职责 | 当前实现 |
| --- | --- | --- |
| `VisionSource` | 输出 `Frame`（框、置信度、深度、相机坐标） | `MockVisionSource`、`TaskARealSenseVision`、`RealSenseYoloVision` |
| `MotionBackend` | 读取机器人状态并执行运动 | `MockMotionBackend`、`RobotMoveMotionBackend` |
| `GripperBackend` | 开爪、合爪、状态反馈 | mock / taskA / MS42DC / serial / TCP / ROS |
| `HoldVerifier` | 合爪后夹持确认 | mock、MS42DC 反馈等 |

## 3. 当前现场测试链路

当前优先测试：

```text
D405 RGB + depth
  -> 正式 V11 40 类权重（加载前校验 SHA256）
  -> 任务目标策略（C only known / D known_and_unseen）/ 置信度 / 深度筛选
  -> 当前末端姿态下的 eye-in-hand 位移
  -> full-pose IK
  -> 每关节最多 1° 插值
  -> /RobotMove
  -> 每次最多 5 mm
  -> 只靠近，不合爪
```

实现文件：

- `vision.py::RealSenseYoloVision`
  - 直接从 D405 读取对齐后的 RGB/深度。
  - 倒装相机场景下，RGB、深度和内参点同时旋转 180°：
    `cx' = width - 1 - cx`、`cy' = height - 1 - cy`。
  - 检测结果携带 `u/fx/cx` 等内参，供视觉伺服使用。
- `approach.py`
  - `select_target()`：过滤类别、置信度、深度和相机坐标。
  - `compute_servo_command()`：像素误差 -> 相机位移 -> 当前工具坐标系 ->
    基座坐标，单步范数限幅。
  - `pose_inside_workspace()`：工作空间硬限位和余量检查。
- `kinematics.py`
  - 与控制器一致的 modified-DH、FK、full-pose IK、方向 IK、关节限位。
- `motion.py::RobotMoveMotionBackend`
  - 从 `/RobotInfo` 读取 `joint_positions`。
  - 默认 `motion_mode: joint_ik`，绝对关节轨迹发到 `/RobotMove`。
  - 运动前检查 `AutoReady`、故障字和实际关节变化。
  - 状态持续 `AutoReady` 且关节不动时，约 2 秒报“控制柜拒绝动作”。
- `nodes/search_approach_node.py`
  - 默认只识别。
  - `--scan` 无 `--execute` 时只计算 IK/限位。
  - 真实搜索需要 `--execute --scan`。
  - 真实靠近需要 `--execute --approach`。
  - 永不发送夹爪命令。

## 4. 配置模型

正式任务 C/D 使用两份独立配置，不能互相代替，也不使用共用配置启动：

| 任务 | 正式配置 | 目标策略 | 未知物 proposer |
| --- | --- | --- | --- |
| C | `config/track1_task_c_v11_official_20261102.yaml` | `known`，仅 40 个已知类 | 关闭 |
| D | `config/track1_task_d_v11_official_20261102.yaml` | `known_and_unseen` | 开启 |

`config/track1_task_cd_40class_v11_official.yaml` 只保留为兼容/回归配置，不是任务
入口。`config/track1_live.yaml` 是底层搜索/靠近测试配置，不是正式 C/D 任务配置。
配置加载会解析相对路径；安装后默认配置通过
`config_paths.package_config_path()` 从 package share 目录定位。

## 5. 任务 A：视觉链路

`VisionDemoPipeline` 每帧：

```text
scan -> detect -> annotate -> evaluate(IoU) -> report
```

- IoU、匹配和评分是纯函数，可脱离相机单测。
- 结果包含 `frames / matched / passed / pass_rate / mean_iou`。
- 赛题 A 的目标口径是每个物体位姿 `IoU >= 0.75`。

## 6. 任务 B/C/D/E：拾放状态机骨架

```text
idle
 -> scan -> detect -> select -> plan
 -> open_gripper
 -> pregrasp -> grasp -> close -> verify_hold
 -> lift -> transfer -> release -> retreat
 -> done / abort
```

- A/B 使用 `any` 策略，C 使用 `known`（只接受 V11 已知 40 类），D 使用
  `known_and_unseen`（已知类 + 高可抓性未知物），E 使用 `ordered`。
- 任务 B 在五个点位逐个抓取并原地放回。只有释放和退开都成功后，才记录
  本轮实际放回 XY；正式配置随后先把六轴返回任务B初始位，再等待操作员按
  Enter。下一点到达观察位后必须重新视觉确认，旧点位仍可见的物品不能触发
  下一轮。
- B/C/D/E 共用 `PickPipeline`，但 C、D 是两个独立任务程序：分别启动、分别
  配置、分别验收，不能合并运行；必须先验收 C，再验收 D。
- **当前这些入口的现场验收尚未全部完成。C 只完成离线/mock 链路；D 的未知物
  正样本和可接受几何候选仍未完成；真实手眼标定、夹爪和碰撞参数也必须按任务
  分别验证。**

## 7. 坐标变换与安全边界

完整拾放链路中，`planner.detection_to_base` 按优先级支持：

1. 检测自带 `base_xyz`。
2. 显式配置的固定相机平移回退（只适合 mock/调试）。
3. 外参文件：固定相机传 `base`，腕部相机传 `flange`，再乘当前末端位姿。

安全约束：

- 开发/测试配置默认 dry-run；正式 C/D 配置由各自配置和启动器真机确认共同控制，不得绕过确认。
- 每次真实运动前检查 `AutoReady` 和 `fault_flag`。
- 单关节插值 <= 1°，视觉伺服单步 <= 5 mm。
- 工作空间、深度下限、总行程超限立即停止。
- 当前没有正式碰撞检测，实机必须有人守急停。
- 手眼外参未完成前禁止直接进入抓取闭环。

## 8. 扩展点

| 想做的事 | 修改位置 |
| --- | --- |
| 换检测模型/相机 | 新增 `VisionSource`，在 `vision.build_vision` 注册 |
| 加 MoveIt 规划 | 新增 `MotionBackend`，在 `motion.build_motion` 注册 |
| 接真实夹爪 | 使用或新增 `GripperBackend` |
| 更可靠的夹持判定 | 替换 `HoldVerifier` |
| 增加新任务 | 增加 `TaskMode`、策略和独立 launch 入口 |
