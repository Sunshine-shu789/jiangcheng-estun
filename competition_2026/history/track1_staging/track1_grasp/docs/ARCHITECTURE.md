# 架构说明（赛道一 A/B/C/D/E）

## 1. 分层

```
                 ┌──────────────────────────────────────────┐
                 │  nodes/  (ROS 2 与离线入口)               │
                 │  track1_node / mock_demo / annotate_node  │
                 └───────────────┬──────────────────────────┘
                                 │ Track1Config
                 ┌───────────────▼──────────────────────────┐
                 │  factory.build_pipeline(config)           │
                 │   task A -> VisionDemoPipeline            │
                 │   task B/C/D/E -> PickPipeline            │
                 └───────────────┬──────────────────────────┘
        ┌────────────────────────┼─────────────────────────────┐
        │                        │                             │
 ┌──────▼───────┐        ┌───────▼────────┐            ┌───────▼───────┐
 │ VisionSource │        │ MotionBackend  │            │GripperBackend │
 │ mock /       │        │ mock /         │            │ mock /        │
 │ taska_       │        │ robotmove      │            │ taska         │
 │ realsense    │        │ (/RobotMove)   │            │               │
 └──────────────┘        └────────────────┘            └───────────────┘
```

编排层只认 `interfaces.py` 里的四个 Protocol，不认任何具体硬件：

| 接口 | 职责 | 现有实现 |
| --- | --- | --- |
| `VisionSource` | 产出一帧 `Frame`（检测框/深度/坐标/真值） | `MockVisionSource`、`TaskARealSenseVision` |
| `MotionBackend` | 查机器人状态、走笛卡尔点位、暴露 `dry_run` | `MockMotionBackend`、`RobotMoveMotionBackend` |
| `GripperBackend` | `open/close/state` | `MockGripperBackend`、`TaskAGripperBackend` |
| `HoldVerifier` | 合爪后判定是否真的夹住 | `MockHoldVerifier` |

## 2. 配置模型

一份 YAML 描述整个赛道，`task.mode` 选子任务。

| 段 | 作用 |
| --- | --- |
| `task` | `mode: A/B/C/D/E`、`max_picks`、`stop_on_error`、采样周期 |
| `vision` | 后端选择、mock 场景、`scenario_files`（按任务给不同场景）、置信度门限 |
| `task_a` | IoU 阈值、真值文件、标注图输出目录、`max_frames`、`min_positions` |
| `task_b` | 放置点位列表/文件、轮次、是否同一目标、点位名 |
| `target` | 目标策略（`any`/`known`/`known_and_unseen`/`ordered`）、顺序来源 |
| `pose` | 相机安装方式（base/flange）、外参文件、回退策略 |
| `motion` | 工作空间硬限位、接近/抬起高度、速度、朝向、话题名、超时 |
| `gripper` | 后端选择、开爪时机、稳定等待 |

`apply_mode(config, "A"|"B"|"C"|"D"|"E")` 会把目标策略切到该任务的默认语义：
A/B/C → `any`，D → `known_and_unseen`，E → `ordered`；顺序仍从 `target.order`
或 `target.order_file` 读取，因此切模式不会丢清单。

## 3. 任务 A：视觉链路

`VisionDemoPipeline` 每帧：

```
scan → detect(按 min_confidence 过滤) → annotate(画框) → evaluate(IoU) → report
```

- `annotator.iou` / `match_frame` / `score_frame` 是纯函数，不依赖 cv2，
  可以脱离相机单测。
- 贪心匹配：每个真值框找当前未被占用的、IoU 最大的检测框；标签不匹配不算，
  但 `object`/`unknown`/空标签视为通用标签放行（服务任务 C/D/E 的通用检测）。
- 每帧产出一条 `IouMatch`，未匹配上的检测框也会记录，便于发现误检。
- 结果 `VisionReport` 给出 `frames / matched / passed / pass_rate / mean_iou`，
  正好对应赛题"A 的十个位置、每个 IoU≥0.75"的证据口径。

## 4. 任务 B/C/D/E：抓取链路

`PickPipeline` 单物体循环：

```
idle
 → scan        取一帧
 → detect      统计原始检测
 → select      按策略选目标
 → plan        相机→基座变换 + 生成点位（做工作空间检查）
 → open_gripper 在进入狭窄接近走廊前先开爪
 → approach    pregrasp
 → grasp       grasp → 到点后合爪
 → verify_hold 夹持确认（失败则开爪 + 退回）
 → lift / transfer / release / retreat
 → done | abort
```

- `plan.ordered_waypoints()` 是唯一的动点来源，mock 与真机走同一份点位。
- 任务 B 每次成功后 `_place_index += 1`，从 `task_b.place_poses` 取下一个点位
  （四角 + 中心），点位名写进 `Plan.place_name` 并回传到 `PickResult`。
- 任务 C/D/E 复用同一循环，只是 `max_picks` / 目标策略不同。
- 任何一步异常都会进入 `abort` 分支，`stop_on_error` 决定是否继续下一件。

## 5. 坐标变换

`planner.detection_to_base` 支持三条路径，按优先级：

1. 检测结果自带 `base_xyz`（mock、或固定相机已在上游换算）→ 直接用。
2. `pose.fallback_camera_to_base: true` → 用 `fixed_camera_to_base_xyz` 平移
   （只在没有标定、跑 mock 时用）。
3. 正常路径：读 `pose.extrinsics_file`（4×4 或 `xyz+rpy`）。
   - `camera_frame: base`：T_base_camera 直接作用。
   - `camera_frame: flange`：T_base_flange（由 `/RobotInfo` 当前位姿算出）再乘
     T_flange_camera。

所有点位在生成后都会过 `_check_workspace`，超界直接判 `REJECTED`，不动机械臂。

## 6. 安全边界

- 全链路默认 `dry_run: true`；只有显式 `--execute` 会关掉。
- 工作空间硬限位 + 余量。
- 置信度门限、可选深度要求。
- 真机模式下先查 `RobotState.ready`，不 ready 直接 `ABORTED`。
- 夹持确认失败必须开爪 + 退回，绝不盲目抬起。

## 7. 扩展点

| 想做的事 | 改哪里 |
| --- | --- |
| 换检测模型/相机 | 新增 `VisionSource` 实现，在 `vision.build_vision` 注册 |
| 加 MoveIt 规划 | 新增 `MotionBackend` 实现，在 `motion.build_motion` 注册 |
| 接真夹爪 | 新增 `GripperBackend` 实现，在 `gripper.build_gripper` 注册 |
| 更可靠的夹持判定 | 换掉 `MockHoldVerifier`（腕部相机位移/夹爪电流/抬起复查） |
| 新子任务 | 加 `TaskMode` 分支 + 策略 + 装配，不动现有状态机 |
