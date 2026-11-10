# 赛道一 · 机器人自主抓取挑战 框架

覆盖赛道一全部子任务（A/B/C/D/E）的**视觉 + 机械臂 + 夹爪**编排框架。
相机和夹爪还没装到机械臂上，所以现在交付的是**框架**：整条链路可以在没有
相机、没有夹爪、没有真机动作的情况下完整跑通并自检，装好硬件后只替换适配器，
任务状态机不用改。

```
假/真相机帧 → 目标策略(A/B/C/D/E) → 相机→基座坐标变换 → 抓取路径
            → 开夹爪 → 接近 → 下探 → 到抓取点才合爪 → 夹持确认
            → 抬起 → 转移 → 放置 → 退回
```

## 赛道一任务与分值

| 任务 | 内容 | 分值 | 框架对应 |
| --- | --- | --- | --- |
| A | 展示相机图并识别标注；同一物体 10 个位姿，每次 **IoU ≥ 0.75** | 30 | `task A`：纯视觉，不动机器人 |
| B | 同一物体在 5 个位置拾放（收纳箱四角 + 中心），滑落 -1 | 10 | `task B`：按点位列表循环拾放 |
| C | 无序抓取，40 件中抽 20 件，滑落 -0.5 | 20 | `task C`：`any` 策略 |
| D | 同 C，但含 5 件未见过的物体 | 20 | `task D`：`known_and_unseen` + 可抓性门控 |
| E | 同 C，但按公布的物品清单顺序拾放 | 20 | `task E`：`ordered` 严格顺序 |

> 任务 A 只做视觉；B 是单物体拾放；C/D/E 是清箱抓取。三者共用一套接口与状态机。

## 目录

```text
track1_grasp/
  track1_grasp/
    domain.py          # 任务/状态/位姿/检测/标注/IoU 结果等纯数据
    interfaces.py      # Vision / Motion / Gripper / HoldVerifier 协议
    config.py          # 一份 YAML 描述整个赛道（task/task_a/task_b/...）
    policy.py          # A/B/C/D/E 目标选择策略
    planner.py         # 相机→基座变换 + 抓取路径点
    annotator.py       # IoU、检测↔真值匹配、画框（cv2 延迟导入）
    vision_pipeline.py # 任务 A：scan→detect→annotate→evaluate
    pipeline.py        # B/C/D/E：可观测抓取状态机
    vision.py          # mock + taskA RealSense/YOLO 适配器
    motion.py          # mock + RobotMove 适配器
    gripper.py         # mock / ms42dc / serial / tcp / ros_* / taskA 后端
    factory.py         # 按任务装配管线
    nodes/             # 离线 demo 与 ROS 2 入口
  config/
    track1_mock.yaml            # 无硬件全任务配置
    track1_real_template.yaml   # 真机模板（默认 dry_run）
    task_b_place_poses.example.yaml
    camera_to_base.example.yaml
    target_order.example.yaml
  scenarios/
    mock_track1_scene.yaml      # B/C/D/E 脚本化观测
    mock_task_a_scene.yaml      # A：同物体 10 个位姿（内置真值做 IoU）
    task_a_ground_truth.example.yaml
  launch/    test/    docs/
```

## 不接硬件先跑

```bash
cd <包目录>
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m track1_grasp.nodes.mock_demo \
  --config config/track1_mock.yaml --task A
```

把 `--task` 换成 `B` / `C` / `D` / `E` 即可逐个任务验证：

```bash
# 任务 A：10 个位姿的 IoU 报告（本 mock 场景内置 2 个故意不达标的位置）
python3 -m track1_grasp.nodes.mock_demo --config config/track1_mock.yaml --task A

# 任务 A：把标注图写盘（绿框=真值，橙框=预测）
python3 -m track1_grasp.nodes.mock_demo --config config/track1_mock.yaml \
  --task A --save-annotated

# 任务 B：四角 + 中心五个点位各拾放一次
python3 -m track1_grasp.nodes.mock_demo --config config/track1_mock.yaml --task B

# 任务 C/D/E：清箱（C 任意 / D 含未见物体 / E 按顺序）
python3 -m track1_grasp.nodes.mock_demo --config config/track1_mock.yaml --task E --max-picks 4
```

跑测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q
```

## ROS 2 构建与运行

```bash
cd /media/sunshine/新加卷/track1/ros2_ws
source /opt/ros/humble/setup.bash
# 数据盘是 NTFS 挂载，用普通构建（不要 --symlink-install）
colcon build --packages-select track1_grasp
source install/setup.bash

ros2 run track1_grasp mock_demo --config src/track1_grasp/config/track1_mock.yaml --task A
ros2 launch track1_grasp track1_mock.launch.py task:=B max_picks:=5
```

真机（默认 dry_run，只算路径不发运动）：

```bash
ros2 run track1_grasp track1_node \
  --config src/track1_grasp/config/track1_real_template.yaml --task C
# 确认真机安全后再加 --execute 关闭 dry-run
```

- `track1_node` / `pick_node`：B/C/D/E 抓取入口。
- `annotate_node`：任务 A 视觉入口（等价 `--task A`）。
- `--execute` 是**唯一的**关闭 dry-run 的方式。

## 状态机

抓取任务（B/C/D/E）每次抓取都会返回事件序列：

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
2. 确认是合法目标后，**先开夹爪**。
3. 机械臂走 `pregrasp -> grasp`。
4. 到抓取点后才合夹爪。
5. 夹持确认通过才抬起，否则开爪并退回。
6. 有放置点位时转移并释放；任务 B 每次成功推进一个点位。

任务 A 是独立视觉链路：

```text
idle -> scan -> detect -> annotate -> evaluate -> report
```

对每一帧按 `IoU ≥ task_a.iou_threshold`（默认 0.75）判定通过，输出整体
通过率、平均 IoU 和逐位置结果；`min_positions` 用来提示是否够 10 个位姿。

## 接硬件时要替换什么

1. **相机安装 + 手眼标定** → 写 `pose.extrinsics_file`，设 `pose.camera_frame`
   （固定相机 `base`，腕部相机 `flange`）。
2. **检测器** → `vision.py` 的 `TaskARealSenseVision` 对接现有 D405+YOLO。
   - A：10 类都要能识别，框要贴合（IoU≥0.75）。
   - D：检测器需输出通用 `object/unknown` 框和可抓性分数。
   - E：检测标签需能匹配 `target_order` 的清单顺序。
3. **夹爪协议** → 硬件已确认为 **WHEELTEC FAE 二指电动夹爪 + MS42DC 步进电控**，
   `backend: ms42dc` 已实现其原生 11 字节串口协议（BCC、角度/速度 ×10、到位标志、
   RCC 校验），手册示例已逐字节复现。只差决定用哪条控制路径（USB-C / TTL /
   控制柜 DO / CAN）——见 `docs/GRIPPER_INTEGRATION.md`。
   其它形态的电动夹爪可用 `serial` / `tcp` / `ros_service` / `ros_topic` 后端，
   命令写在 YAML 里，不用改代码。
4. **夹持确认** → `ms42dc` 已接入真实的夹持确认：控制器「到位标志」+ 可选
   的空抓角度门控（夹到空气会走到硬顶，角度接近满行程），见
   `gripper.hold_min_angle_deg`。
5. **碰撞规划** → 当前是笛卡尔直线点，真机建议让 `MotionBackend` 调 MoveIt。
6. **现场安全** → 先用 `dry_run: true` 核对全部路径点，再逐项 `--execute`。

细节见 `docs/ARCHITECTURE.md` 与 `docs/INTEGRATION_CHECKLIST.md`。

## 当前边界

- 相机和夹爪尚未安装，真实视觉、夹爪反馈、夹持检测都没有进入闭环验证。
- 夹爪的控制路径还没定（USB-C / 接口一 TTL / 控制柜 DO→NPN / CAN 四选一），
  接线前先在台面用 `gripper_bench` 验证方向与行程，见
  `docs/GRIPPER_INTEGRATION.md`。
- 任务 D 的"未见物体"依赖检测器输出通用框与可抓性分数；本仓库提供接口与门控，
  不含新的开放词表模型。
- 规划器只做笛卡尔路径点 + 工作空间检查，不做碰撞规划。
- `robotmove` 适配器复用现有 Codroid UDP 桥思路，实跑前仍需确认话题、
  控制器状态字和末端位姿单位。
- 任务 A 的实时预览（把标注图发到 9100/ROS 话题）尚未接线，目前是
  画框 + 存图 + IoU 报告；接相机时再接图像话题。
