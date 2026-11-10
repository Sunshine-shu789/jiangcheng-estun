# 赛道一 · 机器人自主抓取测试框架

现场桌面入口采用 **任务A旧版视觉 + B/C/D/E共享抓取核心** 的结构：

```text
任务A 视觉识别 -> 旧版 camera_recognition.py（只识别，不运动）
任务B/C/D/E   -> 同一套持续视觉 / IK / 运动 / 夹爪 / 状态机核心
```

`task_a.launch.py` 和 `track1_node --task A` 仅保留为内部测试能力；现场桌面入口
不会用它们启动任务A，避免任务A误启动新的“抓取识别”。

当前现场版本优先完成 **D405 识别 `6-Foam_Brick`、机械臂搜索目标、5 mm 小步靠近、暂不抓取** 的测试链路。所有运动默认 dry-run，只有显式传入 `--execute` 才允许真机运动；测试节点不会发送任何夹爪命令。

> B/C/D/E 的拾放状态机和任务入口已经搭好，但手眼外参、真实夹爪和放置点尚未完成现场闭环验证，**不能直接当作正式抓取程序使用**。

## 1. 当前已搭好的链路

```text
D405 RGB + depth
  -> 训练好的 YOLO v23 模型
  -> 后台持续采集/推理线程（始终只保留最新帧）
  -> 独立实时预览线程（检测框 / 深度 / FPS / 运动 / 抓取状态）
  -> 目标类别/置信度/深度筛选
  -> 当前末端姿态下的 eye-in-hand 小步视觉伺服
  -> RobotMove：full-pose IK + 每关节最多 1° 插值
  -> 每次最多移动 5 mm，并做 AutoReady / fault / workspace / 总行程检查
  -> 只靠近，不合爪
```

关键模块：

| 模块 | 作用 |
| --- | --- |
| `track1_grasp/vision.py` | D405 + YOLO 实时检测；RGB、深度和内参同步旋转 180° |
| `track1_grasp/vision_runtime.py` | 后台持续视觉、最新帧消费、独立实时窗口和状态叠加 |
| `track1_grasp/approach.py` | 目标筛选、像素误差到基座小步位移、深度与工作空间安全门控 |
| `track1_grasp/kinematics.py` | 与控制器一致的 modified-DH、FK、full-pose IK、关节限位 |
| `track1_grasp/motion.py` | `/RobotInfo` 状态读取、关节空间 IK 运动、`/RobotMove` 发布 |
| `track1_grasp/nodes/search_approach_node.py` | 现场识别 / 搜索 / 靠近测试入口 |
| `track1_grasp/pipeline.py` | B/C/D/E 共用的拾放状态机骨架 |
| `track1_grasp/launch_helpers.py` | 五个任务启动文件的共享构造器 |

## 2. 赛道一任务对应关系

| 任务 | 内容 | 当前入口 | 当前状态 |
| --- | --- | --- | --- |
| A | 相机图 + 识别框 + IoU 评分 | 桌面 `任务A_视觉识别.desktop` / 旧版 `视觉识别` | 旧版视觉入口，可测 |
| B | 同一物体在四角 + 中心拾放 | `task_b.launch.py` | 状态机骨架，待标定/夹爪 |
| C | 无序抓取 | `task_c.launch.py` | 状态机骨架，待标定/夹爪 |
| D | 含未见物体的无序抓取 | `task_d.launch.py` | 状态机骨架，待通用检测与夹爪 |
| E | 按指定顺序抓取 | `task_e.launch.py` | 状态机骨架，待顺序清单与夹爪 |

任务A只使用旧版视觉识别；B/C/D/E 复用同一拾放状态机和新的持续实时视觉，
只通过任务模式和配置改变目标策略、次数和放置点。桌面的“抓取识别”是供机械臂
运动、靠近和夹取时手动持续观察的入口，本身不发送运动或夹爪命令。

## 3. 无硬件测试

```bash
cd /media/sunshine/新加卷/track1/track1_grasp
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q
```

独立演示各任务：

```bash
# A：10 个位姿的 IoU 报告
python3 -m track1_grasp.nodes.mock_demo \
  --config config/track1_mock.yaml --task A

# B：四角 + 中心五个拾放点
python3 -m track1_grasp.nodes.mock_demo \
  --config config/track1_mock.yaml --task B

# C/D/E：清箱状态机骨架
python3 -m track1_grasp.nodes.mock_demo \
  --config config/track1_mock.yaml --task C --max-picks 4
```

## 4. ROS 2 构建

数据盘是 NTFS 挂载，**不要使用 `--symlink-install`**。

```bash
source /opt/ros/humble/setup.bash
source /home/sunshine/codex/easton/codroid_bridge/codroid_bridge/install/setup.bash
source /home/sunshine/codex/easton/ros2_ws/install/setup.bash

cd /media/sunshine/新加卷/track1/ros2_ws
colcon build --packages-select track1_grasp
source install/setup.bash
```

配置文件、五个 `task_*.launch.py` 和 `search_approach_node` 都已加入安装清单。

## 5. 当前现场识别 / 靠近测试

### 5.1 每次终端先加载环境

```bash
source /opt/ros/humble/setup.bash
source /home/sunshine/codex/easton/codroid_bridge/codroid_bridge/install/setup.bash
source /home/sunshine/codex/easton/ros2_ws/install/setup.bash
source /media/sunshine/新加卷/track1/ros2_ws/install/setup.bash
```

相机 YOLO 依赖安装在 taskA 的虚拟环境中，现场建议直接用该解释器启动模块：

```bash
cd /media/sunshine/新加卷/track1/track1_grasp
PYTHON=/media/sunshine/新加卷/taskA/venv/bin/python
CONFIG=/media/sunshine/新加卷/track1/track1_grasp/config/track1_live.yaml
```

### 5.2 只识别并持续观察，不运动

```bash
"$PYTHON" -m track1_grasp.nodes.search_approach_node \
  --config "$CONFIG" --watch
```

`--watch` 会在识别/任务结束后继续保持相机、YOLO 和实时窗口运行，直到按
`Ctrl-C`；它不会发送任何运动或夹爪指令。手动移动机械臂或准备抓取时，可用
这个模式持续观察目标。若只需要完成一次识别检查并退出，可去掉 `--watch`。

输出应包含：

```text
[robot] controller state=AutoReady ... joints_deg=[...]
[vision] 6-Foam_Brick conf=... depth=...m camera_xyz=(...)
[result] target detected; no approach requested
```

相机运动和抓取期间会持续刷新窗口，顶部状态会依次显示
`SEARCHING`、`MOVING`、`DETECTING`、`GRASP`、`VERIFY_HOLD`、`DONE`
等任务阶段；画面中还会显示目标框、置信度、深度、检测帧率和帧年龄。

窗口操作：

- `q`：隐藏窗口，后台识别和任务仍继续运行；
- `s`：把当前叠加图保存到 `vision.output_dir`；
- `Ctrl-C`：结束当前程序；使用 `--watch` 时也必须用 `Ctrl-C` 结束。

带框图不再对每一帧自动写盘；需要时按 `s` 保存。若终端没有
`DISPLAY`/`WAYLAND_DISPLAY`，程序会打印提示并继续识别，只是不打开窗口。
若要让识别在机械臂运动期间持续运行，保持：

```yaml
vision:
  continuous: true
  show: true
```

手动快照默认写到：

```text
/media/sunshine/新加卷/track1/runtime/vision/
```

### 5.3 检查搜索视角的 IK，不运动

```bash
"$PYTHON" -m track1_grasp.nodes.search_approach_node \
  --config "$CONFIG" --scan
```

该命令只计算各俯仰角的 IK 和关节限位，并打印 `[search:dry-run]`，不会发布运动。

### 5.4 第一次真机动作：只走一个小步

有人守急停，并确认控制器为 `AutoReady`、`fault_flag=0`：

```bash
"$PYTHON" -m track1_grasp.nodes.search_approach_node \
  --config "$CONFIG" --execute --approach --steps 1 --max-step-mm 5
```

只有目标已在当前视野内才使用上面命令。若目标不在当前视野，先单独确认搜索：

```bash
"$PYTHON" -m track1_grasp.nodes.search_approach_node \
  --config "$CONFIG" --execute --scan --scan-only
```

确认机械臂动作方向和安全距离后，再增加靠近步数。不要第一次就直接跑满 36 步。

### 5.5 五个任务入口

```bash
ros2 launch track1_grasp task_a.launch.py frames:=10
ros2 launch track1_grasp task_b.launch.py
ros2 launch track1_grasp task_c.launch.py
ros2 launch track1_grasp task_d.launch.py
ros2 launch track1_grasp task_e.launch.py
```

B/C/D/E 默认 dry-run。只有确认安全后才加 `execute:=true`，例如：

```bash
ros2 launch track1_grasp task_c.launch.py execute:=true max_picks:=1
```

## 6. 安全规则

- 所有运动入口默认 dry-run；`--execute` / `execute:=true` 是唯一允许真机运动的方式。
- 每次真实运动前检查 `AutoReady`、`fault_flag` 和真实关节位置。
- 接近测试每次最多 5 mm，每次关节插值最多 1°。
- 工作空间、深度下限和总行程超限时立即停止。
- `search_approach_node` 不发送夹爪命令；当前 `track1_live.yaml` 的夹爪后端也是 `mock`。
- 当前没有正式碰撞检测，实机测试必须有人守急停。
- `camera_to_flange.d405_assumed.yaml` 只是倒装相机旋转后的临时单位旋转假设，**不是完成的手眼标定**；`search_approach_node` 的小步伺服不依赖该外参。
- 不要从当前姿态盲目叠加整体大角度扫描；先识别、再 dry-run 检查 IK、最后一次 5 mm 小步验证方向。

## 7. 后续仍需完成

1. 完成 D405 手眼标定，替换 `camera_to_flange.d405_assumed.yaml`。
2. 现场确认 5 mm 小步的 x/y/z 方向，必要时只调整 `servo.camera_to_tool`。
3. 接入真实夹爪并验证开合方向、行程和夹持反馈。
4. 标定 B/C/D/E 的放置点、接近高度和碰撞安全区域。
5. 用足够多的现场样本验证 `6-Foam_Brick` 的识别、深度和遮挡情况。
