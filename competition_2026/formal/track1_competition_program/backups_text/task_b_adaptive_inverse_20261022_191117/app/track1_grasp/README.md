# 赛道一 · 机器人自主抓取测试框架

现场桌面入口采用 **任务A十点位巡检（旧版视觉） + 旧版纯视觉观察 + B/C/D/E共享抓取核心**
的结构：

```text
任务A 十点位巡检 -> 旧版HTTP视觉 + J1当前位姿锚点270°非对称双弧/J2双半径十点 + J5/J6锁定 + 十类识别 + 识别后停3秒 + 回原点
任务A 视觉识别   -> 旧版 camera_recognition.py（只识别，不运动）
任务B/C/D/E     -> 同一套持续视觉 / IK / 运动 / 夹爪 / 状态机核心
```

任务A巡检以启动时六轴反馈为基准，J1以启动角度为锚点左右分配完整的270°
行程：第1弧由一侧端点扫到另一侧端点，第2弧反向扫回，每弧5点。程序先让
左右各请求135°；若一侧触及控制器安全限位，该侧使用全部可用安全行程，由另
一侧补足剩余跨度，而不是把J1锁在启动角中心。当前 J1≈93.324° 时，右侧可用
104.676°、左侧分配165.324°，覆盖 `[-72°, +198°]`，总跨度仍为270°。J2分别叠加
`+15°` 和 `-9°` 偏移形成两条半径弧，第5点到第6点只切换J2并保持J1在同一端。
J1-J4允许运动，J5/J6在每个目标中精确复制启动反馈并全程锁定。执行前完成
10个巡检点加最终回原点的FK、工作空间、关节限位、步长和TCP相邻距离预检查，
任何一项失败都不会发出真实运动。巡检点相邻TCP距离不少于15 cm。旧版视觉程序
在后台持续采集和推理，浏览器实时窗口在整个巡检期间保持刷新；每个位置连续
确认十类中的任意物品后原地停3秒，再自动前往下一点，第10点结束后返回启动时
的精确关节姿态。最终报告1-Cheez-it到10-meat_can十类物品的覆盖情况。
任务A只通过 `http://127.0.0.1:9100/status.json` 读取旧版视觉结果，不会启动
新的抓取识别，也不会导入 `models/best.pt`。

当前所有运动默认 dry-run，只有配置为 `motion.dry_run: false` 且通过启动器
真机确认窗口后才允许真实运动；任务A巡检本身不会发送夹爪命令。

桌面启动器会在 `/RobotInfo` 不可用时自动启动已安装的 `taskA_grasp`
UDP 状态桥接，收到末端位姿后才进入任务；若桥接已经在运行则直接复用。
启动器自己拉起的桥接会在任务结束时关闭。控制器需要把“设置 -> ROS”的
目标 IP 指向本机 UDP 9006 端口。

> B/C/D/E 的拾放状态机和任务入口已经搭好，但手眼外参、真实夹爪和现场点位尚未完成闭环验证，**不能直接当作正式抓取程序使用**。

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
| `track1_grasp/vision.py` | B/C/D/E 的 D405 + YOLO 检测；任务A的旧视觉 HTTP 适配器 |
| `track1_grasp/vision_runtime.py` | 后台持续视觉、最新帧消费、独立实时窗口和状态叠加 |
| `track1_grasp/approach.py` | 目标筛选、像素误差到基座小步位移、深度与工作空间安全门控 |
| `track1_grasp/kinematics.py` | 与控制器一致的 modified-DH、FK、full-pose IK、关节限位 |
| `track1_grasp/motion.py` | `/RobotInfo` 状态读取、关节空间 IK、直线路径全量预检查、`/RobotMove` 发布 |
| `track1_grasp/vision_pipeline.py` | 任务A十段巡检编排，运动期间保持视觉持续刷新 |
| `track1_grasp/nodes/search_approach_node.py` | 现场识别 / 搜索 / 靠近测试入口 |
| `track1_grasp/pipeline.py` | B/C/D/E 共用的拾放状态机骨架 |
| `track1_grasp/launch_helpers.py` | 五个任务启动文件的共享构造器 |

## 2. 赛道一任务对应关系

| 任务 | 内容 | 当前入口 | 当前状态 |
| --- | --- | --- | --- |
| A | J1当前位姿锚点左右分配270°双弧 + J2双半径十点巡检，J5/J6锁定，识别后停3秒，最后回原点，识别十类 | 桌面 `任务A_十点位巡检.desktop` | 离线测试通过；真机启动仍需现场确认 |
| B | 五个点位逐个抓取并原地放回 | 桌面 `任务B_五点位拾放.desktop` | 正式 live 配置；连续轨迹修正已通过离线回归，待低速单点实机确认 |
| C | 无序抓取 | `task_c.launch.py` | 状态机骨架，待标定/夹爪 |
| D | 含未见物体的无序抓取 | `task_d.launch.py` | 状态机骨架，待通用检测与夹爪 |
| E | 按指定顺序抓取 | `task_e.launch.py` | 状态机骨架，待顺序清单与夹爪 |

任务B的五轮流程是严格的逐点闭环：

```text
启动时六轴同步回到任务B初始位（已在0.5°容差内则跳过）
 -> 校验 AutoReady / fault_flag=0 / 六轴反馈 / 到位误差
 -> 当前视野搜索 / 多个安全观察位主动搜索
 -> 连续帧视觉确认当前点物品
 -> 接近 -> 夹取 -> 合爪 -> 夹持确认
 -> 抬起 -> 转移 -> 回到本轮实际抓取 XY -> 释放 -> 退开
 -> 六轴同步回到任务B初始位并验证到位
 -> 显示完成信息，等待操作员按 Enter
 -> 到达下一点观察位后重新执行视觉确认
 -> 确认成功才开始下一轮
```

每个 Cartesian 预检段（`pregrasp / grasp / lift / transfer / release /
retreat`）现在把整段 IK 路径作为一条多点 `JointTrajectory` 一次下发，只在
段终点等待一次 `AutoRunning -> AutoReady`，不再每 2 mm 航点停一次。视觉未
命中后的搜索回程使用六轴关节空间回程，避免在腕部奇异边界重新求 IK。

只有 `release` 和最后的 `retreat` 都成功，程序才记录本轮实际放回 XY 并推进轮次。
下一轮会在任何运动命令发出前，先排除所有已放回物品的实际 XY；如果旧点位物品
仍在视野中，程序只继续等待，绝不会据此开始下一次抓取。`next_point_timeout_s: 0`
表示一直等到下一点被确认或任务被人工停止，不会因为等待超时而自动抓取。
启动初始位由 `task_b.initial_pose_file` 指定，目标值读取
`joint_positions_rad` 并校验 `role: task_b_initial_pose`。正式任务B配置启用
`return_to_initial_after_each_pick: true`：每轮释放并退开后，六轴先同步回到
初始位并验证到位，然后才显示 Enter 提示。偏离容差时六个关节一起插值，单轴
步长继续受 `motion.joint_max_step_deg` 限制。初始位读取失败、控制器未就绪、
限位不通过、命令被拒绝或最终误差超限都会直接中止，不会进入下一点。
如果当前视野没有香蕉，程序会先移动到当前点位上方配置的多个观察位置，并在
每次移动后重新识别；这些搜索位移只在视觉确认失败时发生，不会在 Enter 前执行。
全部观察位置都没有确认时，会回到初始观察位姿继续等待，期间不会发送抓取、
合爪或释放命令。观察偏移当前位于
`config/track1_task_b_final_20261020_live.yaml` 的
`search.observation_offsets_xy_m`，在真机启用前仍必须按现场工作空间和碰撞
边界重新确认。

任务A有两个独立入口：`任务A_十点位巡检.desktop` 固定使用旧版持续实时视觉和
J1当前位姿锚点左右分配270°双弧 + J2双半径十点巡检路线，物品确认后停3秒，第10点后回到启动姿态；
`任务A_视觉识别.desktop` 单独打开同一个旧版视觉程序。
B/C/D/E复用同一拾放状态机和新的持续实时视觉，只通过任务模式和配置改变目标
策略、次数和目标点位。任务A配置文件为 `config/track1_task_a_legacy.yaml`；
任务B桌面入口使用 `config/track1_task_b_final_20261020_live.yaml`。
桌面的“抓取识别”是供机械臂运动、靠近和夹取时手动持续观察的入口，本身不发送
运动或夹爪命令。

## 3. 无硬件测试

```bash
cd /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_grasp
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q
```

独立演示各任务：

```bash
# A：10 个位姿的 IoU 报告
python3 -m track1_grasp.nodes.mock_demo \
  --config config/track1_mock.yaml --task A

# B：五个点位逐个抓取、原地放回，逐点视觉确认
python3 -m track1_grasp.nodes.mock_demo \
  --config config/track1_mock.yaml --task B

# C/D/E：清箱状态机骨架
python3 -m track1_grasp.nodes.mock_demo \
  --config config/track1_mock.yaml --task C --max-picks 4
```

## 4. ROS 2 构建

当前现场启动器直接使用安装好的 ROS 2 环境；若重新构建此包，不要使用
`--symlink-install`，避免非本地文件系统上的符号链接问题。

```bash
source /opt/ros/humble/setup.bash
source /home/sunshine/codex/easton/codroid_bridge/codroid_bridge/install/setup.bash
source /home/sunshine/codex/easton/ros2_ws/install/setup.bash

cd /home/sunshine/codex/easton/ros2_ws
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
```

现场桌面启动器使用 `/usr/bin/python3`，并按需补充 taskA 虚拟环境的
`site-packages`。终端手动测试时使用：

```bash
cd /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_grasp
PYTHON=/usr/bin/python3
CONFIG=/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/config/track1_live.yaml
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
/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/logs/vision/
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
4. 标定 B/C/D/E 的目标参考点、接近高度和碰撞安全区域。
5. 用足够多的现场样本验证 `6-Foam_Brick` 的识别、深度和遮挡情况。
