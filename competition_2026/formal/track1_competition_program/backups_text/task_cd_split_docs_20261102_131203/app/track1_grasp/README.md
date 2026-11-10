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
  -> 正式 V11 40 类权重
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
| B（任意摆放） | 每轮只看当前相机画面，估计物体中心与主轴，生成逆解候选并原地拾放 | 桌面 `任务B_任意摆放抓取.desktop` | 离线 Mock 通过；实机手眼旋转、RobotInfo 帧语义和抓取深度偏移仍待现场标定 |
| C | 无序抓取 | `task_c.launch.py` | V11 + 顺序选择 + 六段抓取安全链；离线 dry-run 通过，待现场标定/夹爪 |
| D | 含未见物体的无序抓取 | `task_d.launch.py` | V11 + 不透明未知物几何候选 + 独立 track + 六段抓取安全链；真实 D405 零目标基线通过，未知物正样本与透明物待验收 |
| E | 按指定顺序抓取 | `task_e.launch.py` | 状态机骨架，待顺序清单与夹爪 |

任务B的五轮流程是严格的逐点闭环：

```text
启动时六轴同步回到任务B初始位（已在0.5°容差内则跳过）
 -> 校验 AutoReady / fault_flag=0 / 六轴反馈 / 到位误差
 -> 当前视野搜索 / 多个安全观察位主动搜索
 -> 连续帧视觉确认当前点物品
 -> 接近 -> 夹取 -> 合爪 -> 夹持确认
 -> 抬起 -> 转移 -> 回到本轮实际抓取 XY -> 释放 -> 退开
 -> 六轴同步回到本轮回位目标并验证到位
    （共享初始位或刚完成点的观察位，由配置二选一）
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
`joint_positions_rad` 并校验 `role: task_b_initial_pose`。每轮回位目标可二选一：
`return_to_initial_after_each_pick: true` 返回共享初始位；
`return_to_observation_after_each_pick: true` 则在释放和最后退开后，直接返回
刚完成点位的 `point_observation_joints_deg`，准备下一次视觉采集。两者互斥，
观察位模式要求每个点位都配置具体六轴姿态。回位时六轴同步插值并受
`motion.joint_max_step_deg` 限制，真机还会按
`observation_pose_tolerance_deg` 或 `initial_pose_tolerance_deg` 验证最终误差；
读取失败、控制器未就绪、限位不通过、命令被拒绝或最终误差超限都会直接中止，
不会进入下一点。
如果当前视野没有香蕉，程序会先移动到当前点位上方配置的多个观察位置，并在
每次移动后重新识别；这些搜索位移只在视觉确认失败时发生，不会在 Enter 前执行。
全部观察位置都没有确认时，会回到初始观察位姿继续等待，期间不会发送抓取、
合爪或释放命令。观察偏移当前位于
`config/track1_task_b_final_20261020_live.yaml` 的
`search.observation_offsets_xy_m`，在真机启用前仍必须按现场工作空间和碰撞
边界重新确认。

任务B的任意摆放入口与五点位入口完全分开。任意模式不读取 `place_poses`、
点位名称、逐点抓取高度、逐点姿态、观察锚点或已完成物体 XY；每一轮都重新从
当前相机画面取得物体中心、主轴和位姿置信度，再生成多个朝向候选。只有整条
`pregrasp -> grasp -> lift -> transfer -> release -> retreat` 路径全部通过 IK、
关节限位、跳变和奇异值预检后，程序才允许开爪或运动；全部候选失败时零运动、
零夹爪命令。当前 `任务B_任意摆放抓取.desktop` 固定使用离线 Mock 配置。实机
启用前必须完成 D405 到法兰的旋转标定、确认 `/RobotInfo.end_positions` 是法兰
还是 TCP 位姿，并标定 PCA 物体中心到夹爪闭合中心的偏移。

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

### 3.1 GraspNet 离线多帧共识

只读取已有批次中的 `offline_candidates.json`，不会导入 ROS、连接硬件或发送运动/夹爪命令。
只把已经通过最终 CAD/净空检查、NMS 和严格 IK 门限的候选送入聚类。默认门禁为
至少 4 个独立帧、TCP 平移不超过 25 mm、平行夹爪对称旋转不超过 15°；不足时
fail-closed。完整连通聚类避免两个外端超出容差的候选通过链式关系混入同一簇。

```bash
BATCH=/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/logs/offline_graspnet_aperture_multiframe_20261031_120309
OUT=/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_graspnet_aperture_multiframe_20261031/diagnostics/consensus
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m track1_grasp.tools.summarize_multiframe_grasp_consensus   "$BATCH" --output-dir "$OUT"   --config config/offline_grasp_consensus.yaml
```

输出包含 `consensus.json`、`consensus.md`、`consensus_clusters.csv` 和
`consensus_members.csv`。共识只表示离线位姿稳定，不是抓取成功率、完整运动规划
或实机授权。

#### 3.1.1 运行时共识门禁（默认关闭）

`grasp_consensus` 可把一份离线 `consensus.json` 绑定到当前 Task C/D 规划出的
TCP 位姿。该功能默认关闭，也不会因为存在一份报告就自动启用；报告必须与当前
场景和物体一一对应。配置示例：

```yaml
grasp_consensus:
  enabled: false
  report_file: ""
  minimum_distinct_frames: 4
  maximum_translation_tolerance_m: 0.025
  maximum_jaw_rotation_tolerance_deg: 15.0
  maximum_pose_translation_error_m: 0.025
  maximum_pose_jaw_rotation_error_deg: 15.0
```

启用时配置校验强制要求：Task C/D、`motion.dry_run=true`、
`gripper.backend=mock`、`search.enabled=false`，并明确 `robot_pose_frame` 为
`flange` 或 `tcp`。门禁会在完整路线预检通过之后、打开夹爪和发布首个运动命令
之前执行。以下任一情况都会拒绝整次抓取，不回退到单帧结果：

- 报告缺失、损坏、超过大小上限或 JSON/schema 不安全；
- 报告中的离线/未连接/未授权标志异常，或阈值比运行时门禁更宽松；
- 共识簇少于所需独立帧，或簇成员、代表位姿、统计量不一致；
- 在线规划 TCP 平移或夹爪对称旋转超出配置容差。

通过时 `PLAN` 事件会记录簇 ID、代表候选、独立帧数、位姿误差、报告 SHA-256、
报告质量标志和 `motion_authorized=false`。即使门禁通过，也仍然只是离线
dry-run/mock 流程，不授权机械臂或夹爪实机动作，也不回退到单帧候选。

### 3.2 Task D 只读视觉验收

`bin/accept_task_d_vision` 只执行 D405 RGB-D、V11、稳定器和 Task D
unknown proposer 的固定帧采集；不要求 `/RobotInfo`，不启动机器人桥、运动后端
或夹爪后端，也不会发布运动/夹爪命令。它不能替代正式 `run_task_d` 的安全门禁。

```bash
ROOT=/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序
OUT="$ROOT/logs/task_d_vision_acceptance_$(date +%Y%m%d_%H%M%S)"
"$ROOT/bin/accept_task_d_vision" \
  --frames 15 \
  --min-valid-depth-ratio 0.80 \
  --output-dir "$OUT"
```

输出包括每帧 RGB、深度伪彩色、原始深度 NPZ、检测叠加图、深度分位数、捕获耗时
和 `summary.json`。采集正样本时再加 `--require-known` 或 `--require-unknown`；
零目标场景不能使用这两个强制检查。

几何证据与几何策略还要分别检查：

- `--require-geometry`：至少一条检测包含同帧物体位姿和几何证据；
- `--require-geometry-candidate`：至少一条检测通过支持面、短边、rim/sidewall
  等全部已启用策略门限。

第二条比第一条严格得多。2026-11-02 的真实 D405 审计共采集 15/15 帧、65 条
几何证据，其中 49 条有支持面，但最高支持面质量仅 `0.524`，低于正式门限
`0.55`，因此几何策略接受数为 **0**。该次运行只能证明证据采集和 fail-closed
审计链可用，不能写成几何候选或实机抓取已经验收。真实 D405 的零目标基线已经
通过，但空场景不能证明未知物召回率；透明/强反光物仍是已知未解决边界。

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
- `camera_to_flange.d405_assumed.yaml` 的平移已按现场数据设为法兰坐标 `(0, -65, 0) mm`；旋转仍是 `rotate_180` 后的单位旋转假设，尚未完成完整手眼验证。
- 不要从当前姿态盲目叠加整体大角度扫描；先识别、再 dry-run 检查 IK、最后一次 5 mm 小步验证方向。

## 7. 后续仍需完成

1. 验证 D405 的旋转外参并完成完整手眼标定，确认后替换 `camera_to_flange.d405_assumed.yaml`。
2. 现场确认 5 mm 小步的 x/y/z 方向，必要时只调整 `servo.camera_to_tool`。
3. 接入真实夹爪并验证开合方向、行程和夹持反馈。
4. 标定 B/C/D/E 的目标参考点、接近高度和碰撞安全区域。
5. 用足够多的现场样本验证 `6-Foam_Brick` 的识别、深度和遮挡情况。

## SGS Modbus 只读基线

新增的 SGS Modbus 接入层默认只读，包含严格报文校验、寄存器表、状态字解码、
本地假控制器和只读探针。详见
[`docs/ESTUN_SGS_MODBUS.md`](docs/ESTUN_SGS_MODBUS.md)。

```bash
./bin/check_sgs_modbus_readonly --offline-smoke
./bin/check_sgs_modbus_readonly --schema
```

没有 `--allow-write` 参数；机器人命令还有第二层独立闸门。现场只读探测通过前
不得接入任务执行链路。
