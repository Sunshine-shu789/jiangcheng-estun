# 装相机 / 夹爪后的接线清单

框架已经在无硬件状态下跑通（`pytest` + 五个任务的 mock demo）。
下面按"装好硬件后最小改动"的顺序列出要做的接线，每一步都给出验收点。

## 0. 前置：确认数据盘与工作区

```bash
cd /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序
./bin/ensure_runtime
source /opt/ros/humble/setup.bash
source /media/sunshine/ESTUN_RUNTIME/ros2_ws/install/setup.bash
```

安装或更新 ROS 包时，构建产物只能写到 ext4 运行镜像中的 ROS 2 工作空间，不得
重新写回主目录。

## 0.1 任务 C/D 独立验收顺序

- [x] C、D 使用独立启动入口：`bin/run_task_c`、`bin/run_task_d`。
- [x] C 绑定 `config/track1_task_c_v11_official_20261102.yaml`，只抓已知 40 类，
      拒绝 `unknown/object`。
- [x] D 绑定 `config/track1_task_d_v11_official_20261102.yaml`，接受已知 40 类
      和高可抓性未知物。
- [x] 全量自动回归通过；C/D mock 分别运行，不合并成一组任务。
- [ ] 先完成 C 的真实 D405 已知物验收、抓取闭环和现场安全检查。
- [ ] C 验收通过后，再开始 D 的真实未知物正样本和几何策略验收。
- [ ] 任何时刻只启动 C 或 D 其中一个，不得两个入口并行抢占相机和机械臂。

## 1. 相机

- [ ] 相机固定（固定式）或装法兰（腕部式），与 `pose.camera_frame` 一致。
- [ ] 手眼标定，结果写入 `pose.extrinsics_file`：
  - 固定相机 → T_base_camera；
  - 腕部相机 → T_flange_camera。
- [ ] `pose.use_detection_base_xyz: false`（真机必须走标定，不吃上游 base_xyz）。
- [ ] `pose.fallback_camera_to_base: false`。

验收：先使用 C 专用 dry-run 副本跑单次规划，日志里的 `[plan]` 点位落在
工作台实际位置附近，且不触发 workspace 拒绝。该副本不是桌面正式配置，验收
C 之前不得启动 D。

## 2. 检测器

现有 D405 + YOLO 已在 `taskA_grasp` 包内；`TaskARealSenseVision` 负责把它
转成框架的 `Frame`。要确认：

- [ ] `vision.backend: taska_realsense`，`vision.taska_config` 指向
      `taskA_grasp/config/grasp_params.yaml`。
- [ ] 任务 A：Subset 1 的 10 类都能识别（缺 1 类丢 1.5 分）。
- [ ] 任务 A：同一物体 10 个位姿，框要贴合，**IoU ≥ 0.75**；
      用 `task_a.ground_truth_file` + `mock_demo --task A` 的评估口径复验。
- [ ] 任务 C：真实 D405 对 V11 已知 40 类的识别、深度和短边抓取闭环验收。
- [x] 任务 D：V11 已知框之外，已接入 RANSAC 支持面 + 前景连通域
      `object/unknown` 候选，并输出保守几何质量到 `Detection.graspability`。
- [x] 任务 D：真实 D405 只读零目标基线通过。入口为
      `bin/accept_task_d_vision`，它不启动 `/RobotInfo`、运动桥或夹爪后端。
- [x] 任务 D：验收工具已输出逐目标几何证据和运行时几何策略决定，并提供
      `--require-geometry` / `--require-geometry-candidate` 两级门禁。
- [ ] 任务 D：真实场景中至少一条观测通过全部几何策略门限。2026-11-02 的
      65 条观测接受数为 0：16 条无可靠支持面，49 条支持面质量低于 `0.55`
      （最高 `0.524`），不能把原始 rim/short-edge 证据当成抓取候选验收。
- [ ] 任务 D：用真实不透明未知物执行 `--require-unknown` 正样本验收，并统计
      多实例、接触粘连、误检和长时间 track 稳定性。
- [ ] 任务 D：透明/强反光物单独做深度修复与抓取验收；当前零目标基线和
      几何 proposer 都不能证明这一项通过。
- [ ] 任务 E：标签能匹配 `target_order` 的清单顺序（必要时用 `label#instance_id`
      区分同类多件）。

## 3. 夹爪

- [ ] 实现 `GripperBackend.open/close/state`（或接现有 RS485/DO 后端）。
- [ ] `gripper.backend: taska`，`gripper.taska_config` 指向抓取参数。
- [ ] `gripper.open_before_approach: true`：确认进入接近走廊前夹爪确实张开。
- [ ] 单人操作安全：确认夹爪空载、无人接触爪指时再点动。

验收：`--task B` 的 `[open_gripper]` 事件必须出现在 `[approach]` 之前；
`grasp` 到点后才出现合爪。

## 4. 夹持确认（最重要，直接对应滑落扣分）

- [ ] 用真实信号替换 `MockHoldVerifier`：
  - 夹爪位置/电流反馈到阈值，或
  - "抬起约 2cm 后复查物体是否仍在原位"（可用腕部相机或深度）。
- [ ] 失败时必须 `open + retreat`（框架已实现，确认真机能执行）。

验收：故意用一个夹不住的物体跑 `--task B`，日志出现
`[recovery] ... open jaws and retreat`，机械臂退回接近点。

## 5. 目标点位（任务 B）

- [ ] 量出五个抓取点位的 base 参考坐标，填 `task_b_place_poses_file`
      或 `task_b.place_poses`。这些坐标是视觉确认窗口，不是五个投放置。
- [ ] `task_b.rounds: 5`，`place_names` 与五个点位一一对应。
- [ ] `task_b.return_to_pick: true`，确保每次释放位置来自实际观测抓取 XY。
- [ ] `task_b.next_point_timeout_s: 0`，等待下一点视觉确认而不是超时自动继续。

验收：五个轮次的 `place_name` 依次为 `pick_1 ... pick_5`；每轮 `release`
XY 与该轮 `grasp` XY 相同；本轮 `release`、`retreat` 完成前看不到下一轮
`pregrasp`；把物品留在旧点位时，程序持续等待且不会发出下一轮运动命令。

## 6. 顺序清单（任务 E）

- [ ] `target_order.example.yaml` → 现场清单，标签与检测器输出一致。
- [ ] `target.order_file` 指向该文件。
- [ ] 确认是否允许按"类"而非"具体某一件"排序；若同一类多件必须区分，
      用 `label#instance_id`。

## 7. 真机使能

- [ ] `--execute` 关闭 dry-run 前，先确认控制器上电、伺服使能、`/RobotInfo`
      有数据（`RobotState.ready == True`）。
- [ ] 首次 Task C 必须使用单次成功硬门禁：
      `TRACK1_TASK_C_ACCEPTANCE_PICKS=1 bin/run_task_c --no-pause`。该门禁在
      一次完整成功并返回固定观察位后退出，不声称整箱已空。`--max-picks`
      对 C/D 清箱循环无效。正式配置主运动上限为 `0.09 m/s`、关节上限为
      `1.8 rad/s`；首次现场验收建议先用“机械臂速度调节”降至 10%–30%，确认
      坐标、限位、路线和碰撞风险后再逐步提高。
- [ ] `/RobotMove` 的字段与控制器协议（Codroid UDP 桥）单位一致：
      位置 m、姿态 rad，六自由度顺序 x,y,z,rx,ry,rz。
- [ ] 建议加 MoveIt 或控制器侧碰撞检查（当前是直线笛卡尔点位）。

## 8. 已知边界（暂未闭环）

- 任务 A 的实时预览（相机图 + 框显示到 9100/ROS 图像话题）尚未接线。
- D 的不透明未知物使用纯几何 proposer，不是开放词表或实例分割网络；接触物粘连、
  线缆/箱壁误检以及透明/强反光物仍未解决。
- 规划器不做碰撞规划。
- 夹爪反馈无法独立证明爪指到位（睿尔曼九字节反馈只含电机角度）。

## 9. 回归测试

任何改动后先跑离线回归，再按任务分别上真机：

```bash
cd /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_grasp
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q \
  test/test_config.py test/test_policy.py test/test_task_cd_short_edge_grasp.py
```

C、D 的 mock 也必须分别运行和分别记录日志；C 未验收时不能把 D 的通过结果当作
C 的验收结果。
