# Task B 箱体任意摆放动态五点抓取离线验收（2026-10-26）

## 结论

**通过。** 已使用正式数据盘配置完成 5 轮动态箱体拟合、每轮五点全路线预检、固定箱体局部点匹配、Mock 抓取与保持验证，结果 **5/5 成功，25/25 条候选路线预检通过**。全程仅离线 Mock/dry-run，未连接实体机械臂，未发送真机运动或夹爪命令。

## 安全边界

- `motion.backend: mock`
- `motion.dry_run: true`
- `gripper.backend: mock`
- 未执行任何真机运动、夹爪开合、手眼现场标定或坐标验证。
- 本验收只证明软件闭环和离线几何流程，不代表现场箱体测量、限位、手眼标定和夹爪阈值已通过。

## 正式资产

- 配置：`config/track1_task_b_box_relative_mock.yaml`
- 五轮场景：`app/track1_grasp/scenarios/task_b_box_relative_mock_20261024.yaml`
- 回归测试：`app/track1_grasp/test/test_box_relative_five_point_mock.py`

场景每一帧均按上一轮完成原位拾放后 Mock 机械臂实际停留的撤回位姿渲染，包括该轮箱体偏航角对应的垂直抓取工具姿态；避免后续帧继续错误使用初始相机位姿。

## 启动器接入与正式入口验收

正式数据盘桌面入口仍为 `任务B_任意摆放抓取.desktop`，显示名称已更新为“任务B 箱体动态五点抓取（离线试运行）”，`Exec` 直接指向数据盘中的 `bin/run_task_b_arbitrary`。

2026-10-26 20:36 测试通过后，已将正式入口同步到 `/home/sunshine/Desktop/赛道一比赛程序/任务B_五点位拾放.desktop`。文件名保持不变，但文件内容、显示名称和执行目标均已切换为箱体动态五点 Mock 入口；同步前后校验值一致。原快捷方式备份在 `backups/home_desktop_task_b_20261026_203603/任务B_五点位拾放.desktop`。

启动器已完成以下接入：

- `bin/run_task_b_arbitrary` 默认配置改为 `config/track1_task_b_box_relative_mock.yaml`，同时保留 `TRACK1_CONFIG` 环境变量以显式选择旧任意摆放配置。
- `bin/track1_task_runner` 同时识别 `arbitrary_placement: true` 和 `box_relative_five_point.enabled: true`，动态任务只验证 B 模式，不再依次套用 A/C/D/E 配置检查，不会误走旧五点位规则。
- 启动标题、运行模式、真机/试运行提示均能正确识别箱体动态五点模式。
- `nodes/track1_node.py` 的人工确认门控已识别箱体动态模式，并使用“重新拟合箱体位姿”的专用提示；当前 Mock 配置不启用轮间暂停。
- 正式入口验收未启动 `/motor_node`，未启动机械臂状态桥接，未发送真机运动或夹爪命令。

正式入口验证命令：

```bash
cd '/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序'
TRACK1_NO_UI=1 TRACK1_NO_PAUSE=1 bin/run_task_b_arbitrary --no-pause
```

最终验收日志：

```text
logs/task_B_20261026_203555.log
requested=5 attempts=5 successes=5 rate=100.00%
all five routes prechecked: 5/5 rounds
completed box_relative_round_N: 5/5 rounds
ERROR/Traceback/Exception/失败: 0
```

## 五轮结果

| 轮次 | 本帧拟合箱心（base m） | 拟合偏航 | 匹配固定局部点 | 五点预检 | Mock 抓取/保持 |
|---|---:|---:|---|---:|---:|
| 1 | `(0.0000, 0.0500, 0.0200)` | `0.00°` | `corner_nw` | 5/5 | 通过 |
| 2 | `(-0.0300, 0.0600, 0.0200)` | `20.00°` | `corner_ne` | 5/5 | 通过 |
| 3 | `(0.0400, 0.0700, 0.0200)` | `-15.00°` | `corner_se` | 5/5 | 通过 |
| 4 | `(0.0200, 0.0300, 0.0200)` | `35.00°` | `corner_sw` | 5/5 | 通过 |
| 5 | `(-0.0200, 0.0800, 0.0200)` | `-30.00°` | `center` | 5/5 | 通过 |

每轮流程满足以下约束：

1. 只使用当前帧拟合箱体位姿，不读取 `place_poses` 或历史抓取位置。
2. 抓拍前后机器人状态完整且稳定，姿态、平移、六关节漂移均在配置阈值内。
3. 五条固定箱体局部点路线全部预检成功后，才进行物品点匹配。
4. 物品匹配点与最终执行路线必须同名。
5. 工具 Z 轴垂直向下，逐段路线在动作前完成预检。
6. 动态模式不保存 `_task_b_completed_xy`。
7. 每轮执行 6 个 waypoint；Mock 夹爪完成张开、闭合、保持验证和释放。

## 自动化验证

- 箱体专项回归：`2 passed in 0.51s`
- 全量回归：`399 passed in 23.75s`
- 全量命令：

```bash
source /media/sunshine/ESTUN_RUNTIME/ros2_ws/install/setup.bash
cd '/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_grasp'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:${PYTHONPATH:-}" \
  /usr/bin/python3 -m pytest -q -p no:cacheprovider
```

## 文件校验

```text
318a6f6e2bff8ded8669823db3b3ca98e7a24f15a83f9e9bc898cbeb9c155861  config/track1_task_b_box_relative_mock.yaml
5248944cda33e8c4cf55b182cca89fc23a8c9baff982cd67cbb0f60754197e3e  app/track1_grasp/scenarios/task_b_box_relative_mock_20261024.yaml
347eeec4fdfc782b4e0b1be4dcc48693f51157310c5aebf0a3a25e83de8d6bb1  app/track1_grasp/test/test_box_relative_five_point_mock.py
5e1ecd7a223bcca0fc47da2cc1423d53491aa5b4665891d19e98f1b2dfb239ca  bin/track1_task_runner
6382894d7cb2a50ba27fcb1a92354f6d17d089ca66806e90df935e96833e1a48  bin/run_task_b_arbitrary
94f7e5713908bca57dedb1c81aa4468c6403e662d14cccded64d2d17cac7357c  app/track1_grasp/track1_grasp/nodes/track1_node.py
6ca7e1902c24301c13abda518b3de5c4c2dd3c5fd27f9b40eae417985706794e  任务B_任意摆放抓取.desktop
17b3a2651f2a0f998be836e203093d7a5b27b1ec69be5e480667bf45f58d09fd  logs/task_B_20261026_203555.log
```

## 现场启用前仍需完成

1. 复核真实箱体内部尺寸、箱心、偏航和支撑面深度质量。
2. 复核相机到法兰外参及当前控制器位姿语义。
3. 验证现场工作空间、关节限位、碰撞边界和工具垂直姿态。
4. 验证 MS42DC 夹爪开口、闭合阈值与保持反馈。
5. 未经用户明确确认，不得切换 `dry_run: false` 或执行实体机械臂运动。
