# Task B/C/D 快速速度收尾与 C/D 超宽预检回归

生成时间：2026-11-03 23:01（Asia/Shanghai）  
状态：离线配置、文档和全量回归完成；未启动 ROS、任务程序、机械臂或夹爪运动。

## 当前正式速度上限

| 任务 | 主笛卡尔速度 | 关节配置速度 | 关节安全上限 | 实际安全关节上限 | 当前执行状态 |
|---|---:|---:|---:|---:|---|
| B | 0.120 m/s | 0.600 rad/s | 0.700 rad/s | 0.600 rad/s | `dry_run=true`，夹爪 `mock` |
| C | 0.090 m/s | 1.800 rad/s | 1.800 rad/s | 1.800 rad/s | 仅离线验证，未实机运行 |
| D | 0.090 m/s | 1.800 rad/s | 1.800 rad/s | 1.800 rad/s | 仅离线验证，未实机运行 |

Task C/D 的视觉伺服 `servo.speed_mps` 独立保持 `0.01 m/s`，不属于主运动提速，
也没有被本次修改。操作员调速工具的 100% 值不能突破上述正式配置上限。

## 正式修改

- `config/track1_task_b_five_point_adaptive_safe_20261102.yaml`
  - 主运动速度 `0.12 m/s`；
  - 关节配置速度 `0.6 rad/s`，安全上限 `0.7 rad/s`；
  - 继续强制 `motion.dry_run: true` 和 `gripper.backend: mock`。
- `config/track1_task_c_v11_official_20261102.yaml`
  - 主运动速度 `0.09 m/s`；
  - 关节配置速度和安全上限均为 `1.8 rad/s`。
- `config/track1_task_d_v11_official_20261102.yaml`
  - 主运动速度由历史 `0.03 m/s` 提升至 `0.09 m/s`；
  - 关节配置速度和安全上限均为 `1.8 rad/s`。
- `app/track1_grasp/test/test_speed_limit.py`
  - 固定校验 A/B/C/D/E 的正式速度上限；
  - 明确校验 Task B 的 `0.6/0.7 rad/s` 配置/安全上限和 dry-run/mock 状态；
  - 防止界面状态文件突破正式配置。
- `使用说明.txt`、`app/track1_grasp/README.md` 和
  `app/track1_grasp/docs/INTEGRATION_CHECKLIST.md`
  - 删除 Task D 仍为 `0.03 m/s / 0.6 rad/s` 的过期说明；
  - 补全当前 B/C/D 速度与关节安全上限；
  - 首次 Task C 现场验收改为从正式 `0.09 m/s` 上限通过调速工具降至
    10%–30% 起步，确认后再逐步提高。

## C/D 超宽预检回归

在 `test_task_cd_fixed_observation_clear.py` 新增端到端离线回归：

```text
measured minor_axis_length_m = 0.070960 m
close_travel_width_margin_m   = 0.005000 m
effective width               = 0.075960 m
CAD opening at 50 mm depth    = 0.072549 m
```

测试结果确认：

- 候选在 `pregrasp` / `grasp` 规划前即被宽度门禁拒绝；
- `pipeline.motion.moves == []`，没有笛卡尔运动命令；
- `pipeline.gripper.events == []`，没有夹爪命令；
- 结果包含明确的 `rejected before motion` 日志和 CAD 开口原因。

该检查位于候选完整路线预检之前；执行阶段仍会再次调用宽度检查，保留双重防护。
本轮未改写历史报告 `task_c_large_workspace_3x_speed_20261103_2039.md` 中的历史
快照，本报告代表当前正式速度状态。

## 保留的灵活抓取能力

本轮未回退以下已完成改造：

- 六路点垂直抓取路线；
- 七路点 `tool_axis_approach`；
- PCA 与 Geometry 倾斜候选；
- 确定的 10 个候选顺序；
- `rotation_from_tool_axis_and_closure()`；
- Task D 灵活扫描和空箱双轮确认。

修改前备份：

```text
backups/flexible_approach_prechange_20261103_214043.tar.gz
SHA256 a5da3f4b918f79006d6b7a35abbd9ed6ed7c3ad3429d4d7740558c7207ebdc12
```

## 离线验证

全量命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=app/track1_grasp \
python3 -m pytest -q app/track1_grasp/test \
  --ignore=app/track1_grasp/test/test_arm_motion_check.py
```

结果：

```text
783 passed, 2 skipped in 81.49s
```

`test_arm_motion_check.py` 因宿主 Python 缺少 `rclpy` 不在本次宿主回归内；ROS 2
运行时内仍应按正式检查流程另行执行。

补充检查：

```text
Python AST 语法检查：103 files OK
bin/run_task_* shell 语法检查：OK
```

`bin/arm_speed_control --check` 只读结果确认：

```text
B: cartesian=0.120 m/s  joint=0.600 rad/s  dry_run=True   gripper=mock
C: cartesian=0.090 m/s  joint=1.800 rad/s  dry_run=False  gripper=ms42dc
D: cartesian=0.090 m/s  joint=1.800 rad/s  dry_run=False  gripper=ms42dc
```

## 文件校验值

```text
270a6326d3c14d7c356e30d083e68803db62643eecd54f39e808703c2e743b1f  config/track1_task_b_five_point_adaptive_safe_20261102.yaml
881e5ff98b73458eee8bf988023ea371914b60a8fd0223208173937e004c4163  config/track1_task_c_v11_official_20261102.yaml
e9d9f432009366ae0d2e907f920cae013bb25f97b384fa525fab40b006ee184e  config/track1_task_d_v11_official_20261102.yaml
a5ba37e04d2702154c77d912077a21468c9f8608db2f1d7021032a9ac8f16373  app/track1_grasp/test/test_speed_limit.py
965d446734f8172a4506a0baf2507d567e145d435bdb986ff2981475b2bae013  app/track1_grasp/test/test_task_cd_fixed_observation_clear.py
08065cc70ed63849785b480a4310901b29b2bd62767935636ba9a4c39ce2c4a1  app/track1_grasp/README.md
d1b2ba044d9144d351f3f2dba6dcbeabf384dc23653a243281ab225d2130563a  使用说明.txt
905aa4d21dec647f820acd6bd8ffde32e1cae9754c4440b5c1e33d45cf7bfd35  app/track1_grasp/docs/INTEGRATION_CHECKLIST.md
```

## 安全状态与下一步

- 本轮没有启动 `run_task_b/c/d`，没有发送 `/RobotMove`，没有打开或关闭实体夹爪。
- Task B 继续保持 dry-run 和 mock，直到现场坐标、限位、手眼标定和夹爪阈值完成。
- Task C 的 `0.09 m/s / 1.8 rad/s` 尚未完成实体验收；不得据此直接运行 Task D。
- 下一次实机运行仍需用户明确现场授权，并先执行 `bin/ensure_runtime`，确认急停、
  工作区无人、控制器使能/无故障且完整路线预检通过。
