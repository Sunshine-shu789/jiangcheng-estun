# Task C/D 单次实机验收硬门禁验证报告

生成时间：2026-11-03 23:14（Asia/Shanghai）  
状态：实现、聚焦回归、正式 ROS 2 运行时全量回归和 C/D 只读启动检查均通过；未启动比赛任务，未发送机械臂或实体夹爪运动命令。

## 1. 门禁目标

为 Task C/D 首次现场验收增加只针对本次操作的硬停止，避免清箱模式持续运行：

- `pipeline.run(..., acceptance_max_successes=N)`；
- 完成 N 次完整成功抓取后退出；
- “完整成功”必须包含抓取、抬升、转移、释放、撤回；
- 退出前必须返回固定观察位；
- 单次验收不声称整箱为空；
- 普通 C/D 清箱流程行为保持不变，旧 `--max-picks` 仍不是清箱成功次数上限。

启动入口：

```bash
TRACK1_TASK_C_ACCEPTANCE_PICKS=1 bin/run_task_c --no-pause
TRACK1_TASK_D_ACCEPTANCE_PICKS=1 bin/run_task_d --no-pause
```

对应节点参数：

```text
--acceptance-max-successes N
```

## 2. 实现与安全边界

修改文件：

```text
app/track1_grasp/track1_grasp/pipeline.py
app/track1_grasp/track1_grasp/nodes/track1_node.py
app/track1_grasp/test/test_task_cd_fixed_observation_clear.py
bin/track1_task_runner
```

实现行为：

1. 门禁仅在已启用的 Task C/D `task_cd` 清箱流程中允许使用；其他模式传入会直接拒绝。
2. 参数必须是严格且非布尔的 Python 整数，且 `N >= 1`。
3. `1.0`、`1.5`、字符串、布尔值、0 和负数均被拒绝。
4. 非整数 CLI 输入由 `argparse` 在任何任务动作前拒绝；CLI 的 `0`/负数即使通过整数语法解析，也会在状态机入口、任何运动或夹爪命令前被拒绝。
5. 成功抓取并完成释放、撤回后，先返回固定观察位，再检查成功次数；达到 N 才退出。
6. 单次验收退出时不再额外采集观察帧，也不会执行空箱确认。
7. 节点仅在最终结果为成功且成功数达到 N 时返回成功；中途异常、失败、拒绝或未达到 N 均保持失败退出。

## 3. 回归验证

### 3.1 聚焦回归

```text
59 passed in 7.97s
```

覆盖内容：

- 单次成功完成后立即硬停止；
- 只执行两次关节运动：进入固定观察位、成功回固定观察位；
- 不增加返回后的额外固定观察帧；
- 已执行完整的 `pregrasp -> grasp -> lift -> transfer -> release -> retreat`；
- 夹爪只执行 `open -> close -> open`；
- 0、负数、布尔值、`1.0`、`1.5`、`"1"`、普通非法字符串均拒绝；
- 原有清箱、失败恢复、重试、空箱确认和多视角测试未回退。

### 3.2 正式 ROS 2 运行时全量回归

命令环境：

```text
Humble + 正式 ext4 ROS 工作空间 + MS42DC vendor 工作空间
CUDA_VISIBLE_DEVICES=
ROS_LOCALHOST_ONLY=1
```

结果：

```text
800 passed in 80.49s
```

### 3.3 启动器和只读检查

```text
bash -n bin/track1_task_runner                              OK
Python compileall                                           OK
Task C runner check                                         OK
Task D runner check                                         OK
CLI acceptance argument syntax                              OK
```

C/D 两项 `track1_task_runner check` 均输出：

```text
Python / ROS / YOLO / 五个任务配置检查通过
```

运行锁和进程状态：

```text
task lock: FREE
task pid file: absent
无 track1_node / track1_task_runner 任务进程
```

## 4. 当前速度档位

`bin/arm_speed_control --check` 只读结果：

```text
B: 100%  cartesian=0.120 m/s  joint=0.600 rad/s  limit=0.120/0.600  dry_run=True   gripper=mock
C:  30%  cartesian=0.027 m/s  joint=0.540 rad/s  limit=0.090/1.800  dry_run=False  gripper=ms42dc
D: 100%  cartesian=0.090 m/s  joint=1.800 rad/s  limit=0.090/1.800  dry_run=False  gripper=ms42dc
```

Task C/D 视觉伺服速度仍为 `0.01 m/s`，未被主运动调速修改。

## 5. 文件校验值

```text
36cd3d995c86c6de8e58d9695165749009afa6f4ec95374f8c4048eac98cefb9  app/track1_grasp/track1_grasp/pipeline.py
9864454e9da101fb06ad8227bdeb247da082635b56c96d93f891a91678612725  app/track1_grasp/track1_grasp/nodes/track1_node.py
a60e4bc0c263629f26877f62d928c02487cc85cc45acd6a11b966f74f8ffe8fb  app/track1_grasp/test/test_task_cd_fixed_observation_clear.py
03493c862c087aa40682474fe1645f026ee11787a2f9120ff509b63c64180867  bin/track1_task_runner
a3a0d27c4be068862df6360ffa05817e1e38e6e98da028f72f41f5cc3478ab4e  bin/run_task_c
928c5f6316163c5efd7ae65bf5f25dc6e7b8b3cccb157f745944c83e15b308ff  bin/run_task_d
881e5ff98b73458eee8bf988023ea371914b60a8fd0223208173937e004c4163  config/track1_task_c_v11_official_20261102.yaml
e9d9f432009366ae0d2e907f920cae013bb25f97b384fa525fab40b006ee184e  config/track1_task_d_v11_official_20261102.yaml
0a38c34f21f4818074227e3419b9026a8b7dd3935548917d1276a95c77282974  使用说明.txt
de6c244f99f52cc67d969ddcd2dbc889b1cd9031b4373645dba4a48d861f5d89  app/track1_grasp/README.md
63501bd574fd61b896093a3c6507ed09ceeaee619090a3644a3efef08c0332f3  app/track1_grasp/docs/INTEGRATION_CHECKLIST.md
4771cffd45b1effa3ed66213427f969534944b01682600e0eef1d287e71ff940  state/arm_speed_overrides.json
a5da3f4b918f79006d6b7a35abbd9ed6ed7c3ad3429d4d7740558c7207ebdc12  backups/flexible_approach_prechange_20261103_214043.tar.gz
```

## 6. 实机前状态与下一步

- Task B 继续保持 `motion.dry_run: true` 和 `gripper.backend: mock`。
- Task C 尚未完成实体验收，Task D 不得先于 Task C 运行。
- 当前没有任务进程、没有持有任务锁、没有实体运动发生。
- 下一次实机运行前仍需执行 `bin/ensure_runtime`。
- 必须由用户明确确认：现场已清场、无人进入工作区、急停在手、控制器 `AutoReady` 且无故障。
- 获得确认后，首次只运行 Task C 单次门禁；不得同时启动 Task D。

```bash
bin/ensure_runtime
TRACK1_TASK_C_ACCEPTANCE_PICKS=1 TRACK1_CONFIRM=1 bin/run_task_c --no-pause
```
