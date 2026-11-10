# Task C/D 单次实机验收硬门禁、现场执行与修复报告

生成时间：2026-11-03 23:14（Asia/Shanghai）  
最后修订：2026-11-03 23:41（Asia/Shanghai）  
当前状态：第三次 Task C 单次实机尝试已于 2026-11-03 23:39 执行并安全结束。单次验收严格只执行一个观察周期；4 个候选全部在运动前被工作空间/法兰高度门禁拒绝，机械臂运动指令为 0，夹爪命令为 0。任务锁已释放、无残留进程。第四次实机运动尚未获授权，Task D 未启动。

## 1. 单次验收门禁目标

为 Task C/D 现场验收增加只针对本次操作的硬停止，避免普通清箱模式持续运行：

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

## 2. 单次验收硬门禁

实现行为：

1. 门禁仅在已启用的 Task C/D `task_cd` 清箱流程中允许使用；其他模式传入会直接拒绝。
2. 参数必须是严格且非布尔的 Python 整数，且 `N >= 1`。
3. `1.0`、`1.5`、字符串、布尔值、0 和负数均被拒绝。
4. 非整数 CLI 输入由 `argparse` 在任何任务动作前拒绝；CLI 的 `0`/负数即使通过整数语法解析，也会在状态机入口、任何运动或夹爪命令前被拒绝。
5. 成功抓取并完成释放、撤回后，先返回固定观察位，再检查成功次数；达到 N 才退出。
6. 单次验收退出时不再额外采集观察帧，也不会执行空箱确认。
7. 节点仅在最终结果为成功且成功数达到 N 时返回成功；中途异常、失败、拒绝或未达到 N 均保持失败退出。
8. 任一首次实机失败均立即停止，不自动继续重试，也不得转而启动 Task D。

## 3. 第一次现场尝试与扫描路线余量修复

第一次现场授权后执行：

```bash
TRACK1_TASK_C_ACCEPTANCE_PICKS=1 TRACK1_CONFIRM=1 bin/run_task_c --no-pause
```

结果：

```text
日志：logs/task_C_20261103_231417.log
退出码：1
成功次数：0
机械臂：停在固定观察位，未执行扫描或抓取运动
夹爪：未打开、未闭合
任务锁/进程：已释放，无任务进程残留
```

程序在进入完整换位扫描路线前执行全路线 IK 预检并安全拒绝：

```text
C/D relocation scan route precheck rejected before any scan motion:
IK failed for target 7 path waypoint:
minimum_singular_value=0.009663 < 0.010000
No arm or gripper command was issued.
```

程序按 fail-closed 设计拒绝继续，没有降低阈值，也没有自动重试。

随后对四份正式配置做只读复算并调整扫描余量：

```text
config/track1_task_c_v11_official_20261102.yaml
config/track1_task_c_v11_official_mustard_single_20261102.yaml
config/track1_task_d_v11_official_20261102.yaml
config/track1_task_cd_40class_v11_official.yaml
```

调整内容：

```text
+Y 轴观察偏移：0.12 m -> 0.11 m
正 Y 两个角点：0.085 m -> 0.075 m
奇异值门限：保持 0.010000，未降低
```

最终 9 个换位视角：

```text
[-0.12, 0, 0]
[0.12, 0, 0]
[0, -0.12, 0]
[0, 0.11, 0]
[-0.085, -0.085, 0]
[0.085, -0.085, 0]
[-0.085, 0.075, 0]
[0.085, 0.075, 0]
[0, 0, 0.08]
```

修改前备份：

```text
backups/task_cd_relocation_scan_margin_20261103_231912/
```

离线 IK/FK 复算结果：

```text
视角数：9
去程/回程总段数：18
最小加权 Jacobian 奇异值：0.014264
正 Y 角点（视角 7 去程）最小奇异值：0.018795
正 Y 角点（视角 8 去程）最小奇异值：0.032978
```

全路线最低值高于控制器 `0.010000` 门限；回归测试额外要求至少 `0.012000`，保留约 20% 安全余量。

## 4. 第二次 Task C 单次实机尝试

第二次现场授权后执行：

```bash
TRACK1_TASK_C_ACCEPTANCE_PICKS=1 TRACK1_CONFIRM=1 bin/run_task_c --no-pause
```

执行日志：

```text
logs/task_C_20261103_232727.log
logs/codroid_bridge_20261103_232727.log
```

现场结果：

```text
RobotInfo：state=AutoReady，fault_flag=0
退出码：130（现场人工 Ctrl-C）
成功次数：0
抓取运动：未进入抓取阶段
夹爪命令：未发送打开或闭合命令
任务锁/进程：退出后已释放，无残留任务进程
```

机械臂行为：

- 固定观察姿态进入和残差校验通过。
- 9 视角、18 段换位扫描路线 IK/FK 预检通过。
- 首轮实际完成 `+X 0.12 m` 扫描去程和回程，速度 `0.027 m/s`、速度档位 30%。
- 后续观察周期在固定视角即发现候选，未继续执行换位运动。

三个观察周期均发现候选，但所有候选均在只读规划/预检阶段被拒绝：

```text
周期 1：11-Padlock、26-Plastic_Strawberry
周期 2：26-Plastic_Strawberry
周期 3：12-Baseball、26-Plastic_Strawberry
```

共 5 次候选拒绝，原因均为：

```text
C/D tilted short-edge candidate lost PCA closure alignment or tool Z
```

每次拒绝均明确记录：

```text
no Cartesian motion or gripper command was issued
```

因此本阶段只发生了观察姿态和扫描路线运动，没有发生抓取、抬升、转移、释放或实体夹爪动作。连续发现候选后，旧单次验收逻辑仍错误地准备下一轮完整扫描，现场人工按下 `Ctrl-C` 中止；runner 退出码为 130，未形成成功抓取。

日志 SHA256：

```text
e70ffd91aee07102ae6650032f4c50b40f229318fc969da137e7bdd138100653  logs/task_C_20261103_232727.log
9af62b12a35ae2ae5f8b54c5df2b08320c4d778c24f67a07c003e954032cb064  logs/codroid_bridge_20261103_232727.log
```

## 5. 第二次现场问题根因与修复

### 5.1 倾斜短边几何错误

问题位于：

```text
app/track1_grasp/track1_grasp/pipeline.py
_task_cd_short_edge_grasp_rpy_variants()
```

原实现直接使用 PCA 主轴的水平投影作为倾斜方向。PCA 主/次轴虽然三维正交，但各自投影到水平面后不保证仍然正交；由此生成的工具 Z 会与闭合轴不垂直，最终触发短边闭合适配拒绝。

修复方式：

- 将 PCA 主轴水平投影到闭合轴的正交平面；
- 使用正交化后的分量生成工具 Z 的有限倾角；
- 保持工具 Z 向下且与闭合适配；
- 若主轴水平投影几乎与闭合轴平行，则跳过不可安全构造的可选倾斜候选；
- 保留原有闭合适配、工具 Z、工作空间、IK、奇异值和碰撞门禁；
- 未降低任何安全阈值；
- 门禁异常日志增加 `tilt_deg`、`closure_z`、`alignment`、`tool_z_max_error` 诊断。

### 5.2 单次验收错误自动重复扫描

修改 `_run_task_cd_clear()`：

- `acceptance_max_successes` 模式只允许一个观察周期；
- 达到成功数后返回固定观察位并立即退出；
- `REJECTED` 立即停止；不释放临时排除项，不启动第二轮扫描；
- `NO_TARGET` 不算验收成功，立即失败退出；
- 可安全恢复的抓取失败先返回观察位，然后退出，不自动重试；
- 普通 C/D 清箱模式行为保持不变。

同步修改 `nodes/track1_node.py`：

- 验收模式只有最终状态为 `SUCCESS` 且达到成功数才返回退出码 0；
- `NO_TARGET`、`REJECTED` 和失败恢复均返回非零退出码。

修改前备份：

```text
backups/task_c_tilt_acceptance_fix_20261103_233302/
```

备份包含修改前四个源码/测试文件及 `SHA256SUMS.before`。

## 6. 回归验证

新增测试：

```text
test_pca_tilt_preserves_closure_when_xy_axis_projections_are_not_orthogonal
test_acceptance_non_success_stops_after_one_scan_without_releasing_exclusions
test_acceptance_recoverable_failure_returns_then_stops_without_retry
```

验证结果：

```text
聚焦回归：103 passed in 17.14s
正式 ROS 2 运行时全量：806 passed in 115.80s
Task C runner check：通过
Task D runner check：通过
bash -n：通过
py_compile：通过
```

全量测试使用正式 ext4 运行时中的 ROS 2 Humble、Track1 安装空间和 MS42DC 安装空间执行。

## 7. 当前速度档位

`bin/arm_speed_control --check` 只读结果：

```text
B: 100%  cartesian=0.120 m/s  joint=0.600 rad/s  limit=0.120/0.600  dry_run=True   gripper=mock
C:  30%  cartesian=0.027 m/s  joint=0.540 rad/s  limit=0.090/1.800  dry_run=False  gripper=ms42dc
D: 100%  cartesian=0.090 m/s  joint=1.800 rad/s  limit=0.090/1.800  dry_run=False  gripper=ms42dc
```

Task C/D 视觉伺服速度仍为 `0.01 m/s`，未被主运动调速修改。

## 8. 文件校验值

配置：

```text
8b26f00bd21ded49cafb3f6e9fed8bfe5bd56f3768f4f48b803499e9642056c5  config/track1_task_c_v11_official_20261102.yaml
196dd6e98577e0873b21cb1b93a422380f320dc2cd488a66ad351a3556089d9a  config/track1_task_c_v11_official_mustard_single_20261102.yaml
158410d4455edb4ce9fa76093157d36192243eeb5379418e3369d60b52de4854  config/track1_task_d_v11_official_20261102.yaml
d040f7082b60bffa0fd1a36bc5b7a7f26d166bddf902dd94370ac219bca93377  config/track1_task_cd_40class_v11_official.yaml
```

本次修复涉及的程序和测试：

```text
afb6ad3dbbcd4fca716239aeec71777ccd83cbac2d66bc98dd989752735037d9  app/track1_grasp/track1_grasp/pipeline.py
541b1e255bc58299c6050b3dfa634c72a09e0f72786a0d47e93fb92a27cf2bd5  app/track1_grasp/track1_grasp/nodes/track1_node.py
2703335e2ecc49049993e7fb696f04ef630e09a01b42bc45b3822162c74429ee  app/track1_grasp/test/test_task_cd_short_edge_grasp.py
0246d60e9d5eb1d92e9df2f6b007479d0030de5c7138f2d64f6d33be3d7b6954  app/track1_grasp/test/test_task_cd_fixed_observation_clear.py
```

前述扫描余量与安全测试文件：

```text
bf1b5511b6a6701b4d73ea9cb0e9e98d777ace38f287644b1bf7eb5d3447c8c0  app/track1_grasp/test/test_task_cd_grasp_safety.py
03493c862c087aa40682474fe1645f026ee11787a2f9120ff509b63c64180867  bin/track1_task_runner
a3a0d27c4be068862df6360ffa05817e1e38e6e98da028f72f41f5cc3478ab4e  bin/run_task_c
928c5f6316163c5efd7ae65bf5f25dc6e7b8b3cccb157f745944c83e15b308ff  bin/run_task_d
808c76dd408be79f1df62000a02e55dfb7f2e1df2e337c6d814af01e24fdcc37  logs/task_C_20261103_231417.log
4771cffd45b1effa3ed66213427f969534944b01682600e0eef1d287e71ff940  state/arm_speed_overrides.json
```

本报告修订前 SHA256：

```text
08aea646553fda3785e7f631f226fd3153309935fd462df3e04f1ce360129380  reports/task_cd_acceptance_guard_20261103.md
```

## 9. 第三次 Task C 单次实机尝试

第三次现场授权后执行：

```bash
TRACK1_TASK_C_ACCEPTANCE_PICKS=1 TRACK1_CONFIRM=1 bin/run_task_c --no-pause
```

执行日志：

```text
logs/task_C_20261103_233900.log
logs/codroid_bridge_20261103_233900.log
```

执行结果：

```text
RobotInfo：state=AutoReady，fault_flag=0
固定观察姿态残差：0.017 deg
9 视角、18 段扫描路线预检：通过
退出码：1
观察周期：1
成功次数：0
最终状态：rejected
机械臂运动命令：0
夹爪命令：0
任务锁：FREE
task.pid：不存在
残留任务进程：无
```

程序在固定视角发现 4 个已确认候选，全部为 `18-Dice`：

```text
track-6  score=0.74
track-4  score=0.72
track-8  score=0.54
track-9  score=0.48
```

每个候选均完成 10 组抓取点/朝向只读预检，共 40 组预检。各候选最终安全拒绝原因：

```text
track-6：pregrasp 水平距离 0.883755 m > 0.820000 m
track-4：抓取法兰 Z 0.142997 m < 0.164621 m
track-8：pregrasp 水平距离 0.890674 m > 0.820000 m
track-9：pregrasp 水平距离 0.965932 m > 0.820000 m
```

程序明确记录：

```text
all 4 confirmed C/D candidate(s) were rejected during read-only planning/precheck;
no Cartesian motion or gripper command was issued
```

本次没有降低工作空间、法兰高度、IK 或奇异值门禁。桥接日志中的 `tx=0`、主日志中机械臂实际运动命令为 0，确认没有向控制器发送 `RobotMove`，也没有发送夹爪动作。

新单次验收门禁已按预期生效：

```text
[run] C/D operator acceptance rejected the observation cycle;
no automatic retry or second full scan will run

observation_cycles=1 successes=0 completed_tracks=0
failed_tracks_pending_retry=0 rejected_tracks_pending_retry=4
final_status=rejected
```

因此本次是安全失败，不是程序循环失控，也没有发生抓取或放置动作。

日志 SHA256：

```text
7b0f229ec46158834b163a6a4206e60d1b6bb23c3343e358f3332414371a16ec  logs/task_C_20261103_233900.log
b5f45444106375ba2ee79a0bbc768e718d4740d3c04ac2de6a937d180190ac20  logs/codroid_bridge_20261103_233900.log
```

## 10. 当前运行状态与下一步

只读检查结果：

```text
task lock: FREE
task pid file: absent
无 track1_node / track1_task_runner / Task C/D runner 任务进程
Task D: 未启动
```

安全门禁：

- Task B 继续保持 `motion.dry_run: true`、`gripper.backend: mock`。
- Task D 在 Task C 验收成功前不得启动。
- 一次只允许一个任务。
- 单次验收只允许一个观察周期；非成功结果不得自动重试或再次扫描。
- 不得降低工作空间、法兰高度、几何、奇异值或工作空间门禁。
- 正式实机命令必须使用单次成功门禁。

第三次授权已经使用；当前没有第四次 Task C 实机运动授权。下一次现场实机前，应先检查现场物体/料箱位置和手眼坐标，确认候选落入已验证安全工作空间，并再次由用户明确确认：

> 现场已清场、无人进入工作区、急停在手、控制器 AutoReady 且无故障，允许执行第四次 Task C 单次抓取。

获得新的明确确认后，才可执行：

```bash
bin/ensure_runtime
TRACK1_TASK_C_ACCEPTANCE_PICKS=1 TRACK1_CONFIRM=1 bin/run_task_c --no-pause
```
