# Task C/D 单次实机验收硬门禁、现场执行与修复报告

生成时间：2026-11-03 23:14（Asia/Shanghai）  
最后修订：2026-11-04 00:36（Asia/Shanghai）  
当前状态：第四次 Task C 单次实机尝试于 2026-11-04 00:29 至 00:34 执行并安全结束。单次验收严格只执行一个观察周期；2 个确认候选均在运动前被安全门禁拒绝，桥接 `tx=0`，机械臂运动指令为 0，夹爪命令为 0。`track-4` 的预抓取水平半径仍为 `0.890152 m`，超过软件 `0.820 m` 包络；`track-1` 在 IK 路点处的最低加权 Jacobian 奇异值为 `0.009753`，低于 `0.010000`。没有降低任何门限，没有自动重试。任务锁已释放、无残留进程，Task D 未启动；第五次实机运动尚未获授权。

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

本次精确向下可达性诊断补丁（修改后）：

```text
2ef20b3b422b49c091f32bef975ea46eb884e828f7b19dc8699e7c04d9456fbf  app/track1_grasp/track1_grasp/kinematics.py
ea9fa68b1419be856c0cc81ee3ba697c5605443b95dd269572ace96dca899800  app/track1_grasp/track1_grasp/motion.py
fbe9a2dbe3e1ce570ba7aa7ce03a34a9bb256a54f8a16d376b2b93820a25c4c0  app/track1_grasp/test/test_kinematics_ik.py
544a2c1c7e6cfa7710541946bb407724a6db61edaf9008c78662e238df4dbd69  app/track1_grasp/test/test_motion_linear.py
```

修改前备份：

```text
backups/task_c_downward_reach_guard_20261104_000528/
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

## 10. 第三次失败后离线根因定位与精确向下可达性门禁

本节全部为正式 DH/FK 复算、源码检查和离线测试；没有启动任务节点、连接控制器、发送机械臂运动命令或夹爪命令。

### 10.1 `track-4` 的真实不可达结论

第三次日志中最接近通过的候选为 `18-Dice / track-4`。其精确向下的 `pregrasp` 法兰目标为：

```text
xyz = (-0.360909768, 0.714530319, 0.245103) m
水平半径 = 0.800505738523 m
姿态 = tool Z 精确向下
```

旧流程在该候选的某条近奇异路径上报告的最低 Jacobian 奇异值为 `0.009745`，仅低于 `0.010000` 门限约 `2.55%`。但进一步按正式 modified-DH/FK 解析后确认：这不是“种子不足”或“中间路点不够”能够修复的问题，而是终点本身在该竖直姿态下不可达。

### 10.2 精确向下姿态的解析可达上界

对 `kinematics.py` 的正式 controller modified-DH 做符号展开后，工具 Z 轴为：

```text
z_x = -sin(q1)cos(q5) + sin(q5)cos(q1)cos(q2+q3-q4)
z_y =  sin(q1)sin(q5)cos(q2+q3-q4) + cos(q1)cos(q5)
z_z = -sin(q5)sin(q2+q3-q4)
```

令工具 Z 精确等于 `(0, 0, -1)`，可得：

```text
q5 = +pi/2, q2 + q3 - q4 = +pi/2
或
q5 = -pi/2, q2 + q3 - q4 = -pi/2
```

`q1` 只旋转水平方向，`q6` 不改变工具 Z，因此水平可达半径只由 `q2/q3` 以及由上述约束确定的 `q4` 决定。对两套分支和全部正式关节范围做连续优化复核，最大水平半径为：

```text
0.585578111247 m
```

对应的最大见证姿态为：

```text
q = [0, 90, 0, 0, 90, 0] deg
及其符号镜像
```

代码采用略高于真实上界的端点筛选值：

```text
EXACT_DOWN_HORIZONTAL_REACH_BOUND_M = 0.5856 m
```

这样只把已证明不可能的目标提前明确拒绝，同时不会把真实最大见证姿态误判；`0.5856 ~ 0.585578111247` 之间的小区间仍会进入原有完整 IK/FK、连续性、工作空间和 Jacobian 校验，不被该诊断直接放行。

当前 `track-4` 的 `0.800506 m` 比真实上界多：

```text
0.800505738523 - 0.585578111247 = 0.214927627276 m
超出比例约 36.70%
```

因此保持当前物体位置和固定观察位时，无论增加多少中间路点、扩大 IK 种子或更换求解器，都不可能把该竖直终点变成可达。必须把物体/料箱向机械臂内移，或重新示教观察位并重新换算手眼坐标。

### 10.3 新增的 fail-closed 前置诊断

代码修改：

- `kinematics.py`
  - 新增 `EXACT_DOWN_HORIZONTAL_REACH_BOUND_M`、`EXACT_DOWN_TOOL_Z_TOLERANCE`；
  - 新增 `is_exact_downward_tool_axis()`；
  - 新增 `exact_downward_horizontal_reach_violation()`。
- `motion.py`
  - 新增 `_validate_exact_downward_reach()`；
  - 在候选端点 `precheck_pose_condition()`、线性路线目标端点、以及细分后的每个 Cartesian 路点执行检查；
  - 在进入 IK/路线细分前给出明确原因，避免把不可达终点误报成近奇异或把规划重试预算耗尽。

该诊断只匹配严格向下的工具 Z；倾斜抓取候选不受这个特定上界限制，仍按原流程进行完整工作空间、IK、连续性、碰撞相关约束和 Jacobian 门禁。此次补丁没有降低以下任何已有门限：

- `pregrasp` 水平安全距离；
- 最小法兰高度；
- 工作空间边界；
- Cartesian IK 连续性/步长；
- 加权 Jacobian 最小奇异值；
- 单次验收失败后不重试、不二次扫描。

### 10.4 回归结果

聚焦回归：

```text
5 passed in 0.35s
```

正式 ROS 2 运行时全量回归：

```text
811 passed in 119.33s
```

覆盖内容包括：精确向下正式关节包络、最大见证姿态、`track-4` 候选端点/线性路线提前拒绝、`0.55 m` 可达向下端点仍进入正常 IK 校验，以及现有全部 Task C/D 安全与执行测试。

### 10.5 下一步现场调整

完成第五次实机运动前，应先完成以下只读/现场调整：

1. 将料箱和骰子整体向机械臂内移；若仍保持本次 `track-4` 的精确向下抓取姿态，径向至少需消除约 `0.215 m` 的超界量，并建议再留现场操作余量。
2. 对 `track-6/8/9` 同时检查 `pregrasp` 水平距离，避免只修复 `track-4` 后仍被 `0.820 m` 安全距离拒绝。
3. 重新采集固定观察位画面，确认所有候选的最终抓取/预抓取法兰目标均通过原有完整门禁。
4. 第五次实机仍需用户新的明确授权；不得因为代码回归通过就自动启动机械臂。

## 11. 当前运行状态与下一步

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
- 精确向下端点超过正式 DH 可达上界 `0.5856 m` 时，应在 IK 前直接拒绝；该诊断不得被绕过。
- 正式实机命令必须使用单次成功门禁。

第四次授权已经使用并完成单次尝试（结果见第 12 节）；当前没有第五次 Task C 实机运动授权。`track-4` 仍未进入预抓取包络，`track-1` 仍存在近奇异路点；下一次现场实机前，应先完成本节 10.5 的调整并确认所有候选通过完整门禁，再由用户明确确认：

> 现场已清场、无人进入工作区、急停在手、控制器 AutoReady 且无故障，允许执行第五次 Task C 单次抓取。

获得新的明确确认后，才可执行：

```bash
bin/ensure_runtime
TRACK1_TASK_C_ACCEPTANCE_PICKS=1 TRACK1_CONFIRM=1 bin/run_task_c --no-pause
```

## 12. 第四次 Task C 单次实机尝试（2026-11-04）

用户明确确认现场清场、无人进入工作区、急停在手、控制器 AutoReady 且无故障后，执行第四次单次验收：

```bash
bin/ensure_runtime
TRACK1_TASK_C_ACCEPTANCE_PICKS=1 TRACK1_CONFIRM=1 bin/run_task_c --no-pause
```

结果：

```text
任务日志：logs/task_C_20261104_002917.log
桥接日志：logs/codroid_bridge_20261104_002917.log
退出码：1
观察周期：1
确认候选：2
成功次数：0
最终状态：rejected
```

本次仍是严格的单次验收：只完成一个观察周期；两个确认候选的全部抓取点和倾角变体均在运动前被拒绝，日志明确记录：

```text
all 2 confirmed C/D candidate(s) were rejected during read-only planning/precheck;
no Cartesian motion or gripper command was issued
```

桥接日志逐次记录的 `tx=0` 也确认没有发送 `RobotMove` 数据报；日志中没有夹爪开爪、闭爪或宽度控制命令。机械臂仍停在进入任务时的固定观察位，未执行扫描换位或抓取运动。

### 12.1 `track-4` 仍然超出预抓取水平包络

`18-Dice / track-4` 的本次目标为：

```text
视觉夹爪尖目标：(-0.405690, 0.792330, 0.000103) m
预抓取法兰：    (-0.405700, 0.792300, 0.245100) m
预抓取水平半径：0.890130 m
```

对应的主候选被拒绝原因：

```text
grasp safety waypoint 'pregrasp' horizontal reach 0.890152 m exceeds 0.820000 m
```

另外多个倾斜变体的预抓取点还触发了配置工作空间边界。该结果说明现场调整后，`track-4` 仍然没有进入软件预抓取包络；不能通过调高 `maximum_horizontal_reach_m`、降低工作空间或忽略该拒绝来解决。

按严格向下的解析上界 `0.585578111 m` 计算，当前目标仍超出约 `0.304574 m`。即使只要求通过当前 `0.820 m` 预抓取包络，也至少需要沿基座 XY 原点方向消除约 `0.070 m` 的径向超界并留出额外余量。下一次调整应直接把料箱和骰子向基座 XY 原点内移，并用新的固定观察画面复核，而不是继续增加路点或扩大 IK 种子。

### 12.2 `track-1` 的近奇异拒绝

另一个确认候选为 `12-Baseball / track-1`：

```text
视觉夹爪尖目标：(-0.098837, 0.553027, 0.000103) m
视觉目标水平半径：0.561790 m
```

其主候选的完整 IK 路点在进入运动前失败：

```text
minimum weighted Jacobian singular value=0.009753 < 0.010000
```

其他倾角变体则分别因最低法兰高度 `0.159481/0.142997 m < 0.164621 m` 被拒绝。该候选不能通过降低 `cartesian_min_jacobian_singular_value` 或忽略法兰下限放行；需要调整固定观察位/抓取姿态，或重新布置物体并保留新的安全余量。

### 12.3 第四次尝试后的运行状态

```text
task.pid：不存在
task.lock：FREE
track1_node / track1_task_runner / run_task_c / codroid_bridge / MS42 进程：均不存在
Task D：未启动
```

第四次授权已经使用。第五次 Task C 实机运动仍需用户新的明确授权；在获得授权前，只允许做只读分析、现场位置调整和重新采集观察画面。

