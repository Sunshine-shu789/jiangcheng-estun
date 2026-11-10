# 任务C 目标类过滤修复与 18:58 运行复盘

- 生成时间：2026-11-02T19:08:48+08:00
- 正式目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序`
- 结论：**Task C 代码修复和离线回归完成；Task C 尚未完成实体抓取验收。Task D 未运行。**

## 18:58 实机运行

- 配置：`config/track1_task_c_v11_official_mustard_single_20261102.yaml`
- 配置 SHA256：`3c488e83f8c00907c080cacb47881dd1644b7b58f5179bc69df5139cb9329178`
- 运行日志：`logs/task_C_20261102_185842.log`
- bridge 日志：`logs/codroid_bridge_20261102_185842.log`
- 该次配置要求 `vision.target_class: 4-Frenchs_Mustard`，但实时 YOLO 仍让 `10-meat_can` 进入候选；程序规划了肉罐头并执行到接近路径 `chunk 1/4`、`chunk 2/4` 后人工中止。
- 夹爪只执行了打开，没有闭合；没有抓取任何物体。
- 中止后无残留 Task C/bridge 进程，`state/task.pid` 不存在。`state/task.lock` 仍是 0 字节残留锁文件，不代表任务仍在运行。
- 本次“允许运行一次任务C”的授权已被这次运行消费，不能复用。

### 只读机械臂复核

`logs/hardware_readonly_robotinfo_20261102_190227.log`：

```text
state=AutoReady
fault_flag=0
finite=True
ready=True
joint_positions_deg=[76.88435, 17.91801, 71.77300, 6.45241, 91.47488, 43.50584]
```

该姿态与中止时第 2/4 段端点一致，说明中止后没有继续执行运动。当前机械臂不在固定观察位。

## 根因

1. `RealSenseYoloVision.capture()` 只按置信度接收原始检测，没有应用 `VisionConfig.accepted_target_labels()`，所以固定目标为芥末酱时，肉罐头仍能成为更高分候选。
2. `PickPipeline._observe_task_cd_candidates()` 没有第二道目标类门禁；即使实时适配器漏过滤，错误类别仍可形成固定观察候选。
3. 本次日志中的芥末酱 PCA `axis ratio 1.014 < 1.020` 发生在移向固定观察位期间（约 18:58:51），不是固定观察位确认后的持续结果。18:20 的固定观察只读诊断中芥末酱 13 帧均通过 PCA（中位轴比 1.45799）。因此本轮没有降低 `min_axis_ratio`，也没有改成沿长轴闭合或绕过短轴 fail-closed。

## 已实施修复

- `app/track1_grasp/track1_grasp/vision.py`
  - 实时 RealSense YOLO 检测在进入 `_make_detection()` 前按 `accepted_target_labels()` 过滤；固定 `target_class` 非空时只允许该精确标签。
  - `target_class` 为空时仍允许配置类别，Task D 的 `unknown` 提议不会被这道过滤误删。
- `app/track1_grasp/track1_grasp/pipeline.py`
  - C/D 固定观察增加第二道门禁：仅当 `vision.target_class` 非空时，标签不等于该值的检测在候选形成前排除。
  - 新增 `off_target_excluded` 计数，便于下次实机日志直接核验。
  - 偏目标不会进入 `TargetPolicy`、路线预检、笛卡尔运动或夹爪命令路径。
- 芥末酱短边路径保持原配置：中心抓取优先，闭合轴对齐 PCA 短轴，closure_axis_tool_xy: [1.0, 0.0]；本轮未修改 PCA 阈值或短边策略。

## 回归验证

- 专项测试：`56 passed`（`test_unknown_proposer.py`、`test_task_cd_fixed_observation_clear.py`）
- 全量测试（先执行 `bin/ensure_runtime`，加载正式 ROS ext4 运行环境，mock/dry-run，无机械臂或夹爪动作）：
  `736 passed in 63.19s`
- 全量日志：`logs/task_c_target_class_filter_full_regression_20261102_191000.log`
- 全量日志 SHA256：`ccd2bb70d8269daac36478d42e7a8fd4c5b46504531dde343bd9515da7b48b5e`
- `vision.py` SHA256：`9a11cbd613b6c7513ce60556501d7bce912ba28bc9ed45908e5624219be005d0`
- `pipeline.py` SHA256：`22acd25c8fcffc44e607a69de5ddf102f81f5b6462d4257a28aae7bf6ed9264a`

## Task C / Task D 状态

- C 和 D 仍是两个独立入口：`bin/run_task_c` 与 `bin/run_task_d`，分别加载 C/D 配置，比赛时一次只运行一个。
- Task C：目标类过滤代码已修复，离线链路通过；实体 Task C 尚未成功抓取并完成放置，不能标记为实机完成。
- Task D：未启动、未运行；在 Task C 完成实体验收前不运行 D。

## 下一次实机前

1. 必须重新获得一次新的明确现场安全授权；18:58 的授权已消费。
2. 先执行 `bin/ensure_runtime`，只读确认 `state=AutoReady`、`fault_flag=0`。
3. 只运行 Task C，核验日志中出现芥末酱候选且不出现 `10-meat_can` 候选；完成抓取、放置、返回观察位后再判定 Task C 成功。
4. 在 Task C 完整成功前不运行 Task D。

## 备份

`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/backups/task_c_target_class_filter_20261102_190503`
