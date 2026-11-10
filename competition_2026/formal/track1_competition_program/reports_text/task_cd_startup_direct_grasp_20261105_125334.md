# Task C/D 启动首帧直接抓取修改报告

- 完成时间：`2026-11-05T12:53:34+08:00`
- 工作目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序`
- 状态：代码、配置与离线回归测试完成；未启动 Task C/D，未发送实体机械臂或夹爪命令
- 备份目录：`backups/task_cd_startup_direct_grasp_20261105_124613/`

## 1. 结论

启动 Task C/D 后不再先进入启动阶段的换位观察流程。程序到达固定观察位后只采集 **1 帧**：

1. 首帧存在可执行候选：立即进入原有完整只读安全预检；预检通过后直接执行抓取。
2. 首帧有物体证据但没有可执行候选：以 `REJECTED` 停止，不换位扫描，并明确“箱子非空、空箱未证明”。
3. 首帧没有候选且没有物体证据：以 `ABORTED` 停止，不换位扫描，并明确“单帧不能证明空箱”。
4. 首帧候选在只读规划/预检中全部被拒绝：以 `REJECTED` 停止；不发抓取路线运动，不发送夹爪命令，不报空箱。

只有启动首轮使用上述直达逻辑。启动首轮成功抓取并返回固定观察位后，后续清箱轮次仍保留原有完整多视角空箱确认。

## 2. 修改内容

### 2.1 配置模型

文件：`app/track1_grasp/track1_grasp/config.py`

- 新增 `TaskCDClearConfig.startup_direct_grasp: bool = False`。
- 从 `task_cd.startup_direct_grasp` 读取，并使用严格布尔解析；字符串或数字不能隐式开启。
- 默认关闭，避免改变未显式启用的其他配置。

### 2.2 启动直达抓取流程

文件：`app/track1_grasp/track1_grasp/pipeline.py`

- 仅在 Task C/D 清箱流程的第一轮传入 `startup_direct_grasp=True`。
- 启动首轮直接调用固定视角采集，强制：
  - `observe_frames_override=1`
  - `confirm_hits_override=1`
- 启动首轮不调用 fast-forward 候选获取，也不调用换位扫描候选获取。
- 首帧候选按原有排序规则处理后，逐个执行原有完整安全预检。
- 首帧失败路径均带明确状态，不把单帧结果解释为空箱。
- 启动轮运行日志标记为 `starting C/D startup direct grasp`。

### 2.3 正式配置

已开启：

```yaml
task_cd:
  startup_direct_grasp: true
```

涉及：

- `config/track1_task_c_v11_official_20261102.yaml`
- `config/track1_task_d_v11_official_20261102.yaml`

同时修正注释，明确“启动首帧失败直接停止”和“成功抓取后的后续轮次仍执行完整多视角空箱确认”。

### 2.4 回归测试

文件：`app/track1_grasp/test/test_task_cd_fixed_observation_clear.py`

新增覆盖：

- 首帧候选直接抓取，只读 1 帧，无扫描运动。
- 首帧为空时不会读取第二帧，不扫描，不声明空箱。
- 首帧有物理物体证据但无候选时，结果不是空箱。
- 候选预检拒绝时，无路线运动、无夹爪动作、无扫描。
- 启动首帧成功抓取后，下一轮仍执行完整多视角空箱扫描。
- 正式 C/D 配置断言 `startup_direct_grasp=True`。
- 严格布尔配置测试包含新字段。

## 3. 安全边界

本次没有删除或放宽任何原有安全门限。启动直达抓取仍必须经过：

- 固定观察位关节限位与到达后反馈检查；
- 深度/候选稳定性与目标过滤；
- 目标策略；
- IK/FK；
- 关节限位、连续性、奇异值检查；
- 全路线碰撞与安全包络；
- 工作空间、夹爪尖端/法兰边界；
- 路线预检；
- AutoReady、故障字和最新机器人状态检查；
- 夹爪后端及抓取前检查。

启动首轮唯一取消的是“首帧失败后继续换位观察”的启动阶段行为；没有取消抓取前安全门禁。

当前正式 C/D YAML 仍为真机配置：`motion.dry_run=False`、`gripper.backend=ms42dc`。因此任何 `run_task_c` / `run_task_d` 启动都属于真机任务，必须在现场安全确认和启动窗口明确确认后才能执行。本次没有启动这些程序。

## 4. 验证结果

### 4.1 运行环境

```text
/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/bin/ensure_runtime
RUNTIME_READY
```

### 4.2 专项测试

```text
app/track1_grasp/test/test_task_cd_fixed_observation_clear.py
93 passed in 11.64s
```

相关四组 C/D 专项：

```text
test_task_cd_fixed_observation_clear.py
test_task_cd_short_edge_grasp.py
test_task_cd_grasp_safety.py
test_config.py
301 passed in 57.19s
```

### 4.3 全量抓取测试基线

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/app/track1_grasp:$PWD:${PYTHONPATH:-}" \
python3 -m pytest -q -p no:cacheprovider app/track1_grasp/test
```

结果：

```text
920 passed in 110.47s (0:01:50)
```

### 4.4 正式配置只读解析

```text
Task C:
  mode=C task_cd.enabled=True
  startup_direct_grasp=True
  dry_run=False gripper.backend=ms42dc
  observe_frames=1 confirm_hits=1

Task D:
  mode=D task_cd.enabled=True
  startup_direct_grasp=True
  dry_run=False gripper.backend=ms42dc
  observe_frames=1 confirm_hits=1
```

### 4.5 进程与备份检查

- 未发现 `run_task_c`、`run_task_d`、`track1_task_runner` 或 `ros2 launch` 任务进程。
- 修改前备份 `SHA256SUMS` 已逐项校验通过。

## 5. 最终文件 SHA256

```text
53a448481e5f4e47bf406ac4e3afdbf0d9691e88533137efe83a60e92b4a837f  app/track1_grasp/track1_grasp/config.py
ac8f0b8ee91bf0735800a0657f34ad44dcc0b6b56d82112470daafc043705e18  app/track1_grasp/track1_grasp/pipeline.py
194816f529f8113fc58e785bba3a553730a22fda3673e11125f2350eb4a2c218  config/track1_task_c_v11_official_20261102.yaml
cb20524c48ccd8b877f8411baa1cdce2d4ce087ff2dcc0a8bc3f5653ebaf612f  config/track1_task_d_v11_official_20261102.yaml
907ab5697bf937370425e32b11daa0966766ae714727863eecc0390faf16e41b  app/track1_grasp/test/test_task_cd_fixed_observation_clear.py
```

## 6. 回滚

在工作目录执行：

```bash
BACKUP="backups/task_cd_startup_direct_grasp_20261105_124613"
cp "$BACKUP/config.py" app/track1_grasp/track1_grasp/config.py
cp "$BACKUP/pipeline.py" app/track1_grasp/track1_grasp/pipeline.py
cp "$BACKUP/config/track1_task_c_v11_official_20261102.yaml" config/track1_task_c_v11_official_20261102.yaml
cp "$BACKUP/config/track1_task_d_v11_official_20261102.yaml" config/track1_task_d_v11_official_20261102.yaml
cp "$BACKUP/test_task_cd_fixed_observation_clear.py" app/track1_grasp/test/test_task_cd_fixed_observation_clear.py
sha256sum -c "$BACKUP/SHA256SUMS"
```

回滚只恢复代码、配置和测试文件，不涉及现场坐标或机器人状态。

## 7. 真机启动前置条件

下次启动 Task C/D 前仍须由用户再次明确确认现场安全。当前代码状态是“已实现并离线验证”，不是“已授权真机运动”。
