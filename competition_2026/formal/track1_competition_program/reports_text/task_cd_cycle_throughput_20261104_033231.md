# Task C / Task D 连续抓取节拍优化报告

- 生成时间：2026-11-04（Asia/Shanghai）
- 正式程序目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序`
- 本轮状态：已完成代码实现、静态检查、单元/回归测试和日志分析
- 安全状态：**未启动实体机械臂，未发送 `/RobotMove`，未发送夹爪命令**

## 1. C/D 共用代码，保留两个独立双击入口

C 和 D 共用同一个核心入口：

```text
bin/track1_task_runner
```

两个双击程序只负责选择任务类型和对应配置：

| 双击入口 | 实际命令 | 配置 |
|---|---|---|
| `任务C_无序抓取.desktop` | `bin/run_task_c` -> `bin/track1_task_runner C` | `config/track1_task_c_v11_official_20261102.yaml` |
| `任务D_未知物体抓取.desktop` | `bin/run_task_d` -> `bin/track1_task_runner D` | `config/track1_task_d_v11_official_20261102.yaml` |

正式目录和桌面上的 `.desktop` 文件 SHA256 一致：

```text
a7116e5e06388952e20337a38c4dc163dfa41ab6927123100076683fe65eb45c  任务C_无序抓取.desktop
b95c0869bb6dedda61c7e6b02f83a088a56bb26aae1b2fb6b449ece2a765af13  任务D_未知物体抓取.desktop
```

因此满足“共用一套核心代码、保留两个独立启动程序”的要求。

## 2. 真正的耗时问题

问题不是夹爪没有夹住，也不是机械臂控制器卡死，而是抓取候选规划中的无效穷举。

### 2.1 抓取后下一次候选规划等待约 37.6 秒

日志：`logs/task_C_20261104_024418.log`

```text
1793731778.013  [task_cd_return_observation_after_track_1] verified
1793731815.643  下一次候选规划完成/继续
```

间隔约 **37.630 秒**。这一阶段包含换位视角航线和多个候选的完整 Cartesian 路线预检，占用了两次抓取之间的大量时间。

### 2.2 近奇异候选曾穷举约 268.9 秒

同一日志中的 `12-Baseball`：

```text
1793732306.009  candidate 2/4: '12-Baseball'
1793732574.953  candidate finally rejected
```

间隔约 **268.944 秒**。规划器连续尝试多个抓取点和腕姿变体，最终均被近奇异安全门拒绝，例如：

```text
minimum weighted Jacobian singular value = 0.0097 < 0.0100
```

该过程没有发送运动命令。真正的问题是：一个必然失败的候选，在完整六/七段路线预检中被反复求解，直到所有变体耗尽才返回。

## 3. 本轮优化

### 3.1 预抓取端点快速筛查

文件：

```text
app/track1_grasp/track1_grasp/pipeline.py
```

新增 `_precheck_task_cd_pregrasp_endpoint()`：

- 在完整路线预检之前，先检查预抓取端点 `waypoints[0][1]`；
- 端点不可达、非有限或低于奇异值安全门时立即拒绝当前抓取点/腕姿变体；
- Mock 或无运动学后端返回 `None`，继续原有完整预检路径；
- 最终完整路线预检、碰撞/盒体净空和全部运动安全门仍然保留。

这不会把安全门降低，只是把“注定失败的完整长路线求解”提前短路。

### 3.2 首个安全 IK 解立即返回

- `kinematics.py` 增加 `stop_after_first_safe`；
- `motion.py` 的连续 Cartesian 路径使用 `stop_after_first_safe=True`；
- 找到通过关节限位、关节步长、加速度、奇异值等全部门限的第一个 IK 解后立即返回，不再无意义地遍历其余解。

### 3.3 固定视角确认后直接规划

C 配置：

```yaml
complete_box_before_planning: false
```

固定观察视角一旦连续确认可抓候选，就直接进入抓取规划，不再先预检全部换位视角航线。只有固定视角没有候选时，才懒加载换位观察作为兜底。

D 配置保持：

```yaml
complete_box_before_planning: true
```

因为 D 的未知物体流程要求先完整覆盖收纳盒再统一决策。

### 3.4 C/D 统一速度档

两份正式配置统一为：

```yaml
speed_mps: 0.18
joint_speed_rad_s: 2.4
joint_speed_safety_limit_rad_s: 2.4
```

相对此前实跑的 `0.0675 m/s`，线速度约提高到 **2.67 倍**。`2.4 rad/s` 仍低于 URDF 最低关节上限 `2.617994 rad/s`，保留约 8.3% 的模型余量。视觉伺服速度独立配置，没有被这次吞吐优化放大。

### 3.5 保持不变的安全门

本轮没有放宽以下限制：

- 关节位置限位；
- 关节步长；
- 加速度限制；
- Cartesian 最小 Jacobian 奇异值 `0.01`；
- 碰撞与盒体净空；
- 手眼标定与工作区检查；
- 变尺寸宽度感知夹持；
- 禁止恢复固定空夹终角；
- 完整路线最终预检。

## 4. GitHub 方案取舍

已核对 GitHub 上常见的开源运动规划路线，例如 MoveIt 2、OMPL、TRAC-IK、cuRobo。它们不能直接作为一个补丁替换到本项目，原因是：

- 本项目不是 MoveIt 在线规划节点，而是向埃斯顿 `/RobotMove` 发送预计算关节轨迹；
- 直接替换规划栈会改变控制接口、依赖、轨迹验证流程和现场标定假设；
- 开源规划器同样需要正确的机器人模型、碰撞场景、TCP 和现场限位；这些不能替代本项目已有的安全门；
- 引入新规划器不能修复本日志中“候选在安全门附近被反复完整预检”的直接原因。

因此本轮采用不改变控制接口、可回归、可逐项验证的本地优化，而不是替换整个运动规划栈。

## 5. 验证结果

定向回归：

```text
5 passed, 40 deselected in 2.19s
```

覆盖：

- PCA 抓取预抓取端点先于完整路线被拒绝；
- 同帧 geometry 抓取预抓取端点先于完整路线被拒绝；
- 被拒绝时完整路线预检次数为 0；
- 无 Cartesian 运动、无夹爪命令；
- 闭合圆柱侧壁策略仍在任何命令前拒绝。

本轮修改完成后的联动与全量回归结果：

```text
联动测试：265 passed in 55.37s
全量测试：840 passed in 108.23s
```

启动脚本检查：

```text
bash -n bin/run_task_c bin/run_task_d bin/track1_task_runner
```

结果正常。正式桌面入口与数据盘副本 SHA256 一致。

## 6. 现场验证时需要观察什么

下一次实机测试必须由用户明确授权后才能启动。建议在日志中对比：

- 从回到观察位到下一次抓取路线开始的耗时；
- `[motion:plan-timing]` 的规划耗时；
- `[motion:linear-precheck]` 的路线预检耗时；
- `pregrasp endpoint passed/rejected` 是否提前淘汰近奇异候选；
- 两次抓取之间的实际时间，而不是只看单段运动速度。

当前可以确认“规划和候选筛选流程已提速”，但**真实连续抓取节拍仍必须由下一次现场日志验证**。本轮没有进行实体机械臂运动。

## 7. 修改前备份

```text
backups/task_cd_cycle_screen_20261104_032725
```

包含：

```text
app/track1_grasp/track1_grasp/pipeline.py
app/track1_grasp/test/test_task_cd_short_edge_grasp.py
SHA256SUMS.before
```

备份索引：

```text
state/last_task_cd_cycle_screen_backup.txt
```
