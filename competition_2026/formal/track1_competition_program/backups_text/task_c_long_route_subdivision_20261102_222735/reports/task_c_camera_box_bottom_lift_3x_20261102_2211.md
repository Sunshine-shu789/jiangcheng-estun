# Task C 抓取后抬升距离与主运动速度调整

生成时间：2026-11-02 22:13（Asia/Shanghai）  
状态：代码、正式配置和离线回归完成；已完成一次授权实机试运行，但全部候选在
运动前被规划门禁拒绝，尚未完成抓取验收。  
本轮未启动 Task C/D、未启动 ROS 任务节点、未连接串口、未向机械臂或夹爪发送命令。

## 结论

Task C 的抓取后抬升目标已从旧的 `0.080 m` 调整为实现 D405 光学中心到收纳盒
底部约 `0.35 m` 所需的 `0.1706 m`。同时，主笛卡尔运动速度已从 `0.01 m/s`
提高到 `0.03 m/s`，即原来的 3 倍；视觉伺服速度 `servo.speed_mps` 保持
`0.01 m/s` 不变。

四个正式配置已同步修改并通过配置、规划器和安全包络回归。最新参数尚未执行
实机验收，不能把本轮离线结果表述为现场运动或抓取已通过。

## 抬升距离核算

现场已确认的最低抓取位记录为：

```text
D405 光学中心到收纳盒底部           0.179600 m
目标直线距离                        约 0.350 m
抓取后沿法兰 Z 的配置抬升量          0.170600 m
```

抓取位与抬升位保持同一 XY 和固定姿态，`lift_height` 表示两点间的法兰 Z 增量，
不是抬升后的绝对高度，也不是垂直抬升 `0.35 m`。按现场几何关系核算，抬升后
相机光轴到盒底的直线距离约为 `0.35 m`；一阶竖直近似为：

```text
0.179600 + 0.170600 = 0.350200 m ≈ 0.350 m
```

因此正式配置写为 `lift_height: 0.1706`。若误写成 `0.35`，会在当前基础上额外
多抬约 `0.1794 m`，并可能触发路线高度安全门禁。

## 正式配置修改

四个正式配置均修改为相同参数：

| 配置 | `lift_height`（旧 → 新） | `motion.speed_mps`（旧 → 新） | `servo.speed_mps` |
|---|---:|---:|---:|
| `track1_task_c_v11_official_20261102.yaml` | `0.08 → 0.1706` | `0.01 → 0.03` | `0.01` |
| `track1_task_c_v11_official_mustard_single_20261102.yaml` | `0.08 → 0.1706` | `0.01 → 0.03` | `0.01` |
| `track1_task_d_v11_official_20261102.yaml` | `0.08 → 0.1706` | `0.01 → 0.03` | `0.01` |
| `track1_task_cd_40class_v11_official.yaml` | `0.08 → 0.1706` | `0.01 → 0.03` | `0.01` |

`motion.speed_mps` 是抓取后主运动、搬运和返回等笛卡尔路径的速度；
`servo.speed_mps` 是视觉伺服靠近速度。本轮只按要求提高主运动速度，未改变视觉
伺服靠近速度。

## 路线高度边界

回归中发现，固定放置位夹爪尖 Z 加 `0.08 m` 接近量后的计算值为：

```text
0.246477463808 + 0.080000000000 = 0.326477463808 m
```

旧安全上限为 `0.32647746380750964 m`，与同一来源的计算值仅相差
`4.9e-13 m`，导致边界断言因二进制浮点表示失败。处理方式是把四个配置的
`maximum_route_tip_z_m` 统一为：

```yaml
maximum_route_tip_z_m: 0.326477463808
```

这是把上限对齐到同一标定/接近量来源值，不是放宽安全门禁；工作空间、最低高度、
工具倾角、路线包络和关节预检均保持有效。

## 离线验证

测试使用正式 ext4 运行环境，只运行 pytest，不启动任务节点或硬件入口：

```bash
/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/bin/ensure_runtime
source /media/sunshine/ESTUN_RUNTIME/ros2_ws/install/setup.bash
cd /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_grasp
PYTHONPATH=".:${PYTHONPATH:-}" python3 -m pytest -q -p no:cacheprovider
```

结果：

```text
专项安全回归：29 passed in 4.24s
全量回归：    745 passed in 73.15s
退出码：      0
```

关键回归
`test_formal_cd_configs_use_camera_box_bottom_lift_and_3x_motion_speed`
逐一覆盖四个正式配置，并校验：

- `lift_height == 0.1706`；
- `motion.speed_mps == 0.03`，同时 `servo.speed_mps == 0.01`；
- 规划出的抬升位与抓取位 XY 相同，Z 增量恰为 `0.1706 m`；
- 规划路线最高点不超过配置的路线安全包络。

全量测试日志：

```text
logs/task_cd_camera_box_bottom_lift_full_20261102_2211.log
SHA-256 c4cf5c2ffc6a26cf09f6ccd7d83f45a7e9857656a9868c8e82c2ec7602f525a5
```

## 文档与备份

已同步更新：

- `使用说明.txt`：标明 `0.1706 m` 是抬升增量、主运动为 3 倍速、伺服速度不变；
- `app/track1_grasp/README.md`：增加 Task C 抬升距离和主运动速度参数说明。

修改前文档备份：

```text
backups/task_c_lift_3x_docs_20261102_2211/
```

`lift_height`、主运动速度及安全测试的修改前备份：

```text
backups/motion_3x_20261102_215526/
```

该备份中的四个配置仍为 `lift_height: 0.08`、`motion.speed_mps: 0.01` 和旧路线
边界值。后续浮点边界修正前的配置备份为：

```text
backups/camera_box_bottom_lift_bound_20261102_221011/
```

备份索引：

```text
state/last_motion_3x_backup
state/last_camera_box_bottom_lift_bound_backup
```

## 关键文件校验

```text
7176479cd0bf640cec9f89a6db12fb6419732b5e301a5e9f5d4da924e8fc73ac  config/track1_task_c_v11_official_20261102.yaml
b4beab171743481231fb12499f563b304d3a853b6e37e6b4f89570caab6a2ea4  config/track1_task_c_v11_official_mustard_single_20261102.yaml
712ad7c150e80170f6b73d13fa2d04fa8baa70cda8fe149295ce5ae75d4be62c  config/track1_task_d_v11_official_20261102.yaml
d05f4a7249318d9346ce0eae1bbf60ea760495f310f59bfc110758a99aad56c9  config/track1_task_cd_40class_v11_official.yaml
9b9167917a40fe5a62ca44905cc2ba3abd1aed1004e9643467feb4324b846c2e  app/track1_grasp/test/test_task_cd_grasp_safety.py
```

## 单次实机试运行结果（2026-11-02 22:19-22:21）

用户在明确确认现场安全后，授权运行一次 Task C。本轮只启动 Task C，未启动
Task D；物理抓取次数上限固定为 `1`。

启动命令：

```bash
./bin/ensure_runtime && \
TRACK1_TASK_C_PICKS=1 TRACK1_CONFIRM=1 TRACK1_NO_PAUSE=1 TRACK1_NO_UI=1 \
./bin/run_task_c
```

### 启动与识别

- 桥接状态为 `AutoReady`、`fault_flag=0`，启动时法兰 XYZ 为
  `(-0.1020, 0.4162, 0.3017) m`。
- 机械臂完成固定观察位进入动作，到位残差 `0.002 deg`。
- 固定视野得到 3 个通过策略门禁的未完成候选：`12-Baseball`、
  `2-Starkist_Tuna`、`14-Sleeve`。
- 抓取安全桩检查已通过，包括最低/最高路线 Z、工具倾角和水平包络；但完整
  六段笛卡尔路线和 IK 预检仍分别触发下列拒绝。

### 候选拒绝原因

| 候选 | 主短边方案 | 等价腕部偏航方案 | 最终结果 |
|---|---|---|---|
| `12-Baseball` | 最小间距处关节步长 `0.623 deg > 0.600 deg` | 需要 `1040` 个基础路点，超过 `600` | 运动前拒绝 |
| `2-Starkist_Tuna` | `pregrasp` 水平半径 `0.606014 m > 0.584605 m` | 同样超过水平包络 | 运动前拒绝 |
| `14-Sleeve` | 最小间距处关节步长 `0.627 deg > 0.600 deg` | 需要 `630` 个基础路点，超过 `600` | 运动前拒绝 |

新增 `0.1706 m` 抬升增加了路线长度，使部分候选更接近或超过现有
`cartesian_max_waypoints: 600` 上限；但三个候选的拒绝原因并不完全相同，
其中 `2-Starkist_Tuna` 是既有水平工作空间门禁拒绝。不得为追求本次抓取而直接
放宽关节步长、水平包络或路点上限。

### 运行结果

```text
attempts=1 successes=0 physical_grasp_attempts=0/1
completed_tracks=0 failed_tracks_excluded=0 final_status=rejected
exit code=1
```

详细结论：

- 机械臂仅执行了固定观察位进入动作，随后只做离线规划；
- 没有发送候选的抓取/预抓取/抬升/搬运/释放笛卡尔路径；
- 没有向夹爪发送开爪或闭爪命令；
- 因而本轮没有执行到新的 `0.1706 m` 抬升，也没有验证 3 倍主运动速度；
- 任务按 fail-closed 策略退出，抓取计数保持 `0/1`，没有自动重试。

任务日志：

```text
logs/task_C_20261102_221940.log
SHA-256 9f36f3308855ff996e6d7581e205c4e0b8d5a7244d989028562dd4f0b0cdef0d
logs/codroid_bridge_20261102_221940.log
SHA-256 3967ddb6b1e0ad5d602995d5a75d52f1ba934585c9ede926429e20a0adda408e
```

运行结束后已检查，没有残留 `track1_node`、`codroid_udp_bridge` 或
`motor_node` 进程。桥接日志末尾包含 rclpy 关闭阶段的异常记录，但桥接进程已
退出，任务没有继续运动。

## 下一次 Task C 实机验收

下一次动作前必须收到用户明确确认：

```text
现场安全已确认，允许运行一次 Task C
```

下一次授权前应先处理本次规划拒绝，不能直接重跑。建议先在不连接实机运动的
离线环境完成：

1. 用相同的三个候选复核 `0.1706 m` 抬升对总路线长度的影响；
2. 评估是否应按路段拆分路线或调整路点上限设计，而不是简单放宽 `600`；
3. 复核 `0.623/0.627 deg` 关节步长是否可通过重新采样解决，不直接放宽
   `0.600 deg`；
4. 保持 `2-Starkist_Tuna` 的水平包络拒绝，不通过扩大工作空间规避。

获授权后只运行 Task C，不同时启动 Task D。现场重点检查：

1. 抓取后实际沿法兰 Z 抬升约 `0.171 m`，不是旧 `0.080 m`；
2. 主运动速度约为原来的 3 倍，视觉伺服靠近速度仍保持原值；
3. 到达放置位后才释放夹爪；
4. 闭爪行程证据、抓取保持、六段路线预检和安全包络均正常；
5. 机械臂状态保持 `AutoReady`、`fault_flag=0`，全程有人守急停。

当前 Task C 状态明确为：**代码及离线回归完成；一次实机试运行因全部候选规划
拒绝而未产生抓取，等待路线预检修复和下一次明确授权。**
