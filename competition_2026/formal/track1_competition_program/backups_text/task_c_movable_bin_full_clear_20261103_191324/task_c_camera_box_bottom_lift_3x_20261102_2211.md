# Task C 抓取后抬升距离与主运动速度调整

生成时间：2026-11-02 22:13（Asia/Shanghai）  
状态：2026-11-03 已完成长直线重叠分块修复和全量离线回归；随后按用户新的
一次性授权执行了一次 Task C 真机试运行。机械臂通过完整六段路线预检并到达
French's Mustard 抓取位，但夹爪宽度安全门禁在闭合前拒绝，未发生物理抓取。  
最新结论为：**本次授权运行已执行，Task C 仍未完成验收**；机械臂最后停在抓取
位姿、夹爪保持张开，后续回位或再次运行均需新的明确授权。
本次运行未启动 Task D；桥接、视觉和串口进程均已退出。

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

2026-11-03 长路线分块修复后的当前 SHA-256：

```text
53b268883c0d6bcb63a23c0c3bba6fa75569347ead353fdabb3f70a410cdf8b0  config/track1_task_c_v11_official_20261102.yaml
b786486271a444008d117ff0955d6a6252beed19b5209adc79d70426187f13eb  config/track1_task_c_v11_official_mustard_single_20261102.yaml
49735084ad8291524ce9a8f8486beff09ecf046410d3a762279ffc41da11ff1a  config/track1_task_d_v11_official_20261102.yaml
8ee608a4d3a8f4f4bb7ec71894282e01e4a58f20780719d90f3d45e9efd0f23b  config/track1_task_cd_40class_v11_official.yaml
c1819c2d91305b225eee9ce84da62e576c322f3f078e8e6dd72ee1bbfa339795  app/track1_grasp/track1_grasp/motion.py
14058d7549cda78f02419dc278109e313722a05a1ee16d83918b90c325343db9  app/track1_grasp/track1_grasp/config.py
f9df1d9aa6d8d230dd67e515e148871d429063964a3cbb77f5f9ae836ab59c01  app/track1_grasp/test/test_motion_linear.py
aff35279d681b2100f1a482b3513b352931a42582dc41a15406601c971b1146b  app/track1_grasp/test/test_task_cd_short_edge_grasp.py
2f7cc0ee6cdd5ed0fe5bc5a7e8629e0d65444e78a390e134bef49ffa6513b27d  logs/task_c_long_route_subdivision_full_20261103.log
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

## 2026-11-03 长直线重叠分块离线修复

### 修复内容

- `motion.py` 不再把 `cartesian_max_waypoints: 600` 当作整条逻辑路段上限，
  而是作为单个内部规划块上限；长路线按不超过 600 点重叠分块，块边界完全
  复用前一块最终关节解，后续块以该精确姿态继续求解。
- 每个内部块仍逐点执行工作空间、IK、关节限位、`0.6 deg` 单步、`5 deg`
  分支跳变、奇异值和 `0.25 deg` 离散加速度检查；任一块失败则整条逻辑路段
  fail-closed，不发布任何命令。
- 新增整段硬资源上限 `cartesian_max_total_waypoints: 3200`，防止异常路线
  无限增长。
- 四个正式配置统一改为：`cartesian_min_step_m: 0.00025`、
  `cartesian_max_waypoints: 600`、`cartesian_max_total_waypoints: 3200`。
  没有放宽 `0.6 deg` 关节步长、`5 deg` 分支跳变、`0.25 deg` 加速度或
  `0.01` 最小奇异值门禁。
- 修改前核心文件备份：
  `backups/task_c_long_route_subdivision_20261102_222735/`；索引：
  `state/last_task_c_long_route_subdivision_backup.txt`。

### 回归结果

```text
test/test_motion_linear.py：48 passed
motion + 正式配置定向回归：49 passed
全量回归：749 passed in 82.92s
退出码：0
```

全量测试日志：

```text
logs/task_c_long_route_subdivision_full_20261103.log
SHA-256 2f7cc0ee6cdd5ed0fe5bc5a7e8629e0d65444e78a390e134bef49ffa6513b27d
```

### 现场日志几何的纯离线回放

回放只调用运动后端的只读 IK/路线预检，不构建任务节点、不连接机械臂、不发送
运动或夹爪命令。

| 候选/分支 | 分块结果 | 最终安全门禁 |
|---|---|---|
| `12-Baseball` 等价腕部偏航 | 1042 个基础步长，2 个内部块 | 末端姿态最小奇异值 `0.009864 < 0.010000`，拒绝 |
| `14-Sleeve` 等价腕部偏航 | 首次 1034 个基础步长、2 个内部块；内部重采样至 1257/1571 点后仍失败 | 末端姿态最小奇异值 `0.009886 < 0.010000`，拒绝 |
| `2-Starkist_Tuna` | 不进入路线求解 | `pregrasp` 水平半径 `0.606014 m > 0.584605 m`，保持拒绝 |

结论：原先“1040/630 点超过 600”的**分块容量问题已修复**。当前记录的两个
候选被拒绝是末端可操作度不足，不是分块缺失；不得通过降低奇异值阈值、放宽
关节步长或扩大水平工作空间强行放行。下一次实机前必须先通过完整六段路线预检，
若同一桌面姿态仍复现，应调整物体/观察位或取得一个满足全部门禁的安全 IK 分支。

## 2026-11-03 授权 Task C 真机试运行

用户明确确认现场安全后，只运行一次 Task C，未启动 Task D。

### 运行结果

```text
任务日志：logs/task_C_20261103_183802.log
任务日志 SHA-256：1403800647a0c5599b64e0477eeb40dbb60eaa37e5c56db257eaef76112b566e
桥接日志：logs/codroid_bridge_20261103_183802.log
桥接日志 SHA-256：cc8b34a67244997013fdaf16f56ca9dd081f632d96cd707aeb87dba57e4fdc41
```

- 启动时 `state=AutoReady`、`fault_flag=0`。
- 固定观察位识别出 7 个原始候选，确认 4 个未完成候选。
- 首选 `4-Frenchs_Mustard` 通过完整六段只读路线预检：
  `1126` 个 Cartesian 路点，最小奇异值 `0.02047 > 0.01000`。
- 夹爪先张开并稳定到 `19.0 deg`；机械臂完成预抓取和抓取段运动。
- 最后下发抓取段终点关节角约为
  `[91.38, 37.33, 75.50, 22.82, 90.00, 80.27] deg`。
- 到达抓取位后，宽度门禁在闭合夹爪前拒绝：

```text
短边宽度 0.070960 m + 安全余量 0.005000 m = 0.075960 m
CAD 50 mm 插入深度处开口 = 0.072549 m
0.075960 > 0.072549
```

- 因此没有发送夹爪闭合帧，没有发生物理抓取；
  `physical_grasp_attempts=0/40`，最终状态 `aborted`，进程退出码 `1`。
- 程序已清理本次启动的桥接和其他子进程；无残留任务 PID，`task.lock` 可重新获取。

### 当前机械臂状态

按最后下发的抓取轨迹终点，机械臂停在抓取位，夹爪保持张开。由于本报告撰写时
只读状态桥接已随任务退出，不能把最后轨迹终点表述为新的实时状态读数。为避免
额外运动，未自动发送回位、抬升或再次抓取命令。

### 发现的问题

宽度可行性检查当前只在到达抓取位、调用 `close_for_width()` 时执行。虽然该检查
正确地 fail-closed，但执行顺序不够理想：应在候选运动前就使用同帧
`minor_axis_length_m` 和同一 CAD 开口证据完成宽度可行性检查，使不兼容候选在
零运动条件下被排除，并继续预检下一个候选。

该修复不改动夹爪开口、CAD 证据、5 mm 安全余量或现有 fail-closed 要求；它只会
把拒绝提前到运动前。完成离线实现和回归后，仍需先处理当前停在抓取位的机械臂，
再取得用户新的单次实机授权。

## 下一次 Task C 实机验收

下一次动作前必须收到用户明确确认：

```text
现场安全已确认，允许运行一次 Task C
```

下一次授权前必须先让完整路线预检通过，不能直接重跑。2026-11-03 已完成：

1. 复核 `0.1706 m` 抬升对总路线长度的影响；
2. 实现长路线重叠分块，保留 `600` 点单块预算并新增 `3200` 点整段硬上限；
3. 用 `0.25 mm` 重采样解决原先 `0.623/0.627 deg` 的基础分辨率问题；
4. 保持 `0.6 deg` 单步、`0.25 deg` 加速度、`0.01` 奇异值和 Starkist 水平
   包络不变。

仍未完成的是：当前 Baseball/Sleeve 记录姿态在路线末端进入
`0.0099 < 0.0100` 的奇异区，因此同一现场摆放不能直接宣称已具备实机放行条件。

获授权后只运行 Task C，不同时启动 Task D。现场重点检查：

1. 抓取后实际沿法兰 Z 抬升约 `0.171 m`，不是旧 `0.080 m`；
2. 主运动速度约为原来的 3 倍，视觉伺服靠近速度仍保持原值；
3. 到达放置位后才释放夹爪；
4. 闭爪行程证据、抓取保持、六段路线预检和安全包络均正常；
5. 机械臂状态保持 `AutoReady`、`fault_flag=0`，全程有人守急停。

当前 Task C 状态明确为：**2026-11-03 的一次授权真机运行已执行，六段路线预检
通过且机械臂到达抓取位，但夹爪宽度安全门禁在闭合前拒绝，物理抓取仍为 0/1；
机械臂最后停在抓取位、夹爪张开，回位和再次运行均需新的明确授权。**
