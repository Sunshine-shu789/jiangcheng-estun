# Task C 闭爪短行程保护修复与离线验收

生成时间：2026-11-02 20:18（Asia/Shanghai）  
本轮未启动 Task C/D、未启动 ROS 任务、未连接串口、未向机械臂或夹爪发送任何命令。

## 结论

已定位并修复“控制器报告闭爪到位，但实际只走了 205.7 deg，程序仍继续抬臂、
搬运和释放”的问题。修复后，Task C/D 的实机闭爪必须同时满足：

1. 开爪后采到稳定的起始角度；
2. 闭爪后采到连续稳定的控制器到位信号；
3. 实测闭爪行程达到按物体 PCA 短边宽度和夹爪 CAD 型线计算的最小行程。

不满足第 3 项时立即进入 `FAULT` 并中止当前抓取，禁止执行后续 `lift`、
`transfer` 和 `release`。裸夹硬限位门限 `1978.0 deg` 保持不变，没有放宽。

## 现场故障证据

最近一次失败日志：`logs/task_C_20261102_194015.log`

```text
19:47:47.496  发出 close，速度字段为 2.0 rad/s
19:47:47.787  控制器报告稳定 arrived，角度 226.7 deg
起始角度      21.0 deg
实际闭爪行程  226.7 - 21.0 = 205.7 deg
耗时          约 0.291 s
```

随后旧逻辑把 `226.7 deg` 当成夹持成功并继续 `lift`。原因是旧的保持检查只在
角度达到裸夹硬限位时判失败；`205.7 deg` 远低于 `1978.0 deg`，因此落入了
“未到空夹门限即视为夹住”的误判路径。单次 `arrived` 状态不能证明爪指已经
接触并夹住物体。

另一个基线是：更早一次全行程闭爪到达 `1978.9 deg`，已由保持检查正确拒绝。
因此现场同时存在两种不同失效形态：

- 全行程到裸夹硬限位：已有门限可以拒绝；
- 极短行程被控制器误报到位：旧门限无法识别，本轮新增行程证据来拒绝。

## 代码修复

### 1. 独立闭爪速度

正式 Task C/D 配置使用：

```yaml
speed_rad_s: 5.0
close_speed_rad_s: 4.0
motion_timeout_s: 25.0
```

闭爪不再沿用导致短行程的 `2.0 rad/s` 历史参数；`4.0 rad/s` 已有空载台架
全行程数据。超时时间覆盖完整行程、启动和堵转判定。

### 2. CAD 宽度门限

`MS42DCGripperBackend.close_for_width()` 和 ROS 变体在闭爪前后分别读取稳定
角度，并要求：

```text
actual_close_travel_deg >= minimum_close_travel_deg
```

以现场 `4-Frenchs_Mustard` 为例：

```text
PCA 短边宽度                    51.64 mm
宽度安全余量                     5.00 mm
用于门限计算的有效宽度          56.64 mm
夹爪 CAD 开口 @ 50 mm 接触深度  72.549 mm
配置闭爪角度                  1957.9 deg
角度测量容差                    80.0 deg
计算最小闭爪行程               349.3406 deg（报告取 349.3 deg）
```

若实测仅 `205.7 deg`，系统会抛出：

```text
MS42DC effective close travel 205.7 deg is below the CAD-derived minimum
349.3 deg ...; refusing to treat the jaws as closed
```

并将夹爪状态置为 `FAULT`。缺少 PCA 短边宽度、缺少 `close_for_width()` 或
CAD 文件不可用时也都是硬错误，不允许退化为未经验证的普通 `close()`。

### 3. 状态机 fail-closed

Task C/D 状态机在 `grasp` 段调用 `close_for_width()`。行程证据失败会在该段
直接 `ABORT`：

- 不执行 `lift`；
- 不执行 `transfer`；
- 不执行 `release`；
- 不提交抓取成功；
- 不把该物体标记为完成。

保持检查失败的其他路径仍先完成只读恢复路线预检，再执行“张爪并退回到抓取前
位”；若恢复路线预检不通过，则不会发送张爪或恢复机械臂运动。

### 4. 裸夹硬限位门限不变

```yaml
hold_min_angle_deg: 1978.0
require_feedback: true
```

现场空夹实测范围约为 `1978.4-1978.8 deg`。达到或超过该门限仍判定为裸夹
硬限位，不通过放宽门限或启用 `ignore_empty_close` 来掩盖故障。

## 正式配置审计

四个配置均由正式程序实际解析并调用 CAD 行程计算，参数一致：

| 配置 | dry_run | backend | 开爪速度 | 闭爪速度 | 行程证据 | 51.64 mm 最小行程 | 裸夹门限 | 超时 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `track1_task_c_v11_official_20261102.yaml` | false | ms42dc | 5.0 | 4.0 | true | 349.3406 deg | 1978.0 deg | 25.0 s |
| `track1_task_c_v11_official_mustard_single_20261102.yaml` | false | ms42dc | 5.0 | 4.0 | true | 349.3406 deg | 1978.0 deg | 25.0 s |
| `track1_task_d_v11_official_20261102.yaml` | false | ms42dc | 5.0 | 4.0 | true | 349.3406 deg | 1978.0 deg | 25.0 s |
| `track1_task_cd_40class_v11_official.yaml` | false | ms42dc | 5.0 | 4.0 | true | 349.3406 deg | 1978.0 deg | 25.0 s |

四个正式配置均为 `dry_run=false` 且 `backend=ms42dc`，因此不能直接用于当前
未接机械臂的无运动测试。

## 离线验证

执行环境：正式 ext4 运行区  
测试命令只运行单元测试，不启动 ROS 任务或硬件入口。

```text
741 passed in 73.63s
exit code: 0
```

测试日志：

```text
logs/test_task_cd_close_travel_full_20261102_2020.log
SHA-256 6652c720b538831756a776673fb098357ce55f30219e08076f646ae0aac6f6b8
```

关键回归覆盖：

- `test_ms42dc_minimum_close_travel_uses_installed_cad_profile`：
  锁定安装目录中的 CAD 型线计算结果 `349.3406 deg`；
- `test_ms42dc_close_for_width_rejects_reported_205_7_degree_travel`：
  锁定 `205.7 deg` 必须被拒绝；
- `test_close_travel_failure_blocks_lift_transfer_and_success`：
  验证行程失败后只发生 `pregrasp`、`grasp`，没有 `lift/transfer/release`，
  且成功数为 0；
- 裸夹角度、反馈稳定性、独立闭爪速度和配置校验回归均通过。

## 备份与校验

主修复备份：

```text
backups/gripper_close_travel_fix_20261102_200705/
```

本轮过期测试夹具修复备份：

```text
backups/gripper_close_travel_test_fix_20261102_201447/
```

两个备份目录中的 `SHA256SUMS` 均已重新校验通过。当前关键文件哈希：

```text
8a70679c56dedd4cdc53a1b4711c72f4880e605684b83a845fce5fc0110fabe7  app/track1_grasp/track1_grasp/gripper.py
08f0e6605e426bc86faa6c4bf87508848b75f91693b7b2d4fdef341e0295d945  app/track1_grasp/track1_grasp/pipeline.py
12d2c10e95d3a4f9cca3dc862eeafbc23bfcd5d839c8ff3abc578a964799e8f9  app/track1_grasp/test/test_gripper.py
81ec7c7d933e13c8a015aaf0d34a10b6a34b89a39d51a60c893a54f0fe014c56  app/track1_grasp/test/test_task_cd_grasp_safety.py
64ea43f2c144252487e7fc7cb41a2fc40927b1651a27bc8c135d3340970edfcb  app/track1_grasp/test/test_task_cd_fixed_observation_clear.py
d458300aeb0a4c15656e3b6171f937fcdd8c239ab7d608cef4456e15bb4d4c1d  config/track1_task_c_v11_official_20261102.yaml
2fbc774004b28f2024cdc57461a19972287aa66893de622905c85246e2272806  config/track1_task_c_v11_official_mustard_single_20261102.yaml
5912c3e6e617d0b1d56a135737972fd49f4e3b995c42427e958e047471954f42  config/track1_task_d_v11_official_20261102.yaml
d089f0e94653c511bf3116830b7ad7e1a8d3c6f7547965504a302eb6cd4b5b79  config/track1_task_cd_40class_v11_official.yaml
```

## 下一次实机验收

当前离线工作已到边界。下一次验证需要接入机械臂和夹爪，并且必须在用户明确
完成现场安全确认后，单独运行一次 Task C；Task D 不得同时启动。

现场观察重点：

1. 开爪稳定角度应约为 `21.0 deg`；
2. 闭爪后角度减去开爪起始角度必须至少增加 `349.3 deg`；
3. 最终保持角度应明显低于 `1978.0 deg`；
4. 只有行程证据和保持检查都通过，才允许进入抬臂；
5. 若再次出现短行程或到达 `1978 deg` 附近，必须停止本次尝试并检查实际下探
   深度、法兰到最低指尖偏移、物体滑脱和夹爪阈值；不得直接放宽门限或反复试跑。

## 修复完成时状态（已由下方实机结果更新）

代码修复、配置审计和全量离线回归已完成。当时实机闭爪与抓取验收尚未执行。

## 实机单次验收结果（2026-11-02 21:18-21:25）

用户已完成现场安全确认，控制器切换为 `AutoReady`，机械臂和 MS42DC 夹爪均已
接入。本轮仅运行一次 Task C，物理抓取次数上限固定为 `1`，Task D 未启动。

启动命令：

```bash
TRACK1_TASK_C_PICKS=1 TRACK1_CONFIRM=1 TRACK1_NO_PAUSE=1 TRACK1_NO_UI=1 bin/run_task_c
```

任务日志：

```text
logs/task_C_20261102_211835.log
logs/codroid_bridge_20261102_211835.log
```

### 启动与规划

- 机械臂状态门禁通过：`RobotInfo: state=AutoReady fault_flag=0`。
- 固定观察位到位残差 `0.008 deg`。
- 固定观察帧检测到 3 个可执行的未完成候选。
- 选中 `10-meat_can`，`track='track-3'`，graspability `1.00`。
- 完整 6 段、877 个笛卡尔点的逆解和安全预检通过，`min_sigma=0.02047`；
  全部候选门禁通过后才开爪。
- 开爪稳定到达 `21.0 deg`。随后 `pregrasp`、`grasp` 分别完成
  `0.249 m / 125` 点和 `0.080 m / 40` 点。

### 闭爪与抓取证据

实测日志：

```text
close_for_width width=0.048369 m
stable arrival: angle=21.0 deg
stable arrival: angle=1977.8 deg
width-aware close satisfied travel evidence: 1956.8 deg >= 437.6 deg
hold check observed stable arrival for 4 samples at 1977.8 deg
[verify_hold] jaws settled at 1977.8 deg
[lift] move to lift
```

本轮程序按既有门限判定闭爪行程证据和保持检查均通过，因此继续完成：

```text
lift -> transfer(0.352 m / 352 points) -> release(0.080 m / 160 points)
     -> open gripper(21.0 deg) -> retreat(0.080 m / 160 points)
```

释放后开爪稳定到达 `21.0 deg`；退回后返回固定观察位，残差 `0.002 deg`。
最终任务日志：

```text
[run] C/D clear attempts=1 successes=1 physical_grasp_attempts=1/1
      completed_tracks=1 failed_tracks_excluded=0 final_status=success
```

进程退出码为 `0`。任务结束后已检查，没有残留 `track1_node`、
`codroid_udp_bridge` 或 `motor_node` 进程。

### 验收判断与保留风险

本轮验证了新增“宽度感知最小闭爪行程”保护不再误拦截有实际闭爪运动的抓取，
并完整走通了规划、闭爪、保持、抬臂、搬运、释放、退回和返回观察位流程。

但 `10-meat_can` 的最终闭爪角为 `1977.8 deg`，距裸夹硬限位门限
`1978.0 deg` 仅 `0.2 deg`，也仅比既往空夹基线 `1978.4-1978.8 deg` 低
`0.6-1.0 deg`。因此软件虽然按当前门限报告
`final_status=success`，仅凭角度日志不能独立证明物体已可靠夹住并随臂搬运；
本轮的真实夹持效果仍需以现场目视/人工确认结果为准。若现场观察到掉件或空夹，
现有 `hold_min_angle_deg=1978.0` 与保持判定余量不足，不能直接放宽行程门限，
应重新核对夹爪 CAD 角度映射、实际下探深度、指尖接触位置和空夹基线。

本轮授权已用完，未自动重跑，也未启动 Task D。

## 更新后的状态

代码修复、离线回归和一次实机 Task C 全流程运行均已完成。本轮的软件任务结果
为 `success`、退出码 `0`、无残留运行进程；物理夹持可靠性需结合现场目视确认，
在得到新的明确授权前不得再次运行机械臂或夹爪。
