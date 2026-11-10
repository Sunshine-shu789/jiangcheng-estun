# Task B 自适应逆运算离线收尾报告（2026-10-22）

## 结论

本轮离线收尾已完成。自适应逆运算链路已经能在纯 Mock/dry-run 条件下完成：

```text
相机目标输入
  -> 目标位姿/PCA 主轴估计
  -> 物体相对抓取候选生成
  -> 全路线几何预检
  -> 候选评分与回退
  -> 竖直接近/抓取序列
  -> Mock 夹爪状态转换
```

Mock 五点全程测试结果为 `5/5 success`。这证明软件链路和离线路线几何验证可以贯通，**不代表实机自适应抓取已经验证通过**。

正式比赛配置仍保持：

```text
config/track1_task_b_final_20261020_live.yaml
adaptive_inverse.enabled: false
```

## 安装几何与坐标定义

现场提供并纳入程序记录的几何关系如下：

- 相机光心到夹爪最低点平面的直线距离：`165 mm`
- 相机光心到法兰中心的直线距离：`65 mm`
- 法兰中心与夹爪最低点共线
- 相机光心在法兰坐标系中的位置：`(X, Y, Z) = (0, -65, 0) mm`

因此当前外参文件使用：

```text
config/camera_to_flange.d405_assumed.yaml
T_flange_camera translation = (0, -0.065, 0) m
```

其中 **只确定了平移量**。旋转部分仍是基于倒装 D405 的假设，文件中的安全标记保持：

```yaml
validated: false
calibration_id: ""
```

在没有完成现场旋转手眼标定前，不能把该外参用于无人监督的真实抓取。

## 本轮离线修改

- 修复视觉工厂在自适应关闭时不应把自适应配置传给旧构造器的问题。
- 修复 PCA 主轴方向翻转导致的 180 度伪角度，改为最短无向轴角差。
- 补齐并稳定自适应逆运算专项测试。
- 增加不连接设备的纯 Mock 五点配置和场景：

```text
config/track1_task_b_adaptive_mock_20261022.yaml
app/track1_grasp/scenarios/task_b_adaptive_mock_20261022.yaml
```

Mock 配置明确使用：

```text
vision.backend = mock
motion.backend = mock
motion.dry_run = true
gripper.backend = mock
adaptive_inverse.enabled = true
require_validated_extrinsics = false
```

其中 `require_validated_extrinsics: false` 仅用于离线 Mock 链路，不是现场安全配置。

## 测试结果

### 专项测试

```text
app/track1_grasp/test/test_adaptive_inverse.py
12 passed
```

覆盖内容包括：PCA 位姿估计、深度异常过滤、主轴符号稳定性、候选数量/角度、视觉抓取 Z、候选失败回退、候选全失败时不发送任何运动/夹爪事件、外参校验门禁，以及正式五点位配置不被 Mock 改动。

### 全量回归

运行环境按要求先执行 `bin/ensure_runtime`，测试使用 Mock/dry-run：

```text
266 passed in 19.58s
```

没有启动 ROS 节点、物理相机、机械臂或真实夹爪，也没有执行 `bin/run_task_b`。

### Mock 五点全程

结果文件：

```text
logs/task_b_adaptive_mock_20261022.log
logs/task_b_adaptive_mock_20261022.json
```

关键结果：

- 请求抓取：`5`
- 尝试：`5`
- 成功：`5`
- 成功率：`100%`
- 每轮生成候选：`5`
- 候选全路线预检通过：`25`
- 执行航点：`30`（5 轮 x 6 个航点）
- 夹爪事件完整：`open, close, open, close, open` 等共 `11` 个事件
- 每轮 `pregrasp -> grasp`：仅 Z 方向变化 `0.080 m`，法兰 XY 和 RPY 不变

## 正式参数保持

正式 Live 配置中的固定五点和运动安全参数未因本轮离线收尾改变：

```text
speed_mps: 0.12
joint_speed_rad_s: 0.6
joint_speed_safety_limit_rad_s: 0.7
workspace: [-0.65, -0.65, 0.02] .. [0.65, 0.65, 0.80]
cartesian_min_jacobian_singular_value: 0.01
```

固定五点法兰 XY 偏移仍为：

```text
(0, 0)
(0.040, 0)
(0.040, -0.015)
(0, -0.030)
(-0.020, -0.030)
```

未修改桌面入口，也未将 Mock 配置设为比赛默认入口。

## 离线通过的边界

Mock 路线预检只验证工作空间范围、航点间距和离线几何构造，**不能替代真实控制器验证**。以下项目仍需在实体设备上逐项验证：

1. 手眼旋转外参标定，不能只依赖当前的 `rotate_180` 假设。
2. `RobotInfo` 的位姿坐标系、单位和姿态语义。
3. 相机测得的视觉抓取高度与实际夹爪最低点是否一致。
4. 每个候选路线在真实 IK 下是否可达。
5. 关节限位、关节分支跳变、加速度限制和 Jacobian 奇异值是否真正通过。
6. 夹爪阈值、闭合到位和持物检测是否可靠。

在上述项目完成前，正式配置继续保持 `adaptive_inverse.enabled: false`，并使用固定五点路径。实机测试必须先得到明确确认，从单点、低速、有人值守开始，不应直接进入无人监督的五点全程。

## 备份

本轮修改前备份目录：

```text
backups/task_b_adaptive_offline_finalize_20261022_210900/
```
