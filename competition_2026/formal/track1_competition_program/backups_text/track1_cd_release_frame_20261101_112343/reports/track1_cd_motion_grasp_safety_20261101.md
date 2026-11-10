# 赛道一 C/D 运动与抓取安全门禁离线验证

- 生成时间：`2026-11-01T11:20:00+08:00`
- 状态：`offline_dry_run_validated_no_motion_authorization`
- 范围：Task C/D 的 V11 视觉目标 → 固定向下抓取位姿 → 夹爪尖/法兰换算 → 全路线安全包络 → 原有运动预检 → dry-run
- 正式工作目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序`
- 正式配置：`config/track1_task_cd_40class_v11_official.yaml`

## 1. 结论

C/D 的主链路已经完成离线安全闭环验证，但仍是**离线 dry-run**：

```text
V11 检测目标（夹爪尖语义）
→ 固定向下抓取姿态 [pi, 0, 0]
→ 视觉夹爪尖目标减去旋转后的法兰偏移
→ 法兰命令位姿
→ 全路线夹爪尖/法兰几何安全包络
→ 原有全路线 IK/FK、关节限位、连续性、奇异值和路线预检
→ dry-run
```

当前没有连接机械臂控制柜，也没有向实体机械臂或实体夹爪发送命令：

```text
实体机械臂命令：未发送
实体夹爪命令：未发送
motion.dry_run: true
gripper.backend: mock
motion_authorized: false
quality_validated: false
```

本次验证不授权实机运动。`575 passed` 只表示离线代码、配置和安全门禁测试通过，不代表实机抓取成功率。

## 2. 本次修改

### 2.1 夹爪尖到法兰的坐标换算

视觉目标被明确解释为**闭合夹爪最低尖端**，不是法兰目标。控制器 `/RobotMove` 仍接收法兰位姿，因此规划时执行：

```text
p_flange = p_tip - R(grasp_rpy) @ gripper_tip_offset_flange
```

当前正式配置：

```yaml
pose:
  robot_pose_frame: flange
  gripper_tip_offset_flange_m: [0.0, 0.0, 0.165]

motion:
  grasp_rpy: [3.14159265, 0.0, 0.0]
  grasp_safety:
    enabled: true
    require_flange_tip_offset: true
    require_tool_down: true
    maximum_tool_tilt_deg: 2.0
    minimum_approach_height_m: 0.05
    maximum_approach_height_m: 0.12
    maximum_approach_xy_error_m: 0.001
    # 现场 Z 和水平安全边界尚未标定，当前为 null
```

`grasp_rpy` 是显式固定值，不从当前机器人姿态继承。这样 `orientation=current` 也不会污染夹爪尖到法兰的旋转换算。

`0.165 m` 的来源限制必须明确：它来自现场装配关系的待复核值，不是 MS42DC 官方成品尺寸。真实抓取前必须用实物/装配图重新确认法兰原点到闭合夹爪最低尖端的装配尺寸和方向。

### 2.2 全路线安全门禁

`planner.validate_grasp_safety_plan()` 在运动路线预检之前检查：

- 每个 waypoint 的 XYZ/RPY 必须是有限值；
- 必须使用 `robot_pose_frame=flange`；
- 必须提供夹爪尖法兰偏移；
- 抓取位姿的 RPY 必须等于显式 `grasp_rpy`；
- 所有 waypoint 都要检查工具 Z 轴朝下和最大倾角；
- 抓取法兰 Z、抓取夹爪尖 Z；
- 全路线法兰 Z、全路线夹爪尖 Z；
- 全路线水平半径；
- 预抓取必须高于抓取法兰位姿；
- 预抓取高度上下限；
- 预抓取和抓取 XY 对齐误差。

安全检查失败时，pipeline 在打开夹爪和发送运动命令之前拒绝候选。专项测试确认：

```text
prechecked_routes == []
pipeline.motion.moves == []
pipeline.gripper.events == []
```

原有 IK/FK、关节限位、连续性、奇异值和全路线预检没有被绕过或放宽。

### 2.3 进近高度符号错误修复

本轮发现并修复了进近高度符号方向的真实错误：

```text
旧：grasp_z - pregrasp_z
新：pregrasp_z - grasp_z
```

旧写法在正常上方进近时会得到负值并被错误拒绝。修复后，正常上方的 `0.08 m` 进近会得到正的 `0.08 m`；同时仍会拒绝倒置或高度不足的路线。

### 2.4 Live fail-closed

非 dry-run 的 C/D 配置在启动时强制要求全部安全边界；任一 Z/水平边界缺失、缺少夹爪偏移、或夹爪仍为 mock，都会在启动阶段失败关闭。不能靠放宽 IK 或安全门禁来让测试通过。

## 3. 正式配置离线审计

未连接相机、机械臂或夹爪，人工构造夹爪尖目标并调用正式配置的 `PosePlanner`：

```text
grasp_tip=(0.10, 0.15, 0.02) m
grasp_flange=(0.10, 0.15, 0.185) m
route_flange_z=(0.08, 0.285) m
route_tip_z=(-0.085, 0.12) m
approach_height=0.08 m
approach_xy_error=0.0 m
max_tool_tilt=0.0 deg
release_pose=(0.25, 0.25, 0.08) m
```

该审计暴露两个必须在真机前处理的风险：

1. **Release 尖端穿桌风险**：当前 `motion.place_pose` 的法兰 Z 是 `0.08 m`，在 `grasp_rpy=[pi,0,0]` 和 `0.165 m` 尖端偏移下，release 夹爪尖 Z 约为 `-0.085 m`，会穿过桌面安全边界。正式 live 配置在填写全路线 Z 限值后必须拒绝该路线；现场需要重新标定 release 法兰位姿/放置高度。
2. **当前 dry-run 的限值为空**：正式 dry-run 配置中全路线 Z 和水平半径仍为 `null`，所以上述 release 路线在当前 dry-run 不会自动拒绝。live 配置会因缺少限值启动失败关闭。不能把当前 dry-run 的成功解释为 release 路径安全。

## 4. 视觉风险

正式配置当前为：

```yaml
vision:
  anchor: top_center
  rotate_180: true
```

`top_center` 只是当前检测/深度锚点名称，不等于经过验证的真实物体顶面中心；`rotate_180: true` 必须用现场 RGB-D 画面确认物理方向。若方向或锚点错误，夹爪尖目标会整体偏移，后续安全门禁只能检查几何边界，无法修正视觉语义错误。

## 5. 测试结果

执行环境：

```bash
/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/bin/ensure_runtime
source /media/sunshine/ESTUN_RUNTIME/ros2_ws/install/setup.bash
cd /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_grasp
export YOLO_CONFIG_DIR=/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/config/ultralytics
```

完整回归：

```text
575 passed in 44.16s
```

C/D 抓取安全专项：

```text
14 passed in 2.27s
```

覆盖：

- 夹爪尖 `0.02 m` + 偏移 `0.165 m` → 法兰 `0.185 m`；
- 固定 `grasp_rpy` 不受当前姿态影响；
- 抓取法兰/夹爪尖 Z 越界；
- release/全路线夹爪尖穿桌；
- 水平半径、工具朝下、工具倾角；
- 进近高度和 XY 对齐；
- live 缺少每个安全限值均启动失败；
- live mock gripper 启动失败；
- 安全拒绝时路线预检、运动和夹爪事件都为空。

## 6. 变更文件与 SHA-256

```text
config/track1_task_cd_40class_v11_official.yaml
138ebc8b2c43c8b85c5e8780d49aec9d136aa2eb09e95e7f0f2d985a37ad9065

app/track1_grasp/track1_grasp/planner.py
438dbbbff9cc87e66b18a130a0e3734096dd18b0696c235cb2dcc6e8488c054a

app/track1_grasp/track1_grasp/pipeline.py
1993c8da0f3e1f0720391d85ff0c12fdc804d0e13575ba06383003e8bb72f24e

app/track1_grasp/track1_grasp/config.py
54db6e2c8493f1d016a492f3242410ee7ca00eacfeb6e21a3e097472ed0b0f9d

app/track1_grasp/test/test_task_cd_grasp_safety.py
700e7042791dac7a4a7d78cd2cf6a48307c7380db7a859cfb7f61da4c493d219
```

正式 V11 权重：

```text
models/official/track1_40class_v11_official_f43234a9.pt
f43234a94f1d71b24cb187082c18eb22fea84c88922c605df5f7b1a79ffcaaa6
```

修改前备份：

```text
backups/track1_cd_motion_safety_20261101_110447/
```

## 7. 第一次实体抓取前仍必须完成

按顺序完成，任何一项未通过都继续保持 `motion.dry_run: true` 和 `gripper.backend: mock`：

1. 复核 D405 手眼外参和 `rotate_180` 后的物理方向；
2. 用实物复核法兰原点到闭合夹爪最低尖端的 `0.165 m` 偏移及方向；
3. 标定桌面 Z、透明箱边界、抓取高度、release 高度和全路线安全包络；
4. 确认夹爪真实开合方向、行程、夹持反馈和安全阈值；
5. 检查全臂碰撞模型、关节限位和现场低速空载轨迹；
6. 由现场人员确认 AutoReady、急停、限位和低速门禁后，才考虑单独一次 `execute` 验证。

当前结论一句话：**C/D 的离线规划和安全拒绝逻辑已完成并通过 575 项回归，但 release 高度、夹爪尖装配尺寸和现场视觉/手眼标定未完成前，不允许进入实体抓取。**
