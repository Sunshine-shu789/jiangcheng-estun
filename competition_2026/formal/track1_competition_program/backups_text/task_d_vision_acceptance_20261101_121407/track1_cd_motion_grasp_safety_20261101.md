# 赛道一 C/D 运动、Task D 未知物候选与抓取安全门禁离线验证

- 更新时间：`2026-11-01T11:35:40+08:00`
- 状态：`offline_dry_run_validated_no_motion_authorization`
- 范围：Task C/D 的 V11 闭集目标与 Task D RGB-D 未知物几何候选 → 固定向下抓取位姿 → 夹爪尖/法兰换算 → 六段路线安全包络 → 原有运动预检 → dry-run
- 正式工作目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序`
- 正式配置：`config/track1_task_cd_40class_v11_official.yaml`

## 1. 结论

C/D 主链路已经完成离线安全闭环验证，但仍是**离线 dry-run**：

```text
D405 RGB-D
→ V11 40 类检测
→ Task D：RANSAC 支持面 + 前景连通域未知物提议
→ 排除与 V11 已知框重叠的 unknown 候选
→ DetectionStabilizer（保留独立 track ID）
→ 统一目标策略
→ 固定向下抓取姿态 [pi, 0, 0]
→ 视觉夹爪尖目标减去旋转后的法兰偏移
→ 法兰命令位姿
→ pregrasp / grasp / lift / transfer / release / retreat 六段检查
→ 全路线夹爪尖与法兰几何安全包络
→ 原有全路线 IK/FK、关节限位、连续性、奇异值和路线预检
→ dry-run
```

当前安全状态：

```text
实体机械臂命令：未发送
实体夹爪命令：未发送
motion.dry_run: true
gripper.backend: mock
motion_authorized: false
quality_validated: false
```

本轮没有连接机械臂控制柜，也没有向实体机械臂发送命令。夹爪虽然已通过 USB 接入并供电，但后端仍为 `mock`，没有发送实体夹爪动作。

`594 passed` 只表示离线代码、配置和安全门禁测试通过，不代表实机抓取成功率，也不等于桌面高度、手眼闭环、未知物现场分割质量或夹爪装配尺寸已经验证。

## 2. 本轮完成

### 2.1 显式 release 坐标语义

新增配置字段：

```yaml
motion:
  place_pose_frame: flange | jaw_tip
```

语义如下：

- `flange`：旧的控制器法兰位姿语义；
- `jaw_tip`：XYZ 表示闭合夹爪最低尖端，规划器换算成控制器所需的法兰命令位姿。

C/D 开启抓取安全时必须显式填写 `place_pose_frame`。缺失、拼写错误或在 `jaw_tip` 模式下缺少 `pose.gripper_tip_offset_flange_m` 都会失败关闭，不再静默按法兰坐标解释。

正式配置现为：

```yaml
pose:
  robot_pose_frame: flange
  gripper_tip_offset_flange_m: [0.0, 0.0, 0.165]

motion:
  place_pose_frame: jaw_tip
  place_pose: [0.25, 0.25, 0.08, 3.14159265, 0.0, 0.0]
  grasp_rpy: [3.14159265, 0.0, 0.0]
  place_approach_height: 0.08
  retreat_height: 0.08
```

固定朝下姿态下执行：

```text
p_flange = p_jaw_tip - R(place_rpy) @ gripper_tip_offset_flange
```

因此正式配置的释放点为：

```text
release_tip_xyz    = [0.250, 0.250, 0.080] m
release_flange_xyz = [0.250, 0.250, 0.245] m
transfer_flange_z  = 0.325 m
retreat_flange_z   = 0.325 m
```

旧实现把 `place_pose.z=0.08` 当作法兰位姿，会生成约 `-0.085 m` 的释放夹爪尖端 Z。该语义问题已经消除。

### 2.2 全路线安全门禁

`planner.validate_grasp_safety_plan()` 在运动路线预检前检查：

- 必须存在 `pregrasp/grasp/lift/transfer/release/retreat` 六个 waypoint；
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

本轮同时保留此前修复：

```text
旧：grasp_z - pregrasp_z
新：pregrasp_z - grasp_z
```

正常上方的 `0.08 m` 进近现在得到正的 `0.08 m`；倒置或高度不足的路线仍会被拒绝。

### 2.4 Live fail-closed

非 dry-run 的 C/D 配置在启动时强制要求全部安全边界。任一 Z/水平边界缺失、缺少夹爪偏移、缺少显式 release frame、非法 release frame，或夹爪仍为 mock，都会在启动阶段失败关闭。

### 2.5 Task D 未知物候选分支

新增 `track1_grasp/unknown_proposer.py`，不再用 V11 的类别名伪造未知物检测：

- 在有效深度点中 RANSAC 拟合支持量最大的平面；
- 将平面上方满足高度范围的点作为前景，按 8 邻域提取独立连通域；
- 过滤深度有效范围、平面内点率/RMS/法向、物体高度、面积、宽高、有效深度率、点数、边缘和位姿质量；
- 每个通过过滤的连通域输出 `label="unknown"`，并带上点云中心、主轴、次轴、轴长度、位姿质量和保守几何质量；
- 与 V11 检测框按 IoU `0.25` 或 containment `0.7` 重叠的前景不会重复生成 unknown；
- 实时链路在 stabilizer 之前合并 V11 与 unknown，后续仍统一经过 `_is_usable`、目标策略、IK、碰撞、路线和原有安全门禁；
- proposer 抛错时只关闭该帧的 unknown 分支，保留 V11 结果，不会绕过任何下游运动或夹爪安全门禁。

`graspability` 在 unknown 分支中明确只是几何质量代理，不是抓取成功概率。

### 2.6 Task C/D 模式门控

- Task D 强制启用 unknown proposer；
- Task A/B/C/E 强制关闭 unknown proposer；
- 门控在 `apply_mode()`、视觉源创建和搜索节点中生效，YAML 中误写 `enabled: true` 也不能把 Task C 或非 D 流程带入未知物分支；
- 正式 YAML 默认写为 `enabled: false`，Task D 运行时由模式门控显式打开。

### 2.7 未知物多实例稳定跟踪

`DetectionStabilizer` 对 `unknown`、`object` 按物理实例处理：

- 不受 V11 已知类别 `max_per_class=1` 的折叠限制；
- 两个空间分离的 unknown 候选可以保留为两个 track；
- IoU/containment 去重仍然生效；
- 短时遮挡恢复后继续使用原 `track-N`，不按类别把不同实例交换；
- 已知类别原有最大实例数和类别切换规则保持不变。

## 3. 正式配置离线审计

未连接相机、机械臂或夹爪，人工构造夹爪尖目标并调用正式配置的 `PosePlanner`。本次直接检查结果：

```text
dry_run=True
gripper_backend=mock
place_pose_frame=jaw_tip
grasp_tip_xyz_m=(0.10, 0.00, 0.02) m
grasp_flange_xyz_m=(0.10, 0.00, 0.185) m
release_tip_xyz_m=(0.25, 0.25, 0.08) m
release_flange_xyz_m=(0.25, 0.25, 0.245) m
minimum_route_tip_z_m=0.02 m
waypoint_count=6
waypoints=pregrasp,grasp,lift,transfer,release,retreat
```

当前正式 dry-run 配置中的：

```yaml
minimum_grasp_tip_z_m: 0.0
minimum_route_tip_z_m: 0.0
```

只是禁止夹爪尖低于机器人基座 `Z=0` 的临时坐标下限，**不是桌面高度标定值**。通过该检查不等于不会穿过现场桌面。

仍未标定或必须现场复核的项目包括：

1. 机器人基座坐标系下的真实桌面 Z；
2. 透明箱边界和放置区域高度；
3. 物体顶面到抓取尖端的目标深度语义；
4. `0.165 m` 法兰到最低闭合夹爪尖的实际装配尺寸与方向；
5. 全路线水平安全包络和现场障碍物。

夹爪资料结论：MS42DC 官方资料主要包含控制、固件、ROS 和电气内容，目前没有找到能证明当前装配正好具有 `0.165 m` 法兰到最低尖端偏移的官方尺寸图。WHEELTEC 通用手册中的外形尺寸不能直接替代该装配偏移。因此 `0.165 m` 仍必须标为“现场装配待复核值”。

## 4. 视觉风险与 Task D 边界

正式配置当前为：

```yaml
vision:
  anchor: top_center
  rotate_180: true
```

`top_center` 只是当前检测/深度锚点名称，不等于经过验证的真实物体顶面中心；`rotate_180: true` 必须用现场 RGB-D 画面确认物理方向。若方向或锚点错误，夹爪尖目标会整体偏移，后续安全门禁只能检查几何边界，无法修正视觉语义错误。

Task D 已有真实 RGB-D 的类别无关候选分支，不再只依赖 V11 的 40 个训练类别，也不是用 `unknown/object` 名称假装开放词表。但当前实现仍有关键边界：

1. **透明物和强反光物未解决**：深度缺失会把物体拆碎或删除，候选会失败关闭；
2. **依赖稳定支撑面**：桌面被大面积遮挡、平面点不足或场景不是平面支撑时，unknown 分支不会输出候选；
3. **接触物可能粘连**：两个相互接触的不透明物体可能形成一个连通域，只生成一个候选；
4. **不是开放词表或实例分割网络**：没有语义理解，纯几何候选也可能把线缆、箱壁或背景结构当成 unknown；
5. **位姿仍是单视角几何近似**：没有经过透明物深度修复、多视角融合或手眼标定实测；
6. **现场数据尚未验收**：当前只有合成/定向单元测试，不能据此宣称比赛现场未知物召回率或抓取成功率。

因此当前准确状态是：**Task D 的不透明未知物几何候选链路已接入；透明/反光/深度洞物体抓取仍未解决。**

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
594 passed in 45.55s
```

Task D 未知物、多实例稳定器和 C/D 抓取安全专项：

```text
41 passed in 3.38s
```

专项覆盖：

- 两个分离未知物生成两个候选；
- 小面积深度噪声被过滤；
- V11 已识别目标不再生成 unknown；
- 无有效深度时失败关闭；
- 低几何质量候选被拒绝；
- Task C 关闭、Task D 开启 unknown proposer；
- 两个 unknown 实例保持两个 track；
- 短暂遮挡后 track ID 不交换；
- proposer 异常时保留 V11 结果；
- 夹爪尖 `0.02 m` + 偏移 `0.165 m` → 法兰 `0.185 m`；
- `jaw_tip` release → 法兰 `Z=0.245 m`；
- 固定 `grasp_rpy` 不受当前姿态影响；
- 抓取法兰/夹爪尖 Z 越界；
- release 和全路线夹爪尖 Z 检查；
- 水平半径、工具朝下、工具倾角；
- 进近高度和 XY 对齐；
- 缺少或非法 `place_pose_frame` 失败关闭；
- `jaw_tip` 缺少夹爪偏移失败关闭；
- live 缺少每个安全限值均启动失败；
- live mock gripper 启动失败；
- 安全拒绝时路线预检、运动和夹爪事件都为空。

## 6. 变更文件与 SHA-256

Task D 未知物接入、C/D 模式门控、多实例跟踪和 C/D 抓取安全相关文件：

```text
app/track1_grasp/track1_grasp/unknown_proposer.py
4135daae04249b858e043a124740062a82281281cf973848851f590314d3deae

app/track1_grasp/track1_grasp/vision.py
298f09df58b0f6f94cfb9c19dbd6fb868e3f5665c57e306b9a6257ed3ee89dae

app/track1_grasp/track1_grasp/detection_stabilizer.py
7f37e0a41184ccbf2f6595affd344352b7b77c7278e0ac825c3e1016fde098e2

app/track1_grasp/track1_grasp/config.py
7313d5c8054de684dbc62f2f40c1e9025280a82aef8e00a20b9ed9577602d035

app/track1_grasp/track1_grasp/factory.py
b674f6ee6c37db4e92754bd73b0de01f33e0c9d6dc7c49183837399a65b990ca

app/track1_grasp/track1_grasp/nodes/search_approach_node.py
024293500d54e47b8c1889a393fdecdc6edaa2215af4e00860883a10b525f11d

app/track1_grasp/track1_grasp/planner.py
fa0d418b7c709c0c10f3f30213e1d557f10a80ce7de2c3ff57414d7e3133597c

app/track1_grasp/track1_grasp/pipeline.py
f9fa9daadf9f415bec3783e53bd931313118f8723e9e37a8f20dd327430278d7

config/track1_task_cd_40class_v11_official.yaml
d50d4e0a9bee57be5a50ff22d94b0968d6bfe48f8035892aaed779663ecca1c3

app/track1_grasp/test/test_unknown_proposer.py
64de0cb359dd7270a7ffe5ae9e77ed99b7c58d63afe0bc33f7fd553b55876eeb

app/track1_grasp/test/test_detection_stabilizer.py
cac5a2ea34d280247eb47a7967b9c1138b0328f940ebd3a157bb2439dfdb3207

app/track1_grasp/test/test_task_cd_grasp_safety.py
eacc5b5283be25e0604448f0c3f40fde6642e7366d7929753d12a4cda322a684
```

正式 V11 权重：

```text
models/official/track1_40class_v11_official_f43234a9.pt
f43234a94f1d71b24cb187082c18eb22fea84c88922c605df5f7b1a79ffcaaa6
```

修改前备份：

```text
backups/track1_cd_motion_safety_20261101_110447/
backups/track1_cd_release_frame_20261101_112343/
backups/track1_task_d_unknown_20261101_113227/
```

## 7. 第一次实体抓取前仍必须完成

按顺序完成，任何一项未通过都继续保持 `motion.dry_run: true` 和 `gripper.backend: mock`：

1. 用当前 D405、现场桌面和实际物体做 Task D 离线 RGB-D 回放，统计未知物召回、误检、连通域粘连、深度有效率和候选稳定性；透明箱/透明物必须单列为未通过项；
2. 复核 D405 手眼外参和 `rotate_180` 后的物理方向；
3. 用实物复核法兰原点到闭合夹爪最低尖端的 `0.165 m` 偏移及方向；
4. 标定桌面 Z、透明箱边界、抓取高度、release 高度和全路线安全包络；
5. 确认夹爪真实开合方向、行程、夹持反馈和安全阈值；
6. 检查全臂碰撞模型、关节限位和现场低速空载轨迹；
7. 由现场人员确认 AutoReady、急停、限位和低速门禁后，才考虑单独一次 `execute` 验证。

当前结论一句话：**C/D 的离线抓取安全链和 Task D 不透明未知物候选分支已经接入并通过 594 项回归，但真实 D405 场景验证、透明物深度恢复、现场标定/装配尺寸和安全包络未完成前，必须保持 dry-run 与 mock，不允许进入实体抓取。**
