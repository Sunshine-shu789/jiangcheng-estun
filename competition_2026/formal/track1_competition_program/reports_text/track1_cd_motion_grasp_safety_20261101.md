# 赛道一 C/D 运动、Task D 未知物候选、抓取安全门禁与 D405 只读验收

- 更新时间：`2026-11-01T12:17:53+08:00`
- 状态：`offline_dry_run_and_real_read_only_camera_validated_no_motion_authorization`
- 范围：真实 D405 只读视觉基线；Task C/D 的 V11 闭集目标与 Task D RGB-D 未知物几何候选 → 固定向下抓取位姿 → 夹爪尖/法兰换算 → 六段路线安全包络 → 原有运动预检 → dry-run
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

真实 D405 的零目标只读验收已通过：15 帧全部采集，V11 权重校验通过，
unknown proposer 已启用，未启动 `/RobotInfo`、机器人桥、运动后端或夹爪后端。
该结果证明相机、深度流、模型加载、稳定器和审计产物可工作，**不证明未知物召回**
，也不改变透明/强反光物未解决、实体动作未授权的状态。

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

`600 passed` 只表示离线代码、配置和安全门禁测试通过，不代表实机抓取成功率，也不等于桌面高度、手眼闭环、未知物现场分割质量或夹爪装配尺寸已经验证。

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

### 2.7 Task D 真实 D405 只读验收入口

新增 `app/track1_grasp/track1_grasp/tools/accept_task_d_vision.py` 和
`bin/accept_task_d_vision`。入口只执行以下链路：

```text
正式 YAML
→ apply_mode(TaskMode.D)
→ D405 RGB-D + V11 + DetectionStabilizer + unknown proposer
→ 固定帧采集与审计输出
```

入口在构建视觉前强制检查 `task.mode=D`、unknown proposer 开启、
`motion.dry_run=true` 和 `gripper.backend=mock`。它不导入运动实现、不创建
`RobotMoveMotionBackend`、不创建夹爪客户端、不启动 ROS 发布器，也不要求
`/RobotInfo`。启动器先挂载 ext4 运行镜像并加载 Humble/`taskA_grasp` 运行路径，
避免把相机工具误连到控制柜。

输出包括逐帧 RGB、深度伪彩色、原始深度 NPZ、检测叠加图、实例/track ID、
深度有效率和分位数、启动到首帧耗时、逐帧捕获耗时及 `summary.json`。
`--require-known` 和 `--require-unknown` 用于正样本失败关闭；空场景不得使用。
该工具只用于视觉验收，不替代正式 `run_task_d` 的 `/RobotInfo` 和实体安全门禁。

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

1. **透明物和强反光物未解决**：真实 D405 深度图中，透明箱内部呈现大面积近似平滑的伪深度面，而不是明显黑洞；透明物可能既没有可靠前景高度，也会污染支撑面拟合；
2. **依赖稳定支撑面**：桌面被大面积遮挡、平面点不足或场景不是平面支撑时，unknown 分支不会输出候选；
3. **接触物可能粘连**：两个相互接触的不透明物体可能形成一个连通域，只生成一个候选；
4. **不是开放词表或实例分割网络**：没有语义理解，纯几何候选也可能把线缆、箱壁或背景结构当成 unknown；
5. **位姿仍是单视角几何近似**：没有经过透明物深度修复、多视角融合或手眼标定实测；
6. **现场正样本尚未验收**：真实 D405 目前只完成空透明箱零目标基线；尚未用不透明未知物执行 `--require-unknown`，不能据此宣称现场未知物召回率、实例稳定性或抓取成功率。

因此当前准确状态是：**Task D 的不透明未知物几何候选链路已接入；透明/反光/深度洞物体抓取仍未解决。**

### 4.1 真实 D405 零目标基线

正式只读验收目录：

```text
logs/task_d_vision_acceptance_20261101_121607/
```

执行结果：

```text
status=passed
captured_frames=15/15
known_observations=0
unknown_observations=0
mean_frame_depth_valid_ratio=0.8559735243
minimum_frame_depth_valid_ratio=0.8528906250
mean_frame_central_depth_valid_ratio=0.9906206597
minimum_frame_central_depth_valid_ratio=0.9895182292
startup_to_first_capture_s=4.1808252910
mean_subsequent_capture_ms=35.5272385000
```

目视核对：画面为空透明箱和下方夹爪尖，没有可抓目标，因此稳定帧零检测符合场景。
首帧 `4.18 s` 包含 D405 启动、V11 权重加载和后台线程初始化；后续 `35.5 ms/帧`
更接近当前 15 FPS 采集链路的消费耗时。

该基线证明：

- D405 640×480@15 FPS、`rotate_180=true` 和深度流可正常启动；
- 正式 V11 权重 SHA-256 校验通过并成功加载；
- `unknown_proposer=on`，稳定器和 Task D 模式门控生效；
- 15 帧 RGB、深度伪彩色、原始深度 NPZ、叠加图和 `summary.json` 均写入成功；
- 工具未启动 `/RobotInfo`、机器人桥、运动后端或夹爪后端。

该基线**不证明** V11 对现场目标的检测率，也**不证明** unknown proposer 的召回率。
下一步必须在同一入口加入真实不透明未知物，并执行 `--require-unknown` 正样本验收。

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
600 passed in 46.13s
```

Task D 未知物、多实例稳定器和 C/D 抓取安全专项：

```text
41 passed in 3.39s
```

Task D 只读视觉验收入口专项：

```text
6 passed in 0.80s
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
- 安全拒绝时路线预检、运动和夹爪事件都为空；
- RGB-D/known/unknown 审计产物完整写入；
- `--require-unknown` 无正样本时失败但仍写报告；
- 相机异常时失败关闭并关闭视觉源；
- 非 dry-run、非 mock 夹爪和非 Task D 配置在创建输出目录前被拒绝。

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

app/track1_grasp/track1_grasp/tools/accept_task_d_vision.py
ce8668f91ba3ca8423210229a05eb2bfe79c9c5f6e7baa4fcf8c312f0ea44217

app/track1_grasp/test/test_accept_task_d_vision.py
73a32e81ced9070a3f9a1aefa398db2ccb9078a6fd915735a2c7bf9dd4e5a2e0

bin/accept_task_d_vision
207c186e257aaa6432de8e89895e0d4a40edc9bfa5ae69b39f8aa54ffa345118

app/track1_grasp/README.md
d97c27faa3ed299cb11dd3dd591c32df127c75618eb556746d081e36ed312db6

app/track1_grasp/docs/INTEGRATION_CHECKLIST.md
805e975cb577322794b693fff89770dc96f87578e54610fd655aea15c4d43134
```

正式 V11 权重：

```text
models/official/track1_40class_v11_official_f43234a9.pt
f43234a94f1d71b24cb187082c18eb22fea84c88922c605df5f7b1a79ffcaaa6
```

真实 D405 零目标验收审计产物：

```text
logs/task_d_vision_acceptance_20261101_121607/summary.json
9b4ea8e823193050eb1fa91d73eec94dee477d4ea2eb706dcc3eba75f2539212

logs/task_d_vision_acceptance_20261101_121607/representative_detection_overlay.jpg
cde8ccba8089a23a3d9233c5c1eabd5f3e7250dbac091af6367302a904a71e2e

logs/task_d_vision_acceptance_20261101_121607/representative_depth_color.jpg
fdfc226815136d5ea97bb470b8a2cfab499be665880683a4060a935656f46af7

logs/task_d_vision_acceptance_20261101_121607/representative_rgb.jpg
8f4e10ae38aa1439ef27d40f3c039146748b9242fb04578f2bb2e906fd629617

logs/task_d_vision_acceptance_20261101_121607/representative_depth_m_npz.npz
0e2de67f17f77017cb2df36e3cecddbd77bfd4744100ee3b6b3b8b3167cb6186
```

修改前备份：

```text
backups/track1_cd_motion_safety_20261101_110447/
backups/track1_cd_release_frame_20261101_112343/
backups/track1_task_d_unknown_20261101_113227/
backups/task_d_vision_acceptance_20261101_121407/
```

## 7. 第一次实体抓取前仍必须完成

按顺序完成，任何一项未通过都继续保持 `motion.dry_run: true` 和 `gripper.backend: mock`：

1. 已完成真实 D405 空透明箱零目标只读基线；下一步必须在同一入口加入真实不透明未知物并运行 `--require-unknown`，统计未知物召回、误检、连通域粘连、深度有效率和候选稳定性；透明箱/透明物继续单列为未通过项；
2. 复核 D405 手眼外参和 `rotate_180` 后的物理方向；
3. 用实物复核法兰原点到闭合夹爪最低尖端的 `0.165 m` 偏移及方向；
4. 标定桌面 Z、透明箱边界、抓取高度、release 高度和全路线安全包络；
5. 确认夹爪真实开合方向、行程、夹持反馈和安全阈值；
6. 检查全臂碰撞模型、关节限位和现场低速空载轨迹；
7. 由现场人员确认 AutoReady、急停、限位和低速门禁后，才考虑单独一次 `execute` 验证。

当前结论一句话：**C/D 的离线抓取安全链和 Task D 不透明未知物候选分支已经接入并通过 600 项回归，真实 D405 空场景只读基线也已通过；但未知物正样本召回、透明物深度恢复、现场标定/装配尺寸和安全包络未完成前，必须保持 dry-run 与 mock，不允许进入实体抓取。**
