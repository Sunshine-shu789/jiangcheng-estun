# 腕部相机“向下搜索 + 靠近”测试原型

这是一个只验证可行性的测试程序，不包含夹爪抓取。当前现场条件是：

- 机械臂接近竖直。
- D405 腕部相机初始为水平直视前方，光轴约为基座
  `[-0.004, 0.995, -0.096]`，即主要朝 `+Y` 并略微向下。
- 物品放在腕部正下方，初始前视画面通常看不到。
- D405 当前为倒置安装，RGB 和深度必须一起旋转 180°。

程序流程：

1. 从当前前视姿态开始，按俯仰角 `0, 20, 40, 60, 80` 度逐步向下看。
2. 每个俯仰角再尝试 `0, -15, +15` 度左右转头。
3. YOLO 找到指定类别后，用对齐后的 RealSense 深度估计目标距离。
4. 图像伺服让目标居中，并以 5 mm 单步靠近到约 180 mm 观察距离。
5. 连续两帧满足中心误差和距离误差后停止。
6. 不控制夹爪，不自动上电。

## 识别模型

程序直接使用之前训练好的权重，不重新训练，也不回退到通用 COCO 模型：

```text
/media/sunshine/新加卷/taskA/runs/
yolo11s_real_ft_v24_brick_specialist_20261016/weights/best.pt
```

已核对模型任务为 `detect`，类别为：

```text
1-Cheez-it, 2-Starkist_Tuna, 3-Scissors, 4-Frenchs_Mustard,
5-Tomato_Soup, 6-Foam_Brick, 7-Clamp, 8-Plastic_Banana,
9-Mug, 10-meat_can
```

当前 `vision.target_class` 是 `6-Foam_Brick`。如果现场测试物品不是泡沫砖，必须把
`target_class` 改成对应类别，或者通过命令行覆盖：

```bash
--target-class "对应类别名"
```

## 搜索角度和正下方目标

按当前实测位姿，初始相机光轴距离竖直向下约 84.5°。向下搜索 80° 后，光轴约为：

```text
[-0.0003, 0.0781, -0.9969]
```

距离竖直向下约 4.5°。D405 的垂直视场约为 63°，因此目标在腕部正下方时通常能进入
画面。配置保留到 80° 而不是继续到 90°，是为了减少腕部不必要的大角度运动。

如果 80° 仍然看不到，优先检查：

- 目标是否被夹爪、腕部或机械臂本体挡住。
- 相机是否真的位于腕部正上方，而不是有较大横向偏置。
- 腕部关节是否已经接近限位。
- YOLO 类别名和模型是否匹配。

不要在没有确认关节限位和碰撞风险时直接把角度加到 90° 以上。

## 安全边界

- 默认 dry-run，只读取当前状态并计算一步，不发布运动。
- 只有显式加 `--execute` 才会运动。
- `--scan-only --execute` 只做搜索，找到目标就停止，不执行靠近。
- 每次运动前都要求控制器状态为 `AutoReady` 且无故障。
- 当前首次实机参数为单步 5 mm、速度 0.01 m/s、累计行程上限 0.35 m。
- 有最小深度和工作空间硬限制。
- 没有正式碰撞检测、奇异点处理和关节限位预测；实机测试必须有人守急停。

## 运行环境

已使用以下环境验证：

- Ubuntu 22.04 + ROS 2 Humble
- 埃斯顿 Codroid S3-60 控制柜与 `taskA_grasp udp_bridge`
- Intel RealSense D405
- Python：
  `/media/sunshine/新加卷/taskA/venv/bin/python`

每次运行前加载 ROS 和项目 overlay：

```bash
source /opt/ros/humble/setup.bash
source /home/sunshine/codex/easton/codroid_bridge/codroid_bridge/install/setup.bash
source /home/sunshine/codex/easton/ros2_ws/install/setup.bash
```

启动控制柜 UDP 状态/运动桥：

```bash
ros2 run taskA_grasp udp_bridge
```

D405 同一时间只能被一个进程打开。运行前确认没有实时预览占用相机：

```bash
fuser -v /dev/video*
```

如果输出里有 `live_camera_preview.py` 等进程，先关闭对应预览程序。

## 1. 无硬件完整仿真

仿真中的物品现在位于腕部正下方。预期在向下 80° 时首次检测到，然后靠近并收敛：

```bash
/media/sunshine/新加卷/taskA/venv/bin/python \
  /home/sunshine/arm_visual_probe/visual_arm_probe.py \
  --mock --config /home/sunshine/arm_visual_probe/config.yaml
```

当前结果：

```text
[mock] moves=18 success=True
```

只验证向下搜索、找到后立即停止：

```bash
/media/sunshine/新加卷/taskA/venv/bin/python \
  /home/sunshine/arm_visual_probe/visual_arm_probe.py \
  --mock --scan-only --config /home/sunshine/arm_visual_probe/config.yaml
```

当前结果是在第 13 个搜索视角、向下 80°、左右 0° 时找到目标，且不执行靠近。

## 2. 只开相机识别

该模式不会初始化 ROS，也不会运动：

```bash
/media/sunshine/新加卷/taskA/venv/bin/python \
  /home/sunshine/arm_visual_probe/visual_arm_probe.py \
  --camera-only --frames 5 \
  --config /home/sunshine/arm_visual_probe/config.yaml
```

当前实机已验证：

```text
[camera] D405 started 640x480@15
[camera] RealSense ready 640x480 rotate180=True
[yolo] loading .../best.pt
```

因为物品在正下方，初始水平前视画面检测为 0 是正常现象。标注图只有在检测到目标时
才会保存到 `/home/sunshine/arm_visual_probe/out/`。

### 相机序列号

`config.yaml` 中 `vision.serial` 已留空，表示自动选择当前唯一 RealSense。不要继续使用
旧相机序列号；固定序列号与当前设备不一致时会直接报 `No device connected`。

### 图像方向

`config.yaml` 中：

```yaml
rotate_180: true
```

表示 RGB 和深度一起旋转 180°，同时把主点修正为：

```text
cx' = width  - 1 - cx
cy' = height - 1 - cy
```

如果相机重新安装后画面变成倒置，应把该值改为 `false`。

## 3. 只检查机器人状态

该模式不打开相机、不发布运动：

```bash
/media/sunshine/新加卷/taskA/venv/bin/python \
  /home/sunshine/arm_visual_probe/visual_arm_probe.py \
  --robot-status --config /home/sunshine/arm_visual_probe/config.yaml
```

当前实测控制柜仍为 `PowerOff`、无故障。位姿约为：

```text
xyz = [-0.0901, 0.3383, 0.6472]
rpy = [-1.6671, -0.0137, 0.0051]
```

## 4. 实机分阶段测试

### 4.1 只搜索、不靠近

先切换到 `AutoReady`，人员守急停，然后运行：

```bash
/media/sunshine/新加卷/taskA/venv/bin/python \
  /home/sunshine/arm_visual_probe/visual_arm_probe.py \
  --execute --scan-only \
  --config /home/sunshine/arm_visual_probe/config.yaml
```

程序会逐级向下俯仰并左右转头。每个搜索动作都会同时等待 XYZ 和末端姿态到位，
姿态容差约 1.15°，单次动作超时为 60 秒。找到目标后停在当前观察姿态，不执行视觉
伺服靠近。首次运行时重点观察：

- 运动方向是否正常。
- 机械臂是否接近关节限位。
- 向下 80° 画面里是否看到目标。
- YOLO 框、深度值和类别是否正确。

### 4.2 在找到目标的姿态做 dry-run

上一步成功后，机械臂会停在检测姿态。此时运行：

```bash
/media/sunshine/新加卷/taskA/venv/bin/python \
  /home/sunshine/arm_visual_probe/visual_arm_probe.py \
  --config /home/sunshine/arm_visual_probe/config.yaml
```

程序只计算一步，打印 `pixel_error`、`depth` 和 `delta_base`，但不运动。确认：

- 目标在画面中的左右、上下与 `delta_base` 方向一致。
- `depth` 与实际距离大致相符。
- 目标没有被夹爪或腕部遮挡。

### 4.3 搜索并小步靠近

确认上一阶段方向和数值合理后：

```bash
/media/sunshine/新加卷/taskA/venv/bin/python \
  /home/sunshine/arm_visual_probe/visual_arm_probe.py \
  --execute --config /home/sunshine/arm_visual_probe/config.yaml
```

首次实机仍必须有人守急停。程序只做笛卡尔平移，不改变观察姿态，也不发送夹爪命令。

## 相机坐标轴设置

`config.yaml` 中 `servo.camera_to_tool` 是旋转后的有效图像坐标系到工具坐标系的旋转
矩阵，当前使用单位阵：

```yaml
camera_to_tool:
  - [1.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0]
  - [0.0, 0.0, 1.0]
```

含义是：

- 图像右方 `+X` = 工具 `+X`
- 图像下方 `+Y` = 工具 `+Y`
- 光轴前方 `+Z` = 工具 `+Z`

这与当前“相机倒置安装、图像旋转 180° 后恢复为工具坐标系方向”的假设一致，但仍必须
通过真实小步运动确认。若目标在左、机械臂却往右，立即急停并修正矩阵。

矩阵必须正交且行列式为 `+1`，程序启动时会检查。

## 可调参数

- `vision.target_class`：YOLO 类别名。
- `vision.rotate_180`：是否旋转倒置的 RGB-D 画面。
- `vision.confidence`：检测置信度。
- `servo.desired_standoff_m`：最终观察距离。
- `servo.max_step_m`：单次最大位移，首次实机建议不超过 10 mm，当前为 5 mm。
- `servo.speed_mps`：运动速度，当前为 0.01 m/s。
- `servo.max_total_travel_m`：相对起始位置的累计行程上限。
- `workspace`：基座坐标系内的硬限位。
- `search.pitch_down_offsets_deg`、`search.yaw_offsets_deg`：搜索姿态网格。

所有搜索姿态都相对本次运行的初始姿态独立计算，不会累计角度漂移。

## 当前限制

- 没有夹爪动作。
- 没有正式手眼标定，只有图像旋转和相机轴方向假设。
- 目标丢失后不会自动重新搜索；程序停止并保留当前位置。
- 控制柜拒绝笛卡尔姿态、关节不可达或发生故障时，程序会退出，不会自行恢复。
- 搜索动作只改变末端姿态，不检查整条机械臂的碰撞情况。
