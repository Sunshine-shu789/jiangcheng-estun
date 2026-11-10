# Task B D405 手眼静态验收与任意摆放离线接入报告（2026-10-25）

## 结论

本轮离线工作已完成：

1. D405 手眼标定及静态闭环验收**通过**。
2. 新标定外参已接入任意摆放 Mock 链路，并启用“外参必须已验证”的严格门禁。
3. 任意摆放 Mock 完成 `5` 次请求、`5` 次尝试、`5` 次成功。
4. 全量离线测试通过：`299 passed`。
5. **正式五点 Live 配置未切换到新外参，任意摆放实机运动仍未启用。**

本次验收没有发布 `/RobotMove`，没有发送夹爪命令，没有执行实体机械臂运动。

## 手眼标定

标定 ID：

```text
hand_eye_d405_20261025_182144_6438b05191
```

标定板：

- 方格：`14 mm`
- 板规格：`11 x 8` 方格
- 内角点：`10 x 7`

标定结果文件：

```text
calibration/hand_eye_20261025/results/hand_eye_d405_20261025_182144_6438b05191/hand_eye_result.json
```

`T_flange_camera` 平移：

```text
[7.976019, -59.531877, 25.355706] mm
```

`T_flange_camera` RPY：

```text
[-0.809087, -0.282351, -0.334400] deg
```

安装距离复核：

```text
期望：65.000000 mm
实测：65.196419 mm
误差：0.196419 mm
```

质量结果：

| 指标 | 结果 |
|---|---:|
| 固定板平移 RMS | 0.713506 mm |
| 固定板旋转 RMS | 0.373595 deg |
| 留一验证平移 RMS | 0.802692 mm |
| 留一验证旋转 RMS | 0.395124 deg |
| 数值校验 `validated` | `true` |

`hand_eye_result.json` 中的：

```text
ready_for_task_b_activation: false
```

是“需要人工确认后才切换正式 Task B”的安全开关，不是标定失败。切换正式 Live 配置前，必须先完成静态闭环验收和获准的现场低速测试。

## 静态闭环验收

验收脚本：

```text
calibration/hand_eye_20261025/validate_static_hand_eye.py
```

验收结果目录：

```text
calibration/hand_eye_20261025/results/static_validation_20261025_182902/
```

主要结果：

| 指标 | 实测 | 阈值 | 结果 |
|---|---:|---:|---|
| 重投影 RMS 最大值 | 0.147433 px | <= 1.000 px | 通过 |
| 相机系板中心稳定跨度 | 0.032080 mm | <= 1.000 mm | 通过 |
| 相机系板姿态稳定跨度 | 0.054737 deg | <= 0.500 deg | 通过 |
| 基座系板平移平均误差 | 0.865389 mm | <= 3.000 mm | 通过 |
| 基座系板旋转平均误差 | 0.206429 deg | <= 1.000 deg | 通过 |
| 采集期间关节角变化 | 0.000000 deg | <= 0.010 deg | 通过 |

共采集并检测 `5` 帧，所有质量门通过。验收期间机械臂为 `PowerOff`，故障标志为 `0`，关节稳定；静态验收只读取 `/RobotInfo` 和 D405 图像，因此掉电状态不影响零运动闭环结果。

实测基座系标定板平均平移：

```text
[0.026450, 0.508088, -0.018927] m
```

期望平移：

```text
[0.026587, 0.507842, -0.018108] m
```

完整报告：

```text
calibration/hand_eye_20261025/results/static_validation_20261025_182902/REPORT_zh.md
```

## 正式候选外参

以下两份文件内容完全一致：

```text
config/camera_to_flange.d405_calibrated_20261025.yaml
app/track1_grasp/config/camera_to_flange.d405_calibrated_20261025.yaml
```

SHA-256：

```text
6cc7e8be0541949e00921636c49cc16f3c86661737c26966f283ed0c26e452ca
```

外参文件包含：

```text
validated: true
calibration_id: hand_eye_d405_20261025_182144_6438b05191
source_manifest_sha256: 6438b051914d6cac78d82a4d221695df0cad277625ba656a3da89c9f272099ef
```

## 任意摆放 Mock 接入

离线配置：

```text
config/track1_task_b_arbitrary_mock.yaml
```

关键设置：

```yaml
pose:
  extrinsics_file: camera_to_flange.d405_calibrated_20261025.yaml

adaptive_inverse:
  enabled: true
  arbitrary_placement: true
  require_validated_extrinsics: true

vision:
  backend: mock

motion:
  backend: mock
  dry_run: true

gripper:
  backend: mock
```

该配置不保留旧五点位的 XY 状态，物体 XY、Z 和抓取姿态由当前视觉位姿及逆运算链路生成。支撑面必须可测量；无法可靠估计时拒绝该帧，不使用危险的回退 Z。

离线结果：

```text
请求抓取：5
尝试：5
成功：5
```

测试文件：

```text
app/track1_grasp/test/test_arbitrary_placement.py
```

测试已验证：

- 使用 `camera_to_flange.d405_calibrated_20261025.yaml`。
- 外参 `validated` 为 `true` 且 `calibration_id` 非空。
- `require_validated_extrinsics` 为 `true`。
- 任意摆放模式不读取或保存固定点位 XY。
- 五轮 Mock 流程全部成功。

## 全量回归

运行环境按要求先执行：

```text
bin/ensure_runtime
```

测试结果：

```text
299 passed in 19.47s
```

此前唯一失败是测试仍断言旧的 `gripper.backend == "ms42dc_ros"`，但正式配置已经根据现场串口问题改为直连 USB-C 的 `ms42dc`。本轮只修正过期测试断言，并增加以下检查：

```text
gripper.backend == "ms42dc"
gripper.serial_port == "/dev/motor_serial"
gripper.require_feedback == true
```

未修改五点位姿、XY 偏移、速度、限位、工作空间或正式夹爪参数。

测试修改前备份：

```text
backups/task_b_test_direct_gripper_20261025_183151/test_task_b.py
```

## 正式配置保持

正式五点 Live 配置：

```text
config/track1_task_b_final_20261020_live.yaml
```

当前仍保持：

```text
adaptive_inverse.enabled: false
pose.extrinsics_file: camera_to_flange.d405_assumed.yaml
motion.dry_run: false
gripper.backend: ms42dc
```

本轮**没有**把正式五点程序自动切换到新外参或任意摆放模式，也没有修改桌面入口。

## 尚未完成的实机边界

以下项目仍需实体设备、明确授权和有人值守测试：

1. 正式五点 Live 配置切换新外参后的单点静态验证。
2. 真实 D405 对待抓物品的识别、支撑面和目标位姿稳定性验证。
3. 视觉目标到法兰位姿的实机坐标闭环。
4. 真实 IK、关节限位、关节分支跳变和 Jacobian 奇异检查。
5. 低速单点任意摆放接近、抓取、抬升和夹爪反馈验证。
6. 通过单点验收后，再逐级扩展到多点流程。

在明确确认开始实机测试前，Task B 实机运动保持不启动，任意摆放正式 Live 激活保持关闭。
