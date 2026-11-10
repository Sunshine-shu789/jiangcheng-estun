# Task B/C/D GPU 主通道与应急 CPU 兜底

- 时间：2026-11-05 20:54:24 CST
- 范围：Task B/C/D 视觉推理设备门禁、启动器、节点入口、测试和现场说明
- 结论：正常入口仍强制 GPU；仅在任务启动前由操作员显式启用应急 CPU
- 回滚备份：`backups/task_bcd_emergency_cpu_20261105_204336/`
- 原文件清单：`backups/task_bcd_emergency_cpu_20261105_204336/SHA256SUMS.before`
- 修改后清单：`backups/task_bcd_emergency_cpu_20261105_204336/SHA256SUMS.after`

## 最终策略

1. 正常启动 Task B/C/D 时，正式 YOLO 配置必须是 `vision.device: 0`、
   `cuda` 或 `cuda:N`，启动前必须通过 CUDA 设备与真实张量运算自检。
2. GPU 不可用、CUDA 初始化失败或运行时视觉/GPU 异常时，程序 fail-closed，
   不静默回退 CPU。
3. 任务运行期间不进行 CPU 热切换。异常直接退出，避免较慢的 CPU 视觉继续向
   机械臂提供过期坐标。
4. 只有在任务启动前，操作员确认 GPU 不可用并显式选择应急 CPU，才允许：
   `./bin/run_task_b --emergency-cpu`、
   `./bin/run_task_c --emergency-cpu`、
   `./bin/run_task_d --emergency-cpu`。
   也支持 `TRACK1_CPU_EMERGENCY=1` 环境变量。
5. 应急模式会清空 `CUDA_VISIBLE_DEVICES`，在终端/窗口标题标记“应急CPU”，
   且只接受 B/C/D 的 `realsense_yolo` 正式配置。
6. YAML 中的 `vision.cpu_emergency` 被忽略，不能通过改文件绕过 GPU 门禁。
7. 直接调用节点且不传 `--task` 时，也会按配置中的 task mode 重新执行 GPU
   门禁；节点层的 `--emergency-cpu` 同样限制为 B/C/D。

## 修改文件

- `app/track1_grasp/track1_grasp/config.py`
  - 增加仅内存可设置的 `VisionConfig.cpu_emergency`；
  - `apply_mode()` 区分正式 GPU 与显式应急 CPU；
  - B/C/D 普通 CPU 配置继续被拒绝。
- `app/track1_grasp/track1_grasp/nodes/track1_node.py`
  - 增加 `--emergency-cpu`；
  - `prepare_config()` 无条件重新应用任务模式门禁；
  - 应急 CPU 仅允许 B/C/D；
  - 运行中 GPU/视觉异常记录“不热切换 CPU”并退出。
- `bin/track1_task_runner`
  - 支持参数或环境变量启用应急 CPU；
  - 应急时隐藏 CUDA，启动自检使用 CPU 应急配置；
  - 正常启动继续执行 GPU 自检，禁止回退；
  - `check --emergency-cpu` 参数已提前解析并可用于离线预检。
- `app/track1_grasp/test/test_config.py`
  - 覆盖普通 CPU 拒绝、应急 CPU 接受、非 CPU 设备拒绝、YAML 绕过拒绝、
    节点默认门禁、Task A/E 应急拒绝及启动器静态安全约束。
- `使用说明.txt`
  - 新增第九节，写明现场 GPU 故障时的应急操作和禁止事项。

## 验收结果

### 全量离线回归

加载正式 ROS 2 runtime 后执行完整测试目录：

```text
982 passed in 113.20s
```

### 应急 CPU 启动自检

Task B/C/D 三个正式配置分别执行：

```text
TRACK1_CONFIG=<正式配置> ./bin/track1_task_runner check --emergency-cpu
```

三个配置均通过，且均输出“CUDA 已隐藏，仅用于 GPU 不可用的现场应急”。

### 应急 CPU 合成帧推理

使用正式 V11 40 类权重，在 `CUDA_VISIBLE_DEVICES=` 下执行无相机合成帧推理：

```text
CPU inference passed: predictor=cpu params=175/CPU detections=0 torch=2.14.0+cu130
```

### 正常 GPU 自检

`./bin/check_task_bcd_gpu` 通过：

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU
- PyTorch：2.14.0+cu130，CUDA 13.0
- 42 个正式 B/C/D YOLO 配置全部显式指定 CUDA
- 5 套不同模型/尺寸均完成合成帧推理，predictor 和模型参数均在 `cuda:0`
- 未打开相机或连接机器人

## 文件 SHA256

```text
8023a7e36f8f21ca942b2eabb2b061b1751a9389b7dfe6a268fc14f36eb97cb8  app/track1_grasp/track1_grasp/config.py
f1346a8ad55d36bf0752cfa02bdba4c5098900509da87b5afe50571d91eab204  app/track1_grasp/track1_grasp/nodes/track1_node.py
9b69760aab08d86b705ff91a4265c6f84aebb5f8a187c8dcfcf7b2183ad2b211  bin/track1_task_runner
32667ad06a3711dff78ef452acacaaa04ee2fca1111f60139e26fda896421e82  app/track1_grasp/test/test_config.py
68821a6412dcb752fbe9c7d39ba9761107ba7899ed47d75b9e7d920eea1870f0  使用说明.txt
```

## 安全声明

本次工作只执行静态检查、自动测试、启动配置自检、GPU 合成帧推理和 CPU 合成帧
推理。未打开实体相机，未连接 RobotInfo/RobotBridge，未构造抓取流水线，未发布
机械臂或夹爪命令。Task B 仍保持 `motion.dry_run: true` 和
`gripper.backend: mock`。
