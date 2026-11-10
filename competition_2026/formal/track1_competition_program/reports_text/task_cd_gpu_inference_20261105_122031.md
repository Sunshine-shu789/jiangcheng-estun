# Task C/D GPU 推理切换与验收

- 时间：2026-11-05 12:20:31 CST
- 范围：Task C / Task D 正式启动配置及启动器 GPU 自检
- 原配置：`vision.device: cpu`
- 现配置：`vision.device: 0`（Ultralytics 解析为 `cuda:0`）
- 回滚备份：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/backups/gpu_inference_20261105_121006/`

## 修改内容

1. `config/track1_task_c_v11_official_20261102.yaml`：`vision.device` 改为 `0`。
2. `config/track1_task_d_v11_official_20261102.yaml`：`vision.device` 改为 `0`。
3. `bin/track1_task_runner`：
   - 仅在配置明确为 CPU 时隐藏 CUDA；
   - GPU 配置下执行 CUDA 可用性、设备设置和张量运算自检；
   - GPU 不可用时失败退出，不静默回退 CPU。

## 验收结果

- 运行区挂载：已通过 `bin/ensure_runtime`。
- Task C 启动检查：`GPU 配置检查通过：NVIDIA GeForce RTX 5060 Laptop GPU [cuda:0]`。
- Task D 启动检查：`GPU 配置检查通过：NVIDIA GeForce RTX 5060 Laptop GPU [cuda:0]`。
- 正式 40 类模型、640x480 合成帧推理：
  - `predictor.device = cuda:0`
  - `predictor.model.model` 参数设备 = `cuda:0`
  - CUDA 显存已实际分配。
- 完整离线测试：`908 passed in 114.22s`。
- 本次验收未打开实体相机、未连接或驱动实体机械臂、未执行夹爪动作。

## 文件 SHA256

```text
550842430fc465a9cbe2a54cc3ddfd17170a2fb6f004ec52e0d9f50bb8a755ed  /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/config/track1_task_c_v11_official_20261102.yaml
7b6917f46ec98c10155e16a8ee11c2d37a2ccad5fde7c7621650d18c8bf35992  /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/config/track1_task_d_v11_official_20261102.yaml
a5734f5caa9eec6629b413ab9396a0f3c2a2e62b82bfee5225a478d601b94013  /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/bin/track1_task_runner
```
