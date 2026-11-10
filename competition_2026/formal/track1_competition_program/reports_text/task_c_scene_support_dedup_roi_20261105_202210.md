# Task C 支撑平面去重与保守 ROI 修复报告

- 完成时间：`2026-11-05T20:23:43+08:00`
- 正式工作目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序`
- 状态：修复完成；离线回归与 8 帧只读相机诊断通过
- 实体动作：未执行。未连接 RobotInfo/RobotBridge，未规划或发布机械臂运动，未构造抓取流水线，未向夹爪发送命令
- 备份目录：`backups/task_c_scene_support_dedup_roi_20261105_201904/`

## 1. 结论

Task C 原先“已检测到物体但无抓取候选”的根因已修复，主要问题是：

1. 支撑平面采样前直接使用原始检测框，同一物体的重复检测框会扩大目标保护区并遮蔽桌面支撑平面采样；
2. 固定 D405 相机约 16 度倾角下，原支撑平面 ROI 边缘有效桌面像素偏窄，部分帧不足稳定的 300 点门槛。

修复后没有降低 300 点最小内点、RMS、内点率、组件共识、目标间隙、策略置信度或几何质量 `0.55` 等安全门槛。只对支撑平面采样用的检测框增加去重，并采用保守的显式 ROI 外扩与局部法向约束。

最新 8 帧只读诊断结果为 `passed`：

- 已知目标观测：`32`
- 几何通过：`28`
- 策略接受：`28`
- 物理目标簇：`4`
- `stable_geometry` 目标簇：`4/4`

首帧相机冷启动约 `7.95 s` 且支撑平面失败；第 2 至第 8 帧均实现 `4/4` 几何通过。8 帧窗口仍让每个目标达到 7/8 通过，显著高于稳定判定要求的 5 帧，因此不再由冷启动首帧导致整轮验收失败。

## 2. 修改内容

### 2.1 检测框去重复用

文件：`app/track1_grasp/track1_grasp/detection_stabilizer.py`

- 提取纯函数 `boxes_are_duplicates`。
- 新增 `select_non_duplicate_box_indices`。
- 检测稳定器与支撑平面采样共用同一套重复框判断，避免两份实现漂移。

### 2.2 支撑平面采样去重

文件：`app/track1_grasp/track1_grasp/vision.py`

- 只在 `scene_support_items` 构建支撑平面采样保护区前去除重复检测框。
- 最终检测结果、跟踪结果和诊断目标列表保持不变，避免改变下游抓取对象集合。

### 2.3 保守 ROI 与组件约束

文件：`config/track1_task_c_v11_official_20261102.yaml`

```yaml
support_plane_scene_margin_x_px: 35
support_plane_scene_margin_y_px: 45
support_plane_min_local_normal_z: 0.99
support_plane_roi_far_min_local_normal_z: 0.97
support_plane_roi_far_depth_quantile: 0.80
support_plane_component_min_inlier_ratio: 0.70
support_plane_min_support_components: 5
```

### 2.4 回归测试

文件：

- `app/track1_grasp/test/test_detection_stabilizer.py`
- `app/track1_grasp/test/test_unknown_proposer.py`

新增重复框纯函数、去重索引选择及支撑平面采样场景覆盖。

## 3. 验证结果

### 3.1 固定观察清箱专项

```bash
APP="/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_grasp"
SITE="/media/sunshine/新加卷/taskA/venv/lib/python3.10/site-packages"
PYTHONPATH="$APP:$SITE" /usr/bin/python3 -m pytest -q \
  "$APP/test/test_task_cd_fixed_observation_clear.py"
```

结果：

```text
102 passed in 11.60s
```

### 3.2 支撑平面与抓取几何回归

本轮累计定向回归通过：

```text
test_detection_stabilizer.py + test_unknown_proposer.py: 26 passed
test_support_plane.py + test_adaptive_inverse.py + test_grasp_geometry.py: 71 passed
test_diagnose_task_c_geometry.py + test_task_cd_grasp_safety.py: 56 passed
```

合计 `255` 项通过。

### 3.3 历史帧离线复验

使用修复前保存的诊断帧，按新配置逐帧重新拟合：

```text
frame 1: pass, points=346, RMS=0.000961 m, quality=0.602
frame 2: pass, points=318, RMS=0.000768 m, quality=0.709
frame 3: pass, points=381, RMS=0.000909 m, quality=0.719
```

三帧均从原始 5 个框去重为 3 个框，并全部通过平台质量 `0.55`。

### 3.4 最新只读相机诊断

执行：

```bash
./bin/ensure_runtime
./bin/diagnose_task_c_geometry \
  --frames 8 \
  --require-known \
  --require-geometry-candidate \
  --require-stable-geometry
```

结果：`status=passed`

| 帧 | 原始/去重检测 | 已知目标 | 几何通过 | 平台质量 | 内点 | RMS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 / 4 | 4 | 0 | 冷启动失败 | - | - |
| 2 | 4 / 4 | 4 | 4 | 0.8019 | 355 | 0.685 mm |
| 3 | 4 / 4 | 4 | 4 | 0.8144 | 317 | 0.641 mm |
| 4 | 4 / 4 | 4 | 4 | 0.7634 | 325 | 0.754 mm |
| 5 | 4 / 4 | 4 | 4 | 0.6836 | 350 | 0.978 mm |
| 6 | 4 / 4 | 4 | 4 | 0.7767 | 343 | 0.718 mm |
| 7 | 4 / 4 | 4 | 4 | 0.7864 | 334 | 0.731 mm |
| 8 | 4 / 4 | 4 | 4 | 0.7194 | 443 | 0.749 mm |

稳定目标簇：

```text
C001 10-meat_can          8/8 tracks, 7/8 geometry, stable_geometry
C002 2-Starkist_Tuna      8/8 tracks, 7/8 geometry, stable_geometry
C003 21-Haribo_Golden_Bear 8/8 tracks, 7/8 geometry, stable_geometry
C004 25-Flat_Screwdriver   8/8 tracks, 7/8 geometry, stable_geometry
```

诊断产物：

```text
logs/task_c_geometry_diagnostic_20261105_202210/summary.json
```

安全元数据复核：

```text
motion_dry_run=true
motion_commands_published=false
motion_backend_constructed=false
pose_planner_constructed=false
pick_pipeline_constructed=false
robot_bridge_started=false
gripper_backend_constructed=false
gripper_commands_sent=false
execution_authorized=false
```

## 4. 最终文件 SHA256

```text
ac83e274a5e6da9f3aa902b3e98fcae2815e50cf884d727cb2c1206d29380ae6  app/track1_grasp/track1_grasp/detection_stabilizer.py
fd7d6546b1eba72380c33e1bb8a328490852e03bc6300b7118579571991fd8ac  app/track1_grasp/track1_grasp/vision.py
09b481c47aed2ee92577ab0e6ec9d6ec47639ff32e87a24e6081ba018713b358  app/track1_grasp/test/test_detection_stabilizer.py
f08686b792ac4eb44dc387d40a121adf40b7c0b62eb424255f047f300b40857b  app/track1_grasp/test/test_unknown_proposer.py
07d45ea87c78588b32e96b7d428d03e9023f18ecc0cc630f0c75a605e326598b  config/track1_task_c_v11_official_20261102.yaml
```

变更前后清单：

```text
backups/task_c_scene_support_dedup_roi_20261105_201904/SHA256SUMS.before
backups/task_c_scene_support_dedup_roi_20261105_201904/SHA256SUMS.after
```

两组清单均已逐项校验通过。

## 5. 回滚

```bash
BACKUP="backups/task_c_scene_support_dedup_roi_20261105_201904"
cp "$BACKUP/app/track1_grasp/track1_grasp/detection_stabilizer.py" app/track1_grasp/track1_grasp/detection_stabilizer.py
cp "$BACKUP/app/track1_grasp/track1_grasp/vision.py" app/track1_grasp/track1_grasp/vision.py
cp "$BACKUP/app/track1_grasp/test/test_detection_stabilizer.py" app/track1_grasp/test/test_detection_stabilizer.py
cp "$BACKUP/app/track1_grasp/test/test_unknown_proposer.py" app/track1_grasp/test/test_unknown_proposer.py
cp "$BACKUP/config/track1_task_c_v11_official_20261102.yaml" config/track1_task_c_v11_official_20261102.yaml
(cd "$BACKUP" && sha256sum -c SHA256SUMS.before)
```

## 6. 边界与后续

- 本次只验证视觉检测、支撑平面、几何策略和候选生成，没有执行真机 Task C。
- 相机冷启动首帧仍需约 8 秒，当前 8 帧验收窗口已能吸收该问题；正式任务应按现有稳定帧门槛读取足够帧数，不应降低门槛。
- 现场坐标、限位、手眼标定、夹爪阈值和实体抓取仍需现场验证。
- 未发现需要 GitHub 外部方案才能解决的未决代码问题。
