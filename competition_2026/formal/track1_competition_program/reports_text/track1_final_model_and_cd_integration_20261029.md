# 赛道一最终模型与 C/D 接入状态（2026-10-29）

## 1. 已选模型

- 模型：v5 40 类检测模型
- 权重：
  `/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/runs/track1_train_40class_v5/track1_40class_v5_20261029_234800/weights/best.pt`
- SHA256：
  `cdafa5f39f8e0b22fe1a608a126283eba52a81c2daf7a9f20f4352bb55396099`
- 训练：20 轮配置，第 9 轮早停；最佳 fitness 在第 1 轮
- 总体指标：mAP50 `0.936`，mAP50-95 `0.720`
- 类别槽：40；第 17 项 `17-Eggs_Plush` 和第 26 项 `26-Plastic_Strawberry` 保留空槽
- 可识别活动类别：38

### 已知限制

前 10 类严格离线回归中，25 个 GT 全部正确匹配，漏检 0、错类 0；但 1 个罐顶画面同时输出了金枪鱼和番茄汤两个高度重叠框。该问题不再继续训练修补，作为已知重复框限制保留。

## 2. 40 类验收入口

`config/track1_acceptance_40class.yaml` 已切换到 v5 权重并写入对应 SHA256。

- 自检结果：`self_test=PASS`
- 模型类别数：40
- 有效类别数：38
- 保留空槽：16、25
- 验收程序仍只读 D405，不连接机械臂或夹爪
- 修改前配置备份：
  `backups/track1_acceptance_40class_before_v5_20261029_235621.yaml`

## 3. 任务 C/D 安全接入

新增：

`config/track1_task_cd_38class_safe_20261029.yaml`

配置内容：

- 显式加载上述 v5 权重
- 38 个有效已知类别，排除第 17、26 项空槽
- D405 使用 `camera_to_flange.d405_calibrated_20261025.yaml`
- `motion.dry_run: true`
- `gripper.backend: mock`
- Task C 目标策略：`any`
- Task D 目标策略：`known_and_unseen`

`bin/run_task_c` 和 `bin/run_task_d` 默认显式使用该配置；仍可通过 `TRACK1_CONFIG` 环境变量覆盖。修改前启动器备份：

`backups/track1_cd_launcher_before_final_model_20261029_235723/`

## 4. 当前边界

- 未执行机械臂运动
- 未连接或控制夹爪
- 未修改桌面“视觉识别”和“照片采集”两个原有程序
- 未覆盖 `models/best.pt`

Task D 的 `unknown/object` 分支目前仍需要独立的类别无关分割器提供候选；现有 40 类 YOLO 只能提供已知 38 类检测结果，不能把未训练物体可靠地当作 unknown 输出。
