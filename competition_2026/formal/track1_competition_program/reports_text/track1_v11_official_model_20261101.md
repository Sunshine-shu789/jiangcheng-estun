# 赛道一 V11 正式抓取模型固化报告（2026-11-01）

## 结论

V11 已固化为赛道一 **Task C / Task D 默认正式抓取模型**。本次不进行任何训练，不修改 Task A / Task B 的默认启动配置，不授权实体机械臂运动。

- 正式权重：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/models/official/track1_40class_v11_official_f43234a9.pt`
- SHA256：`f43234a94f1d71b24cb187082c18eb22fea84c88922c605df5f7b1a79ffcaaa6`
- 类别数：40
- 正式配置：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/config/track1_task_cd_40class_v11_official.yaml`
- 默认启动器：`bin/run_task_c`、`bin/run_task_d`
- 模型元数据：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/models/official/track1_40class_v11_official.json`
- 校验文件：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/models/official/track1_40class_v11_official.sha256`

## 已实施

1. 将 V11 `best.pt` 复制到正式模型目录，复制后重新计算 SHA256 并核对一致。
2. 新增 40 类 C/D 正式配置，类别顺序与 V11 完全一致，补齐：
   - `17-Eggs_Plush`
   - `26-Plastic_Strawberry`
3. C/D `max_picks` 调整为 40，`target.known_classes` 和 `target.order` 均使用完整 40 类顺序。
4. 为视觉配置增加可选 `weights_sha256`；真实 YOLO 加载前先校验权重文件哈希，不一致立即拒绝启动。
5. `bin/run_task_c` 和 `bin/run_task_d` 的默认配置切换到 V11 正式配置。
6. Task A/B 默认启动器和配置未切换。
7. 安装前文件已备份至：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/backups/track1_v11_official_install_20261101_103025`。

## 安全状态

- `motion.dry_run: true`
- `gripper.backend: mock`
- 本次没有启动相机、没有启动 ROS 任务、没有发布机械臂运动指令、没有控制夹爪。
- “正式模型”表示 C/D 的默认识别权重已确定，不等同于真机抓取安全验收或运动授权。

## 验证命令

```bash
cd /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=app/track1_grasp \
  python3 -m pytest -p no:cacheprovider -q \
  app/track1_grasp/test/test_v11_official_model.py
TRACK1_CONFIG="$PWD/config/track1_task_cd_40class_v11_official.yaml" \
  TRACK1_NO_UI=1 bin/track1_task_runner check
```

## 回滚

```bash
cp "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/backups/track1_v11_official_install_20261101_103025/app/track1_grasp/track1_grasp/config.py" app/track1_grasp/track1_grasp/config.py
cp "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/backups/track1_v11_official_install_20261101_103025/app/track1_grasp/track1_grasp/vision.py" app/track1_grasp/track1_grasp/vision.py
cp "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/backups/track1_v11_official_install_20261101_103025/bin/run_task_c" bin/run_task_c
cp "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/backups/track1_v11_official_install_20261101_103025/bin/run_task_d" bin/run_task_d
```
