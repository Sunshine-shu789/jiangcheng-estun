# Task B 固定五点约束与物体位姿自适应离线验收报告

- 生成时间：2026-11-02
- 状态：**离线验收通过；未授权、未执行实体机械臂或实体夹爪运动**
- 正式程序目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/`
- 说明：`新加卷` 与 `新加卷1` 为同一数据盘目录的不同挂载/链接名称。

## 1. 验收结论

Task B 已完成到可离线验收状态，正式启动入口已切换到五点自适应安全配置。当前实现满足：

1. 五个抓取位置使用固定教学参考窗口，不在运行中移动或拟合收纳盒。
2. 物体位于中间或四角时，抓取中心与姿态由当前 RGB-D 检测自适应生成。
3. PCA 自适应候选全部失败时，才使用教学姿态附近的有界救援候选。
4. 每个候选必须通过完整六段 Cartesian 路线预检，首个安全候选生效。
5. 每点完成抓取、提升、原位释放和退开后，回到共享初始六轴位，再进入下一点。
6. 第五点完成后同样回到初始位，不重复执行最终回位。
7. 任一点失败立即停止，不进入后续抓取点。
8. 当前仍保持 `motion.dry_run: true` 和 `gripper.backend: mock`。

本结论是**软件与离线回归验收**，不是实体抓取验收。

## 2. 实现行为

### 2.1 固定五点区域约束

- `task_b.place_names` 固定为 `pick_1` 至 `pick_5`。
- 五点沿用既有教学参考窗口，当前检测按最近参考窗口完成点位归属。
- 抓取 XY 来自当前 RGB-D 物体检测，并叠加标定的物体相对夹爪尖端偏移。
- 收纳盒不拟合、不移动、不作为动态抓取坐标系。
- 视觉深度不替换抓取法兰 Z；各点使用配置中的安全 Z，且受硬下限
  `0.166534092 m` 保护。

### 2.2 物体位姿自适应

候选顺序如下：

1. 根据物体平面 PCA 主轴生成自适应抓取方向。
2. PCA 主分支使用 `[0, +45, -45, +90, -90] deg` 偏航候选。
3. 仅当全部 PCA 候选路线预检失败后，才尝试教学姿态附近的救援分支。
4. 救援分支使用 `[0, -15, +15, -30, +30] deg` 偏航和
   `[0, -5, -10] deg` 俯仰，共 15 个候选。
5. 所有候选均执行相同的完整路线预检；首个安全候选生效。

未放宽 IK、工作空间、Z 下限、关节限位、Jacobian 奇异阈值、步长、
加速度或路线预检条件。

### 2.3 连续五点和回位流程

```text
初始六轴位
→ pick_1 观察位 → 抓取/提升/原位释放/退开 → 初始位
→ pick_2 观察位 → 抓取/提升/原位释放/退开 → 初始位
→ pick_3 观察位 → 抓取/提升/原位释放/退开 → 初始位
→ pick_4 观察位 → 抓取/提升/原位释放/退开 → 初始位
→ pick_5 观察位 → 抓取/提升/原位释放/退开 → 初始位
```

关键配置：

```text
rounds=5
return_to_pick=true
return_to_initial_after_each_pick=true
return_to_observation_after_each_pick=false
return_to_initial_after_task=false
operator_pause_after_each_pick=false
```

`return_to_initial_after_task=false` 仅用于避免第五点后重复回位；
第五点已在每点回位逻辑中回到共享初始位。

## 3. 修改文件

- `bin/run_task_b`
  - 默认配置切换为 `config/track1_task_b_five_point_adaptive_safe_20261102.yaml`。
- `config/track1_task_b_five_point_adaptive_safe_20261102.yaml`
  - 固定五点约束、物体 PCA 自适应、教学姿态救援、安全 Z 下限和连续回位配置。
  - 修正过期注释，明确每点返回共享初始位。
- `app/track1_grasp/track1_grasp/pipeline.py`
  - 失败的 `PickResult` 保留 `place_name`，使第三点等失败报告能明确显示失败点。
- `app/track1_grasp/test/test_adaptive_inverse.py`
  - 增加五点连续抓取、第二点救援、第三点失败立即停机测试。
- `app/track1_grasp/test/test_task_b.py`
  - 更新正式启动入口断言。
- `app/track1_grasp/test/test_arbitrary_placement.py`
  - 更新正式启动入口断言。

历史固定配置测试常量保持不变，历史固定模式仍由原配置覆盖。

## 4. 测试结果

在挂载的正式 ESTUN ROS runtime 中执行完整离线测试：

```bash
source /media/sunshine/ESTUN_RUNTIME/ros2_ws/install/setup.bash
cd /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_grasp
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:$PYTHONPATH python3 -m pytest -q
```

结果：

```text
709 passed in 60.81s (0:01:00)
```

测试覆盖的关键场景：

- 五个抓取点连续完成，产生 30 段 Cartesian 动作。
- 产生 11 次关节动作：初始位，加每点观察位/回初始位。
- 五个点完成抓取、原位释放和退开，最终六轴状态等于初始位。
- 第二点 5 个 PCA 路线全部失败后，教学姿态救援路线成功并继续完成后续点。
- 第三点全部候选失败时：
  - 只到达第三点观察位；
  - 未产生第三点 Cartesian 动作；
  - 未产生第三点夹爪命令；
  - 未进入第四、第五点；
  - 失败结果明确标记为 `pick_3`。
- 第一、第四点候选数为 18，是 `±90 deg` 在无向夹爪轴下去重后的正常结果；
  每点仍保留完整 15 个救援候选。
- 启动脚本 `bash -n bin/run_task_b` 通过。

## 5. 当前安全状态

```yaml
motion:
  backend: robotmove
  dry_run: true
gripper:
  backend: mock
```

本轮未启动 ROS 任务节点，未发布 `RobotMove`，未连接或操作实体机械臂，
未向实体夹爪发送命令。

## 6. 实机前仍需验证

以下项目尚未实机验收，不能据此宣称实体抓取已经完成：

1. 现场五点坐标和观察位姿复核。
2. 手眼标定在当前安装位置、温度和相机连接下的复验。
3. 关节限位、工作空间边界、奇异区、步长和加速度的现场路线矩阵验证。
4. MS42DC 夹爪开口、闭合阈值、反馈和物体保持力验证。
5. 中间位及四角随机位姿的真实 RGB-D 检测稳定性与深度误差验证。
6. 五点连续抓取、原位释放和每点回位的低风险分段实机验证。
7. 在另行获得明确授权前，保持 `dry_run: true` 和 `gripper.backend: mock`。

## 7. 正式文件校验值

```text
15602675170d3071f740e58ff74cb8fad18d23cf338ef65ff12a1344b015413c  bin/run_task_b
99079cc249460d5d393f22baa94a4782ee1b9113766216931a5546a7e4afe365  config/track1_task_b_five_point_adaptive_safe_20261102.yaml
cab759bbddd248276ec168d005c89cd36ca37d7ad20d1e551d5a619f2450b489  app/track1_grasp/track1_grasp/pipeline.py
4660bf54b1e83ded1180ef2198f2d6a4508101d269b2ea6327e6f78fe9346fdb  app/track1_grasp/test/test_adaptive_inverse.py
f219d2cc522d0a1662de3de835ff95065e45ec54a181db8714fd57bdfe360a07  app/track1_grasp/test/test_task_b.py
9f014ccf2e710fc79abd760d20aa0872b6b5494e3b8b7d347762f5e40ccb2781  app/track1_grasp/test/test_arbitrary_placement.py
```

## 8. 最终判定

```text
software_ready_for_offline_acceptance=true
five_point_continuous_flow=true
object_pose_adaptive=true
taught_pose_rescue=true
storage_box_moved=false
motion_attempted=false
hardware_connected=false
physical_grasp_validated=false
motion_authorized=false
```
