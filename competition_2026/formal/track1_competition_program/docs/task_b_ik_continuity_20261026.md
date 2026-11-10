# Task B IK 分支连续性整改（2026-10-26）

## 结论

已完成 Task B 多起点 IK 的关节连续性整改。新增的两个确定性回归场景在旧实现上均失败，
修复后通过。专项、非 ROS 全量、动态 Mock、随机矩阵、倾斜固定矩阵和桌面默认入口均通过。

本轮没有执行实体机械臂、实体夹爪或真机运动，也没有切换桌面默认配置。

## 安全边界

- `vision.backend: mock`
- `motion.backend: mock`
- `motion.dry_run: true`
- `gripper.backend: mock`
- 矩阵工具只做离线 FK/IK/路径预检，不创建 ROS 节点，不发布机械臂或夹爪命令。

## 修改内容

正式源码：

- `app/track1_grasp/track1_grasp/kinematics.py`
- `app/track1_grasp/test/test_kinematics_ik.py`

修复点：

1. `_candidate_rank()` 改为关节运动连续性优先，再比较归一化残差、最小奇异值和种子序号。
   所有候选均满足容差时，不再优先选择残差略好但关节运动更大的分支。
2. `_fast_candidate_is_robust()` 增加 `accept_delta_rad` 条件。快速接受最多使用允许步长余量的
   一半，避免第一次阻尼尝试直接接受接近上限的远分支。
3. 保留原有位姿容差、关节限位、奇异值门限和 fail-closed 行为，没有放宽安全参数。
4. 增加两个回归测试：
   - 容差内优先选择 `0.20 deg` 候选，而不是残差略好但运动 `1.90 deg` 的候选。
   - 快速接受不得在第一轮接受 `0.45 deg` 候选，而应继续选择 `0.05 deg` 候选。

整改前备份：

```text
backups/task_b_ik_continuity_20261026_230311/
├── track1_grasp/kinematics.py
└── test/test_kinematics_ik.py
```

备份位置同时记录在 `state/last_task_b_ik_continuity_backup`。

## 验证结果

| 项目 | 结果 |
|---|---:|
| 新回归在旧实现上 | `2 failed`（预期失败） |
| IK / 线性运动 / 倾斜 Mock 专项 | `73 passed in 7.34s` |
| 非 ROS 全量回归 | `402 passed, 2 skipped in 40.56s` |
| 动态 Mock 五轮 | `5/5 success` |
| 随机矩阵，30 样本，seed `20261026` | `30/30 passed` |
| 倾斜固定矩阵 | `50/50 passed` |
| 桌面默认入口，动态 Mock 五轮 | `5/5 success`；`25` 次完整路线预检 |

全量回归排除了当前环境缺少 `rclpy` 的
`app/track1_grasp/test/test_arm_motion_check.py`，不是本轮代码失败。

动态 Mock 日志：

```text
logs/task_B_20261026_230630.log
requested=5 attempts=5 successes=5 rate=100.00%
```

随机矩阵：

```text
logs/task_b_arbitrary_random30_seed20261026_ik_continuity_20261026.json
max_total_s=54.9487946
max_joint_step=0.499977611 deg
min_sigma=0.022717268
```

倾斜固定矩阵：

```text
logs/task_b_arbitrary_matrix_tilted_sym_ik_continuity_20261026.json
max_total_s=65.5842755
max_joint_step=0.499977424 deg
min_sigma=0.022623153
```

两组矩阵均无失败样本。动态 Mock 日志未出现 `Traceback`、`Exception`、ERROR 或
`possible IK branch jump`；5 条完整路线预检和 5 轮 Mock 流程均完成。

桌面默认入口最终回归：

- 入口：`bin/run_task_b_arbitrary`
- 配置：`config/track1_task_b_box_relative_mock.yaml`
- 配置保持 `vision.backend: mock`、`motion.backend: mock`、`motion.dry_run: true`、
  `gripper.backend: mock`；`adaptive_inverse.arbitrary_placement: false`，
  `box_relative_five_point.enabled: true`。
- 结果：`requested=5 attempts=5 successes=5 rate=100.00%`，每轮完成五条箱体局部路线预检，
  合计 `25` 次完整路线预检。
- 日志：`logs/task_B_20261026_231610.log`
- 日志 SHA256：`6f055cedf077d88502ccee9ba858458052d20f0822dfd6c7496c4a9b3246ffb9`
- 未出现 `Traceback`、`Exception`、`ERROR` 或 `possible IK branch jump`。

## 文件校验

```text
5433581ded12e546c2166178f3d01e58baec69dc9c086ea132775555737b42a2  app/track1_grasp/track1_grasp/kinematics.py
0b0d9cb19e0e19c7faa1f204a2a4b368df06474f2a81387aa8d399d15b50d42a  app/track1_grasp/test/test_kinematics_ik.py
82ee247c1c9ea71048582dc1ee2852f3aa00e81c70452190c89840a9ea6cf3f1  logs/task_B_20261026_230630.log
6f055cedf077d88502ccee9ba858458052d20f0822dfd6c7496c4a9b3246ffb9  logs/task_B_20261026_231610.log
0e84f633348862f48d7156c5ced03d70a6d9b6dbd63ec3025e0b6e6458979571  logs/task_b_arbitrary_random30_seed20261026_ik_continuity_20261026.json
17bae295ce4d7b0e3d8dce40948cf4eb16f03b930b0985e45c08690767cef087  logs/task_b_arbitrary_matrix_tilted_sym_ik_continuity_20261026.json
```

与备份逐行对比后，生产代码只包含候选排序和快速接受门控两处预期改动，测试文件只包含
两个新回归及其辅助构造函数。

## 尚需现场确认

本轮只证明离线数值与 Mock 链路通过，不代表现场坐标、限位、手眼标定、碰撞几何或夹爪阈值
已完成验证。倾斜 Mock 配置尚未切换为桌面默认配置；后续真机测试仍需用户明确确认。
