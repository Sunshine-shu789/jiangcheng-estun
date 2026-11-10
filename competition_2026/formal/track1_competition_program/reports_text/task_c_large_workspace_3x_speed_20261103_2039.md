# Task C 扩大软件工作区并将主运动提高 3 倍

生成时间：2026-11-03 20:39（Asia/Shanghai）  
状态：离线配置、IK 核算和全量回归完成；未发送任何实体机械臂运动命令。

## 正式修改

只修改主 Task C 正式配置
`config/track1_task_c_v11_official_20261102.yaml`：

```yaml
motion:
  workspace_min: [-0.80, -0.80, 0.02]
  workspace_max: [0.80, 0.80, 0.80]
  workspace_margin: 0.01
  speed_mps: 0.09
  joint_speed_rad_s: 1.8
  joint_speed_safety_limit_rad_s: 1.8
  grasp_safety:
    maximum_horizontal_reach_m: 0.82
servo:
  speed_mps: 0.01
```

- 主笛卡尔速度由 `0.03 m/s` 提高到 `0.09 m/s`，即 3 倍。
- 主关节速度由 `0.6 rad/s` 提高到 `1.8 rad/s`，即 3 倍。
- 视觉伺服速度独立保持 `0.01 m/s`，未同步提速。
- Task C mustard-single、Task D 和 C/D 兼容配置保持原速度与原工作区，避免把
  主 Task C 的提速意外带入 Task D。
- 工具倾角、抓取/路线 Z 包络、关节限位、分支跳变、奇异值、关节步长和
  离散加速度门禁未放宽。

## 六轴工作区的实际含义

六个自由度表示姿态调整能力，不表示该型号拥有很大的物理半径。控制器
modified-DH 模型核算结果：

```text
全姿态搜索最大水平法兰半径                 0.644650 m
法兰 Z=0.245103 m 预抓取高度最大水平半径   0.640993 m
```

2026-11-03 被拒绝的骰子点：

```text
jaw_tip = (-0.364879265, 0.716331185, 0.000103) m
flange pregrasp = (-0.364879265, 0.716331185, 0.245103) m
水平半径 = 0.803907485 m
```

该骰子点比预抓取高度的模型最大水平半径远约 `0.162914 m`。使用现场观察关节
姿态和两个等价短边腕部 yaw 分别做 IK，两个姿态均失败，最佳末端位置残差约
`0.167 m`，与几何超出量一致。因此：

- 旧 `0.65 m` 工作空间边界确实会过早拒绝一些点，现已扩大；
- 但 `0.804 m` 的骰子点本身超出机械臂物理可达范围；
- 不应继续扩大软件边界来伪造可达性，需将收纳盒或机械臂底座调整到可达范围。
  按当前几何，骰子至少需要向机械臂靠近约 `0.17 m` 才能进入预抓取 IK 的可行
  区域，现场应以实际重新观察和完整路线预检结果为准。

## 离线回归

```text
正式 Task C/C/D 参数专项：4 passed
速度上限与专项联合：      21 passed
全量回归：                767 passed in 75.40s
```

关键测试已更新为：

- 主 Task C：`speed_mps=0.09`、`joint_speed=1.8 rad/s`、工作区 X/Y=`±0.80 m`、
  水平包络 `0.82 m`；
- 其他正式配置：仍保持 `speed_mps=0.03`；
- 全部正式配置：`servo.speed_mps=0.01`；
- 速度调节 UI 的 Task C 100% 上限读取新的 `0.09/1.8`，Task D 仍为 `0.03/0.6`。

全量测试命令：

```bash
/media/sunshine/新加卷1/埃斯顿比赛_2026/赛道一比赛程序/bin/ensure_runtime
source /media/sunshine/ESTUN_RUNTIME/ros2_ws/install/setup.bash
cd /media/sunshine/新加卷1/埃斯顿比赛_2026/赛道一比赛程序/app/track1_grasp
PYTHONPATH=".:${PYTHONPATH:-}" python3 -m pytest -q -p no:cacheprovider
```

## 校验值

```text
5acd6096395dea454fd5080e5d0c1c4aae85294a629adfe4432353680abf5023  config/track1_task_c_v11_official_20261102.yaml
0dd9ffbcdd8ffb145348e8885d7e5aede229b35ea22feec19a10a3daa4370ee0  app/track1_grasp/test/test_task_cd_grasp_safety.py
79acf02a6cf60a523f71b194eb9fcb843d68ef3a6ee2994e61c731e10db1b4f7  app/track1_grasp/test/test_speed_limit.py
3cbf04a84a6ac604b13c4734083518c6b7becc79a0cd14112690e0034642bee3  使用说明.txt
2c84b67401593d71000f660fe67874b56ed8ed5c28322524caec6af293eebaf4  app/track1_grasp/README.md
```

修改前备份：

```text
backups/task_c_large_workspace_speed_20261103_203321
```

备份索引：

```text
state/last_task_c_large_workspace_speed_backup.txt
```

## 安全状态

- 本轮未执行 `run_task_c`，未启动 ROS 运动节点，未发布 `/RobotMove`。
- 最新 `0.09 m/s`、`1.8 rad/s` 仅完成离线配置与回归，尚未做实体运动验收。
- 下一次实体运行仍需用户明确授权，并应现场确认急停、无人进入工作区、控制器
  `AutoReady`、无故障和路线预检通过。
- Task B 继续使用 `motion.dry_run: true` 和 `gripper.backend: mock`。
