# 规划 → 执行闭环验证（2026-09-11）

`plan_execute_demo.py` 把"给定目标 → MoveIt 规划 → joint_trajectory_controller →
Gazebo 里真的走一遍"整条链路跑通，执行完再读 `/joint_states` 量跟踪误差。
这一页记录实测结果和过程中修掉的两个配置问题。

## 实测结果（Humble + MoveIt 2.5.9 + Gazebo 11，`gui:=false`）

| 目标 | 目标位姿 / 关节角 | IK 姿态 | 轨迹 | 实际移动量 | 跟踪误差 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| SRDF `home` | 全零 | - | 16 点 / 1.42 s | 40.3° | 0.55° | 通过 |
| SRDF `demo` | [0.4, -0.5, 0.7, 0, 0.5, 0] | - | 16 点 / 1.42 s | 40.3° | 0.46° | 通过 |
| 物品 #3 某物品 | (+0.00, +0.33, +0.37) | tilt0_yaw0 | 28 点 / 2.69 s | 118.9° | 0.36° | 通过 |
| 物品 #7 某物品 | (-0.16, +0.53, +0.47) | tilt60_yaw180 | 34 点 / 3.29 s | 156.2° | 0.52° | 通过 |

物品 #7 在远排（水平距离 0.55 m），默认俯视姿态解不出来，脚本自动在 32 个候选姿态
里挑中了 `tilt60_yaw180`，和 `docs/track1_reachability_all.md` 里 IK 检查的结论一致。

跟踪误差 0.3°~0.6° 是 Gazebo 位置控制器的稳态误差（`demo_trajectory.py` 单独发轨迹时
也是这个量级），不是规划问题。

## 过程中修掉的两个配置问题

这两个问题都只在"真去执行"的时候才暴露：只做 IK/工作空间检查时看不出来。

### 1. 规划流水线参数没有挂到命名空间下

`ompl_planning.yaml` 原来是平铺加载的（参数直接挂在 `/move_group` 根上，
即 `planning_plugins`、`request_adapters`……），但 MoveIt 2 读的是流水线命名空间里的
`ompl.planning_plugin`、`ompl.request_adapters`、`ompl.arm.*`。

结果 move_group 找不到 OMPL 配置，按"有多个插件可用就随便挑一个"处理，实测挑中了
CHOMP，日志里能看到：

```
Multiple planning plugins available. You should specify the '~planning_plugin' parameter.
Using 'chomp_interface/CHOMPPlanner' for now.
MoveGroup context using planning plugin chomp_interface/CHOMPPlanner
```

修法：launch 里改成 `{"ompl": load_yaml("config/ompl_planning.yaml")}`，配置文件也改成
MoveIt 2 的键名（`planning_plugin` 单数 + `planner_configs` 用 `RRTConnect`/`RRTstar`/
`BKPIECE` 这组官方名字）。

### 2. 缺时间参数化，控制器直接拒收

适配器名字写成了 `default_planning_request_adapters/...`（少了一个 `planner` 的
`request`），而且 `AddTimeOptimalParameterization` 被放在 `response_adapters` 段里
（MoveIt 2 没有这一段）。结果 OMPL 给出的几何路径没有被加时间戳，轨迹里所有
`time_from_start` 都是 0：

```
[joint_trajectory_controller]: Time between points 0 and 1 is not strictly increasing,
it is 0.000000 and 0.000000 respectively
[moveit_ros.trajectory_execution_manager]: Completed trajectory execution with status ABORTED
```

MoveIt 侧的表象是 `MoveGroup` 返回 `-4`（`CONTROL_FAILED`），容易误判成控制器坏或
机械臂没动。

修法：把 `AddTimeOptimalParameterization` 挪进 `request_adapters`，用正确的前缀
`default_planner_request_adapters/`，并去掉不存在的 `response_adapters`。

## 复现

```bash
# 1) 仿真 + 控制器
ros2 launch s3_60_eco_2024 gazebo.launch.py gui:=false
# 2) MoveIt
ros2 launch s3_60_eco_2024_moveit_config move_group.launch.py
# 3) 闭环自检（退出码非 0 表示失败，可直接进 CI/脚本）
ros2 run s3_60_eco_2024_moveit_config plan_execute_demo.py --named demo
ros2 run s3_60_eco_2024_moveit_config load_track1_items.py --subset 1
ros2 run s3_60_eco_2024_moveit_config plan_execute_demo.py --grasp-item 3
ros2 run s3_60_eco_2024_moveit_config plan_execute_demo.py --grasp-item 7
```

`--plan-only` 只规划不执行；`--tilt-deg/--yaw-deg` 可以手动指定接近姿态；
`--velocity-scaling/--accel-scaling` 控制速度（默认 0.3，改大更接近比赛节拍）。

## 边界

- 链路终点是"法兰到达抓取位姿上方"，**没有爪子、没有合爪、没有把物品拎起来**；
  抓取动作要等简化两指夹爪 + 假抓取（接触后把物体固定到 `link6`）做出来再验证；
- 只对上面 4 个目标做过实测，40 件物品没有逐件跑规划+执行（IK 层的结果见
  `track1_reachability_all.md`）；
- 规划出的轨迹只避开规划场景里的碰撞体（桌面 + 物品包围盒），和真实物体的
  形状差异还是有的。
