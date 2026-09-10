# 改动记录：ROS 1 → ROS 2 移植 + Gazebo 仿真

- 原始包（**未改动**，作为对照基线）：`../../S3-60 Eco 2024/S3-60 Eco 2024`（ROS 1 catkin）
- 本包位置：`ros2_ws/src/s3_60_eco_2024`（ROS 2 Humble，ament_cmake）

记录分三批：**1–4** 是移植到 ROS 2（2026-09-10 19 时），**5–7** 是接入 Gazebo
Classic + ros2_control（同日 21 时），**8–13** 是跑通仿真时排查出来的修正与新增
（同日 23 时）。

## 1. 包名与网格路径

包名 `S3-60 Eco 2024` → `s3_60_eco_2024`。ROS 2 包名不允许大写和空格，原
`package://S3-60 Eco 2024/...` 在 ROS 2 里解析不了；URDF 里的机器人名同步改名，
网格路径改为 `package://s3_60_eco_2024/meshes/...`。

## 2. 关节限位

SolidWorks 导出的是全零占位值（`lower=0 upper=0 effort=0 velocity=0`，RViz 滑条
拖不动）。改为控制器配置里的实际值：

- 位置：J1/J2/J4/J5/J6 ±200°，J3 ±163°（来自 `base.yaml` 的 `jointPositionLimit`）
- 速度：J1–J3 150°/s，J4–J6 180°/s（来自 `S3-60-PRO.yaml` 的 `jointMaxVelocity`）
- 力矩：80/100/80/50/50/50 N·m（来自 `base.yaml` 的 `outLimitTrqTrack`）

这些是控制器软件里的安全限位，不等于机械极限。

## 3. RViz 显示

`launch/display.launch.py` 改为 ROS 2 Python launch（`robot_state_publisher` +
`joint_state_publisher_gui` + `rviz2`），并新增 `rviz/display.rviz`（原包里
`display.launch` 引用的 `urdf.rviz` 实际不存在）。

## 4. 未移植的部分

原 `gazebo.launch` 是 ROS 1 写法，且 URDF 没有 transmission/controller 配置，
当时未移植（后来按 5–7 条重新实现）。

## 5. Gazebo Classic + ros2_control

- `urdf/s3_60_eco_2024.gazebo.urdf.xacro`：在原始 URDF 上叠一层——`world`
  固定关节（把机械臂焊死在地上）、`ros2_control` 系统（六个关节：位置指令 +
  位置/速度状态）、`gazebo_ros2_control` 插件。
- `config/controllers.yaml`：`joint_state_broadcaster` +
  `joint_trajectory_controller`（位置指令、闭环校验、`hold_joints: true`）。
- `launch/gazebo.launch.py`：起 Gazebo → 生成模型 → 等模型进世界后再 spawn
  两个控制器。

## 6. SolidWorks 导出模型在 Gazebo 里的三个坑

- **惯量**：导出值里有 0 和负数的主惯量，ODE 求解器无法处理（表现为机械臂完全
  不理指令）。启动时把主惯量下限抬到 `1e-3 × 质量`，其余元素不变。
- **`package://` 路径**：Gazebo Classic 不解析它，网格打不开时整个模型（含插件）
  都不会加载；启动时统一改写成 `file://`。
- **XML 注释**：`gazebo_ros2_control` 把 `robot_description` 当参数覆盖规则传给
  controller manager，rcl 按 YAML 解析，注释里的 `: ` 会解析失败；所以在给
  `robot_state_publisher` 之前把注释剥掉。

## 7. 新增演示/自检脚本

`scripts/demo_trajectory.py`（`ros2 run s3_60_eco_2024 demo_trajectory.py`）：
读 `/joint_states` → 发 JointTrajectory → 打印跟踪误差，跟不上则非零退出。
`CMakeLists.txt` 加 `install(PROGRAMS ...)`，`package.xml` 补 `rclpy`、
`control_msgs`、`trajectory_msgs`、`sensor_msgs` 依赖。

## 8. 修正：基座插入地面导致机械臂"乱飞"

**现象**：启动后机械臂被甩到一个固定姿态，关节速度到几十 rad/s，位置看着却像
没动（采样混叠）。

**根因**：导出的 `base_link.STL` 比 `base_link` 原点低 15 mm，按 `z=0` 生成时基座
埋进地面，接触求解器把能量注入整条机械臂。

**修正**：`launch/gazebo.launch.py` 新增 `z` 参数，默认 `0.016`（留 1 mm 间隙）。

## 9. 修正：机械臂"软腿"下垂

**原因**：① 从模型生成到控制器激活之间有约 0.7 s 没有东西保持关节；
② `joint_trajectory_controller` 在轨迹结束、目标为静止时不再持续输出保持力矩。

**修正**：xacro 新增 `gravity` 参数（默认 `false`），给 `link1`~`link6` 打
`<gazebo><gravity>false</gravity></gazebo>`，像真机伺服抱闸一样停在指令位置。
需要动力学仿真时用 `ros2 launch ... gazebo.launch.py gravity:=true`。

## 10. 试过又撤回：关节 PID 增益

给六个关节配过 `pid_gains`（`kp=1000 / ki=0 / kd=1`），实测比直接写位置指令更差
（机械臂卡在地面姿态、速度抖动）。已从 `controllers.yaml` 删除，只在注释里记录
原因，避免以后有人照着乱调。

## 11. 清理

删除 `launch/__pycache__`（源码目录和 install 目录各一份，被
`install(DIRECTORY)` 一起装进包里了）以及测试时生成的 `scripts/__pycache__`；
已在 `.gitignore` 中忽略。

## 12. 文档

`README.md` 增加 Gazebo 运行方法、launch 参数表、三个坑的说明、"关于重力"、
"关于生成高度"、"已验证"、"未包含/待做"；截图 `docs/gazebo-run.png`。

## 13. 已知边界

- 只有位置指令接口，**没有**力矩接口，不能做力控/接触抓取；
- 默认关了机械臂重力，抓取类物理仿真需要自己补重力补偿；
- 没装 MoveIt，没有 SRDF、没有逆解与避障规划；
- Gazebo Classic 已于 2025-01 停止维护，长期建议迁到新版 Gazebo（gz sim）。

## 验证结果（2026-09-10 实测）

- 启动后不发指令，六个关节 40 s 内稳定在 0（位置、速度都是 0.000）；
- `demo_trajectory.py` 走 `[0.4, -0.5, 0.7, 0, 0.5, 0]`，跟踪误差 0.00°；
- 轨迹结束后 15 s 仍停在目标位；
- 打开重力（`gravity:=true`）时同样位置能站住，只有毫米级/千分弧度级漂移。

## 改动过程记录在哪

- 本仓库的 git 历史（从本次基线提交开始，之后每一步都有 diff、可回滚）；
- 原始包 `../../S3-60 Eco 2024/S3-60 Eco 2024` 保持不动，可逐文件对照；
- 每轮对话的工具记录：`~/.codex/sessions/2026/09/10/rollout-*.jsonl`
  （含每次文件补丁的完整文本和每条命令）；
- 构建日志：`ros2_ws/log/build_*/`；运行日志：`~/.ros/log/`。
