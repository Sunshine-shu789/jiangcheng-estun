# 江城埃斯顿 · S3-60 Eco 2024（ROS 2 Humble）

埃斯顿酷卓 S3-60 Eco 2024 六轴机械臂的 ROS 2 移植、Gazebo Classic 仿真、
MoveIt 2 规划，以及第三届“埃斯顿”大学生机器人大赛赛道一 40 件物品的可达性与
“IK → 规划 → 执行”验证。

## 包里有什么

| 包 | 说明 |
| --- | --- |
| `s3_60_eco_2024` | 机器人 URDF、Gazebo Classic 仿真（`ros2_control` 轨迹控制器）、RViz 显示 |
| `s3_60_eco_2024_moveit_config` | MoveIt 2 配置（OMPL + KDL）、赛道一 40 件物品的碰撞模型、可达性/规划执行脚本与报告 |

## 怎么跑

本仓库就是 ROS 2 工作区的 `src` 目录，依赖 ROS 2 Humble、Gazebo Classic 和 MoveIt 2：

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone <本仓库地址> .
cd ..
colcon build
source install/setup.bash
```

三个终端：

```bash
# 1) 仿真 + 控制器（常驻）
ros2 launch s3_60_eco_2024 gazebo.launch.py gui:=true

# 2) MoveIt（常驻；rviz:=true 同时打开带 MotionPlanning 显示的 RViz）
ros2 launch s3_60_eco_2024_moveit_config move_group.launch.py rviz:=true

# 3) 把 40 件物品摆成 2x2 展台加载进规划场景（一次性）
ros2 run s3_60_eco_2024_moveit_config load_track1_items.py --all
```

其他常用命令：

```bash
# 逐件检查“从上方抓”的位姿能否解出 IK，输出 4 个 Subset 的汇总报告
ros2 run s3_60_eco_2024_moveit_config check_reachability.py --all

# 逐件跑 IK → OMPL 规划 → 控制器执行，量跟踪误差（约 8 分钟）
ros2 run s3_60_eco_2024_moveit_config plan_execute_demo.py --sweep-all

# 扫描工作空间，输出可达区域
ros2 run s3_60_eco_2024_moveit_config workspace_map.py --z 0.35
```

## 实测结果（2026-09-11，Humble + MoveIt 2.5.9）

- 赛道一 40 件物品可达性：近排 20/20，远排 7/20，合计 **27/40**；
- 逐件 “IK → OMPL 规划 → 控制器执行”：**27/40** 走通，凡是 IK 有解的全部
  规划成功并在 Gazebo 里到位，跟踪误差 0.33°~0.60°；
- 明细报告见 `s3_60_eco_2024_moveit_config/docs/`。

## 说明与限制

- 物品模型是按比赛清单做的**包围盒近似**（box/cylinder/sphere），尺寸多为估计值、
  未核对实物，只用于碰撞检查与可达性分析；**不涉及抓取本身**（没有爪指、摩擦模型）。
- URDF 和 meshes 来自厂家 SolidWorks 导出，版权归埃斯顿酷卓所有，此处仅用于
  兼容性开发；请勿用于商业用途。
- 仿真用 Gazebo Classic（Gazebo 11），未在 Gazebo Sim / Ignition 上验证。
