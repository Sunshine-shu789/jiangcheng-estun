# s3_60_eco_2024_moveit_config

埃斯顿酷卓 S3-60 Eco 2024 的 MoveIt 2 配置，外加赛道一 40 件物品的简化碰撞模型。
解决的是"给一个物品位姿，机械臂能不能解出关节角、能不能规划出不撞的轨迹"，
**不涉及抓取本身**（没有爪子、没有摩擦，见文末限制）。

## 依赖

```bash
sudo apt install -y ros-humble-moveit
```

## 怎么跑

```bash
cd <工作区>/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select s3_60_eco_2024 s3_60_eco_2024_moveit_config
source install/setup.bash
```

三个终端（或后台）：

```bash
# 1) 仿真 + 控制器
ros2 launch s3_60_eco_2024 gazebo.launch.py gui:=true

# 2) MoveIt
ros2 launch s3_60_eco_2024_moveit_config move_group.launch.py

# 3) 把某个 Subset 的物品摆进规划场景，然后检查可达性
ros2 run s3_60_eco_2024_moveit_config load_track1_items.py --subset 1
ros2 run s3_60_eco_2024_moveit_config check_reachability.py --subset 1
```

工作空间扫描（决定桌子摆哪、物品放哪）：

```bash
ros2 run s3_60_eco_2024_moveit_config workspace_map.py --z 0.35 --x-step 0.1 --y-step 0.05
```

## 目录

| 文件 | 说明 |
| --- | --- |
| `config/s3_60_eco_2024.srdf` | 规划组 `arm`（base_link→link6）、fixed 虚拟关节、`home`/`demo` 预设 |
| `config/kinematics.yaml` | KDL 逆解（`attempts: 20`、`timeout: 0.1`，见下面"坑") |
| `config/joint_limits.yaml` | 位置/速度/力矩来自 URDF；**加速度是估计值** |
| `config/ompl_planning.yaml` | RRTConnect / RRTstar / BKPIECE |
| `config/moveit_controllers.yaml` | 通过 action 调 Gazebo 的 `joint_trajectory_controller` |
| `config/track1_items.yaml` | 40 件物品（4 个 Subset）的包围尺寸 + 商品链接 + 桌子/摆放 |
| `scripts/load_track1_items.py` | 把物品（含桌面）加载进规划场景 |
| `scripts/check_reachability.py` | 逐件检查"从上方抓"的位姿能否解出 IK |
| `scripts/workspace_map.py` | 扫描 x-y 平面，输出可达区域图 |
| `launch/move_group.launch.py` | 起 move_group（`use_sim_time:=false` 可接真机） |
| `docs/track1_reachability.md` | Subset 1 的可达性结果 |
| `docs/workspace_map_z035.md` | z=0.35 m 平面的可达区域图 |

## 实测结论（2026-09-11，Humble + MoveIt 2.5.9）

1. **桌子近边不能超过 y≈0.22 m**。大臂（link2）在零位时前伸到 y≈0.21 m，桌子压上去
   机器人就"泡在桌子里"，任何带碰撞检查的 IK/规划全部失败（表现为可达率 1/10）。
   本配置用 y=0.26 m 作近边。
2. **能用俯视抓取的区域很窄**：z=0.35 m 平面上大约 y 0.26~0.50 m、|x| <= 0.25 m
   （见 `docs/workspace_map_z035.md`）。远一点的位置得靠 40°~60° 倾斜接近才够得到，
   正上方下不去。
3. **物品间距必须大于物品宽度**。把 10 件摆成 0.12 m 间距时，物品自己的碰撞体互相
   挤住，可达率反而从 7/10 掉到 5/10；拉开到 0.16 m 才恢复。真做抓取演练建议
   **一次只摆 5 件（一排）、间距 >=0.18 m**。
4. **KDL 参数很重要**：默认 `attempts: 3 / timeout: 0.05` 时 Subset 1 只有 1/10
   能解出来，改成 `20 / 0.1` 并给 32 个候选姿态后是 7/10。KDL 是局部求解器，
   配置保守会把"其实够得着"的目标误判成不可达。
5. Subset 1 结果：**近排 5/5 全部可达，远排（y=0.53）2/5**，合计 7/10。
   够不到的 3 件是某物品、某物品、某物品（水平距离 0.55~0.62 m，已在臂展边缘）。

## 限制与注意

- **没有爪子**：IK 目标是"法兰 link6 位于物品顶面 + 5 cm"，真装上爪后目标要再抬高
  一个爪长（`--approach-height` 可调），一般会让远一点的目标更容易够到。
- **尺寸多为估计**：`track1_items.yaml` 里每件都有 `source` 字段
  （`ycb近似` 11 件 / `估计` 18 件 / `待实测` 11 件），正式用之前按实物或商品页核对。
- **IK 成功 ≠ 轨迹能走**：IK 只说明那一瞬间的位姿可达，实际还要过 OMPL 规划与
  执行（目前还没做闭环验证）。
- 物品只是 MoveIt 规划场景里的碰撞体，**没有摆进 Gazebo**（Gazebo 里只有机器人）。

## 下一步

1. 用实测尺寸替换 `估计/待实测`；
2. 把"规划 + 执行"跑通：OMPL 规划 → `joint_trajectory_controller` → Gazebo 里真的动；
3. 需要演示抓取流程时，再加简化两指夹爪 + 假抓取（接触后把物体固定到 link6）；
4. 跑其余 3 个 Subset 的可达性，汇总成一张总表。
