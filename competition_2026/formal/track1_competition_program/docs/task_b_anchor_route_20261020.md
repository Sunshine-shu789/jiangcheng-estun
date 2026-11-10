# Task B 五点位拾放：观察锚点、路线预检与搜索安全修正

> 后续校准更新（2026-10-20）：2 号点已通过
> `point_grasp_xy_from_observation: true` 改为从示教观察列垂直下降，且
> 2～5 号点抓取高度均已提高 5 mm。最新参数与预检结果见
> `docs/task_b_point2_vertical_grasp_20261020.md`。

日期：2026-10-20  
正式目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序`

## 1. 结论

2 号点出现的近奇异/路线预检问题不是 2 号点独有。原搜索配置使用
`0、±100 mm X、±100 mm Y` 五个 Cartesian 观察视图；这些外圈视图在
2～5 号点均可能进入雅可比条件较差的区域：

- 2 号点：主要在 `+Y` 以及左侧偏移区域出现失败。
- 3、4 号点：`+100 mm Y` 备用视图失败。
- 5 号点：`+60～80 mm Y` 区域出现失败。
- 日志中的 `target=2` 是路线航点索引，不是“2 号点”。

现已把正式配置的 Cartesian 搜索视图改为经验证安全的三视图：

```yaml
observation_offsets_xy_m:
  - [0.00, 0.00]
  - [0.08, 0.00]
  - [0.00, -0.08]
```

雅可比阈值仍为 `0.010000`，没有为了通过检查而降低阈值。

## 2. 观察锚点和六段拾放路线

每个 Task B 点位使用现场示教的 `point_observation_joints_deg`。程序先做
FK，得到该点的观察位 Cartesian 锚点；以下四段高位航点统一使用该锚点：

1. `pregrasp`
2. `lift`
3. `transfer`
4. `retreat`

`grasp` 和 `release` 仍使用视觉 XY、该点标定抓取高度和抓取姿态。物体原位
释放，不跨点搬运。这样避免让 IK 在大范围平移和姿态变化中穿过条件很差的腕部
构型。

此前已核对六段完整路线 `pregrasp → grasp → lift → transfer → release →
retreat`：

| 点位 | min_sigma | 航点数 | 最大关节步长 |
|---|---:|---:|---:|
| pick_2 | 0.017237 | 123 | 0.749° |
| pick_3 | 0.057409 | 152 | 0.558° |
| pick_4 | 0.060262 | 196 | 0.646° |
| pick_5 | 0.049869 | 112 | 0.459° |

观察位关节插值切换也已核对：

| 移动 | min_sigma |
|---|---:|
| initial → pick_2 | 0.017237 |
| pick_2 → pick_3 | 0.017237 |
| pick_3 → pick_4 | 0.057409 |
| pick_4 → pick_5 | 0.049869 |
| pick_5 → initial | 0.029609 |

## 3. 视觉 XY 偏移边界扫描

扫描范围：`dx/dy = -80…80 mm`，只取半径 `≤80 mm`，步长 `20 mm`；
检查完整拾放路线和笛卡尔预检，不是只看单个 IK 点。

| 点位 | 通过/总数 | 最差通过 min_sigma | 失败特征 |
|---|---:|---:|---|
| pick_2 | 30/49 | 0.011683 | 19 个失败点，`+Y` 和左侧危险区 |
| pick_3 | 49/49 | 0.021101 | 80 mm 扫描范围内全部通过 |
| pick_4 | 49/49 | 0.047130 | 80 mm 扫描范围内全部通过 |
| pick_5 | 43/49 | 0.018218 | 6 个失败点，`Y=+60…80 mm` |

如果香蕉落在预检拒绝区域，程序会在动作前停止并输出点位、视觉 XY、失败航点
和搜索视图信息。正确处理是移动香蕉到点位安全内区或重新示教，不是降低
`cartesian_min_jacobian_singular_value`。

## 4. 正式配置复验结果

使用两份实际配置和三视图搜索范围重新运行只读预检，全部 PASS：

### `track1_task_b_resume2_20261020_live.yaml`（2～5 号点）

| 点位 | min_sigma | 最大关节步长 | 搜索航点数 |
|---|---:|---:|---:|
| pick_2 | 0.017237 | 0.749° | 98 |
| pick_3 | 0.057409 | 0.664° | 66 |
| pick_4 | 0.054508 | 0.750° | 85 |
| pick_5 | 0.044059 | 0.745° | 101 |

### `track1_task_b_20261020.yaml`（完整 1～5 号点）

| 点位 | min_sigma | 最大关节步长 | 搜索航点数 |
|---|---:|---:|---:|
| pick_1 | 0.044380 | 0.499° | 135 |
| pick_2 | 0.017237 | 0.499° | 147 |
| pick_3 | 0.057409 | 0.442° | 98 |
| pick_4 | 0.054507 | 0.497° | 127 |
| pick_5 | 0.044059 | 0.499° | 150 |

结果文件：

```text
logs/task_b_search_route_check.json
logs/task_b_search_route_check_actual_config_20261020_034732.json
logs/task_b_offset_scan_20261020.json
```

## 5. 运行时保护

- 在第一次 Cartesian 搜索运动前，完整预检所有有效搜索视图。
- 预检失败时报告 `failed_search_view=N/M`、该视图 XYZ、参考 XY 和失败航点。
- 预检失败时不发送机械臂或夹爪动作。
- 全路线预检失败时，在抓取动作前拒绝运行。
- 每个点抓取并原位放回后，仍须重新视觉确认下一个点的香蕉，才进入下一次抓取。
- 所有点完成后回到配置的初始位置。
- 阈值保持 `0.01`，不得为了让危险路线通过而降低。

## 6. 代码与回归测试

主要代码：

```text
app/track1_grasp/track1_grasp/planner.py
app/track1_grasp/track1_grasp/pipeline.py
```

正式排查工具：

```text
app/track1_grasp/track1_grasp/tools/check_task_b_search_routes_offsets.py
app/track1_grasp/track1_grasp/tools/task_b_offset_scan.py
```

修改前备份：

```text
backups/task_b_safe_search_views_20261020_034724/
```

全量测试结果：

```text
191 passed in 12.48s
```

重新运行正式搜索路线预检：

```bash
cd /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序
PYTHONPATH="$PWD/app/track1_grasp${PYTHONPATH:+:$PYTHONPATH}" \
python3 app/track1_grasp/track1_grasp/tools/check_task_b_search_routes_offsets.py
```

该工具创建只读运动预检对象，不发布 `/RobotMove`，不控制夹爪。

## 7. 本轮动作声明

本轮只进行了配置修改、离线 IK/工作空间/限位/连续性/近奇异预检和单元测试。
**没有发送任何机械臂运动或夹爪动作命令，没有重启现场 ROS 节点。**

`bin/run_task_b` 当前仍默认使用：

```text
config/track1_task_b_20261020.yaml
motion.dry_run: true
gripper.backend: mock
```

在没有新的“允许实机运行”明确授权前，不应改用 live 配置。获得授权后，先确认
现场安全、香蕉位置、急停和点位，再执行完整五点位连续实机测试。
