# GraspNet 抓取后处理借鉴与离线验证结果

日期：2026-10-29  
状态：离线只读验证通过；**未接入实时 ROS 抓取链路，未授权真机运动**

## 1. 已借鉴的 GraspNet 机制

已参考 `graspnet-baseline/utils/collision_detector.py`、GraspNet 的 17 维抓取表示和 `graspnetAPI` 的 NMS 做法，在当前工程实现：

1. 解析 GraspNet 17 维数组：`score, width, height, depth, rotation(9), translation(3), object_id`。
2. 把候选转换到局部夹爪坐标系：局部 `+X` 为接近方向，`+Y` 为夹口开合方向，`+Z` 为夹爪高度方向。
3. 对左指、右指、掌部和接近走廊做体素化障碍碰撞估计。
4. 检查夹腔是否为空、左右指是否形成对向接触、目标物宽度是否适配、最大开口/保留量是否足够。
5. 增加目标穿透检查，避免只用“有接触”而放过夹指切入目标物的情况。
6. 使用 3D 平移 + 旋转 NMS，按质量分数去重并保留高分候选。
7. 输出可解释的质量分、碰撞 IoU、接触点数量、测量宽度、开口余量、清除间距和拒绝原因。

## 2. 代码与入口

- 评分器：`app/track1_grasp/track1_grasp/grasp_quality.py`
- 离线 CLI：`app/track1_grasp/track1_grasp/tools/score_grasp_quality.py`
- 估算配置：`app/track1_grasp/config/grasp_quality_ms42dc_estimated.yaml`
- 单测：`app/track1_grasp/test/test_grasp_quality.py`
- 演示报告：`reports/grasp_quality_demo_20261029.json`

运行方式：

```bash
cd /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/app/track1_grasp
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \
  -m track1_grasp.tools.score_grasp_quality \
  --config config/grasp_quality_ms42dc_estimated.yaml \
  --demo --output ../../reports/grasp_quality_demo_20261029.json
```

## 3. 验证结果

- `test/test_grasp_quality.py`：`11 passed`
- `bin/ensure_runtime`：通过；已挂载正式 ext4 ROS 运行镜像
- 加载 ROS 运行环境后的全量回归：`448 passed`（2026-10-29）
- 全量回归只加载 Python 包和运行依赖，没有启动 ROS 节点、夹爪或机械臂；没有发送任何运动/夹爪命令。

内置合成场景结果：

- 5 个候选
- 2 个通过几何/接触检查
- 3 个被拒绝
- NMS 后选中 1 个候选
- 选中候选质量分：`0.9663`
- 近重复候选被 NMS 抑制
- `demo_collision`：左指障碍碰撞
- `demo_narrow`：目标穿透
- `demo_empty`：夹腔为空且左右无接触

这些是合成几何回归，不代表真实抓取成功率。

## 4. 当前安全边界

`config/grasp_quality_ms42dc_estimated.yaml` 明确标记：

```yaml
validated: false
geometry_source: estimated_not_cad_or_measured
```

当前 MS42DC 尺寸是估算值，不能用于授权真机动作。实时抓取链路尚未强行接入评分器，因为当前候选对象没有可靠的逐候选目标点云/障碍点云；强行接入会制造伪精度。

## 5. 接入真机还缺什么

按优先级直接列出：

1. **MS42DC CAD 或实测几何**：最大开口、左右指内侧面、指尖长度、指厚、指高、掌部尺寸、接近走廊尺寸。
2. **开合命令到实际几何的映射**：夹爪指令/角度到实际夹口宽度和指尖位置的标定曲线。
3. **完整坐标链**：指尖、掌部、TCP、法兰、相机、基座之间的变换；至少给出评分用的 `T_pointcloud_gripper`。
4. **逐候选点云**：目标物点云、桌面/障碍点云、目标分割掩码或 object id；所有点必须与候选处于同一坐标系。
5. **真机抓取标签**：成功/失败、滑落、碰撞、空抓记录，用于校准质量权重、接触阈值和开口阈值。
6. **现场静止场景数据**：先录制一组 RGB-D、候选位姿和人工确认结果，做离线回放；这一步通过后再做 dry-run。

## 6. 推荐下一步

收到上述 CAD/标定和一组现场点云后：

1. 替换估算几何，重新跑离线回放。
2. 用真实成功/失败标签调阈值，确认 `validated: true` 的准入条件。
3. 保持 `motion.dry_run: true`、`gripper.backend: mock` 做 dry-run。
4. 只有在现场复核、限位、手眼标定和夹爪阈值全部通过后，才讨论受控真机接入。

在此之前，本次交付的定位是：**可用于离线候选筛选和质量排序，不可作为真机抓取授权。**
