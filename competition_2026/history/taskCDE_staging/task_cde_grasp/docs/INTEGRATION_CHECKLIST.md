# 真机接入清单

## 1. 相机安装

- 固定相机：相机与机器人底座相对位置必须固定。
- 腕部相机：确认相机线缆、法兰安装件和末端负载。
- 记录相机型号、分辨率、帧率、序列号和安装角度。

## 2. 手眼标定

固定相机输出 `T_base_camera`，腕部相机输出 `T_flange_camera`。

至少使用 9 个空间分布良好的标定点，记录：

- 每点机器人末端 `x/y/z/rx/ry/rz`
- 图像像素 `u/v`
- 有效深度
- 拟合残差与最大误差

把结果写入 `pose.extrinsics_file`，不要使用示例单位矩阵直接上真机。

## 3. 检测器输出契约

每个检测至少提供：

```text
label
confidence
bbox = [x0, y0, x1, y1]
depth_m                  # 可选但真机抓取建议必须有
camera_xyz               # 3D 点，单位 m
graspability             # 0..1，Task D 未知目标必须提供
instance_id              # Task E 同类多物体时建议提供
```

Task E 的 `order_file` 使用探测器的实际标签。若要区分两个同类别实例，写 `label#instance_id`。

## 4. 夹爪接入

实现或包装以下接口：

```python
state: GripperState
open() -> None
close() -> None
close_backend() -> None
```

现有 `taskA_grasp` 已有多种夹爪后端，可在 `gripper.backend: taska` 后面复用。当前六针夹爪的 RS485 通路仍需实物验证。

## 5. 机器人接入

`robotmove` 后端约定：

- 订阅 `/RobotInfo`
- 发布 `/RobotMove`
- `/RobotMove.joint_names = [x, y, z, rx, ry, rz]`
- 只允许在 `AutoReady` 且无 fault 时运动

上真机前必须确认单位、姿态欧拉角约定、最大线速度和控制器坐标系。

## 6. 夹持确认

优先级建议：

1. 夹爪反馈位置到位且电机电流达到阈值。
2. 闭合宽度与目标尺寸一致。
3. 抬起 1-2 cm 后用腕部相机复查物体是否随动。
4. 最后才接受“命令已发送”作为成功。

## 7. 比赛前验收

- C：随机 20 件，无序取放。
- D：15 件已知物体 + 5 件未见物体。
- E：按公布清单严格顺序执行，验证重复类别和遗漏处理。
- 对贴边、遮挡、堆叠、反光/透明物体分别记录失败原因。
- 统计每次抓取时间、成功率和滑落次数。
