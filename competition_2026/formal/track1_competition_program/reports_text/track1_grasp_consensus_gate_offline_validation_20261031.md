# GraspNet 多帧共识门禁接入与离线验证

- 生成时间：`2026-10-31T12:47:31+08:00`
- 状态：`offline_gate_validated_no_motion_authorization`
- 范围：Task C/D 离线 dry-run 共识门禁

## 结论

共识门禁已接入离线 Task C/D dry-run。它在完整路线预检之后、打开夹爪和发布首个运动命令之前执行；任何报告缺失、损坏、安全标记异常、阈值弱化、簇成员/代表位姿篡改，或在线 TCP/夹爪旋转不匹配，都会拒绝整次抓取，不回退到单帧结果。

```text
实体运动未执行
机械臂/控制柜未连接
夹爪未连接
motion_authorized=false
quality_validated=false
```

## 已完成

- 配置层新增逐场景 `grasp_consensus`，默认关闭；启用时强制 Task C/D、`motion.dry_run=true`、`gripper.backend=mock`、`search.enabled=false`，并要求明确的 `robot_pose_frame`。
- factory 在启用时构造共识门禁并注入抓取 pipeline。
- pipeline 门禁位置已修正到固定点位、自适应和 box-relative 全部规划分支之后，因此所有 Task C/D 路线都会经过门禁。
- 报告解析 fail-closed：校验安全标志、schema、阈值方向和上限、簇计数、成员位姿、代表位姿、最大成对平移/旋转以及在线规划位姿绑定。
- 成功事件记录 cluster ID、代表候选、独立帧数、位姿误差、报告 SHA-256、质量标志和 `motion_authorized=false`。
- `config/track1_mock.yaml` 与正式 38 类 Task C/D 安全配置均加入禁用的门禁模板，防止误默认启用。
- README 说明报告必须与当前场景/物体对应；离线位姿稳定不是抓取成功率，也不是实机授权。

## 正式参考报告绑定校验

- 报告：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_graspnet_aperture_multiframe_20261031/diagnostics/consensus/consensus.json`
- SHA-256：`39a58f5bd2d27eae27e3f82929b32608a4a2a4656f0fe222ce4e522958f43db8`
- 簇：`consensus_001`
- 代表候选：`frame_19/graspnet_00050`
- 独立帧：`5`
- 最大成对 TCP 平移：`0.021385 m`
- 最大成对夹爪对称旋转：`0.000000 deg`
- 代表位姿运行门禁结果：`accepted=true`
- TCP 误差：`0.000000 m`
- 夹爪旋转误差：`0.000000 deg`
- 报告质量标志：`quality_validated=false`
- 授权标志：`motion_authorized=false`

该验证只证明门禁能读取并绑定现有离线报告，不代表实机抓取成功，也不代表 38 类任务都有有效共识证据。

## 测试结果

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:${PYTHONPATH:-}" python3 -m pytest -q
556 passed in 47.95s
```

新增门禁测试覆盖：

- 缺报告、损坏 JSON、安全标记异常；
- 报告最小帧数/平移/旋转阈值被放宽，以及非有限阈值；
- 簇成员和代表位姿被篡改；
- 在线 TCP 平移和夹爪对称旋转超限；
- 匹配代表位姿时接受但绝不授权运动；
- 自适应 Task C 在开夹爪和首动作前拒绝；
- factory 确实构造并注入门禁，接受后才进入原有 mock/dry-run 状态机。

## 当前限制

- 现有共识报告只对应一个离线场景，不能作为 38 类 Task C/D 的通用现场证据。
- 尚无针对每个真实现场物体的逐场景报告、在线规划位姿一致性和夹爪阈值验证。
- 在获得用户明确确认且完成现场坐标、限位、手眼标定、夹爪阈值和安全检查前，不得启用实体运动。
