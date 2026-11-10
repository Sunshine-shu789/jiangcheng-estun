# 赛道一 C/D 几何策略与 Task D D405 只读验收（2026-11-02）

## 1. 结论

本轮完成了 C/D 共用视觉与几何策略的验收工具补强、专项测试和真实 D405
只读复核。**离线代码与 fail-closed 审计链通过，但真实几何候选验收尚未通过。**

- 全量回归：`712 passed in 61.02s`。
- Task D 视觉验收工具专项：`9 passed in 1.33s`。
- 真实 D405：15/15 帧采集成功，65 条已知目标观测均生成物体位姿与几何证据。
- 几何策略：评估 65 条，接受 0 条；49 条支持面质量不足，16 条无可靠支持面。
- 未观察到未知物正样本，不能宣称 Task D 开放物体召回已通过。
- 本轮未发布 `/RobotMove`，未发送夹爪开合命令，也未授权实体运动。

## 2. 本轮修改

修改文件：

```text
app/track1_grasp/track1_grasp/tools/accept_task_d_vision.py
app/track1_grasp/test/test_accept_task_d_vision.py
app/track1_grasp/README.md
app/track1_grasp/docs/INTEGRATION_CHECKLIST.md
```

验收工具现在：

1. 对每条检测写入完整 `geometry_evidence`；
2. 直接调用运行时 `select_grasp_geometry()`，避免验收实现与抓取实现分叉；
3. 写入 accepted/reason/reason_category、候选模式、轴、置信度和证据摘要；
4. 聚合支持面、rim、sidewall、策略接受/拒绝和 track ID 指标；
5. 提供 `--require-geometry` 与 `--require-geometry-candidate` 两级门禁；
6. 仍然只构建视觉源，不启动运动桥、运动后端或夹爪后端。

新增专项测试覆盖：

- 支持面缺失时 fail-closed，仍写出完整 `summary.json`；
- 合格测量短边被接受为 `vertical_short_edge`；
- 同一 track 跨帧聚合正确；
- 已知和无几何证据的检测被准确归入 `geometry_evidence_missing`。

## 3. 真实 D405 只读结果

审计产物：

```text
logs/task_d_geometry_readonly_20261102_122705/
logs/task_d_geometry_readonly_20261102_122705.console.log
```

配置：

```text
config/track1_task_d_vision_geometry_readonly_20261102_121500.yaml
```

| 指标 | 结果 |
|---|---:|
| 采集帧 | 15/15 |
| 有检测帧 | 13/15 |
| 已知观测 | 65 |
| 未知观测 | 0 |
| 位姿观测 | 65 |
| 几何证据观测 | 65 |
| rim 原始证据 | 41 |
| sidewall 原始证据 | 36 |
| 支持面有效观测 | 49 |
| 支持面质量均值 | 0.2648 |
| 支持面质量最高值 | 0.5240 |
| 策略评估观测 | 65 |
| 策略接受观测 | 0 |
| 策略拒绝观测 | 65 |
| 深度有效率均值 / 最低 | 0.8459 / 0.8413 |
| 中央深度有效率均值 / 最低 | 0.9296 / 0.9178 |

拒绝原因：

- `support_plane_invalid`：16 条；
- `support_plane_quality_below_policy`：49 条；
- 正式策略要求支持面质量至少 `0.55`，本次最高仅 `0.524`。

因此：

- 带 `--require-geometry` 的验收通过，证明同帧证据链可工作；
- 带 `--require-geometry-candidate` 的验收会按设计失败；
- 原始 rim/short-edge 特征命中不等于抓取候选通过，当前不能绕过支持面门限。

## 4. 识别风险

五个稳定 track 中：

```text
38-Jumbo_Pencil_Cup  track-1
10-meat_can          track-2
4-Frenchs_Mustard    track-3
4-Frenchs_Mustard    track-4
27-Coca_cola_full    track-5
```

`track-4` 对应的现场物体在上一轮硬件只读记录中被判断为粉色 Foam_Brick，
但模型以约 `0.696` 置信度稳定分类为 `4-Frenchs_Mustard`。这再次说明持续 track
稳定不等于类别正确，不能把五个检测框直接当作五个正确类别验收。

## 5. 状态边界

- **Task C**：共用抓取核心、离线安全检查与回归已通过；仍不是实机抓取完成，
  必须先完成手眼、桌面 Z、夹爪尺寸/方向、限位和低速空载验证。
- **Task D**：未知物提议代码已接入，但真实不透明未知物 `--require-unknown`
  正样本验收未完成；本轮真实场景也未能通过任何几何策略候选。
- **透明/强反光物**：仍属于深度恢复未解决项，不得启用实体抓取。
- 继续保持 `motion.dry_run: true` 和 `gripper.backend: mock`，直到现场逐项验收
  并获得用户明确运动授权。

## 6. 校验值

```text
192c5b72bd99573ce1d69c745dc90e7d1eeef5318d99b595a85de96bd77f1d26  config/track1_task_d_vision_geometry_readonly_20261102_121500.yaml
277b40e648ecaf5c2f03d21c8d5f085fcca754288a882bc9231e38eb4ca5e37a  logs/task_d_geometry_readonly_20261102_122705/summary.json
e7a42223b1da90c0423de509fefed518414f531581664d9d8efab5e74a0bedcc  logs/task_d_geometry_readonly_20261102_122705.console.log
7cf701d05621a654080d82bf0809a00326b0d613578a0cf932e440d58a76d253  app/track1_grasp/track1_grasp/tools/accept_task_d_vision.py
f62faf2227b4c707ac04969fb057fe7efeb0abd9f1ec52fb290e9ed6f58685ae  app/track1_grasp/test/test_accept_task_d_vision.py
```

一句话结论：**C/D 共用的几何验收工具和离线安全链已收口并通过 712 项回归；
真实 D405 的几何证据链可用，但候选接受数仍为 0，Task D 未知物正样本和实机
标定尚未完成，因此现在只能保持只读/dry-run。**
