# Task B 已夹住却被固定终角误判修复

## 现场事实

- 用户确认现场香蕉已经被夹住，视觉画面也显示香蕉位于两指之间。
- 本次只读状态检查没有启动任务、机械臂运动、夹爪开合或自动重试。
- 实机动作已经结束，机械臂仍保持抓取位，夹爪闭合。

## 失败原因

实机日志：

- `logs/task_B_20261106_190531.log`
- `logs/codroid_bridge_20261106_190531.log`

关键数据：

```text
banana close width       = 0.037216 m
close start angle        = 19.2 deg
close final angle        = 1977.9 deg
close travel             = 1958.7 deg
CAD minimum travel       = 1719.5 deg
CAD travel margin        = 239.2 deg
legacy fixed ceiling     = 1970.0 deg
```

`close_for_width` 已经验证 CAD 行程通过，但后续固定终角 `1970.0 deg` 无条件覆盖了
有效行程证据，导致真实夹持被误判为空抓并中止。问题不在 YOLO、香蕉位姿、短轴抓取
路线或夹爪闭合。

## 修改

修改文件：

- `app/track1_grasp/track1_grasp/gripper.py`
- `app/track1_grasp/track1_grasp/config.py`
- `app/track1_grasp/test/test_gripper.py`
- `app/track1_grasp/docs/GRIPPER_INTEGRATION.md`

行为变化：

1. 闭合证据先检查当前物体宽度对应的 CAD 最小行程。
2. `require_close_travel_evidence=true` 时，CAD 行程是权威证据；固定
   `close_contact_max_angle_deg` 不再否决已经通过的行程。
3. CAD 行程不足仍然 fail-closed。
4. 固定终角上限只保留给 `require_close_travel_evidence=false` 的兼容模式。
5. 新增真实香蕉数值回归测试，并新增兼容模式仍保留固定终角门限的测试。

## 回归验证

相关安全链：

```text
300 passed in 55.85s
```

完整测试：

```text
1051 passed in 115.17s
```

配置校验：

```text
pick1 low-speed live: speed=0.01 m/s, dry_run=false, backend=ms42dc
five-point formal:    speed=0.12 m/s, dry_run=true, backend=mock
```

正式速度和安全模式未修改。

## 现场限制

当前 MS42DC 没有独立的电流/转矩夹持信号。宽度感知模式可以避免固定终角误判，
但若要绝对区分“夹住”和“空夹到硬限位”，后续仍应增加电流/堵转证据，或短距离
抬升后的视觉复核。任何抬升、释放、复位或再次执行 `pick_1` 都必须等待用户新的
明确授权。
