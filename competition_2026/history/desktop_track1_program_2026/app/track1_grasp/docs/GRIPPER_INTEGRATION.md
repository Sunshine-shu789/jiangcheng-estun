# 夹爪接入说明（二指电动夹爪 + 电控）

## 1. 你的夹爪是什么

从已归档的资料确认，实际硬件是 **WHEELTEC FAE 电动柔性机械爪（2 指）+ 电控**，
电控就是 **MS42DC 驱控一体步进电机**（V3.0，TB67S539FTG + MT6816 磁编码器）。

资料位置：

- `/media/sunshine/新加卷/爪子/MS42DC步进电机版柔性机械爪用户资料_V3.0.-20251031/`
- 工作区副本：`/home/sunshine/codex/easton/fae爪子/爪子/…`
- 要点报告：`/home/sunshine/codex/easton/taskA/reports/夹爪_新爪MS42DC要点_20261015.md`

它**不是**旧的那只睿尔曼夹爪：

| | 旧爪（睿尔曼版） | 现在的爪（MS42DC 版） |
| --- | --- | --- |
| 通信 | RS485，9 字节厂商协议 | **没有 485** |
| 接口 | PH2.0-6Pin | CAN / TTL 串口 / USB-C / NPN 开关量 |
| 反馈 | 实测读不到任何字节 | 9 字节状态帧，有「到位标志」 |
| 模式 | 位置 + 速度 | **只有位置模式** |

`taskA_grasp/GRIPPER_RS485.md` 里那套 `E5` 帧协议是为旧爪写的，**对新爪无用**，
不要照着发。旧的 `estun_do` 占位后端也不要再对这只爪发 DO 脉冲。

## 2. 框架里已经准备好什么

编排状态机只依赖三个调用：`open()`、`close()`、`state`，外加一个抓取后
的 `HoldVerifier`。所以「怎么控制夹爪」是完全可替换的一层，现在有六种后端：

| `gripper.backend` | 用于 | 状态 |
| --- | --- | --- |
| `mock` | 无硬件跑通全流程（默认） | 已用，25+ 项测试覆盖 |
| **`ms42dc`** | **你这只爪，TTL 串口 / USB-C 原生协议** | **已实现，协议按手册示例逐字节校验** |
| `serial` | 任意「串口发两条命令」的电控夹爪，命令写成 YAML | 已实现 |
| `tcp` | 网口控制盒 | 已实现 |
| `ros_service` | 厂商 ROS 驱动提供 `std_srvs/SetBool` 服务 | 已实现 |
| `ros_topic` | 厂商 ROS 驱动订阅 `std_msgs/Bool` | 已实现 |
| `taska` | 复用 `taskA_grasp` 里已有的可插拔夹爪 | 保留 |

### `ms42dc` 后端

```yaml
gripper:
  backend: ms42dc
  serial_port: /dev/motor_serial   # 或 /dev/ttyUSB0
  serial_baudrate: 115200          # USB 固定 115200
  device_id: 1
  subdivide: 32
  motion_mode: 2                   # 2=相对角度，4=绝对角度
  open_angle_deg: 1872.0           # 5.2 圈；按实际刷的固件改
  close_angle_deg: 1872.0
  speed_rad_s: 20.0
  hold_min_angle_deg: 0.0          # 见下面「空抓检测」
  settle_time_s: 0.3
```

它做三件旧代码没做的事：

1. **自己算 BCC**，角度/速度按 `×10` 编码，你只填物理量。
2. `open()`/`close()` 发完运动帧后**等「到位标志」**，返回时爪指已经停稳，
   不是「命令已发出」。
3. 提供 `read_status()`，把 9 字节反馈解成地址 / 到位标志 / 速度 / 角度，
   并校验 RCC 异或和。

## 3. 控制路径还没定 —— 这是唯一需要你决策的地方

要点报告第 6 节列的四个选项，对应到框架里的做法：

| 方案 | 接线 | 框架里的写法 | 评价 |
| --- | --- | --- | --- |
| ③ Type-C 直连电脑 | 一根 USB 线到工控机 | `backend: ms42dc`, `serial_port: /dev/ttyUSB0` | **最快**，零额外硬件，先这样跑通 |
| ② 接口一 TTL | M8/控制柜 TTL ↔ 夹爪 RX/TX | 加一个 USB↔TTL 后同上 | 和 ③ 同协议 |
| ① 控制柜 DO → NPN | DO 经三极管/光耦拉低接口三 pin1/pin3 | `backend: ros_service` 或 `ros_topic` | 只能开/合，**没有反馈**，无法做夹持确认 |
| ④ CAN | 控制柜或转接板，1 Mbit/s，ID 1 | 需要再加一个 CAN 后端 | 多爪级联时才值得 |

建议顺序：**先用 ③ 把 `ms42dc` 后端在台面上跑通**（USB 拨码 2 打到 ON），
再决定要不要改成控制柜 DO 或 CAN。改成 ① 时状态机的运动时序不用动，
但会丢掉夹持确认能力。

## 4. 协议参考（已按手册与厂商 `motor_node.cpp` 双重核对）

11 字节，TTL 串口与 USB 一致：

```
7B | ID | mode | dir | 细分 | POS_H POS_L | SPD_H SPD_L | BCC | 7D
```

- `dir`：`0` = 张开（逆时针），`1` = 闭合（顺时针）
- `mode`：运动帧 `2` = 相对角度，`4` = 绝对角度；状态帧 `0`
- `POS` / `SPD`：物理值 × 10
- `BCC`：前 9 字节异或；帧尾 `7D`
- 手册示例（本框架已逐字节复现）：
  - 闭合 `7b 01 02 01 20 49 20 00 c8 f8 7d`（1872° @ 20 rad/s）
  - 张开 `7b 01 02 00 20 49 20 00 c8 f9 7d`
  - 状态 `7b 01 00 00 00 00 00 00 00 7a 7d`

状态回复 9 字节：

```
ID | 到位标志 | SPD_H SPD_L | ANGLE(4B 大端, ×10) | RCC
```

`RCC` = 前 8 字节异或。

### 空抓检测

手册写明「堵转保护：夹到最紧即停」。这意味着：

- 夹到物体 → 爪指被挡住，**提前停**，角度小于满行程；
- 夹到空气 → 一直走到硬顶，**角度 ≈ 满行程**。

所以 `hold_min_angle_deg` 设成略小于满行程的值（例如 1872° 行程设 1500°）
就能把「空抓」判成失败，触发状态机的开爪 + 退回分支。

**上电实测前先别设这个值**：先在台面上量一次空合的角度，再填进来。
在没量之前保持 `hold_min_angle_deg: 0.0`，此时只校验「到位标志」。

## 5. 上电前的台面验证步骤

夹爪**还没有装到机械臂上**，先在台面用 USB 验证，机械臂不动：

```bash
# 1. 看串口有没有出来（USB 拨码 2 打到 ON）
ls -l /dev/serial/by-id/ /dev/ttyUSB* 2>/dev/null

# 2. 只读状态，不发运动帧 —— 确认能收到 9 字节且 RCC 正确
python3 -m track1_grasp.tools.gripper_bench --config <你的配置> status

# 3. 空载张开 / 闭合，确认方向没接反
python3 -m track1_grasp.tools.gripper_bench --config <你的配置> cycle
```

`gripper_bench` 默认只读；发运动帧要显式加 `--really`，和
`taskA_grasp.estun_rs485` 同样的安全门控。

先确认的第一件事：**方向**。如果 `open` 实际在夹、`close` 实际在松，
把 `open`/`close` 两条运动帧的 `dir` 对调即可（配置项
`invert_direction: true`）。

## 6. 还需要你确认的事（未确认前不接线、不动作）

1. **型号**：电控板上是不是 Type-C + 2 位拨码 + 两个 XH2.54-4Pin + 一个
   XH1.25-4Pin？（是 → MS42DC 版）
2. **控制路径**：③ USB / ② TTL / ① 控制柜 DO / ④ CAN，四选一（见第 3 节）
3. **电控形态**：装在电机尾部一体，还是独立电控盒
4. **供电**：是否从法兰 M8 pin5(24V+)/pin8(GND) 取电（M8 的 24V 可给 2A，
   夹爪最大 1.2A，余量够）
5. **固件行程**：现在刷的是几圈（4.5/4.7/4.9/5.1/5.3/5.5/5.8/6.1/7.0/7.5/8.5/9.5 圈
   各有一份 hex），决定 `open_angle_deg` / `close_angle_deg`
6. **按压力量**：有无需要限制（当前协议没有力量参数，只有位置和速度）

安全约定：先接线后上电，不支持热插拔；首次动作用小角度低速试，
避免硬顶。
