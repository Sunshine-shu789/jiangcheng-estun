# 赛道一任务 C / D 独立程序拆分报告（2026-11-02）

## 1. 结论

任务 C 和任务 D 已经拆成**两个分别启动、分别配置、分别验收的任务程序**，不是
一个 C/D 合并任务。底层视觉、IK、运动、夹爪和状态机库继续共用，但正式入口和
配置不共用。

当前状态必须分开表述：

| 任务 | 程序拆分 | 离线 / mock | 实机验收 |
| --- | --- | --- | --- |
| C | 已完成 | 已完成 | 尚未完成，不能宣称任务 C 已整体收口 |
| D | 已完成 | 已完成（含 mock `object`） | 未完成，真实未知物正样本和几何候选尚未通过 |

因此，对“任务 C 是不是弄好了”的准确回答是：**C 的独立程序、已知类策略、
离线/mock 链路和安全门禁已经完成；C 的真实现场闭环验收尚未完成，所以任务 C
还不能标记为整体完成。D 也不是和 C 一起完成，必须在 C 验收通过后单独继续。**

本轮没有发布 `/RobotMove`，没有发送夹爪开合命令，没有让实体机械臂或夹爪运动。

## 2. 正式入口与配置

| 项目 | 任务 C | 任务 D |
| --- | --- | --- |
| 桌面入口 | `任务C_无序抓取.desktop` | `任务D_未知物体抓取.desktop` |
| 启动程序 | `bin/run_task_c` | `bin/run_task_d` |
| 正式配置 | `config/track1_task_c_v11_official_20261102.yaml` | `config/track1_task_d_v11_official_20261102.yaml` |
| `task.mode` | `C` | `D` |
| 目标策略 | `known` | `known_and_unseen` |
| 已知类 | V11 已知 40 类 | V11 已知 40 类 |
| 未知物 | `unknown/object` 一律拒绝 | 接受高可抓性 `unknown/object` |
| unknown proposer | `enabled: false` | `enabled: true` |
| 模型 | V11 40 类 | 同一 V11 40 类 |

启动器绑定规则：

```text
bin/run_task_c -> track1_task_runner C -> config/track1_task_c_v11_official_20261102.yaml
bin/run_task_d -> track1_task_runner D -> config/track1_task_d_v11_official_20261102.yaml
```

`config/track1_task_cd_40class_v11_official.yaml` 只作为历史兼容和公共回归配置，
不是桌面正式入口。两份正式配置使用同一模型文件及其 SHA256 校验，但任务策略和
未知物 proposer 明确不同。

## 3. 已完成修改

- 为 C、D 各建立独立正式配置，分别声明任务模式、目标策略和未知物策略。
- 为 C、D 各建立独立启动包装器，桌面快捷方式直接指向 `bin/run_task_c` 或
  `bin/run_task_d`，没有合并入口。
- 实际桌面快捷方式已修正为直接指向数据盘正式目录；C、D 的 `Exec`、`TryExec`、
  `Path` 均不再指向主目录，并通过 `desktop-file-validate`。
- `apply_mode()` 中任务 C 已从旧的宽松策略改为 `known`，只接受
  `target.known_classes` 中的 40 类；任务 D 保持 `known_and_unseen`。
- 新增 C/D 分开的 mock 场景：C 的场景只含 `mug/bowl/book`，D 的场景额外包含
  可抓性 `object`。
- 修正了 8 个仍假定“任务 C 可接受任意标签”的旧短边抓取用例；测试夹具现在
  显式提供测试用已知类，不削弱生产环境的 C 策略。
- 更新 README、架构、接线清单和 `使用说明.txt`，明确“C 先完成并验收，再开始
  D”以及“同一时间只能运行其中一个”。

共享核心仍包括：

```text
track1_grasp/vision.py
track1_grasp/pipeline.py
track1_grasp/planner.py
track1_grasp/kinematics.py
track1_grasp/motion.py
track1_grasp/gripper.py
```

共享代码是为了避免复制实现，不代表任务 C、D 可以合并启动。

## 4. 自动验证

| 验证项 | 结果 | 证据 |
| --- | --- | --- |
| 全量回归 | `714 passed in 61.45s` | `logs/track1_task_cd_separate_full_regression_20261102.log` |
| 配置/策略/模型/清箱专项 | `130 passed in 11.48s` | `logs/track1_task_cd_split_specialized_20261102.log` |
| Task C mock | 5/5 成功，目标均为已知类 | `logs/task_C_separate_mock_20261102_123203.log` |
| Task D mock | 6/6 成功，包含未知 `object` | `logs/task_D_separate_mock_20261102_123203.log` |
| C 启动环境检查 | 通过，退出码 0 | `logs/track1_task_cd_launcher_checks_20261102.log` |
| D 启动环境检查 | 通过，退出码 0 | `logs/track1_task_cd_launcher_checks_20261102.log` |

启动环境检查只验证 Python、ROS、YOLO、模型和配置，不启动任务节点，不发布运动
或夹爪命令。

## 5. 任务 C 的真实状态

已完成：

- 独立入口、独立配置和桌面绑定；
- 40 类已知目标策略；
- `unknown/object` 拒绝；
- unknown proposer 关闭；
- 公共抓取核心、短边抓取、六段安全链和 mock 清箱；
- 专项与全量离线回归。

尚未完成：

- 真实 D405 对现场已知物品的可重复识别、深度和短边抓取闭环验收；
- 真实手眼、夹爪尺寸/方向、限位、低速单点和故障恢复的完整现场验收；
- 透明/强反光物的抓取边界。

结论：**C 的程序拆分和离线开发已完成，但“任务 C 已验收完成”仍不能成立。**

## 6. 任务 D 的真实状态与已知风险

D 的代码已接入未知物 proposer 和 `known_and_unseen` 策略，但真实数据仍未通过：

```text
最近一次真实 D405 只读验收：
  采集帧：15/15
  已知观测：65
  未知观测：0
  几何证据：65
  几何策略接受：0
  支持面质量不足：49
  无可靠支持面：16
  最高支持面质量：0.524
  正式支持面质量门限：0.55
```

因此：

- 不能用离线 mock 的 `object` 成功冒充真实未知物验收；
- 真实未知物 `--require-unknown` 正样本验收仍未完成；
- `--require-geometry-candidate` 仍会按设计失败；
- 透明/强反光物的深度修复和抓取仍未解决；
- 现有真实场景中还存在稳定 track 但类别识别错误的风险，持续跟踪稳定不等于
  类别正确。

## 7. 现场执行顺序

1. 只运行任务 C，完成其真实 D405、点位、夹爪、限位和低速单点验收。
2. 任务 C 验收通过并形成独立报告后，才允许开始任务 D。
3. 任务 D 先做只读未知物正样本验收，再做几何候选验收，最后才做实体抓取。
4. C、D 任何时刻只能启动一个，不能并行占用 D405、机械臂或夹爪。
5. 未获得明确运动授权时，不得运行正式真机入口。

当前两份正式配置都声明了真实运动后端和 MS42DC 夹爪，因此本次只做了静态环境
检查，没有启动 `run_task_c` 或 `run_task_d` 的任务主体。

## 8. 文件校验值

```text
93db1acaa0c587967d9f8e0e855e2ada2b62fbf5621231f42d7e67d01f94ea66  config/track1_task_c_v11_official_20261102.yaml
af907710c4709f0d693e7d21f7b1fb3b06cc88a82474de3a5d28212939733c7d  config/track1_task_d_v11_official_20261102.yaml
a3a0d27c4be068862df6360ffa05817e1e38e6e98da028f72f41f5cc3478ab4e  bin/run_task_c
928c5f6316163c5efd7ae65bf5f25dc6e7b8b3cccb157f745944c83e15b308ff  bin/run_task_d
f43234a94f1d71b24cb187082c18eb22fea84c88922c605df5f7b1a79ffcaaa6  models/official/track1_40class_v11_official_f43234a9.pt
fa71549efe6790ad8c3a990bf46c11dacaa5bb69d2be640e72dd0443e3417a51  logs/track1_task_cd_split_specialized_20261102.log
d267f1ff96dd99b2be24deb801a36baa1ba523367f9ba2b3ca1b8ff0dc435839  logs/track1_task_cd_separate_full_regression_20261102.log
3d5329e4cb620ad238e6e29f7fcb7e3c6214dd258c41915b0b6d1ac32e046ae8  logs/track1_task_cd_launcher_checks_20261102.log
63efc92ea3940673177620ce2bd579a0aaa8bb02ff9afab24ec3d8c81d39d630  logs/task_C_separate_mock_20261102_123203.log
fa8865544effa99f7ef448b393a7acfd0537eee86a06bb55f0bc04beda6b4a91  logs/task_D_separate_mock_20261102_123203.log
a26ffd413bd16d98d149d490ee0cdf545e99f6c86813f01218c92374e3723467  app/track1_grasp/README.md
f465b41ad3315925304537a6c6e77dbebf8e134df5b3d7bbac5f3ee10e856290  app/track1_grasp/docs/ARCHITECTURE.md
1095c04ca860813cd9eb36e02c45ef0478b95fd0f25dcf16ff8fde425cfa5025  app/track1_grasp/docs/INTEGRATION_CHECKLIST.md
cc53662341fcb2622bf5e2e5d66dc076dcec35041132d8cd82a8a61bb1fb4f7c  使用说明.txt
a7116e5e06388952e20337a38c4dc163dfa41ab6927123100076683fe65eb45c  /home/sunshine/Desktop/赛道一比赛程序/任务C_无序抓取.desktop
b95c0869bb6dedda61c7e6b02f83a088a56bb26aae1b2fb6b449ece2a765af13  /home/sunshine/Desktop/赛道一比赛程序/任务D_未知物体抓取.desktop
```
