# 第三届“埃斯顿”大学生机器人大赛（2026）自研资料

本目录汇总了“江城队”在第三届埃斯顿大学生机器人大赛中的**自研代码、配置、算法、标定资料、实验记录、训练日志、自训练模型和历史版本**。

上游仓库原有的 `s3_60_eco_2024` 与 `s3_60_eco_2024_moveit_config` 两个 ROS 2 包保持不变；2026 年比赛资料统一放在 `competition_2026/`，不会覆盖原有提交和历史。

## 目录说明

| 路径 | 内容 |
| --- | --- |
| `formal/track1_competition_program/` | 数据盘正式运行目录中的自研程序快照：任务 A–E、视觉抓取、训练/验收工具、配置、桌面入口、标定、文本报告、日志和备份源码 |
| `formal/track1_competition_program/app/` | Python 应用、视觉验收、数据采集、训练准备、抓取逻辑 |
| `formal/track1_competition_program/bin/` | 正式启动、诊断、训练、验收和测试脚本 |
| `formal/track1_competition_program/config/` | 任务及训练配置（敏感凭据已在发布前扫描） |
| `formal/track1_competition_program/calibration/` | 手眼标定脚本、结果、验证图和报告 |
| `formal/track1_competition_program/reports_text/` | Git 中的文本版实验报告（JSON/Markdown/CSV 等） |
| `formal/track1_competition_program/logs/` | Git 中的 `.log` 运行记录 |
| `formal/track1_competition_program/backups_text/` | 历史备份中的源码和文本文件 |
| `formal/track1_competition_program/robot_description_text/` | 自研机器人描述/夹爪集成代码与配置；厂家 CAD、STEP 和 mesh 不入库 |
| `history/` | 主目录和早期数据盘中的历史自研版本 |
| `models/` | 可再分发的自训练模型及元数据 |
| `patches/` | 针对第三方依赖的最小兼容补丁和来源记录，不包含第三方完整源码或权重 |

## 自训练模型

| 文件 | 说明 |
| --- | --- |
| `models/best.pt` | 正式赛道一 40 类检测模型 |
| `models/official/track1_40class_v11_official_f43234a9.pt` | v11 正式模型及对应 JSON/SHA256 元数据 |
| `models/yolo11s_backbone_40class_init_v10aug.pt` | 由本项目自训练结果导出的后续训练初始化权重 |

第三方 CLIP、基础 YOLO、MobileSAM、YCB 模型和 GraspNet checkpoint 均不纳入本仓库。

## 大数据与完整实验产物

比赛目录中约 24 GB 的原始数据不适合直接写入普通 Git 历史。完整自采数据集、全部训练输出、完整日志/报告媒体、历史备份和快照将作为 **GitHub Releases 分卷资产**发布。

每个大归档都附带 SHA-256 清单。下载后按下面的方式还原：

```bash
# 将同一归档的所有 .part-* 文件放在同一目录
cat datasets.tar.zst.part-* > datasets.tar.zst
sha256sum -c SHA256SUMS
tar --zstd -xf datasets.tar.zst
```

Release 资产与仓库中的 `MANIFEST.sha256` 一起构成完整的交付清单。

## 安全与运行说明

- 本仓库只保存代码和实验资料，不包含可启动实体机械臂所需的安全确认。
- Task B 的正式运行策略仍应保持 `motion.dry_run: true` 和 `gripper.backend: mock`，直到现场坐标、限位、手眼标定与夹爪阈值全部复核通过。
- 运行正式程序前应先执行数据盘中的 `bin/ensure_runtime`；不要把比赛程序重新写回主目录。
- 仓库中的日志可能包含本机路径、ROS 话题名、相机序列号或实验阶段的标定数值，这些是复现实验所需的技术记录，不包含 GitHub/API 凭据。

## 官方和第三方内容

官方培训材料、比赛任务书、厂家 ROS 包、CAD/STEP/mesh、控制器固件、运行时 ext4 镜像以及第三方依赖和预训练权重**没有上传**。完整排除清单见 [`EXCLUDED_ASSETS.md`](EXCLUDED_ASSETS.md)。

## 来源快照

- 正式目录：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/`（实际挂载到 `新加卷1`）
- 快照日期：2026-11-10
- 上游仓库默认分支：`main`
- 主目录中的旧版本仅做只读复制，没有移动、删除或重新写入主目录。
