# 赛道一 C/D — v5 模型 38 类就绪矩阵（2026-10-30）

- 权重：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/runs/track1_train_40class_v5/track1_40class_v5_20261029_234800/weights/best.pt`
- SHA256：`cdafa5f39f8e0b22fe1a608a126283eba52a81c2daf7a9f20f4352bb55396099`
- 数据集：`datasets/track1_training_merged_v5` 的 `val` 划分
- 真实前向参数：conf=0.15，NMS IoU=0.60，匹配 IoU=0.50，imgsz=640
- 统计：38 个有效类别，156 个 GT；TP=141，FN=15；漏检 5、错类 8、定位 2、额外框 18
- 严格结论：**未通过全类别零错误验收**；这不阻塞 C/D 软件联调，但会限制自主抓取可用类别。

## 结论

- `READY`：21 类，可进入 C/D 已知类别受控测试。
- `CAUTION`：7 类，无真实漏检/错类，但有额外框或已知罐顶重复框；必须依赖稳定器去重。
- `BLOCKED`：10 类，存在漏检、错类或定位错误，暂不应作为无人值守的自主目标。
- 前 10 类在本次严格匹配中仍然漏检 0、错类 0；但番茄汤/金枪鱼罐顶存在已知重叠双框，按用户决定不再训练修补。
- 不接相机/机械臂时，C/D 状态机 mock 已分别 4/4 通过；真实 D405 识别验收仍需硬件恢复后执行。

## 就绪矩阵

| 官方序号 | 类别 | GT | TP | FP | Recall | Precision | AP50 | 状态 | 说明 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1-Cheez-it | 2 | 2 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 3 | 3-Scissors | 3 | 3 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 4 | 4-Frenchs_Mustard | 2 | 2 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 6 | 6-Foam_Brick | 2 | 2 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 7 | 7-Clamp | 2 | 2 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 8 | 8-Plastic_Banana | 2 | 2 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 9 | 9-Mug | 1 | 1 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 10 | 10-meat_can | 4 | 4 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 12 | 12-Baseball | 3 | 3 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 14 | 14-Sleeve | 2 | 2 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 16 | 16-ICRA_Duckie | 4 | 4 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 20 | 20-Scotch_Tape | 2 | 2 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 21 | 21-Haribo_Golden_Bear | 4 | 4 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 27 | 27-Coca_cola_full | 5 | 5 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 28 | 28-Kong_Duck_Dog_Toy | 14 | 14 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 32 | 32-Wine_Glass | 3 | 3 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 33 | 33-Key | 2 | 2 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 34 | 34-Nail | 2 | 2 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 35 | 35-Book | 3 | 3 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 37 | 37-T-shirt | 2 | 2 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 39 | 39-Glove | 2 | 2 | 0 | 1.0 | 1.0 | 0.995 | **READY** | 离线检测通过 |
| 2 | 2-Starkist_Tuna | 3 | 3 | 0 | 1.0 | 1.0 | 0.995 | **CAUTION** | 已知番茄汤/金枪鱼罐顶重叠双框；运行态依赖稳定器去重 |
| 5 | 5-Tomato_Soup | 4 | 4 | 1 | 1.0 | 0.8 | 0.945 | **CAUTION** | 已知番茄汤/金枪鱼罐顶重叠双框；运行态依赖稳定器去重 |
| 13 | 13-Bowl | 3 | 3 | 1 | 1.0 | 0.75 | 0.995 | **CAUTION** | GT 匹配正常，但有 1 个额外预测；需稳定帧和运行态去重 |
| 15 | 15-Coca_cola_half | 4 | 4 | 1 | 1.0 | 0.8 | 0.995 | **CAUTION** | GT 匹配正常，但有 1 个额外预测；需稳定帧和运行态去重 |
| 19 | 19-Elmers_Glue | 10 | 10 | 3 | 1.0 | 0.76923 | 0.995 | **CAUTION** | GT 匹配正常，但有 3 个额外预测；需稳定帧和运行态去重 |
| 36 | 36-Adjustable_Wrench | 2 | 2 | 1 | 1.0 | 0.66667 | 0.995 | **CAUTION** | GT 匹配正常，但有 1 个额外预测；需稳定帧和运行态去重 |
| 40 | 40-Pringles | 4 | 4 | 1 | 1.0 | 0.8 | 0.945 | **CAUTION** | GT 匹配正常，但有 1 个额外预测；需稳定帧和运行态去重 |
| 11 | 11-Padlock | 3 | 2 | 0 | 0.66667 | 1.0 | 0.665 | **BLOCKED** | 存在漏检/错类/定位错误，暂不用于自主抓取 |
| 18 | 18-Dice | 3 | 2 | 1 | 0.66667 | 0.66667 | 0.665 | **BLOCKED** | Dice 有 1 个漏检；AP50 0.665 |
| 22 | 22-E27_Adapter | 9 | 8 | 3 | 0.88889 | 0.72727 | 0.87278 | **BLOCKED** | E27 与 GU24 相互错类；需定向修正 |
| 23 | 23-GU24_Adapter | 7 | 5 | 1 | 0.71429 | 0.83333 | 0.75357 | **BLOCKED** | E27 与 GU24 相互错类，并有漏检 |
| 24 | 24-Spoon | 13 | 10 | 0 | 0.76923 | 1.0 | 0.74775 | **BLOCKED** | Spoon 13 个 GT 漏 3；AP50 0.748 |
| 25 | 25-Flat_Screwdriver | 9 | 7 | 2 | 0.77778 | 0.77778 | 0.70821 | **BLOCKED** | Flat Screwdriver 2 个定位错误并有额外框；AP50 0.708 |
| 29 | 29-Paper_Towels | 3 | 2 | 0 | 0.66667 | 1.0 | 0.995 | **BLOCKED** | Paper Towels 被错识别为 Elmers Glue |
| 30 | 30-Stabilo_OHPen | 5 | 4 | 1 | 0.8 | 0.8 | 0.795 | **BLOCKED** | Stabilo OHPen 被错识别为 Jumbo Pencil Cup |
| 31 | 31-Plastic_White_Cup | 4 | 2 | 0 | 0.5 | 1.0 | 0.745 | **BLOCKED** | Plastic White Cup 4 个 GT 错 2 个为 Elmers Glue |
| 38 | 38-Jumbo_Pencil_Cup | 4 | 3 | 2 | 0.75 | 0.6 | 0.845 | **BLOCKED** | Jumbo Pencil Cup 被错识别为 Bowl，并有额外框 |

## C/D 使用建议

- Task C：`READY` 类别可进入“任意目标”自动选择；`CAUTION` 类别固定单目标并依赖连续帧稳定；`BLOCKED` 类别不要进入自动选择列表。
- Task D：已知类别走同一判定；`unknown/object` 目前仍无独立的类别无关分割器，不能把 YOLO 的低置信度结果直接当未知物体。
- 罐类重复框不继续训练。运行态依赖 `detection_stabilizer` 去重；现场必须观察连续帧是否只保留一个框。
- 下一步不接硬件可做：给 C/D 增加验收白名单/黑名单配置并补测试；接上 D405 后再做真实画面逐类复核。

## 文件

- 报告：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_cd_38class_readiness_20261030/track1_cd_38class_readiness_20261030.md`
- CSV：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_cd_38class_readiness_20261030/readiness_matrix.csv`
- JSON：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_cd_38class_readiness_20261030/readiness_matrix.json`
- 原始 38 类离线验收：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/reports/track1_allclass_offline_audit_v5_conf015_20261030/allclass_offline_audit_val.md`
