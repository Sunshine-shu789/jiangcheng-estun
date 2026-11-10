# 赛道一 v9 Eggs 困难负样本诊断

- 生成时间：2026-10-31T02:29:02+08:00
- 数据集：`/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/datasets/track1_training_merged_v7_oversample_new2/dataset.yaml`
- 模式：只读静态图片；不打开相机、不导入 ROS、不控制机械臂或夹爪。
- 验证集图片只用于失败诊断，v9 禁止复制到 train。

## 数量

- train 唯一原始帧：401
- val 唯一原始帧：121
- train 10-meat_can 唯一帧：6，现有数据集副本 23
- train 绿色正面 Eggs 唯一帧：15，现有副本 60
- train 橙色背面 Eggs 唯一帧：20，现有副本 80
- val 绿色正面 Eggs 仅诊断：5

## 判断

- v8 将绿色肉罐头以高置信度同时报为 Eggs，必须用训练集肉罐头困难负样本压制。
- v8 漏检的 5 张 val 图均为绿色正面 Eggs；训练集存在绿色正面帧，但 v9 应提高这类帧的重复次数。
- 不能删除、过滤或降低阈值来掩盖同框误报。

## 产物

- `meat_can_train_unique_frames.jpg`
- `eggs_train_green_front_unique_frames.jpg`
- `eggs_train_orange_back_unique_frames.jpg`
- `eggs_val_green_front_diagnostic_only.jpg`
- `strawberry_train_unique_frames.jpg`
- `unique_frame_inventory.csv`
- `diagnostics.json`
