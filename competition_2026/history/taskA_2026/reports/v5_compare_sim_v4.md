# 检测评估：sim_val / labels

- 数据：`/media/sunshine/新加卷/taskA/dataset_real_ft_v5_grouped_20260926`
- 图片：60，GT 实例：600
- 通过阈值：IoU ≥ 0.75
- 匹配阈值：IoU ≥ 0.5（用于 recall）
- 预测来源：外部 YOLO 标签

## 总览

| 指标 | 数值 |
| --- | --- |
| 通过率（IoU≥0.75） | 85.7% |
| 平均 best IoU | 0.830 |
| Recall@IoU0.5 | 92.2% |
| 误检（FP 框） | 13 |

## 按类别

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| class 0 | 60 | 60 | 100.0% | 0.953 |
| class 1 | 60 | 42 | 70.0% | 0.706 |
| class 2 | 60 | 42 | 70.0% | 0.714 |
| class 3 | 60 | 56 | 93.3% | 0.888 |
| class 4 | 60 | 53 | 88.3% | 0.834 |
| class 5 | 60 | 50 | 83.3% | 0.798 |
| class 6 | 60 | 52 | 86.7% | 0.829 |
| class 7 | 60 | 49 | 81.7% | 0.813 |
| class 8 | 60 | 55 | 91.7% | 0.883 |
| class 9 | 60 | 55 | 91.7% | 0.878 |

## 按可见度

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| unknown | 600 | 514 | 85.7% | 0.830 |

## 按相机

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| sim_left | 200 | 175 | 87.5% | 0.842 |
| sim_middle | 200 | 166 | 83.0% | 0.826 |
| sim_right | 200 | 173 | 86.5% | 0.820 |

## 按光照

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| directional_light | 200 | 176 | 88.0% | 0.851 |
| point_light | 200 | 167 | 83.5% | 0.822 |
| spot_light | 200 | 171 | 85.5% | 0.816 |

## 最差的 10 个实例

| 图片 | 类别 | 可见度 | best IoU |
| --- | --- | --- | --- |
| sim_left_camera_directional_light_scene_1651_rgb.png | class 4 | - | 0.000 |
| sim_left_camera_directional_light_scene_4401_rgb.png | class 2 | - | 0.000 |
| sim_left_camera_point_light_scene_1301_rgb.png | class 9 | - | 0.000 |
| sim_left_camera_point_light_scene_1301_rgb.png | class 1 | - | 0.000 |
| sim_left_camera_point_light_scene_401_rgb.png | class 2 | - | 0.000 |
| sim_left_camera_point_light_scene_901_rgb.png | class 2 | - | 0.000 |
| sim_left_camera_spot_light_scene_1551_rgb.png | class 2 | - | 0.000 |
| sim_left_camera_spot_light_scene_1551_rgb.png | class 3 | - | 0.000 |
| sim_left_camera_spot_light_scene_1551_rgb.png | class 5 | - | 0.000 |
| sim_left_camera_spot_light_scene_2251_rgb.png | class 1 | - | 0.000 |
