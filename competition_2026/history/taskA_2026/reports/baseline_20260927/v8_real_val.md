# 检测评估：real_val / labels

- 数据：`/media/sunshine/新加卷/taskA/dataset_real_ft_v9_grouped_20260927`
- 图片：192，GT 实例：192
- 通过阈值：IoU ≥ 0.75
- 匹配阈值：IoU ≥ 0.5（用于 recall）
- 预测来源：外部 YOLO 标签

## 总览

| 指标 | 数值 |
| --- | --- |
| 通过率（IoU≥0.75） | 15.6% |
| 平均 best IoU | 0.302 |
| Recall@IoU0.5 | 35.4% |
| 误检（FP 框） | 186 |

## 按类别

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| class 0 | 32 | 20 | 62.5% | 0.727 |
| class 2 | 32 | 0 | 0.0% | 0.000 |
| class 3 | 32 | 5 | 15.6% | 0.492 |
| class 6 | 32 | 0 | 0.0% | 0.243 |
| class 7 | 32 | 0 | 0.0% | 0.000 |
| class 9 | 32 | 5 | 15.6% | 0.348 |

## 按可见度

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| unknown | 192 | 30 | 15.6% | 0.302 |

## 按相机

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| real_v5 | 120 | 20 | 16.7% | 0.337 |
| real_v9 | 72 | 10 | 13.9% | 0.242 |

## 按光照

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| unknown | 192 | 30 | 15.6% | 0.302 |

## 最差的 10 个实例

| 图片 | 类别 | 可见度 | best IoU |
| --- | --- | --- | --- |
| real_real_c0_v5_cheezit_1-Cheez-it_20260926_134900_531382.jpg | class 0 | - | 0.000 |
| real_real_c2_v5_scissors_3-Scissors_20260926_134253_096711.jpg | class 2 | - | 0.000 |
| real_real_c2_v5_scissors_3-Scissors_20260926_134253_402927.jpg | class 2 | - | 0.000 |
| real_real_c2_v5_scissors_3-Scissors_20260926_134253_707784.jpg | class 2 | - | 0.000 |
| real_real_c2_v5_scissors_3-Scissors_20260926_134254_012707.jpg | class 2 | - | 0.000 |
| real_real_c2_v5_scissors_3-Scissors_20260926_134254_317347.jpg | class 2 | - | 0.000 |
| real_real_c2_v5_scissors_3-Scissors_20260926_134254_622411.jpg | class 2 | - | 0.000 |
| real_real_c2_v5_scissors_3-Scissors_20260926_134254_928790.jpg | class 2 | - | 0.000 |
| real_real_c2_v5_scissors_3-Scissors_20260926_134255_233726.jpg | class 2 | - | 0.000 |
| real_real_c2_v5_scissors_3-Scissors_20260926_134255_538848.jpg | class 2 | - | 0.000 |
