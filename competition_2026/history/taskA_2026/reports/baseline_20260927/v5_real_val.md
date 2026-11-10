# 检测评估：real_val / labels

- 数据：`/media/sunshine/新加卷/taskA/dataset_real_ft_v9_grouped_20260927`
- 图片：192，GT 实例：192
- 通过阈值：IoU ≥ 0.75
- 匹配阈值：IoU ≥ 0.5（用于 recall）
- 预测来源：外部 YOLO 标签

## 总览

| 指标 | 数值 |
| --- | --- |
| 通过率（IoU≥0.75） | 0.0% |
| 平均 best IoU | 0.110 |
| Recall@IoU0.5 | 13.5% |
| 误检（FP 框） | 33 |

## 按类别

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| class 0 | 32 | 0 | 0.0% | 0.168 |
| class 2 | 32 | 0 | 0.0% | 0.000 |
| class 3 | 32 | 0 | 0.0% | 0.000 |
| class 6 | 32 | 0 | 0.0% | 0.059 |
| class 7 | 32 | 0 | 0.0% | 0.412 |
| class 9 | 32 | 0 | 0.0% | 0.020 |

## 按可见度

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| unknown | 192 | 0 | 0.0% | 0.110 |

## 按相机

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| real_v5 | 120 | 0 | 0.0% | 0.138 |
| real_v9 | 72 | 0 | 0.0% | 0.062 |

## 按光照

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| unknown | 192 | 0 | 0.0% | 0.110 |

## 最差的 10 个实例

| 图片 | 类别 | 可见度 | best IoU |
| --- | --- | --- | --- |
| real_real_c0_v5_cheezit_1-Cheez-it_20260926_134855_934461.jpg | class 0 | - | 0.000 |
| real_real_c0_v5_cheezit_1-Cheez-it_20260926_134856_545165.jpg | class 0 | - | 0.000 |
| real_real_c0_v5_cheezit_1-Cheez-it_20260926_134857_766568.jpg | class 0 | - | 0.000 |
| real_real_c0_v5_cheezit_1-Cheez-it_20260926_134858_071649.jpg | class 0 | - | 0.000 |
| real_real_c0_v5_cheezit_1-Cheez-it_20260926_134858_376874.jpg | class 0 | - | 0.000 |
| real_real_c0_v5_cheezit_1-Cheez-it_20260926_134858_699721.jpg | class 0 | - | 0.000 |
| real_real_c0_v5_cheezit_1-Cheez-it_20260926_134859_004758.jpg | class 0 | - | 0.000 |
| real_real_c0_v5_cheezit_1-Cheez-it_20260926_134859_309910.jpg | class 0 | - | 0.000 |
| real_real_c0_v5_cheezit_1-Cheez-it_20260926_134859_615213.jpg | class 0 | - | 0.000 |
| real_real_c0_v5_cheezit_1-Cheez-it_20260926_134859_920512.jpg | class 0 | - | 0.000 |
