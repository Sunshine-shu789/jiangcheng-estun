# 检测评估：real_val / labels

- 数据：`/media/sunshine/新加卷/taskA/dataset_real_ft_v5_grouped_20260926`
- 图片：120，GT 实例：120
- 通过阈值：IoU ≥ 0.75
- 匹配阈值：IoU ≥ 0.5（用于 recall）
- 预测来源：外部 YOLO 标签

## 总览

| 指标 | 数值 |
| --- | --- |
| 通过率（IoU≥0.75） | 0.0% |
| 平均 best IoU | 0.068 |
| Recall@IoU0.5 | 5.0% |
| 误检（FP 框） | 18 |

## 按类别

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| class 0 | 20 | 0 | 0.0% | 0.267 |
| class 2 | 20 | 0 | 0.0% | 0.000 |
| class 3 | 20 | 0 | 0.0% | 0.000 |
| class 6 | 20 | 0 | 0.0% | 0.143 |
| class 7 | 20 | 0 | 0.0% | 0.000 |
| class 9 | 20 | 0 | 0.0% | 0.000 |

## 按可见度

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| unknown | 120 | 0 | 0.0% | 0.068 |

## 按相机

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| real_real_c0_cheezit_1-Cheez-it_20260926_134855_934461.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134856_239938.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134856_545165.jpg | 1 | 0 | 0.0% | 0.165 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134856_850702.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134857_156148.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134857_461460.jpg | 1 | 0 | 0.0% | 0.174 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134857_766568.jpg | 1 | 0 | 0.0% | 0.142 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134858_071649.jpg | 1 | 0 | 0.0% | 0.190 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134858_376874.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134858_699721.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134859_004758.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134859_309910.jpg | 1 | 0 | 0.0% | 0.499 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134859_615213.jpg | 1 | 0 | 0.0% | 0.571 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134859_920512.jpg | 1 | 0 | 0.0% | 0.562 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134900_226393.jpg | 1 | 0 | 0.0% | 0.492 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134900_531382.jpg | 1 | 0 | 0.0% | 0.480 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134900_836743.jpg | 1 | 0 | 0.0% | 0.501 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134901_141992.jpg | 1 | 0 | 0.0% | 0.500 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134901_447465.jpg | 1 | 0 | 0.0% | 0.519 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134901_752652.jpg | 1 | 0 | 0.0% | 0.556 |
| real_real_c2_scissors_3-Scissors_20260926_134253_096711.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134253_402927.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134253_707784.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134254_012707.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134254_317347.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134254_622411.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134254_928790.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134255_233726.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134255_538848.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134255_843571.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134256_148415.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134256_453276.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134256_758172.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134257_063006.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134257_370029.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134257_674747.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134257_979482.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134258_284276.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134258_589116.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134258_897879.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135734_837513.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135735_142215.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135735_448815.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135735_753347.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135736_058091.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135736_362640.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135736_667328.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135736_971983.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135737_276797.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135737_582765.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135737_889864.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135738_194426.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135738_499097.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135738_803637.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135739_108200.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135739_414133.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135739_719071.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135740_023647.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135740_328267.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c3_mustard_4-Frenchs_Mustard_20260926_135740_632882.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135227_523778.jpg | 1 | 0 | 0.0% | 0.497 |
| real_real_c6_clamp_7-Clamp_20260926_135227_828416.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135228_133168.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135228_441273.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135228_746108.jpg | 1 | 0 | 0.0% | 0.469 |
| real_real_c6_clamp_7-Clamp_20260926_135229_050847.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135229_355617.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135229_661297.jpg | 1 | 0 | 0.0% | 0.355 |
| real_real_c6_clamp_7-Clamp_20260926_135229_966069.jpg | 1 | 0 | 0.0% | 0.237 |
| real_real_c6_clamp_7-Clamp_20260926_135230_270799.jpg | 1 | 0 | 0.0% | 0.252 |
| real_real_c6_clamp_7-Clamp_20260926_135230_575637.jpg | 1 | 0 | 0.0% | 0.347 |
| real_real_c6_clamp_7-Clamp_20260926_135230_880187.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135231_184792.jpg | 1 | 0 | 0.0% | 0.272 |
| real_real_c6_clamp_7-Clamp_20260926_135231_489585.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135231_795152.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135232_100169.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135232_405197.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135232_709777.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135233_019927.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c6_clamp_7-Clamp_20260926_135233_324403.jpg | 1 | 0 | 0.0% | 0.432 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134549_728020.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134550_032833.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134550_337370.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134550_642071.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134550_946843.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134551_251707.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134551_608695.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134551_913311.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134552_218200.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134552_523111.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134552_828605.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134553_133542.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134553_438595.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134553_743565.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134554_048518.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134554_353138.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134554_659946.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134554_964939.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134555_269893.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c7_banana_8-Plastic_Banana_20260926_134555_574689.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135507_118708.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135507_423471.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135507_729581.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135508_034232.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135508_338897.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135508_643735.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135508_948224.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135509_257413.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135509_564041.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135509_868801.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135510_173550.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135510_478354.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135510_783084.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135511_088878.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135511_393588.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135511_698188.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135512_002881.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135512_307750.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135512_612483.jpg | 1 | 0 | 0.0% | 0.000 |
| real_real_c9_meat_can_10-meat_can_20260926_135512_917310.jpg | 1 | 0 | 0.0% | 0.000 |

## 按光照

| 分组 | 实例数 | IoU≥阈值 | 通过率 | 平均 best IoU |
| --- | --- | --- | --- | --- |
| unknown | 120 | 0 | 0.0% | 0.068 |

## 最差的 10 个实例

| 图片 | 类别 | 可见度 | best IoU |
| --- | --- | --- | --- |
| real_real_c0_cheezit_1-Cheez-it_20260926_134855_934461.jpg | class 0 | - | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134856_239938.jpg | class 0 | - | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134856_850702.jpg | class 0 | - | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134857_156148.jpg | class 0 | - | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134858_376874.jpg | class 0 | - | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134858_699721.jpg | class 0 | - | 0.000 |
| real_real_c0_cheezit_1-Cheez-it_20260926_134859_004758.jpg | class 0 | - | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134253_096711.jpg | class 2 | - | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134253_402927.jpg | class 2 | - | 0.000 |
| real_real_c2_scissors_3-Scissors_20260926_134253_707784.jpg | class 2 | - | 0.000 |
