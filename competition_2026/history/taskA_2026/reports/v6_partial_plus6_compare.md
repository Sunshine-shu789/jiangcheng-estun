# Random real-scene model comparison

- Evaluation split: 8 permanently isolated scenes / 24 instances
- Evaluation image size: 640
- Prediction confidence: 0.25
- Pass threshold: IoU >= 0.75

| Model | Pass rate | Mean best IoU | Recall@0.5 | False positives |
| --- | ---: | ---: | ---: | ---: |
| v5 | 4.2% | 0.170 | 16.7% | 5 |
| v6_partial | 29.2% | 0.437 | 58.3% | 4 |
| v6_partial_plus6_nomosaic | 33.3% | 0.480 | 62.5% | 9 |

## New model details

- Weights: `/media/sunshine/新加卷/taskA/runs/yolo11s_random_real_v6_partial_plus6_nomosaic_20260926/weights/best.pt`
- Training data: `/home/sunshine/codex/easton/taskA/datasets/random_real_scenes_v6_partial_plus6_20260926/data.yaml`
- Initial weights: v5 `best.pt`
- Epochs: 50
- Image size: 640
- Batch size: 8
- Optimizer: AdamW
- Initial learning rate: `5e-5`
- Frozen layers: 10
- Mosaic: disabled

## Per-class result

| Class | Passed | Total | Pass rate | Mean best IoU |
| --- | ---: | ---: | ---: | ---: |
| 0 Cheez-it | 3 | 4 | 75.0% | 0.652 |
| 2 Scissors | 1 | 4 | 25.0% | 0.638 |
| 3 Mustard | 2 | 4 | 50.0% | 0.388 |
| 6 Clamp | 2 | 4 | 50.0% | 0.410 |
| 7 Banana | 0 | 4 | 0.0% | 0.648 |
| 9 meat_can | 0 | 4 | 0.0% | 0.147 |

## Decision

The new model is retained as a candidate but is not promoted to the live camera.
The live camera remains on v5 until the candidate has a clear IoU and
false-positive improvement on additional unseen scenes.

## Follow-up diagnostics

| Experiment | Pass rate | Mean best IoU | Recall@0.5 | False positives |
| --- | ---: | ---: | ---: | ---: |
| 640, freeze 10, 50 epochs | 33.3% | 0.480 | 62.5% | 9 |
| 640, all layers unfrozen | 20.8% | 0.206 | 37.5% | 9 |
| 768 inference | 29.2% | 0.551 | 75.0% | 14 |
| 960 inference | 20.8% | 0.454 | 54.2% | 25 |

Unfreezing the full network overfits immediately. Increasing inference resolution
does not provide a balanced gain: 960 helps banana reach 2/4 passes but degrades
the other classes and creates many more false positives. The next useful step is
additional real-scene diversity, with priority on meat can and banana views that
are not near-duplicates of the current training scenes.
