# Random real-scene model comparison: v7

- Evaluation split: 8 permanently isolated scenes / 24 instances
- Evaluation image size: 640
- Prediction confidence: 0.25
- Pass threshold: IoU >= 0.75

| Model | Pass rate | Mean best IoU | Recall@0.5 | False positives |
| --- | ---: | ---: | ---: | ---: |
| v5 | 4.2% | 0.170 | 16.7% | 5 |
| v6 | 33.3% | 0.480 | 62.5% | 9 |
| v7 | 45.8% | 0.490 | 62.5% | 8 |

## v7 training

- Initial weights: v5 `best.pt`
- Training data: 37 scenes / 103 instances
- Validation data: 8 unchanged holdout scenes / 24 instances
- Epochs: 50
- Image size: 640
- Batch size: 8
- Optimizer: AdamW
- Initial learning rate: `5e-5`
- Frozen layers: 10
- Mosaic: disabled

## v7 per-class result

| Class | Passed | Total | Pass rate | Mean best IoU |
| --- | ---: | ---: | ---: | ---: |
| 0 Cheez-it | 3 | 4 | 75.0% | 0.634 |
| 2 Scissors | 1 | 4 | 25.0% | 0.633 |
| 3 Mustard | 2 | 4 | 50.0% | 0.387 |
| 6 Clamp | 2 | 4 | 50.0% | 0.412 |
| 7 Banana | 2 | 4 | 50.0% | 0.669 |
| 9 meat_can | 1 | 4 | 25.0% | 0.202 |

## Inference diagnostic

| Setting | Pass rate | Mean best IoU | Recall@0.5 | False positives |
| --- | ---: | ---: | ---: | ---: |
| 640, conf 0.25 | 45.8% | 0.490 | 62.5% | 8 |
| 768, conf 0.25 | 41.7% | 0.610 | 79.2% | 11 |

Confidence sweep at 640 found `conf=0.25` to be the best pass-rate setting.
The live camera uses v7 at 640 with confidence 0.25.
