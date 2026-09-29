# Object Classification for Low-Cost Edge AI

This project explores low-cost edge AI for image-based object and plant
classification. The initial goal is to develop a lightweight vision model
that can identify common fruits and vegetables using inexpensive edge
hardware.

The project will progressively move from controlled object classification
toward real-world plant identification, damage/health assessment, and
eventually object detection.

---

## Current Status

### Baseline model

- Model: MobileNetV3-Small
- Pretrained on ImageNet
- Precision: FP32
- Input resolution: 224 × 224
- Number of classes: 8
- Parameters: 1.53M
- FP32 checkpoint size: 5.95 MB

### Controlled test performance

- Test accuracy: 99.40%
- Macro F1-score: 0.9906

### Inference benchmark

Measured on an Apple Silicon Mac using PyTorch MPS:

- Batch size: 1
- Mean latency: 6.439 ms
- Median latency: 5.858 ms
- P95 latency: 7.648 ms
- Throughput: 155.29 images/s

These measurements represent the software baseline on the development
machine and should not be interpreted as edge-device performance.

---

## Objectives

The project is being developed in stages:

1. Object classification
2. Real-world image classification
3. Plant identification
4. Plant damage and health assessment
5. Care recommendations
6. Object detection
7. Deployment on low-cost edge hardware

The primary design goal is to investigate the tradeoff between:

- Accuracy
- Model size
- Inference latency
- Memory usage
- Energy consumption
- Hardware cost

---

## Dataset

The initial experiments use the Fruits-360 100x100 dataset.

Dataset source:

https://github.com/fruits-360/fruits-360-100x100

Eight classes are currently selected:

- Garlic
- Ginger
- Onion Red
- Onion White
- Potato Red
- Potato Sweet
- Potato White
- Tomato

The dataset is used only for the initial controlled classification
experiments.

The dataset is not included in this repository.

---

## Model

The baseline model is MobileNetV3-Small pretrained on ImageNet.

The final classification layer is modified for the eight target classes.

The current baseline uses FP32 inference.

Quantization and other deployment optimizations may be evaluated later.

---

## Training

The current training configuration includes:

- Input size: 224 × 224
- Batch size: 64
- Training epochs: 10
- Initial learning rate: 1e-3
- 80/20 training-validation split
- Random horizontal flip
- Random rotation
- Color jitter
- ImageNet normalization
- Class-weighted cross entropy

The ImageNet-pretrained feature extractor is initially frozen while the
classification layer is trained.

The best validation checkpoint is saved to:

models/mobilenet_v3_small_fp32_best.pth

Model checkpoints are excluded from Git.

---

## Results

The current FP32 model achieves:

| Metric | Result |
|---|---:|
| Test accuracy | 99.40% |
| Macro F1 | 0.9906 |
| Parameters | 1,526,056 |
| Model size | 5.95 MB |
| Mean latency | 6.439 ms |
| Median latency | 5.858 ms |
| P95 latency | 7.648 ms |
| Throughput | 155.29 images/s |

The controlled Fruits-360 test set is highly structured, so these results
should not be considered representative of real-world camera performance.

---

## Real-World Evaluation

Real-world images are being collected using a phone camera to evaluate
robustness to conditions not represented in the controlled dataset.

The real-world evaluation will vary:

- Background
- Lighting
- Object orientation
- Object size
- Camera distance
- Camera angle
- Shadows
- Different physical objects

The real-world dataset is kept locally and is not included in the
repository.

The current model has successfully classified several initial real-world
images, with some individual misclassifications observed.

A systematic real-world evaluation will be performed as the dataset grows.

---

## Inference

To classify an individual image:

```bash
python src/inference.py path/to/image.jpg