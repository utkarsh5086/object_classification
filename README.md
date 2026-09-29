# Object Classification for Low-Cost Edge AI
This project explores low-cost edge AI for real-world image classification. The initial focus is on developing lightweight vision models for identifying common fruits and vegetables, with a long-term goal of extending the system to plant identification, health and damage assessment, care recommendations, and object detection.

## Quick Start

### 1. Clone the repository

git clone git@github.com:utkarsh5086/object_classification.git
cd object_classification

### 2. Create a virtual environment

python -m venv .venv
source .venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Prepare the dataset

Download the [Fruits-360 100x100](https://github.com/fruits-360/fruits-360-100x100) dataset and place it under:

data/raw/fruits-360-100x100/

### 5. Train the model

python src/train.py

### 6. Evaluate the model

python src/evaluate.py

### 7. Run inference on an image

python src/inference.py path/to/image.jpg

### 8. Benchmark inference performance

python src/benchmark.py

## Prerequisites

- Python 3.10+
- PyTorch
- torchvision
- A CPU, CUDA GPU, or Apple Silicon Mac
- ~3 GB of storage for the dataset

## Dataset Setup

This project uses the [Fruits-360 100x100](https://github.com/fruits-360/fruits-360-100x100) dataset for initial model development and evaluation.

Download the dataset and place it in:

data/raw/fruits-360-100x100/

The expected directory structure is:

data/
└── raw/
    └── fruits-360-100x100/
        ├── Training/
        ├── Test/
        ├── LICENSE
        └── README.md

The `Training` and `Test` directories are used for model training and evaluation, respectively.

> **Note:** The dataset is not included in this repository.

## Classes

The initial model classifies images into eight categories:

| Class | Description |
|---|---|
| Garlic | Garlic bulbs |
| Ginger | Ginger root |
| Onion Red | Red onions |
| Onion White | White onions |
| Potato Red | Red-skinned potatoes |
| Potato Sweet | Sweet potatoes |
| Potato White | White potatoes |
| Tomato | Tomatoes |

Some classes combine multiple corresponding folders from the original Fruits-360 dataset into a single logical class.

## Training

The baseline model uses an ImageNet-pretrained MobileNetV3-Small. The
feature extractor is frozen, and only the final classifier is trained
for the eight target classes.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | MobileNetV3-Small |
| Pretraining | ImageNet |
| Input size | 224 × 224 |
| Batch size | 64 |
| Epochs | 10 |
| Learning rate | 1 × 10⁻³ |
| Optimizer | Adam |
| Loss | Class-weighted Cross Entropy |
| Train/Validation split | 80/20 |
| Random seed | 42 |
| Feature extractor | Frozen |
| Data augmentation | Horizontal flip, rotation, color jitter |

To train the model:
```bash
python src/train.py
```

The best model checkpoint is saved to:
models/mobilenet_v3_small_fp32_best.pth


## Evaluation
## Inference
## Benchmark
## Real-World Evaluation
## Current Results
## Repository Structure
## Troubleshooting
## Roadmap
## Reproducibility
## License