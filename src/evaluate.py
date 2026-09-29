import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
)

from dataset import (
    FruitsVegetablesDataset,
    CLASS_NAMES,
)


# ============================================================
# Configuration
# ============================================================

TEST_DIR = "data/raw/fruits-360-100x100/Test"

MODEL_PATH = "models/mobilenet_v3_small_fp32_best.pth"

BATCH_SIZE = 64

NUM_WORKERS = 0

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# Transform
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Load test dataset
# ============================================================

print("\nLoading test dataset...")

test_dataset = FruitsVegetablesDataset(
    TEST_DIR,
    transform=test_transform,
)

print(f"Test images: {len(test_dataset)}")


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# Load model
# ============================================================

print("\nLoading model...")

model = mobilenet_v3_small(
    weights=None
)

num_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    num_features,
    len(CLASS_NAMES),
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


print(
    f"Loaded checkpoint from epoch "
    f"{checkpoint['epoch']}"
)

print(
    f"Validation accuracy: "
    f"{checkpoint['val_accuracy']:.2f}%"
)


# ============================================================
# Parameter count
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nModel parameters:")
print(
    f"Total parameters:     {total_parameters:,}"
)
print(
    f"Trainable parameters: {trainable_parameters:,}"
)


# ============================================================
# Model size
# ============================================================

model_size_bytes = os.path.getsize(
    MODEL_PATH
)

model_size_mb = (
    model_size_bytes / (1024 ** 2)
)

print(
    f"FP32 checkpoint size: "
    f"{model_size_mb:.2f} MB"
)


# ============================================================
# Evaluation
# ============================================================

print("\nEvaluating...")
print("=" * 70)

all_labels = []
all_predictions = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# Overall accuracy
# ============================================================

accuracy = 100.0 * correct / total

print(
    f"\nTest accuracy: "
    f"{accuracy:.2f}%"
)

print(
    f"Correct: {correct:,} / {total:,}"
)

print(
    f"Incorrect: {total - correct:,}"
)


# ============================================================
# Macro F1
# ============================================================

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
)

print(
    f"Macro F1: "
    f"{macro_f1:.4f}"
)


# ============================================================
# Per-class accuracy
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
)

print("\nPer-class accuracy:")
print("-" * 50)

class_accuracies = []

for i, class_name in enumerate(CLASS_NAMES):

    class_total = cm[i].sum()

    class_correct = cm[i, i]

    class_accuracy = (
        100.0 * class_correct / class_total
    )

    class_accuracies.append(
        class_accuracy
    )

    print(
        f"{class_name:15s}: "
        f"{class_accuracy:6.2f}% "
        f"({class_correct}/{class_total})"
    )


# ============================================================
# Classification report
# ============================================================

print("\nClassification report:")
print("=" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4,
)

print(report)


# ============================================================
# Confusion matrix
# ============================================================

print("\nConfusion matrix:")
print("=" * 70)

print("Rows = actual")
print("Columns = predicted")
print()

print(
    f"{'':15s}"
    + "".join(
        f"{name[:8]:>10s}"
        for name in CLASS_NAMES
    )
)

for i, class_name in enumerate(CLASS_NAMES):

    row = ""

    for j in range(len(CLASS_NAMES)):
        row += f"{cm[i, j]:10d}"

    print(
        f"{class_name:15s}{row}"
    )


# ============================================================
# Save confusion matrix
# ============================================================

cm_file = os.path.join(
    RESULTS_DIR,
    "confusion_matrix_fp32.csv",
)

np.savetxt(
    cm_file,
    cm,
    delimiter=",",
    fmt="%d",
)

print(
    f"\nSaved confusion matrix to:"
    f"\n{cm_file}"
)


# ============================================================
# Save classification report
# ============================================================

report_file = os.path.join(
    RESULTS_DIR,
    "classification_report_fp32.txt",
)

with open(
    report_file,
    "w",
) as f:

    f.write(
        f"Test accuracy: {accuracy:.4f}%\n"
    )

    f.write(
        f"Macro F1: {macro_f1:.4f}\n"
    )

    f.write(
        f"Total parameters: "
        f"{total_parameters}\n"
    )

    f.write(
        f"FP32 checkpoint size: "
        f"{model_size_mb:.2f} MB\n\n"
    )

    f.write(report)

print(
    f"Saved classification report to:"
    f"\n{report_file}"
)

print("\nEvaluation complete.")