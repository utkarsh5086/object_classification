import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

from src.dataset import (
    FruitsVegetablesDataset,
    CLASS_NAMES,
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data/raw/fruits-360-100x100/Training"
MODEL_DIR = "models"

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

VAL_RATIO = 0.20
SEED = 42

NUM_WORKERS = 0

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
# Image transforms
# ============================================================

weights = MobileNet_V3_Small_Weights.DEFAULT

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std,
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std,
    ),
])


# ============================================================
# Load dataset
# ============================================================

print("\nLoading dataset...")

full_dataset = FruitsVegetablesDataset(
    DATA_DIR,
    transform=None,
)

print(f"Total images: {len(full_dataset)}")


# ============================================================
# Train / validation split
# ============================================================

val_size = int(len(full_dataset) * VAL_RATIO)
train_size = len(full_dataset) - val_size

generator = torch.Generator().manual_seed(SEED)

train_indices, val_indices = random_split(
    range(len(full_dataset)),
    [train_size, val_size],
    generator=generator,
)

train_indices = train_indices.indices
val_indices = val_indices.indices


# ============================================================
# Dataset wrapper
# ============================================================

class SubsetWithTransform(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset,
        indices,
        transform,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        original_idx = self.indices[idx]

        image_path, label = self.dataset.samples[
            original_idx
        ]

        from PIL import Image

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


train_dataset = SubsetWithTransform(
    full_dataset,
    train_indices,
    train_transform,
)

val_dataset = SubsetWithTransform(
    full_dataset,
    val_indices,
    val_transform,
)


print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# ============================================================
# Data loaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# Calculate class weights
# ============================================================

class_counts = np.zeros(len(CLASS_NAMES))

for _, label in full_dataset.samples:
    class_counts[label] += 1

class_weights = (
    len(full_dataset)
    / (len(CLASS_NAMES) * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
)

print("\nClass counts:")

for i, name in enumerate(CLASS_NAMES):
    print(
        f"{name:15s}: "
        f"{int(class_counts[i]):5d} "
        f"(weight = {class_weights[i]:.3f})"
    )


# ============================================================
# Model
# ============================================================

print("\nLoading MobileNetV3-Small...")

model = mobilenet_v3_small(
    weights=weights
)

# Replace ImageNet classifier
num_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    num_features,
    len(CLASS_NAMES),
)

model = model.to(device)


# ============================================================
# Freeze feature extractor
# ============================================================

for parameter in model.features.parameters():
    parameter.requires_grad = False


print(
    "\nFeature extractor frozen."
)

print(
    "Training classifier only."
)


# ============================================================
# Loss and optimizer
# ============================================================

class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Training functions
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    class_correct = np.zeros(
        len(CLASS_NAMES)
    )

    class_total = np.zeros(
        len(CLASS_NAMES)
    )

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        for label, prediction in zip(
            labels,
            predicted,
        ):

            label = label.item()
            prediction = prediction.item()

            class_total[label] += 1

            if label == prediction:
                class_correct[label] += 1

    loss = running_loss / total

    accuracy = 100.0 * correct / total

    class_accuracy = (
        100.0 * class_correct / class_total
    )

    return loss, accuracy, class_accuracy


# ============================================================
# Training loop
# ============================================================

best_val_accuracy = 0.0

print("\nStarting training...")
print("=" * 70)

for epoch in range(NUM_EPOCHS):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy, class_accuracy = validate()

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Acc:   {val_accuracy:.2f}%"
    )

    print("Per-class validation accuracy:")

    for name, acc in zip(
        CLASS_NAMES,
        class_accuracy,
    ):
        print(
            f"  {name:15s}: {acc:.2f}%"
        )

    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        model_path = os.path.join(
            MODEL_DIR,
            "mobilenet_v3_small_fp32_best.pth",
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1,
            },
            model_path,
        )

        print(
            f"Saved best model → {model_path}"
        )


print("\n" + "=" * 70)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)