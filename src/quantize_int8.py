import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/mobilenet_v3_small_fp32_best.pth"
OUTPUT_PATH = "models/mobilenet_v3_small_int8_dynamic.pth"

NUM_CLASSES = 8


# ============================================================
# Load FP32 model
# ============================================================

print("Loading FP32 model...")

model = mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.cpu()
model.eval()

print("FP32 model loaded.")


# ============================================================
# FP32 statistics
# ============================================================

fp32_params = sum(p.numel() for p in model.parameters())
fp32_size_mb = os.path.getsize(MODEL_PATH) / (1024 ** 2)

print("\nFP32 model:")
print(f"Parameters: {fp32_params:,}")
print(f"Checkpoint size: {fp32_size_mb:.2f} MB")


# ============================================================
# Dynamic INT8 quantization
# ============================================================

print("\nQuantizing model to INT8...")

quantized_model = torch.ao.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

print("INT8 quantization complete.")


# ============================================================
# Save INT8 model
# ============================================================

torch.save(
    quantized_model.state_dict(),
    OUTPUT_PATH
)

int8_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)

print("\nINT8 model:")
print(f"Checkpoint size: {int8_size_mb:.2f} MB")

print("\nSize reduction:")
print(
    f"{fp32_size_mb / int8_size_mb:.2f}x smaller"
)

print(
    f"{(1 - int8_size_mb / fp32_size_mb) * 100:.1f}% reduction"
)

print(f"\nSaved INT8 model to:")
print(OUTPUT_PATH)