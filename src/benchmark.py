import os
import time
import csv

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/mobilenet_v3_small_fp32_best.pth"
OUTPUT_FILE = "results/benchmark_fp32.csv"

NUM_CLASSES = 8
IMAGE_SIZE = 224

BATCH_SIZE = 1
WARMUP_RUNS = 50
BENCHMARK_RUNS = 200


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
# Load model
# ============================================================

print("\nLoading model...")

model = mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


# ============================================================
# Model statistics
# ============================================================

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

model_size_mb = os.path.getsize(MODEL_PATH) / (1024 ** 2)

print("\nModel statistics:")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"FP32 checkpoint size: {model_size_mb:.2f} MB")


# ============================================================
# Dummy input
# ============================================================

dummy_input = torch.randn(
    BATCH_SIZE,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
    device=device
)


# ============================================================
# Synchronization helper
# ============================================================

def synchronize():
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# Warm-up
# ============================================================

print("\nWarming up...")

with torch.inference_mode():
    for _ in range(WARMUP_RUNS):
        _ = model(dummy_input)

synchronize()

print(f"Warm-up complete ({WARMUP_RUNS} runs)")


# ============================================================
# Benchmark
# ============================================================

print(f"\nRunning benchmark ({BENCHMARK_RUNS} runs)...")

latencies_ms = []

with torch.inference_mode():

    for _ in range(BENCHMARK_RUNS):

        synchronize()

        start = time.perf_counter()

        _ = model(dummy_input)

        synchronize()

        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies_ms.append(latency_ms)


# ============================================================
# Statistics
# ============================================================

latencies_ms.sort()

mean_latency_ms = sum(latencies_ms) / len(latencies_ms)

median_latency_ms = latencies_ms[len(latencies_ms) // 2]

p95_index = int(0.95 * len(latencies_ms)) - 1
p95_latency_ms = latencies_ms[p95_index]

min_latency_ms = min(latencies_ms)
max_latency_ms = max(latencies_ms)

images_per_second = BATCH_SIZE / (mean_latency_ms / 1000)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 65)
print("FP32 BENCHMARK RESULTS")
print("=" * 65)

print(f"Device:              {device}")
print(f"Batch size:          {BATCH_SIZE}")
print(f"Input size:          {IMAGE_SIZE}x{IMAGE_SIZE}")

print(f"\nMean latency:        {mean_latency_ms:.3f} ms")
print(f"Median latency:      {median_latency_ms:.3f} ms")
print(f"P95 latency:         {p95_latency_ms:.3f} ms")
print(f"Minimum latency:     {min_latency_ms:.3f} ms")
print(f"Maximum latency:     {max_latency_ms:.3f} ms")

print(f"\nThroughput:          {images_per_second:.2f} images/s")

print(f"\nParameters:          {total_params:,}")
print(f"Model size:          {model_size_mb:.2f} MB")

print("=" * 65)


# ============================================================
# Save results
# ============================================================

os.makedirs("results", exist_ok=True)

results = {
    "model": "MobileNetV3-Small",
    "precision": "FP32",
    "device": str(device),
    "batch_size": BATCH_SIZE,
    "input_size": IMAGE_SIZE,
    "mean_latency_ms": mean_latency_ms,
    "median_latency_ms": median_latency_ms,
    "p95_latency_ms": p95_latency_ms,
    "min_latency_ms": min_latency_ms,
    "max_latency_ms": max_latency_ms,
    "images_per_second": images_per_second,
    "parameters": total_params,
    "model_size_mb": model_size_mb,
}

with open(OUTPUT_FILE, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results.keys()
    )

    writer.writeheader()
    writer.writerow(results)

print(f"\nSaved benchmark results to:")
print(OUTPUT_FILE)