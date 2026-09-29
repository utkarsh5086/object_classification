import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/mobilenet_v3_small_fp32_best.pth"

CLASS_NAMES = [
    "Garlic",
    "Ginger",
    "Onion Red",
    "Onion White",
    "Potato Red",
    "Potato Sweet",
    "Potato White",
    "Tomato",
]

IMAGE_SIZE = 224
TOP_K = 5


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

model = mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(CLASS_NAMES)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# Inference function
# ============================================================

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.inference_mode():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    top_probabilities, top_indices = torch.topk(
        probabilities,
        min(TOP_K, len(CLASS_NAMES))
    )

    print("\n" + "=" * 50)
    print("PREDICTION")
    print("=" * 50)

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        class_name = CLASS_NAMES[index.item()]
        confidence = probability.item() * 100

        print(
            f"{class_name:<15} {confidence:6.2f}%"
        )

    predicted_class = CLASS_NAMES[
        top_indices[0].item()
    ]

    print("=" * 50)
    print(f"Predicted class: {predicted_class}")
    print("=" * 50)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:\n"
            "python src/inference.py <image_path>\n"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    predict(image_path)