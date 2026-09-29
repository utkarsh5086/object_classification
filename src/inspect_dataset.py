import os
import random
from PIL import Image, ImageOps, ImageDraw

BASE_DIR = "data/raw/fruits-360-100x100/Training"
OUTPUT_DIR = "results"

GROUPS = {
    "Garlic": ["garlic bulb 1"],

    "Ginger": [
        "Ginger Root 1",
        "Ginger 2",
    ],

    "Onion Red": [
        "Onion Red 1",
        "Onion Red 2",
        "Onion Red 3",
    ],

    "Onion White": [
        "Onion White 1",
        "Onion White 2",
    ],

    "Potato Red": [
        "Potato Red 1",
        "Potato Red 2",
    ],

    "Potato Sweet": [
        "Potato Sweet 1",
    ],

    "Potato White": [
        "Potato White 1",
    ],

    "Tomato": [
        "Tomato 1",
        "Tomato 10",
        "Tomato 11",
        "Tomato 2",
        "Tomato 3",
        "Tomato 4",
        "Tomato 5",
        "Tomato 7",
        "Tomato 8",
        "Tomato 9",
    ],
}

NUM_SAMPLES = 5
IMAGE_SIZE = 150
LABEL_HEIGHT = 30
PADDING = 10

random.seed(42)


def get_images(folder):
    """Return all image paths in a folder."""
    if not os.path.isdir(folder):
        return []

    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def make_contact_sheet(label, image_paths):
    """Create a contact sheet for one class."""

    samples = random.sample(
        image_paths,
        min(NUM_SAMPLES, len(image_paths))
    )

    sheet_width = (
        NUM_SAMPLES * IMAGE_SIZE
        + (NUM_SAMPLES + 1) * PADDING
    )

    sheet_height = IMAGE_SIZE + LABEL_HEIGHT + 2 * PADDING

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, image_path in enumerate(samples):

        img = Image.open(image_path).convert("RGB")

        # Make image square while preserving aspect ratio
        img = ImageOps.fit(
            img,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        x = PADDING + i * (IMAGE_SIZE + PADDING)
        y = PADDING

        sheet.paste(img, (x, y))

    draw.text(
        (PADDING, IMAGE_SIZE + PADDING),
        label,
        fill="black"
    )

    return sheet


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Inspecting selected dataset classes...")
    print("=" * 60)

    sheets = []

    for label, folders in GROUPS.items():

        image_paths = []

        for folder in folders:
            path = os.path.join(BASE_DIR, folder)
            image_paths.extend(get_images(path))

        print(f"{label:15s}: {len(image_paths)} images")

        if len(image_paths) == 0:
            print("  WARNING: No images found!")
            continue

        sheet = make_contact_sheet(label, image_paths)
        sheets.append((label, sheet))

    # Create one large contact sheet
    total_height = sum(
        sheet.height + PADDING
        for _, sheet in sheets
    )

    max_width = max(
        sheet.width
        for _, sheet in sheets
    )

    final_sheet = Image.new(
        "RGB",
        (max_width, total_height),
        "white"
    )

    y = 0

    for label, sheet in sheets:
        final_sheet.paste(sheet, (0, y))
        y += sheet.height + PADDING

    output_file = os.path.join(
        OUTPUT_DIR,
        "dataset_samples.jpg"
    )

    final_sheet.save(output_file, quality=95)

    print("=" * 60)
    print(f"Saved sample images to:")
    print(output_file)


if __name__ == "__main__":
    main()