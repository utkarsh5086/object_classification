import os

BASE_DIR = "data/raw/fruits-360-100x100/Training"

GROUPS = {
    "Garlic": ["garlic bulb 1"],
    "Ginger": ["Ginger Root 1", "Ginger 2"],
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


def count_images(folder):
    """Count image files in a folder."""
    if not os.path.isdir(folder):
        return None

    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def main():
    print("Dataset:", BASE_DIR)
    print("=" * 60)

    total_images = 0

    for label, folders in GROUPS.items():
        print(f"\n{label}")

        class_total = 0

        for folder in folders:
            path = os.path.join(BASE_DIR, folder)
            count = count_images(path)

            if count is None:
                print(f"  {folder}: NOT FOUND")
            else:
                print(f"  {folder}: {count}")
                class_total += count

        print(f"  TOTAL: {class_total}")
        total_images += class_total

    print("\n" + "=" * 60)
    print(f"Total images: {total_images}")


if __name__ == "__main__":
    main()