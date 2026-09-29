import os
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# Dataset configuration
# ============================================================

CLASS_FOLDERS = {
    "Garlic": [
        "garlic bulb 1",
    ],

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


CLASS_NAMES = list(CLASS_FOLDERS.keys())

CLASS_TO_IDX = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

IDX_TO_CLASS = {
    idx: name
    for name, idx in CLASS_TO_IDX.items()
}


# ============================================================
# Dataset
# ============================================================

class FruitsVegetablesDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        for class_name, folders in CLASS_FOLDERS.items():

            label = CLASS_TO_IDX[class_name]

            for folder in folders:

                folder_path = os.path.join(
                    root_dir,
                    folder
                )

                if not os.path.isdir(folder_path):
                    print(
                        f"WARNING: folder not found: "
                        f"{folder_path}"
                    )
                    continue

                for filename in sorted(
                    os.listdir(folder_path)
                ):

                    if filename.lower().endswith(
                        (".jpg", ".jpeg", ".png")
                    ):

                        image_path = os.path.join(
                            folder_path,
                            filename
                        )

                        self.samples.append(
                            (image_path, label)
                        )

        print(
            f"Loaded {len(self.samples)} images "
            f"from {len(CLASS_NAMES)} classes."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label