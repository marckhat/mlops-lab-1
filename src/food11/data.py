"""Prepare the Food-11 dataset for ResNet training.

Reads data/food11_raw/<split>/<label>_<n>.jpg and writes two copies,
resized to 128x128 RGB and sorted into one folder per class:

    data/food11_processed/<split>/<class>/*.jpg        (all images)
    data/food11_processed_mini/<split>/<class>/*.jpg   (at most 100 per class)

Usage:
    uv run python src/food11/data.py
"""

import shutil
from pathlib import Path

from PIL import Image

# Index i is the class with label i in the raw file names.
CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

SPLITS = ["training", "validation", "evaluation"]
IMAGE_SIZE = 128
MINI_PER_CLASS = 100

RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")


def category_from_filename(filename: str) -> str:
    """'3_120.jpg' -> 'Egg'"""
    label = Path(filename).stem.split("_")[0]
    if not label.isdigit() or int(label) >= len(CLASSES):
        raise ValueError(f"No Food-11 class label in file name: {filename!r}")
    return CLASSES[int(label)]


def resize_image(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Returns an RGB image of size x size."""
    # convert() first: some raw images are grayscale or CMYK, and the model
    # expects 3 channels.
    return img.convert("RGB").resize((size, size))


def prepare_split(split: str) -> None:
    """Resize every raw image of one split into the processed and mini folders."""
    mini_counts = {name: 0 for name in CLASSES}

    for img_path in sorted((RAW_DIR / split).glob("*")):
        try:
            category = category_from_filename(img_path.name)
        except ValueError:
            continue  # not a Food-11 image

        processed_path = PROCESSED_DIR / split / category / img_path.name
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(img_path) as img:
            resize_image(img).save(processed_path)

        if mini_counts[category] < MINI_PER_CLASS:
            mini_path = MINI_DIR / split / category / img_path.name
            mini_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(processed_path, mini_path)
            mini_counts[category] += 1


def main() -> None:
    for target in [PROCESSED_DIR, MINI_DIR]:
        if target.exists():
            shutil.rmtree(target)

    for split in SPLITS:
        prepare_split(split)

    print("Finished creating datasets.")


if __name__ == "__main__":
    main()
