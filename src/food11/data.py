"""Prepare the Food-11 dataset for ResNet training.

Reads data/food11_raw/<split>/<label>_<n>.jpg and writes two copies,
resized to 128x128 RGB and sorted into one folder per class:

    data/food11_processed/<split>/<class>/*.jpg        (all images)
    data/food11_processed_mini/<split>/<class>/*.jpg   (at most 100 per class)

The raw Food-11 data contains byte-identical images in several splits (a
train/test leak). Each image is kept only once, in the first split of
DEDUP_PRIORITY, so no image is in two splits.

Usage:
    uv run python src/food11/data.py
"""

import hashlib
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
# A duplicated image is kept in the first split of this list and dropped from
# the others: training stays complete, and the test split wins over validation.
DEDUP_PRIORITY = ["training", "evaluation", "validation"]
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


def find_duplicates(raw_dir: Path, priority: list[str] = DEDUP_PRIORITY) -> set[Path]:
    """Return the raw files to skip so that every image is kept only once.

    Splits are read in `priority` order and files in name order; a file whose
    content (MD5) was already seen is a duplicate.
    """
    seen = set()
    duplicates = set()
    for split in priority:
        for path in sorted((raw_dir / split).glob("*")):
            if not path.is_file():
                continue
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if digest in seen:
                duplicates.add(path)
            else:
                seen.add(digest)
    return duplicates


def prepare_split(split: str, skip: set[Path]) -> None:
    """Resize every raw image of one split into the processed and mini folders."""
    mini_counts = {name: 0 for name in CLASSES}

    for img_path in sorted((RAW_DIR / split).glob("*")):
        if img_path in skip:
            continue  # same image already kept in another split
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

    duplicates = find_duplicates(RAW_DIR)
    for split in SPLITS:
        dropped = sum(1 for p in duplicates if p.parent.name == split)
        print(f"{split}: dropping {dropped} duplicated images")
        prepare_split(split, duplicates)

    print("Finished creating datasets.")


if __name__ == "__main__":
    main()
