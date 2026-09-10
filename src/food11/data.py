from pathlib import Path
from PIL import Image
import shutil

LABELS = {
    "0": "Bread",
    "1": "Dairy product",
    "2": "Dessert",
    "3": "Egg",
    "4": "Fried food",
    "5": "Meat",
    "6": "Noodles-Pasta",
    "7": "Rice",
    "8": "Seafood",
    "9": "Soup",
    "10": "Vegetable-Fruit"
}

RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

for target in [PROCESSED_DIR, MINI_DIR]:
    if target.exists():
        shutil.rmtree(target)

for split in ["training", "evaluation", "validation"]:

    counters = {name: 0 for name in LABELS.values()}

    for img_path in (RAW_DIR / split).glob("*"):

        label = img_path.stem.split("_")[0]

        if label not in LABELS:
            continue

        category = LABELS[label]

        processed_category = PROCESSED_DIR / split / category
        processed_category.mkdir(parents=True, exist_ok=True)

        with Image.open(img_path) as img:
            img = img.resize((128, 128))
            img.save(processed_category / img_path.name)

        if counters[category] < 100:

            mini_category = MINI_DIR / split / category
            mini_category.mkdir(parents=True, exist_ok=True)

            with Image.open(img_path) as img:
                img = img.resize((128, 128))
                img.save(mini_category / img_path.name)

            counters[category] += 1

print("Finished creating datasets.")