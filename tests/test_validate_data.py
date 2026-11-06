"""Tests for the data contract, on a tiny dataset built in a temporary folder.

CI cannot see the real data (it lives in DVC), so these tests make sure the
checks themselves work: a clean dataset passes, a broken one fails.
"""

import shutil

import pytest
from PIL import Image

from src.food11.data import CLASSES
from src.food11.validate_data import SPLITS, main, validate

PER_CLASS = 3


@pytest.fixture
def dataset(tmp_path):
    """A valid dataset: 3 splits x 11 classes x 3 distinct 128x128 RGB images."""
    index = 0
    for split in SPLITS:
        for category in CLASSES:
            folder = tmp_path / split / category
            folder.mkdir(parents=True)
            for i in range(PER_CLASS):
                color = (index % 256, index // 256, 100)  # unique color -> unique file
                Image.new("RGB", (128, 128), color).save(folder / f"{i}.png")
                index += 1
    return tmp_path


def results_by_number(dataset):
    results = validate(dataset, min_per_class=PER_CLASS, tau=0.05)
    return {number: r.passed for number, r in enumerate(results, start=1)}


def test_clean_dataset_passes(dataset):
    assert all(results_by_number(dataset).values())
    assert main([str(dataset), "--min-per-class", str(PER_CLASS)]) == 0


def test_missing_class_folder_fails(dataset):
    shutil.rmtree(dataset / "validation" / "Rice")
    assert results_by_number(dataset)[1] is False


def test_wrong_image_size_and_mode_fail(dataset):
    Image.new("L", (64, 64)).save(dataset / "training" / "Bread" / "bad.png")
    assert results_by_number(dataset)[2] is False


def test_too_few_images_fails(dataset):
    (dataset / "evaluation" / "Egg" / "0.png").unlink()
    assert results_by_number(dataset)[3] is False


def test_train_test_duplicate_fails(dataset):
    # The same mistake as in the lab: copy a training image into evaluation.
    shutil.copy(dataset / "training" / "Bread" / "0.png", dataset / "evaluation" / "Bread" / "copy.png")
    assert results_by_number(dataset)[4] is False
    assert main([str(dataset), "--min-per-class", str(PER_CLASS)]) == 1


def test_class_imbalance_fails(dataset):
    for i in range(PER_CLASS, 30):
        Image.new("RGB", (128, 128), (i, 200, 200)).save(dataset / "training" / "Soup" / f"{i}.png")
    assert results_by_number(dataset)[5] is False
