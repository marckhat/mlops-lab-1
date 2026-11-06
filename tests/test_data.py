import pytest
from PIL import Image

from src.food11.data import (
    CLASSES,
    category_from_filename,
    find_duplicates,
    resize_image,
)


def test_category_from_filename():
    assert category_from_filename("0_12.jpg") == "Bread"
    assert category_from_filename("3_120.jpg") == "Egg"
    assert category_from_filename("10_3.jpg") == "Vegetable-Fruit"


def test_category_from_filename_rejects_unknown_labels():
    with pytest.raises(ValueError):
        category_from_filename("11_1.jpg")
    with pytest.raises(ValueError):
        category_from_filename("readme.txt")


def test_resize_image_shape_and_mode():
    img = Image.new("L", (512, 384))  # a grayscale, non-square image
    out = resize_image(img)
    assert out.size == (128, 128)
    assert out.mode == "RGB"


def test_eleven_classes():
    assert len(CLASSES) == 11


def test_classes_match_imagefolder_order():
    # ImageFolder sorts class folders alphabetically. serve.py maps the model's
    # output index to CLASSES, so both orders must be the same.
    assert CLASSES == sorted(CLASSES)


def test_find_duplicates_keeps_each_image_once(tmp_path):
    for split in ["training", "validation", "evaluation"]:
        (tmp_path / split).mkdir()
    (tmp_path / "training" / "0_1.jpg").write_bytes(b"A")
    (tmp_path / "training" / "0_2.jpg").write_bytes(b"B")
    (tmp_path / "evaluation" / "0_9.jpg").write_bytes(b"A")  # copy of a training image
    (tmp_path / "evaluation" / "0_5.jpg").write_bytes(b"C")
    (tmp_path / "validation" / "0_7.jpg").write_bytes(b"C")  # copy of an evaluation image
    (tmp_path / "validation" / "0_8.jpg").write_bytes(b"D")

    duplicates = find_duplicates(tmp_path, ["training", "evaluation", "validation"])

    # training is kept complete, evaluation wins over validation
    assert duplicates == {tmp_path / "evaluation" / "0_9.jpg", tmp_path / "validation" / "0_7.jpg"}
