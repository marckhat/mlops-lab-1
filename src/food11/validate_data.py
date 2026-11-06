"""Check a processed Food-11 dataset against the data contract.

Checks:
  1. each split (training, validation, evaluation) has exactly the 11 class folders
  2. every image opens with PIL, is RGB and is 128x128
  3. every class has at least --min-per-class images in every split
  4. no image appears in two splits (same MD5 hash of the file content)
  5. each class's share of the training split is within --tau of its share
     of the validation split

Prints one PASS/FAIL line per check and exits with status 1 if any check failed.

Usage:
    uv run python src/food11/validate_data.py data/food11_processed_mini
    uv run python src/food11/validate_data.py data/food11_processed --min-per-class 50
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

try:
    from src.food11.data import CLASSES, IMAGE_SIZE
except ModuleNotFoundError:  # run as a script: python src/food11/validate_data.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.food11.data import CLASSES, IMAGE_SIZE

SPLITS = ["training", "validation", "evaluation"]
MAX_EXAMPLES = 5  # problems printed per check


@dataclass
class CheckResult:
    name: str
    passed: bool
    summary: str
    problems: list[str] = field(default_factory=list)


def list_images(dataset: Path, split: str, category: str) -> list[Path]:
    folder = dataset / split / category
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file())


def all_images(dataset: Path):
    """Yield (split, category, path) for every image in the expected folders."""
    for split in SPLITS:
        for category in CLASSES:
            for path in list_images(dataset, split, category):
                yield split, category, path


def check_class_folders(dataset: Path) -> CheckResult:
    problems = []
    for split in SPLITS:
        split_dir = dataset / split
        if not split_dir.is_dir():
            problems.append(f"{split}: split folder is missing")
            continue
        found = {p.name for p in split_dir.iterdir() if p.is_dir()}
        for name in sorted(set(CLASSES) - found):
            problems.append(f"{split}: missing class folder '{name}'")
        for name in sorted(found - set(CLASSES)):
            problems.append(f"{split}: unexpected folder '{name}'")
    return CheckResult(
        "class folders",
        not problems,
        f"{len(CLASSES)} classes expected in each of {len(SPLITS)} splits",
        problems,
    )


def check_images(dataset: Path, size: int = IMAGE_SIZE) -> CheckResult:
    problems = []
    total = 0
    for split, category, path in all_images(dataset):
        total += 1
        rel = path.relative_to(dataset).as_posix()
        try:
            with Image.open(path) as img:
                img.load()  # decode the pixels, not just the header
                if img.mode != "RGB":
                    problems.append(f"{rel}: mode {img.mode}, expected RGB")
                if img.size != (size, size):
                    problems.append(f"{rel}: size {img.size}, expected ({size}, {size})")
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
            # OSError covers unreadable, truncated and unidentified image files.
            problems.append(f"{rel}: cannot open ({error})")
    return CheckResult(
        "images RGB 128x128",
        not problems and total > 0,
        f"{total} images checked, {len(problems)} bad" if total else "no images found",
        problems,
    )


def check_min_per_class(dataset: Path, min_per_class: int) -> CheckResult:
    problems = []
    smallest = None
    for split in SPLITS:
        for category in CLASSES:
            count = len(list_images(dataset, split, category))
            smallest = count if smallest is None else min(smallest, count)
            if count < min_per_class:
                problems.append(f"{split}/{category}: {count} images < {min_per_class}")
    return CheckResult(
        f"at least {min_per_class} images per class",
        not problems,
        f"smallest class has {smallest} images",
        problems,
    )


def check_no_cross_split_duplicates(dataset: Path) -> CheckResult:
    paths_by_hash = defaultdict(list)
    for split, category, path in all_images(dataset):
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        paths_by_hash[digest].append((split, path))

    problems = []
    for entries in paths_by_hash.values():
        if len({split for split, _ in entries}) > 1:
            names = ", ".join(p.relative_to(dataset).as_posix() for _, p in entries)
            problems.append(f"same image in several splits: {names}")
    return CheckResult(
        "no image in two splits",
        not problems,
        f"{len(problems)} images shared between splits",
        problems,
    )


def class_shares(dataset: Path, split: str) -> dict[str, float]:
    counts = {c: len(list_images(dataset, split, c)) for c in CLASSES}
    total = sum(counts.values())
    return {c: (n / total if total else 0.0) for c, n in counts.items()}


def check_class_balance(dataset: Path, tau: float) -> CheckResult:
    train = class_shares(dataset, "training")
    val = class_shares(dataset, "validation")
    problems = []
    worst = 0.0
    for category in CLASSES:
        gap = abs(train[category] - val[category])
        worst = max(worst, gap)
        if gap > tau:
            problems.append(
                f"{category}: {train[category]:.1%} of training vs "
                f"{val[category]:.1%} of validation (gap {gap:.1%} > {tau:.1%})"
            )
    return CheckResult(
        f"class shares train vs val within {tau:.0%}",
        not problems,
        f"largest gap {worst:.1%}",
        problems,
    )


def validate(dataset: Path, min_per_class: int = 50, tau: float = 0.05) -> list[CheckResult]:
    """Run every check of the data contract on one processed dataset folder."""
    dataset = Path(dataset)
    return [
        check_class_folders(dataset),
        check_images(dataset),
        check_min_per_class(dataset, min_per_class),
        check_no_cross_split_duplicates(dataset),
        check_class_balance(dataset, tau),
    ]


def print_report(dataset: Path, results: list[CheckResult]) -> None:
    print(f"Data contract for {dataset}")
    for number, result in enumerate(results, start=1):
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status}  {number}. {result.name} - {result.summary}")
        for problem in result.problems[:MAX_EXAMPLES]:
            print(f"          {problem}")
        if len(result.problems) > MAX_EXAMPLES:
            print(f"          ... and {len(result.problems) - MAX_EXAMPLES} more")
    failed = sum(not r.passed for r in results)
    if failed:
        print(f"FAILED: {failed} of {len(results)} checks failed")
    else:
        print(f"PASSED: all {len(results)} checks passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a processed Food-11 dataset.")
    parser.add_argument("dataset", type=Path, help="e.g. data/food11_processed_mini")
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args(argv)

    if not args.dataset.is_dir():
        print(f"Dataset folder not found: {args.dataset}")
        return 1

    results = validate(args.dataset, args.min_per_class, args.tau)
    print_report(args.dataset, results)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
