"""Quality gate for the food11 model registry.

Moves the `champion` alias to a candidate version only if the candidate is
clearly better than the current champion:

    acc_candidate >= acc_champion + 2 * se,   se = sqrt(p * (1 - p) / n)

where p is the champion's test accuracy and n the size of the test split.
Otherwise the candidate gets the `challenger` alias and `champion` stays put.
In both cases the decision and both accuracies are written as tags on the
candidate version.

Usage:
    uv run python src/food11/promote.py --version 8
"""

import argparse
import math
import os
import sys
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "food11"
CHAMPION = "champion"
CHALLENGER = "challenger"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FOLDERS = {"mini": "food11_processed_mini", "processed": "food11_processed"}


def standard_error(p: float, n: int) -> float:
    """Standard error of an accuracy p measured on n test images."""
    return math.sqrt(p * (1 - p) / n)


def gate(acc_candidate: float, acc_champion: float, n: int) -> tuple[bool, float, float]:
    """Return (promote?, threshold, se) for the 2-standard-error rule."""
    se = standard_error(acc_champion, n)
    threshold = acc_champion + 2 * se
    return acc_candidate >= threshold, threshold, se


def count_test_images(dataset: str) -> int | None:
    """Count the images in the evaluation split of a local dataset folder."""
    folder = DATASET_FOLDERS.get(dataset)
    if folder is None:
        return None
    evaluation = PROJECT_ROOT / "data" / folder / "evaluation"
    if not evaluation.is_dir():
        return None
    return sum(1 for p in evaluation.glob("*/*") if p.is_file())


def get_test_set_size(run, override: int | None) -> int:
    """n: --n-test, else the run's test_size param, else count the local files."""
    if override:
        return override
    params = run.data.params
    if "test_size" in params:
        return int(params["test_size"])
    n = count_test_images(params.get("dataset", ""))
    if n:
        return n
    sys.exit(
        f"Cannot find the test-set size for run {run.info.run_id}: "
        "pass it with --n-test."
    )


def get_metric(client: MlflowClient, run_id: str, metric: str, label: str) -> float:
    metrics = client.get_run(run_id).data.metrics
    if metric not in metrics:
        sys.exit(f"The {label} run {run_id} has no '{metric}' metric.")
    return metrics[metric]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Apply the quality gate to a model version.")
    parser.add_argument("--version", type=int, required=True, help="candidate model version")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--metric", default="test_accuracy")
    parser.add_argument("--n-test", type=int, default=None, help="size of the test split")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    name = args.model_name

    candidate = client.get_model_version(name, str(args.version))
    try:
        champion = client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        champion = None

    acc_candidate = get_metric(client, candidate.run_id, args.metric, "candidate")

    if champion is None:
        client.set_registered_model_alias(name, CHAMPION, candidate.version)
        client.set_model_version_tag(name, candidate.version, "gate_decision", "promoted")
        client.set_model_version_tag(
            name, candidate.version, "gate_reason", "no previous champion"
        )
        client.set_model_version_tag(
            name, candidate.version, "gate_candidate_accuracy", f"{acc_candidate:.4f}"
        )
        print(f"No champion yet: version {candidate.version} is now '{CHAMPION}'.")
        return

    if champion.version == candidate.version:
        print(f"Version {candidate.version} is already '{CHAMPION}'. Nothing to do.")
        return

    champion_run = client.get_run(champion.run_id)
    candidate_run = client.get_run(candidate.run_id)
    champion_data = champion_run.data.params.get("dataset")
    candidate_data = candidate_run.data.params.get("dataset")
    if champion_data and candidate_data and champion_data != candidate_data:
        sys.exit(
            f"Not comparable: champion was tested on '{champion_data}', "
            f"candidate on '{candidate_data}'."
        )

    acc_champion = get_metric(client, champion.run_id, args.metric, "champion")
    n = get_test_set_size(champion_run, args.n_test)
    promote, threshold, se = gate(acc_candidate, acc_champion, n)

    if promote:
        client.set_registered_model_alias(name, CHAMPION, candidate.version)
        # A champion is no longer a challenger.
        try:
            if client.get_model_version_by_alias(name, CHALLENGER).version == candidate.version:
                client.delete_registered_model_alias(name, CHALLENGER)
        except MlflowException:
            pass
        decision = "promoted"
    else:
        client.set_registered_model_alias(name, CHALLENGER, candidate.version)
        decision = "rejected"

    tags = {
        "gate_decision": decision,
        "gate_metric": args.metric,
        "gate_candidate_accuracy": f"{acc_candidate:.4f}",
        "gate_champion_accuracy": f"{acc_champion:.4f}",
        "gate_champion_version": str(champion.version),
        "gate_threshold": f"{threshold:.4f}",
        "gate_n_test": str(n),
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, candidate.version, key, value)

    print(f"Candidate  version {candidate.version}: {args.metric} = {acc_candidate:.4f}")
    print(f"Champion   version {champion.version}: {args.metric} = {acc_champion:.4f}")
    print(f"n = {n}, se = {se:.4f}, threshold = champion + 2*se = {threshold:.4f}")
    if promote:
        print(f"PROMOTED: '{CHAMPION}' now points to version {candidate.version}.")
    else:
        print(
            f"REJECTED: version {candidate.version} is now '{CHALLENGER}'; "
            f"'{CHAMPION}' stays on version {champion.version}."
        )


if __name__ == "__main__":
    main()
