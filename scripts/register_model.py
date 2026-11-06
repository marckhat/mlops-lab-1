"""Register a Lab 2 model in the (containerized) MLflow registry.

The mlflow container starts with an empty volume, so the models registered in
Labs 2-3 (stored in the host's mlflow.db / mlruns/) are not in it. This script
loads one of the already-trained Lab 2 models from the local mlruns/ folder,
logs it to the tracking server, registers it as a new version of `food11`
and points an alias (default: champion) at that version. No retraining needed.

Usage (with `docker compose up` running):
    uv run python scripts/register_model.py --model runner-up
    uv run python scripts/register_model.py --model best
"""

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTERED_MODEL_NAME = "food11"

# Logged models from the Lab 2 experiment runs (experiment id 1 in the host mlflow.db).
PRESETS = {
    "best": {
        "model_id": "m-39b167fc70bb498e971a7986771b241e",
        "source_run_name": "skillful-midge-64",
        "source_run_id": "5e23509ae8e8413bab36cb2a2530ee5d",
        "lr": 0.0001,
        "batch_size": 32,
        "val_accuracy": 0.7263,
        "test_accuracy": 0.7518248175182481,  # 824 / 1096 on the mini evaluation split
        "test_loss": 0.7436415672737317,
    },
    "runner-up": {
        "model_id": "m-f724849858784d7898be7c01dc5f8a00",
        "source_run_name": "invincible-shad-365",
        "source_run_id": "0737831bf9ab4a5c97737c4e0a375ad5",
        "lr": 0.001,
        "batch_size": 64,
        "val_accuracy": 0.6122,
        "test_accuracy": 0.6642335766423357,  # 728 / 1096 on the mini evaluation split
        "test_loss": 1.3641694466563037,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=PRESETS, default="best")
    parser.add_argument("--alias", default="champion")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preset = PRESETS[args.model]

    model_dir = PROJECT_ROOT / "mlruns" / "1" / "models" / preset["model_id"] / "artifacts"
    if not (model_dir / "MLmodel").exists():
        raise FileNotFoundError(f"No MLflow model found in {model_dir}")

    print(f"Loading local model {preset['source_run_name']} from {model_dir}")
    model = mlflow.pytorch.load_model(str(model_dir))

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("food11")

    with mlflow.start_run(run_name=f"import-{preset['source_run_name']}"):
        mlflow.log_params(
            {
                "source_run_name": preset["source_run_name"],
                "source_run_id": preset["source_run_id"],
                "source_model_id": preset["model_id"],
                "lr": preset["lr"],
                "batch_size": preset["batch_size"],
                "model": "resnet18",
                # Same keys as train.py, so promote.py can compare this run with new ones.
                "dataset": "mini",
                "test_size": 1096,
            }
        )
        mlflow.log_metrics(
            {
                "val_accuracy": preset["val_accuracy"],
                "test_accuracy": preset["test_accuracy"],
                "test_loss": preset["test_loss"],
            }
        )

        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    client = MlflowClient()
    version = model_info.registered_model_version
    if version is None:  # fallback: newest version of the registered model
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        version = max(int(v.version) for v in versions)

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, args.alias, version)
    print(
        f"Registered {REGISTERED_MODEL_NAME} version {version} "
        f"({preset['source_run_name']}) and set alias '{args.alias}' -> {version}"
    )


if __name__ == "__main__":
    main()
