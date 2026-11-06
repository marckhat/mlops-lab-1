import argparse
import hashlib
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    """Read the training settings entered in the terminal."""
    parser = argparse.ArgumentParser(description="Train a Food-11 classifier")

    parser.add_argument(
        "--dataset",
        choices=["processed", "mini"],
        default="mini",
        help="Choose the full processed dataset or the mini dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of times the model processes the training dataset.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate used to update the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images processed together.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Also register the trained model as a new version of 'food11'.",
    )

    return parser.parse_args()


def folder_fingerprint(folder):
    """MD5 of every file path and content in a folder: changes if any image changes."""
    digest = hashlib.md5()
    for path in sorted(p for p in Path(folder).rglob("*") if p.is_file()):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(hashlib.md5(path.read_bytes()).digest())
    return digest.hexdigest()


def create_data_loaders(dataset_root, batch_size):
    """Load the training, validation, and evaluation datasets."""

    image_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(
        dataset_root / "training",
        transform=image_transform,
    )
    val_dataset = datasets.ImageFolder(
        dataset_root / "validation",
        transform=image_transform,
    )
    test_dataset = datasets.ImageFolder(
        dataset_root / "evaluation",
        transform=image_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


def build_model(num_classes):
    """Load pretrained ResNet-18 and adapt it to Food-11."""

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    number_of_inputs = model.fc.in_features

    model.fc = nn.Linear(
        in_features=number_of_inputs,
        out_features=num_classes,
    )

    return model


def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    """Train the model once on the complete training dataset."""

    model.train()

    total_loss = 0.0
    total_images = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(images)
        loss = loss_function(predictions, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

    return total_loss / total_images


def evaluate_model(model, data_loader, loss_function, device):
    """Calculate loss and accuracy without changing the model."""

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = loss_function(predictions, labels)

            predicted_classes = predictions.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predicted_classes == labels).sum().item()
            total_images += batch_size

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images

    return average_loss, accuracy


def main():
    args = parse_args()
    torch.manual_seed(42)

    project_root = Path(__file__).resolve().parents[2]

    if args.dataset == "mini":
        dataset_name = "food11_processed_mini"
    else:
        dataset_name = "food11_processed"

    dataset_root = project_root / "data" / dataset_name

    for split in ["training", "validation", "evaluation"]:
        split_path = dataset_root / split

        if not split_path.exists():
            raise FileNotFoundError(f"Missing dataset folder: {split_path}")

    # Same default as before, but can point to another server
    # (e.g. the Compose mlflow service or a hosted DagsHub MLflow).
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("food11")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = create_data_loaders(dataset_root, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run() as run:
        model = build_model(
            num_classes=len(train_dataset.classes)
        ).to(device)

        loss_function = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )

        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
                "device": str(device),
                "num_classes": len(train_dataset.classes),
                # Needed by promote.py to compute the standard error of test_accuracy.
                "test_size": len(test_dataset),
                # Identifies the exact test images: promote.py only compares runs
                # whose accuracy was measured on the same test set.
                "test_set_md5": folder_fingerprint(dataset_root / "evaluation"),
            }
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Device: {device}")
        print(f"Training images: {len(train_dataset)}")
        print(f"Validation images: {len(val_dataset)}")
        print(f"Evaluation images: {len(test_dataset)}")
        print("Starting training...")

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                loss_function,
                optimizer,
                device,
            )

            val_loss, val_accuracy = evaluate_model(
                model,
                val_loader,
                loss_function,
                device,
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_accuracy={val_accuracy:.4f}"
            )

        test_loss, test_accuracy = evaluate_model(
            model,
            test_loader,
            loss_function,
            device,
        )

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        print(f"Final test loss: {test_loss:.4f}")
        print(f"Final test accuracy: {test_accuracy:.4f}")
        print("Saving trained model to MLflow...")

        model = model.to("cpu")
        model.eval()

        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
            registered_model_name="food11" if args.register else None,
        )

        print("Model saved successfully.")
        if args.register:
            print(f"Registered as food11 version {model_info.registered_model_version}")


if __name__ == "__main__":
    main()