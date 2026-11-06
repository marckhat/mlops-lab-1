import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import mlflow
import mlflow.pyfunc
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.food11.data import CLASSES

# The class order must match the ImageFolder class order used during training
# (alphabetical folder names, which is also the order of CLASSES).
CLASS_NAMES = CLASSES

MODEL_URI = "models:/food11@champion"

# The model is loaded once, at startup, into this module-level variable.
# Tests replace it with a fake model.
model = None


def load_model():
    """Load the champion model from MLflow, unless SKIP_MODEL is set."""

    global model

    if os.environ.get("SKIP_MODEL"):
        print("SKIP_MODEL is set: starting without a model.")
        return

    # Locally this uses 127.0.0.1.
    # Docker can override it with the MLFLOW_TRACKING_URI environment variable.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Loading model: {MODEL_URI}")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("Model loaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts (same role as @app.on_event("startup")).
    load_model()
    yield


app = FastAPI(
    title="Food-11 Classification API",
    description="Classifies uploaded food images into one of 11 categories.",
    version="1.0.0",
    lifespan=lifespan,
)

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


@app.get("/health")
def health():
    """Confirm that the API is running."""

    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    """Classify one uploaded food image."""

    if model is None:
        raise HTTPException(status_code=503, detail="No model is loaded.")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    # Add a batch dimension:
    # [3, 128, 128] becomes [1, 3, 128, 128].
    image_tensor = image_transform(image).unsqueeze(0)

    try:
        predictions = model.predict(image_tensor.numpy())
    except Exception as error:  # noqa: BLE001 - any model failure becomes a 500 with details
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        )

    # The model returns raw scores called logits.
    # Softmax converts them into probabilities.
    probabilities = torch.softmax(
        torch.as_tensor(predictions, dtype=torch.float32),
        dim=1,
    )

    predicted_index = int(probabilities.argmax(dim=1).item())
    confidence = float(probabilities[0, predicted_index].item())

    return {
        "category": CLASS_NAMES[predicted_index],
        "confidence": confidence,
    }
