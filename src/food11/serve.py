import os
from io import BytesIO

import mlflow
import mlflow.pyfunc
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


# The class order must match the ImageFolder class order used during training.
CLASS_NAMES = [
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

MODEL_URI = "models:/food11@champion"

# Locally this uses 127.0.0.1.
# Docker can override it with the MLFLOW_TRACKING_URI environment variable.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"Loading model: {MODEL_URI}")

# The model is loaded once when the API starts.
model = mlflow.pyfunc.load_model(MODEL_URI)

print("Model loaded successfully.")

app = FastAPI(
    title="Food-11 Classification API",
    description="Classifies uploaded food images into one of 11 categories.",
    version="1.0.0",
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

    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify one uploaded food image."""

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
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        )

    # The model returns raw scores called logits.
    # Softmax converts them into probabilities.
    probabilities = torch.softmax(
        torch.as_tensor(predictions),
        dim=1,
    )

    predicted_index = int(probabilities.argmax(dim=1).item())
    confidence = float(probabilities[0, predicted_index].item())

    return {
        "category": CLASS_NAMES[predicted_index],
        "confidence": confidence,
    }