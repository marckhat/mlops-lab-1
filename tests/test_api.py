import numpy as np
from fastapi.testclient import TestClient

from src.food11 import serve


class FakeModel:
    """Stands in for the MLflow pyfunc model.

    Like the real model, it returns one row of 11 raw scores (logits) per image
    as a numpy array. Index 0 (Bread) always has the highest score.
    """

    def predict(self, x):
        assert x.shape == (1, 3, 128, 128)  # serve.py must send one 128x128 RGB image
        return np.array([[0.90] + [0.01] * 10], dtype=np.float32)


def test_health(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_json(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())
        with open("tests/fixtures/bread.jpg", "rb") as f:
            r = client.post("/predict", files={"file": ("bread.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "Bread"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", None)
        with open("tests/fixtures/bread.jpg", "rb") as f:
            r = client.post("/predict", files={"file": ("bread.jpg", f, "image/jpeg")})
    assert r.status_code == 503


def test_predict_rejects_non_image(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())
        r = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
