"""
tests/test_api.py
──────────────────
FastAPI unit tests using the test client (no live ONNX model needed).
Patches the ONNX session with a random predictor.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# We patch _get_session before importing the app
class _FakeSession:
    """Returns random logits for 6 classes."""
    def run(self, output_names, feed_dict):
        logits = np.random.randn(1, 6).astype(np.float32)
        return [logits]


@pytest.fixture()
def client():
    with patch("api.main._get_session", return_value=_FakeSession()):
        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_point_missing_patch_no_ee(client):
    """
    Without GEE_PROJECT set, /predict/point without a pre-built patch
    should return 503 (EE pull fails), not 500.
    """
    resp = client.post("/predict/point", json={
        "lat": 0.0, "lon": 37.0, "flood_risk": False, "slope_deg": 5.0
    })
    # Without EE creds the live pull fails → 503
    assert resp.status_code == 503


def test_predict_point_with_patch(client):
    import base64
    patch_arr = np.zeros((10, 64, 64), dtype=np.float32)
    b64 = base64.b64encode(patch_arr.tobytes()).decode()
    resp = client.post("/predict/point", json={
        "lat": 0.0, "lon": 37.0, "patch_b64": b64
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "recommendation" in body
    assert body["recommendation"]["action"] in ("settle", "farm", "protect")


def test_county_not_found(client):
    resp = client.get("/predict/county/nonexistent_county_xyz")
    assert resp.status_code == 404
