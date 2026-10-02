"""
tests/test_model.py
────────────────────
Unit tests for the CNN model factory and ONNX export.
These run on CPU with random weights (no trained checkpoint needed).
"""
import numpy as np
import pytest
import torch

from training.model import build_model, export_onnx


def test_efficientnet_forward():
    model = build_model(backbone="efficientnet_b0", pretrained="none")
    model.eval()
    x = torch.randn(2, 10, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 6), f"Expected (2, 6), got {out.shape}"


def test_resnet18_forward():
    model = build_model(backbone="resnet18", pretrained="none")
    model.eval()
    x = torch.randn(2, 10, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 6), f"Expected (2, 6), got {out.shape}"


def test_onnx_export(tmp_path):
    model = build_model(backbone="resnet18", pretrained="none")
    out   = tmp_path / "test.onnx"
    export_onnx(model, out)
    assert out.exists()


def test_onnx_inference(tmp_path):
    import onnxruntime as ort
    model = build_model(backbone="resnet18", pretrained="none")
    out   = tmp_path / "test.onnx"
    export_onnx(model, out)

    sess   = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    dummy  = np.random.randn(1, 10, 64, 64).astype(np.float32)
    logits = sess.run(None, {"sentinel2_patch": dummy})[0]
    assert logits.shape == (1, 6)
