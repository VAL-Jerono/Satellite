"""
training/model.py
──────────────────
Sentinel-2 land-cover CNN.

Backbones: EfficientNet-B0 (default) or ResNet18.
First conv layer is replaced to accept 10 input channels instead of 3.
Final FC layer replaced for num_classes outputs.

Also exports the model to ONNX for CPU serving.
"""

from __future__ import annotations
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

_ROOT = Path(__file__).resolve().parents[1]
_CFG  = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
MC    = _CFG["model"]


def build_model(
    backbone: str | None    = None,
    in_channels: int | None = None,
    num_classes: int | None = None,
    pretrained: str | None  = None,
    dropout: float | None   = None,
) -> nn.Module:
    """
    Returns a torchvision backbone adapted for multi-spectral input.

    Parameters
    ----------
    backbone    : "efficientnet_b0" | "resnet18"
    in_channels : number of input bands (default 10 for Sentinel-2)
    num_classes : land-cover classes (default 6)
    pretrained  : "imagenet" | "none"
    dropout     : dropout before head (default 0.3)
    """
    backbone    = backbone    or MC["backbone"]
    in_channels = in_channels or MC["in_channels"]
    num_classes = num_classes or MC["num_classes"]
    pretrained  = pretrained  or MC["pretrained"]
    dropout     = dropout     if dropout is not None else MC["dropout"]

    use_pretrained = (pretrained == "imagenet")
    weights = "DEFAULT" if use_pretrained else None

    if backbone == "resnet18":
        m = models.resnet18(weights=weights)
        orig = m.conv1
        new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_pretrained:
            with torch.no_grad():
                rep = orig.weight.repeat(1, (in_channels // 3) + 1, 1, 1)[:, :in_channels]
                new_conv.weight.copy_(rep * (3.0 / in_channels))
        m.conv1 = new_conv
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(m.fc.in_features, num_classes),
        )

    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        orig = m.features[0][0]   # first Conv2d
        new_conv = nn.Conv2d(in_channels, orig.out_channels,
                             kernel_size=orig.kernel_size, stride=orig.stride,
                             padding=orig.padding, bias=orig.bias is not None)
        if use_pretrained:
            with torch.no_grad():
                rep = orig.weight.repeat(1, (in_channels // 3) + 1, 1, 1)[:, :in_channels]
                new_conv.weight.copy_(rep * (3.0 / in_channels))
        m.features[0][0] = new_conv
        in_feats = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Use 'resnet18' or 'efficientnet_b0'.")

    return m


def export_onnx(model: nn.Module, out_path: str | Path, patch_size: int = 64, in_channels: int = 10):
    """Export model to ONNX with dynamic batch axis."""
    model.eval()
    dummy = torch.zeros(1, in_channels, patch_size, patch_size)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["sentinel2_patch"],
        output_names=["class_logits"],
        dynamic_axes={"sentinel2_patch": {0: "batch"}, "class_logits": {0: "batch"}},
        opset_version=_CFG["serving"]["onnx_opset"],
        export_params=True,
    )
    print(f"ONNX saved → {out_path}")
