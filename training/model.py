"""
training/model.py
──────────────────
Sentinel-2 Multi-Spectral Land Cover CNN Architecture.

Features:
- 10-channel Sentinel-2 input adaptation
- Spectral Channel Attention (Squeeze-and-Excitation) module for remote sensing
- Support for EfficientNet-B0, ResNet-18, and ResNet-34 backbones
- ONNX model export for low-latency CPU serving
"""

from __future__ import annotations
from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

_ROOT = Path(__file__).resolve().parents[1]
_CFG_PATH = _ROOT / "configs" / "config.yaml"
if _CFG_PATH.exists():
    _CFG = yaml.safe_load(_CFG_PATH.read_text())
    MC = _CFG["model"]
else:
    MC = {
        "backbone": "efficientnet_b0",
        "in_channels": 10,
        "num_classes": 6,
        "pretrained": "imagenet",
        "dropout": 0.3,
    }


class SpectralChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel Attention block adapted for multi-spectral input bands.
    Dynamically recalculates feature weights across the 10 Sentinel-2 bands.
    """
    def __init__(self, in_channels: int = 10, reduction: int = 2):
        super().__init__()
        reduced_dim = max(4, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        weights = self.fc(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * weights


class SpectralCNN(nn.Module):
    """
    CNN architecture wrapping a torchvision backbone with Spectral Channel Attention.
    """
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        in_channels: int = 10,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.spectral_attn = SpectralChannelAttention(in_channels=in_channels, reduction=2)
        weights = "DEFAULT" if pretrained else None

        if backbone == "resnet18" or backbone == "resnet34":
            m = models.resnet18(weights=weights) if backbone == "resnet18" else models.resnet34(weights=weights)
            orig_conv = m.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    rep = orig_conv.weight.repeat(1, (in_channels // 3) + 1, 1, 1)[:, :in_channels]
                    new_conv.weight.copy_(rep * (3.0 / in_channels))
            m.conv1 = new_conv
            m.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(m.fc.in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2.0),
                nn.Linear(128, num_classes),
            )
            self.backbone = m

        elif backbone == "efficientnet_b0":
            m = models.efficientnet_b0(weights=weights)
            orig_conv = m.features[0][0]
            new_conv = nn.Conv2d(in_channels, orig_conv.out_channels,
                                 kernel_size=orig_conv.kernel_size, stride=orig_conv.stride,
                                 padding=orig_conv.padding, bias=orig_conv.bias is not None)
            if pretrained:
                with torch.no_grad():
                    rep = orig_conv.weight.repeat(1, (in_channels // 3) + 1, 1, 1)[:, :in_channels]
                    new_conv.weight.copy_(rep * (3.0 / in_channels))
            m.features[0][0] = new_conv
            in_feats = m.classifier[1].in_features
            m.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2.0),
                nn.Linear(128, num_classes),
            )
            self.backbone = m
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn = self.spectral_attn(x)
        return self.backbone(x_attn)


def build_model(
    backbone: str | None    = None,
    in_channels: int | None = None,
    num_classes: int | None = None,
    pretrained: str | None  = None,
    dropout: float | None   = None,
) -> nn.Module:
    backbone    = backbone    or MC["backbone"]
    in_channels = in_channels or MC["in_channels"]
    num_classes = num_classes or MC["num_classes"]
    pretrained  = pretrained  or MC["pretrained"]
    dropout     = dropout     if dropout is not None else MC["dropout"]

    use_pretrained = (pretrained == "imagenet")

    return SpectralCNN(
        backbone=backbone,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=use_pretrained,
        dropout=dropout,
    )


def export_onnx(model: nn.Module, out_path: str | Path, patch_size: int = 64, in_channels: int = 10):
    """Export model to ONNX with dynamic batch axis."""
    model.eval()
    dummy = torch.zeros(1, in_channels, patch_size, patch_size)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    opset_ver = _CFG["serving"]["onnx_opset"] if _CFG_PATH.exists() else 17

    def _export():
        torch.onnx.export(
            model, dummy, str(out_path),
            input_names=["sentinel2_patch"],
            output_names=["class_logits"],
            dynamic_axes={"sentinel2_patch": {0: "batch"}, "class_logits": {0: "batch"}},
            opset_version=opset_ver,
            export_params=True,
            dynamo=False,
        )

    try:
        _export()
    except (ModuleNotFoundError, Exception) as exc:
        if "onnxscript" in str(exc) or "onnx" in str(exc).lower():
            import subprocess, sys
            print("Installing onnxscript & onnx for ONNX export...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript"])
            _export()
        else:
            raise exc

    print(f"ONNX saved → {out_path}")
