"""src/vision/model.py - MobileNetV3 binary classifier for ECG images."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models

MobileNetArch = Literal["small", "large"]


def build_mobilenet_v3(
    arch: MobileNetArch = "large",
    pretrained: bool = False,
    num_outputs: int = 1,
    dropout: float = 0.2,
) -> nn.Module:
    """Build a MobileNetV3 with a fresh ``num_outputs``-logit head.

    Args:
        arch: 'small' (~1.5M params) or 'large' (~4.2M params).
        pretrained: Initialize the backbone with torchvision ImageNet weights
            (downloaded on first use). Strongly recommended for real training.
        num_outputs: Number of output logits (1 for binary SHD classification).
        dropout: Dropout before the final linear layer.
    """
    if arch == "small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights, dropout=dropout)
    elif arch == "large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights, dropout=dropout)
    else:
        raise ValueError(f"Unknown MobileNetV3 arch '{arch}' (expected 'small' or 'large')")

    head = model.classifier[-1]
    assert isinstance(head, nn.Linear)
    model.classifier[-1] = nn.Linear(head.in_features, num_outputs)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run ``model`` in eval mode and return a 1-D tensor of logits."""
    model.eval()
    return model(images).reshape(-1)
