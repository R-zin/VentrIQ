"""src/vision/inference.py - Predict structural heart disease from a user's ECG image.

``SHDPredictor`` loads a checkpoint written by ``train_classifier`` and applies the
exact preprocessing used during training:

    user photo / scan -> ECGImageCanonicalizer (rectify, resize, enhance)
                      -> eval transform (resize to model input, ImageNet normalization)
                      -> MobileNetV3 -> sigmoid probability

Research use only: this is not a validated medical device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from src.vision.canonical import CanonicalizationConfig, ECGImageCanonicalizer, RectifyMode
from src.vision.dataset import build_image_transform
from src.vision.model import build_mobilenet_v3
from src.vision.training import resolve_device

ImageInput = Union[str, Path, Image.Image, np.ndarray]


@dataclass
class SHDPrediction:
    probability: float
    predicted_label: int
    threshold: float
    label: str
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "label": self.label,
            "probability": round(self.probability, 6),
            "threshold": round(self.threshold, 6),
            "predicted_label": self.predicted_label,
        }


class SHDPredictor:
    """Load a trained MobileNetV3 checkpoint and score ECG images."""

    def __init__(
        self,
        model: torch.nn.Module,
        canonicalization: CanonicalizationConfig,
        image_size: Sequence[int],
        threshold: float = 0.5,
        label: str = "shd",
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.canonicalizer = ECGImageCanonicalizer(canonicalization)
        self.transform = build_image_transform(
            (int(image_size[0]), int(image_size[1])), train=False
        )
        self.threshold = float(threshold)
        self.label = label

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
        threshold: Optional[float] = None,
    ) -> "SHDPredictor":
        dev = resolve_device(device)
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        model = build_mobilenet_v3(ckpt["arch"], pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        return cls(
            model=model,
            canonicalization=CanonicalizationConfig.from_dict(ckpt["canonicalization"]),
            image_size=ckpt["image_size"],
            threshold=ckpt.get("threshold", 0.5) if threshold is None else threshold,
            label=ckpt.get("label", "shd"),
            device=dev,
        )

    def preprocess(self, image: ImageInput, rectify: Optional[RectifyMode] = None) -> torch.Tensor:
        """Return the normalized ``(3, H, W)`` model input for ``image``."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                pil = img.convert("RGB")
        elif isinstance(image, np.ndarray):
            pil = Image.fromarray(image).convert("RGB")
        else:
            pil = image.convert("RGB")
        canonical = self.canonicalizer(pil, rectify=rectify)
        return self.transform(canonical)

    @torch.no_grad()
    def predict(
        self,
        images: Union[ImageInput, Sequence[ImageInput]],
        rectify: Optional[RectifyMode] = None,
    ) -> List[SHDPrediction]:
        """Score one or more ECG images (paths, PIL images or uint8 arrays)."""
        if isinstance(images, (str, Path, Image.Image, np.ndarray)):
            images = [images]
        batch = torch.stack([self.preprocess(img, rectify) for img in images]).to(self.device)
        probs = torch.sigmoid(self.model(batch).reshape(-1)).cpu().numpy()
        return [
            SHDPrediction(
                probability=float(p),
                predicted_label=int(p >= self.threshold),
                threshold=self.threshold,
                label=self.label,
                source=str(img) if isinstance(img, (str, Path)) else None,
            )
            for img, p in zip(images, probs)
        ]
