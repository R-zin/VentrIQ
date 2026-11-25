"""src/vision/dataset.py - PyTorch dataset over converted ECG images.

Reads the ``image_manifest.csv`` written by ``convert_echonext_to_images`` and yields
``(image_tensor, label)`` samples for MobileNetV3 training. Images on disk are already
in the canonical view, so only cheap, label-preserving augmentations are applied
online (small affine jitter, color/contrast jitter, grayscale prints, slight blur).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.preprocessing.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.vision.conversion import LABEL_COLUMNS, MANIFEST_FILENAME

# Model input resolution as (width, height). Keeps roughly the paper aspect ratio;
# MobileNetV3 is fully convolutional + global pooling, so non-square inputs are fine.
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (640, 448)


def build_image_transform(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    train: bool = False,
) -> Callable[[Image.Image], torch.Tensor]:
    """Return the PIL -> normalized float tensor transform used for training or evaluation.

    ``train=False`` is the exact transform used at inference time.
    """
    width, height = int(image_size[0]), int(image_size[1])
    ops: list = [v2.ToImage(), v2.Resize((height, width), antialias=True)]
    if train:
        ops += [
            v2.RandomAffine(degrees=2.0, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=255),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.02),
            v2.RandomGrayscale(p=0.1),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        ]
    ops += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
    return v2.Compose(ops)


class ECGImageDataset(Dataset):
    """Labelled ECG image dataset backed by a conversion manifest.

    Each sample is a dict with ``image`` (float32 ``(3, H, W)``), ``label`` (float32
    scalar 0/1), ``record_id`` and ``patient_key``.
    """

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame],
        split: Optional[str] = None,
        image_root: Optional[Union[str, Path]] = None,
        label: str = "shd",
        transform: Optional[Callable[[Image.Image], Any]] = None,
    ) -> None:
        if label not in LABEL_COLUMNS:
            raise ValueError(f"Unknown label '{label}'. Available: {list(LABEL_COLUMNS)}")

        if isinstance(manifest, pd.DataFrame):
            df = manifest.copy()
            if image_root is None:
                raise ValueError("image_root is required when passing a DataFrame manifest")
        else:
            manifest_path = Path(manifest)
            if manifest_path.is_dir():
                manifest_path = manifest_path / MANIFEST_FILENAME
            df = pd.read_csv(manifest_path)
            image_root = image_root or manifest_path.parent

        self.image_root = Path(image_root)
        self.label = label
        self.label_column = LABEL_COLUMNS[label]

        if split is not None:
            df = df[df["split"] == split]
        df = df[df[self.label_column].notna()].reset_index(drop=True)
        df[self.label_column] = df[self.label_column].astype(int)
        self.frame = df
        self.transform = transform or build_image_transform(train=False)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def labels(self) -> np.ndarray:
        return self.frame[self.label_column].to_numpy(dtype=np.int64)

    def positive_rate(self) -> float:
        return float(self.labels.mean()) if len(self) else 0.0

    def pos_weight(self) -> float:
        """``n_negative / n_positive`` for ``BCEWithLogitsLoss(pos_weight=...)``."""
        labels = self.labels
        pos = int(labels.sum())
        neg = int(len(labels) - pos)
        return float(neg / pos) if pos > 0 else 1.0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(self.image_root / row["image_path"]) as img:
            image = img.convert("RGB")
        return {
            "image": self.transform(image),
            "label": torch.tensor(float(row[self.label_column]), dtype=torch.float32),
            "record_id": str(row["record_id"]),
            "patient_key": str(row["patient_key"]),
        }
