"""src/preprocessing/transforms.py - Vision Transformations for MobileNetV3 Edge Inference.

This module standardizes, resizes, and normalizes rectified ECG recordings into
model-ready PyTorch tensors formatted for MobileNetV3 backbones.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from src.preprocessing.image import ECGImageEnhancer
from src.preprocessing.perspective import DocumentRectifier
from src.rendering.renderer import ECGImage

logger = logging.getLogger(__name__)

# Standard ImageNet normalization parameters
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def image_to_mobilenet_tensor(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    normalize_imagenet: bool = True,
    channels_first: bool = True,
) -> torch.Tensor:
    """Resize, convert, and normalize an image into a PyTorch float32 tensor for MobileNetV3.

    Args:
        image: Input image (PIL Image or uint8 NumPy array).
        target_size: (width, height) resolution (default 224x224 for MobileNetV3).
        normalize_imagenet: If True, applies standard ImageNet mean and std standardization.
        channels_first: If True, output has shape (3, H, W). Otherwise (H, W, 3).

    Returns:
        torch.FloatTensor of shape (3, target_h, target_w).
    """
    if isinstance(image, Image.Image):
        pil_img = image
    else:
        pil_img = Image.fromarray(image)

    # Convert to RGB if not already
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize to target resolution with high-quality Lanczos resampling
    resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(resized, dtype=np.float32) / 255.0  # Scale to [0.0, 1.0]

    tensor = torch.from_numpy(arr)  # Shape: (H, W, 3)

    if channels_first:
        tensor = tensor.permute(2, 0, 1)  # (3, H, W)

    if normalize_imagenet:
        mean = (
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
            if channels_first
            else torch.tensor(IMAGENET_MEAN, dtype=torch.float32)
        )
        std = (
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
            if channels_first
            else torch.tensor(IMAGENET_STD, dtype=torch.float32)
        )
        tensor = (tensor - mean) / std

    return tensor.contiguous()


def extract_12_lead_crops(
    rectified_image: Union[Image.Image, np.ndarray],
    patch_size: Tuple[int, int] = (224, 112),
    normalize_imagenet: bool = True,
) -> torch.Tensor:
    """Extract individual lead box crops from a rectified 3x4 + 1 rhythm strip ECG recording.

    Args:
        rectified_image: Input rectified ECG image.
        patch_size: Tuple (width, height) in pixels (default (224, 112)).
        normalize_imagenet: Whether to apply ImageNet mean and std.

    Standard 3x4 lead layout geometry:
    - Row 0: Lead I, aVR, V1, V4
    - Row 1: Lead II, aVL, V2, V5
    - Row 2: Lead III, aVF, V3, V6

    Returns:
        torch.FloatTensor of shape (12, 3, patch_height, patch_width) = (12, 3, 112, 224).
    """
    if isinstance(rectified_image, Image.Image):
        pil_img = rectified_image
    else:
        pil_img = Image.fromarray(rectified_image)

    w, h = pil_img.size
    # 3 lead rows occupy top ~75% of height; bottom ~25% is rhythm strip and margins
    top_margin = h * 0.08
    row_height = (h * 0.72) / 3.0
    left_margin = w * 0.04
    col_width = (w * 0.92) / 4.0

    lead_crops: List[torch.Tensor] = []

    # 3 Rows x 4 Columns = 12 Leads
    for r in range(3):
        y0 = int(round(top_margin + r * row_height))
        y1 = int(round(y0 + row_height))
        for c in range(4):
            x0 = int(round(left_margin + c * col_width))
            x1 = int(round(x0 + col_width))

            crop = pil_img.crop((x0, y0, x1, y1))
            t = image_to_mobilenet_tensor(
                crop,
                target_size=patch_size,
                normalize_imagenet=normalize_imagenet,
                channels_first=True,
            )
            lead_crops.append(t)

    # Stack along lead dimension -> shape (12, 3, patch_h, patch_w)
    return torch.stack(lead_crops, dim=0)


class MobileNetV3Preprocessor:
    """Complete Vision Preprocessing & Rectification Pipeline for MobileNetV3."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        apply_rectification: bool = True,
        apply_enhancement: bool = True,
        normalize_imagenet: bool = True,
        extract_lead_crops: bool = False,
        rectified_resolution: Tuple[int, int] = (1654, 1122),
    ) -> None:
        self.target_size = target_size
        self.apply_rectification = apply_rectification
        self.apply_enhancement = apply_enhancement
        self.normalize_imagenet = normalize_imagenet
        self.extract_lead_crops = extract_lead_crops
        self.rectified_resolution = rectified_resolution

        self.rectifier = (
            DocumentRectifier(target_size=rectified_resolution) if apply_rectification else None
        )
        self.enhancer = ECGImageEnhancer() if apply_enhancement else None

    def __call__(
        self,
        image_or_wrapper: Union[Image.Image, np.ndarray, ECGImage],
    ) -> torch.Tensor:
        """Execute end-to-end vision preprocessing.

        Returns:
            torch.FloatTensor:
                - Shape (3, 224, 224) if extract_lead_crops=False
                - Shape (12, 3, 112, 224) if extract_lead_crops=True
        """
        # Unwrap ECGImage if provided
        if isinstance(image_or_wrapper, ECGImage):
            img = image_or_wrapper.pil_image
        elif isinstance(image_or_wrapper, np.ndarray):
            img = Image.fromarray(image_or_wrapper)
        elif isinstance(image_or_wrapper, Image.Image):
            img = image_or_wrapper
        else:
            raise TypeError(f"Unsupported input type: {type(image_or_wrapper)}")

        # Step 1: Perspective Rectification
        if self.rectifier is not None:
            rectified = self.rectifier(img)
            img = rectified.image

        # Step 2: Illumination Normalization & Contrast Enhancement
        if self.enhancer is not None:
            img = self.enhancer(img)

        # Step 3: Multi-Crop or Standard Vision Tensor
        if self.extract_lead_crops:
            return extract_12_lead_crops(img, normalize_imagenet=self.normalize_imagenet)

        return image_to_mobilenet_tensor(
            img,
            target_size=self.target_size,
            normalize_imagenet=self.normalize_imagenet,
            channels_first=True,
        )
