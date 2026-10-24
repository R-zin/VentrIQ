"""tests/test_image_preprocessing.py - Unit Tests for Illumination Correction & MobileNetV3 Transforms.

Tests cover:
1. Background illumination field estimation.
2. Division normalization removing cast shadows.
3. CLAHE adaptive histogram equalization.
4. MobileNetV3 tensor formatting and ImageNet normalization.
5. 12-lead multi-crop patch extraction.
6. MobileNetV3Preprocessor callable pipeline.
7. DataLoader batch integration.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.image import (
    ECGImageEnhancer,
    apply_clahe,
    estimate_background_illumination,
    normalize_illumination,
)
from src.preprocessing.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MobileNetV3Preprocessor,
    extract_12_lead_crops,
    image_to_mobilenet_tensor,
)
from src.rendering.augmentation import apply_cast_shadow


@pytest.fixture
def shadowed_ecg_canvas() -> Image.Image:
    """Generate a synthetic white ECG paper canvas with an intense cast shadow."""
    img = Image.new("RGB", (400, 300), "#ffffff")
    draw = ImageDraw.Draw(img)
    # Add grid
    for x in range(0, 400, 20):
        draw.line([(x, 0), (x, 300)], fill="#fecaca", width=1)
    for y in range(0, 300, 20):
        draw.line([(0, y), (400, y)], fill="#fecaca", width=1)
    # Add ECG trace
    draw.line([(0, 150), (100, 150), (130, 60), (160, 220), (190, 150), (400, 150)], fill="#111827", width=2)

    # Cast shadow across top-left corner
    poly = [(0, 0), (220, 0), (0, 180)]
    return apply_cast_shadow(img, shadow_intensity=0.55, polygon=poly)


def test_background_illumination_estimation(shadowed_ecg_canvas):
    """Verify background illumination estimator captures smooth low-frequency lighting."""
    arr = np.array(shadowed_ecg_canvas)
    gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

    bg = estimate_background_illumination(gray, sigma=25.0)
    assert bg.shape == gray.shape
    assert np.all(bg >= 10.0)

    # Shadowed region (top-left) must have lower background estimate than illuminated region (bottom-right)
    assert np.mean(bg[:50, :50]) < np.mean(bg[200:, 250:])


def test_normalize_illumination_attenuates_shadow(shadowed_ecg_canvas):
    """Verify that division normalization removes cast shadows and restores uniform white."""
    norm_img = normalize_illumination(shadowed_ecg_canvas, sigma=20.0, target_background=245.0)
    assert norm_img.size == shadowed_ecg_canvas.size

    arr_orig = np.array(shadowed_ecg_canvas, dtype=float)
    arr_norm = np.array(norm_img, dtype=float)

    # In original, top-left was dark (< 150), bottom-right was bright (~255)
    orig_shadow_mean = np.mean(arr_orig[:40, :40])
    orig_bright_mean = np.mean(arr_orig[200:, 250:])
    assert orig_bright_mean - orig_shadow_mean > 80.0

    # In normalized image, background should be balanced
    norm_shadow_mean = np.mean(arr_norm[:40, :40])
    norm_bright_mean = np.mean(arr_norm[200:, 250:])
    diff = abs(norm_bright_mean - norm_shadow_mean)
    # The gradient difference should be significantly attenuated
    assert diff < 35.0


def test_apply_clahe_contrast_enhancement(shadowed_ecg_canvas):
    """Verify CLAHE sharpens trace edges without blowing out color channels."""
    clahe_img = apply_clahe(shadowed_ecg_canvas, clip_limit=3.0, tile_grid_size=(4, 4))
    assert clahe_img.size == shadowed_ecg_canvas.size
    assert clahe_img.mode == "RGB"

    arr = np.array(clahe_img)
    # Standard deviation of local trace area should be higher due to increased contrast
    trace_patch = arr[130:170, 100:200]
    assert np.std(trace_patch) > 10.0


def test_ecg_image_enhancer_pipeline(shadowed_ecg_canvas):
    """Verify complete ECGImageEnhancer combines division normalization and CLAHE."""
    enhancer = ECGImageEnhancer(illumination_sigma=20.0, enable_clahe=True)
    enhanced = enhancer(shadowed_ecg_canvas)

    assert isinstance(enhanced, Image.Image)
    assert enhanced.size == shadowed_ecg_canvas.size


def test_image_to_mobilenet_tensor(shadowed_ecg_canvas):
    """Verify resizing, ImageNet standardization, and PyTorch tensor shape."""
    tensor = image_to_mobilenet_tensor(
        shadowed_ecg_canvas,
        target_size=(224, 224),
        normalize_imagenet=True,
        channels_first=True,
    )

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

    # ImageNet standardized values span approximately [-2.5, 2.5]
    assert -3.0 <= float(tensor.min())
    assert float(tensor.max()) <= 3.5


def test_extract_12_lead_crops(shadowed_ecg_canvas):
    """Verify extraction of 12 individual lead crop patches."""
    lead_tensor = extract_12_lead_crops(
        shadowed_ecg_canvas,
        patch_size=(224, 112),
        normalize_imagenet=True,
    )

    assert isinstance(lead_tensor, torch.Tensor)
    # Expected shape: (12, 3, 112, 224) (12 leads, 3 channels, 112 height, 224 width)
    assert lead_tensor.shape == (12, 3, 112, 224)
    assert lead_tensor.dtype == torch.float32


def test_mobilenet_preprocessor_end_to_end(shadowed_ecg_canvas):
    """Verify MobileNetV3Preprocessor executes full rectification and standardization."""
    preprocessor = MobileNetV3Preprocessor(
        target_size=(224, 224),
        apply_rectification=False,  # Already planar
        apply_enhancement=True,
        normalize_imagenet=True,
    )

    t = preprocessor(shadowed_ecg_canvas)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 224, 224)


def test_dataloader_with_mobilenet_preprocessor():
    """Verify PyTorch DataLoader collates batches of MobileNetV3 preprocessed tensors."""
    class DummyImageDataset(Dataset):
        def __init__(self, count=4):
            self.count = count
            self.preprocessor = MobileNetV3Preprocessor(target_size=(224, 224), apply_rectification=False)

        def __len__(self):
            return self.count

        def __getitem__(self, idx):
            img = Image.new("RGB", (200, 150), "#ffffff")
            tensor = self.preprocessor(img)
            return {"image": tensor, "label": torch.tensor(1.0)}

    ds = DummyImageDataset(count=4)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 224, 224)
    assert batch["label"].shape == (2,)
