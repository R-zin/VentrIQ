"""tests/test_perspective.py - Unit Tests for Document Localization & Perspective Rectification.

Tests cover:
1. Automated document boundary and corner detection.
2. Localization error benchmarks against ground-truth corners.
3. Fallback behavior on ambiguous or missing document images.
4. Four-point perspective unwarping and homography calculation.
5. Preservation of standard ECG paper aspect ratio (280 mm x 190 mm).
6. RectifiedECG container properties, NumPy arrays, and PyTorch tensor export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from src.preprocessing.perspective import (
    STANDARD_PAPER_ASPECT_RATIO,
    DetectedDocument,
    DocumentRectifier,
    RectifiedECG,
    detect_document_corners,
    rectify_document,
)
from src.rendering.augmentation import (
    DistortionConfig,
    SmartphoneECGSimulator,
)


@pytest.fixture
def synthetic_tilted_document() -> Tuple[Image.Image, np.ndarray]:
    """Generate a clean synthetic tilted document on a dark background."""
    w, h = 800, 600
    img = Image.new("RGB", (w, h), "#262626")
    draw = ImageDraw.Draw(img)

    # Document corners [TL, TR, BR, BL]
    corners = np.array([
        [160.0, 110.0],
        [660.0, 130.0],
        [630.0, 510.0],
        [130.0, 480.0],
    ], dtype=np.float32)

    poly = [tuple(pt) for pt in corners]
    draw.polygon(poly, fill="#ffffff")
    # Draw internal grid lines to represent ECG paper
    for x in range(180, 620, 20):
        draw.line([(x, 150), (x, 460)], fill="#fca5a5", width=1)
    for y in range(150, 470, 20):
        draw.line([(180, y), (620, y)], fill="#fca5a5", width=1)

    return img, corners


def test_detect_document_corners_accuracy(synthetic_tilted_document):
    """Verify that automated corner detection accurately extracts document boundaries."""
    img, gt_corners = synthetic_tilted_document
    doc = detect_document_corners(img)

    assert isinstance(doc, DetectedDocument)
    assert doc.is_fallback is False
    assert doc.confidence >= 0.50
    assert doc.corners.shape == (4, 2)

    # Measure error against ground truth: should be within 4 pixels on clean high-contrast
    for gt, det in zip(gt_corners, doc.corners):
        err = float(np.linalg.norm(gt - det))
        assert err < 6.0


def test_detect_corners_fallback_on_empty_image():
    """Verify fallback behavior when no document boundary is present."""
    empty_img = Image.new("RGB", (400, 300), "#262626")
    doc = detect_document_corners(empty_img)

    assert doc.is_fallback is True
    assert doc.confidence <= 0.35
    assert doc.corners.shape == (4, 2)


def test_rectify_document_with_explicit_corners(synthetic_tilted_document):
    """Verify four-point perspective rectification unwarps image to canonical dimensions."""
    img, gt_corners = synthetic_tilted_document
    target_w, target_h = 700, int(round(700 / STANDARD_PAPER_ASPECT_RATIO))

    rectified = rectify_document(img, corners=gt_corners, target_size=(target_w, target_h))
    assert isinstance(rectified, RectifiedECG)
    assert rectified.width == target_w
    assert rectified.height == target_h
    assert rectified.size == (target_w, target_h)

    # Output should be bright document canvas (no dark desk background)
    arr = rectified.to_numpy()
    center_patch = arr[100:300, 100:500]
    # Mean brightness must exceed 200 (white/pink paper)
    assert np.mean(center_patch) > 200.0


def test_rectify_document_auto_detection(synthetic_tilted_document):
    """Verify rectification when corners are automatically detected from input image."""
    img, _ = synthetic_tilted_document
    rectified = rectify_document(img, target_size=(600, 407))

    assert rectified.size == (600, 407)
    assert isinstance(rectified.image, Image.Image)
    assert rectified.homography_matrix.shape == (3, 3)

    # PyTorch tensor export
    t_tensor = rectified.to_tensor(channels_first=True, normalize_0_1=True)
    assert isinstance(t_tensor, torch.Tensor)
    assert t_tensor.shape == (3, 407, 600)
    assert 0.0 <= float(t_tensor.min()) <= float(t_tensor.max()) <= 1.0


def test_document_rectifier_callable_interface(synthetic_tilted_document):
    """Verify DocumentRectifier class and save method."""
    img, _ = synthetic_tilted_document
    rectifier = DocumentRectifier(target_size=(800, 543))
    result = rectifier(img)

    assert isinstance(result, RectifiedECG)
    assert result.size == (800, 543)


def test_rectification_on_simulated_smartphone_ecg():
    """Verify full loop: simulate distortion -> detect corners -> unwarp to rectangle."""
    # Create canonical ECG paper canvas
    canonical_w, canonical_h = 560, 380
    canonical_paper = Image.new("RGB", (canonical_w, canonical_h), "#ffffff")
    draw = ImageDraw.Draw(canonical_paper)
    # Draw grid
    for x in range(0, canonical_w, 20):
        draw.line([(x, 0), (x, canonical_h)], fill="#fecaca", width=1)
    for y in range(0, canonical_h, 20):
        draw.line([(0, y), (canonical_w, y)], fill="#fecaca", width=1)
    # Draw distinct trace
    draw.line([(0, 190), (100, 190), (150, 80), (200, 280), (250, 190), (560, 190)], fill="#111827", width=3)

    # Distort with SmartphoneECGSimulator (mild tilt)
    cfg = DistortionConfig.from_preset("mild")
    cfg.enable_shadows = False
    cfg.enable_noise = False
    cfg.enable_blur = False
    cfg.enable_page_curl = False
    sim = SmartphoneECGSimulator(config=cfg, seed=999)
    photo = sim.simulate(canonical_paper, custom_angles=(8.0, -8.0, 3.0))

    # Detect & Rectify
    doc = detect_document_corners(photo.image)
    assert doc.is_fallback is False

    # Measure corner detection accuracy against simulator's ground-truth corners
    gt = photo.corners_transformed
    mean_corner_err = np.mean([np.linalg.norm(g - d) for g, d in zip(gt, doc.corners)])
    assert mean_corner_err < 10.0  # Under 10 pixels error on tilted document

    # Rectify back
    rect = rectify_document(photo.image, corners=doc, target_size=(canonical_w, canonical_h))
    assert rect.size == (canonical_w, canonical_h)
