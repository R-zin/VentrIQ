"""src/preprocessing/image.py - Illumination Correction & Contrast Enhancement.

This module provides background illumination estimation, shadow-division normalization,
and contrast-limited adaptive histogram equalization (CLAHE) to restore crisp waveform
and grid visibility from photographed ECG recordings.
"""

from __future__ import annotations

import logging
from typing import Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)


def estimate_background_illumination(
    gray: np.ndarray,
    sigma: float = 30.0,
) -> np.ndarray:
    """Estimate low-frequency background illumination using large-scale morphological/Gaussian filtering.

    Args:
        gray: 2D float32 grayscale image array in [0, 255].
        sigma: Standard deviation for Gaussian smoothing.

    Returns:
        2D float32 array representing the background illumination field.
    """
    # Morphological dilation with flat disk/box to remove dark waveform traces and grid
    # before Gaussian smoothing
    kernel_size = max(5, int(round(sigma * 0.4)))
    dilated = ndimage.maximum_filter(gray, size=kernel_size)
    bg = ndimage.gaussian_filter(dilated, sigma=sigma)
    return np.maximum(bg, 10.0)


def normalize_illumination(
    image: Union[Image.Image, np.ndarray],
    sigma: float = 30.0,
    target_background: float = 245.0,
) -> Image.Image:
    """Normalize non-uniform illumination and eliminate cast shadow gradients via background division.

    Args:
        image: Input RGB or grayscale image.
        sigma: Gaussian smoothing scale for background illumination estimation.
        target_background: Target paper brightness value (default 245.0 for clean paper).

    Returns:
        Illumination-normalized PIL Image.
    """
    if isinstance(image, Image.Image):
        arr = np.array(image, dtype=np.float32)
    else:
        arr = np.array(image, dtype=np.float32)

    # Compute luminance / grayscale representation
    if arr.ndim == 3:
        gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = arr

    bg = estimate_background_illumination(gray, sigma=sigma)

    # Division normalization: I_norm = (I / bg) * target_background
    if arr.ndim == 3:
        norm_arr = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            norm_arr[:, :, c] = (arr[:, :, c] / bg) * target_background
    else:
        norm_arr = (arr / bg) * target_background

    norm_arr = np.clip(norm_arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(norm_arr)


def apply_clahe(
    image: Union[Image.Image, np.ndarray],
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> Image.Image:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Enhances contrast locally across contextual tiles with a clip limit to prevent
    noise amplification while sharpening grid lines and cardiac waveform deflections.

    Args:
        image: Input image (PIL Image or NumPy array).
        clip_limit: Threshold for contrast limiting (higher = more contrast).
        tile_grid_size: Number of contextual grid tiles (rows, cols).

    Returns:
        Contrast-enhanced PIL Image.
    """
    if isinstance(image, Image.Image):
        arr = np.array(image, dtype=np.float32)
    else:
        arr = np.array(image, dtype=np.float32)

    # Process in YUV / Luminance space for RGB images to preserve color fidelity
    is_rgb = arr.ndim == 3 and arr.shape[2] == 3
    if is_rgb:
        # Convert RGB to YCbCr
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
        luminance = y
    else:
        luminance = arr

    h, w = luminance.shape[:2]
    num_tiles_y, num_tiles_x = tile_grid_size
    tile_h = h // num_tiles_y
    tile_w = w // num_tiles_x

    # Compute clipped CDF for each tile
    cdfs = np.zeros((num_tiles_y, num_tiles_x, 256), dtype=np.float32)

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            y_start = ty * tile_h
            y_end = (ty + 1) * tile_h if ty < num_tiles_y - 1 else h
            x_start = tx * tile_w
            x_end = (tx + 1) * tile_w if tx < num_tiles_x - 1 else w

            tile = luminance[y_start:y_end, x_start:x_end].astype(np.uint8)
            hist, _ = np.histogram(tile.ravel(), bins=256, range=(0, 256))

            # Clip histogram
            actual_clip = int(round(clip_limit * (tile.size / 256.0)))
            clipped = np.minimum(hist, actual_clip)
            excess = np.sum(hist - clipped)
            bonus = excess / 256.0
            hist_clipped = clipped + bonus

            # Cumulative Distribution Function
            cdf = np.cumsum(hist_clipped)
            cdf = (cdf - cdf.min()) / max(1e-6, (cdf.max() - cdf.min()))
            cdfs[ty, tx] = cdf * 255.0

    # Bilinear interpolation between tile CDFs
    eq_lum = np.zeros_like(luminance)
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)

    # Tile center coordinates
    tile_centers_y = (np.arange(num_tiles_y) + 0.5) * tile_h
    tile_centers_x = (np.arange(num_tiles_x) + 0.5) * tile_w

    # Find tile indices for interpolation
    ty0 = np.clip(np.floor((y_coords - tile_centers_y[0]) / tile_h).astype(int), 0, num_tiles_y - 2)
    tx0 = np.clip(np.floor((x_coords - tile_centers_x[0]) / tile_w).astype(int), 0, num_tiles_x - 2)
    ty1 = ty0 + 1
    tx1 = tx0 + 1

    # Fractional weights
    wy = np.clip((y_coords - tile_centers_y[ty0]) / tile_h, 0.0, 1.0)
    wx = np.clip((x_coords - tile_centers_x[tx0]) / tile_w, 0.0, 1.0)

    val = np.clip(luminance, 0, 255).astype(int)

    # Bilinear blend of mapped values
    m00 = cdfs[ty0, tx0, val]
    m01 = cdfs[ty0, tx1, val]
    m10 = cdfs[ty1, tx0, val]
    m11 = cdfs[ty1, tx1, val]

    top = (1.0 - wx) * m00 + wx * m01
    bot = (1.0 - wx) * m10 + wx * m11
    eq_lum = (1.0 - wy) * top + wy * bot

    if is_rgb:
        # Convert YCbCr back to RGB
        y_eq = eq_lum
        r_eq = y_eq + 1.402 * (cr - 128.0)
        g_eq = y_eq - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
        b_eq = y_eq + 1.772 * (cb - 128.0)

        out_rgb = np.stack([r_eq, g_eq, b_eq], axis=-1)
        out_rgb = np.clip(out_rgb, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(out_rgb)
    else:
        out_gray = np.clip(eq_lum, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(out_gray)


class ECGImageEnhancer:
    """Pipeline for illumination gradient normalization and contrast enhancement."""

    def __init__(
        self,
        illumination_sigma: float = 30.0,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.5,
        target_background: float = 245.0,
    ) -> None:
        self.illumination_sigma = illumination_sigma
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.target_background = target_background

    def __call__(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Run complete illumination normalization and contrast enhancement."""
        # 1. Division normalization
        norm_img = normalize_illumination(
            image,
            sigma=self.illumination_sigma,
            target_background=self.target_background,
        )

        # 2. CLAHE Contrast enhancement
        if self.enable_clahe:
            norm_img = apply_clahe(norm_img, clip_limit=self.clahe_clip_limit)

        return norm_img
