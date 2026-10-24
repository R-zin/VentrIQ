"""src/preprocessing/perspective.py - Document Localization & Homography Rectification.

This module provides automated document boundary detection, corner extraction,
and four-point perspective rectification for photographed ECG recordings, warping
perspective-distorted smartphone photographs back into canonical rectilinear planar coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage
import torch

from src.rendering.augmentation import solve_homography_coefficients

logger = logging.getLogger(__name__)

# Standard ECG Paper Dimensions (280 mm x 190 mm -> Aspect ratio 1.4737)
STANDARD_PAPER_ASPECT_RATIO: float = 280.0 / 190.0


@dataclass
class DetectedDocument:
    """Detected document boundaries and corner localization metadata."""

    corners: np.ndarray      # Shape (4, 2): [TL, TR, BR, BL] in image pixels
    confidence: float        # Localization confidence score in [0.0, 1.0]
    aspect_ratio: float      # Estimated width / height ratio
    is_fallback: bool        # True if boundary detection failed and fallback was used

    @property
    def top_left(self) -> np.ndarray:
        return self.corners[0]

    @property
    def top_right(self) -> np.ndarray:
        return self.corners[1]

    @property
    def bottom_right(self) -> np.ndarray:
        return self.corners[2]

    @property
    def bottom_left(self) -> np.ndarray:
        return self.corners[3]


@dataclass
class RectifiedECG:
    """Rectified planar ECG recording container."""

    image: Image.Image
    detected_document: DetectedDocument
    target_size: Tuple[int, int]
    homography_matrix: np.ndarray

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @property
    def size(self) -> Tuple[int, int]:
        return self.image.size

    def to_numpy(self) -> np.ndarray:
        """Convert rectified image to uint8 RGB array of shape (H, W, 3)."""
        return np.array(self.image, dtype=np.uint8)

    def to_tensor(self, channels_first: bool = True, normalize_0_1: bool = True) -> torch.Tensor:
        """Convert rectified image to PyTorch tensor."""
        arr = self.to_numpy()
        tensor = torch.from_numpy(arr).float() / 255.0 if normalize_0_1 else torch.from_numpy(arr)
        if channels_first:
            tensor = tensor.permute(2, 0, 1)
        return tensor.contiguous()

    def save(self, path: Union[str, Path], quality: int = 95) -> Path:
        """Save rectified image to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(p, quality=quality)
        return p


def _otsu_threshold(gray: np.ndarray) -> float:
    """Compute Otsu's optimal bimodal threshold."""
    hist, bin_edges = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    current_max, threshold = 0.0, 128.0
    sum_total = np.dot(np.arange(256), hist)
    weight_background, sum_background = 0.0, 0.0

    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += i * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_variance = weight_background * weight_foreground * ((mean_background - mean_foreground) ** 2)
        if between_variance > current_max:
            current_max = between_variance
            threshold = float(i)

    return threshold


def detect_document_corners(
    image: Union[Image.Image, np.ndarray],
    min_paper_area_ratio: float = 0.15,
) -> DetectedDocument:
    """Automatically detect the 4 corners of an ECG document in a smartphone photo.

    Employs Otsu bimodal segmentation, gradient magnitude analysis, and extremal
    quadrilateral fitting to locate the paper document boundary.

    Args:
        image: Input RGB or grayscale image.
        min_paper_area_ratio: Minimum expected paper area relative to total canvas.

    Returns:
        DetectedDocument instance containing ordered corners [TL, TR, BR, BL] and confidence.
    """
    if isinstance(image, Image.Image):
        arr = np.array(image)
    else:
        arr = np.array(image)

    h, w = arr.shape[:2]

    # Convert to grayscale
    if arr.ndim == 3:
        gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    else:
        gray = arr.astype(np.float32)

    # Blur slightly to eliminate fine high-frequency noise/text
    blurred = ndimage.gaussian_filter(gray, sigma=2.0)

    # 1. Intensity Thresholding (ECG paper is bright against typical clinic desks)
    thresh = _otsu_threshold(blurred)
    bright_mask = blurred > max(60.0, thresh * 0.85)

    # 2. Gradient Magnitude via Sobel to detect strong boundary edges
    sobel_h = ndimage.sobel(blurred, axis=0)
    sobel_v = ndimage.sobel(blurred, axis=1)
    grad_mag = np.hypot(sobel_h, sobel_v)
    edge_mask = grad_mag > np.percentile(grad_mag, 85)

    # Combine intensity and edge information
    combined_mask = bright_mask | edge_mask
    # Morphological closing to seal perimeter
    closed_mask = ndimage.binary_closing(combined_mask, structure=np.ones((7, 7)))

    # Find connected components and select largest area
    labeled, num_features = ndimage.label(closed_mask)
    if num_features > 0:
        sizes = ndimage.sum(closed_mask, labeled, range(1, num_features + 1))
        largest_label = int(np.argmax(sizes)) + 1
        paper_mask = labeled == largest_label
        area = float(np.sum(paper_mask))
    else:
        paper_mask = bright_mask
        area = float(np.sum(paper_mask))

    total_pixels = w * h
    area_ratio = area / total_pixels

    # Check validity
    if area_ratio < min_paper_area_ratio or not np.any(paper_mask):
        # Fallback to padded canvas boundaries
        pad_x, pad_y = w * 0.05, h * 0.05
        fallback_corners = np.array([
            [pad_x, pad_y],
            [w - pad_x, pad_y],
            [w - pad_x, h - pad_y],
            [pad_x, h - pad_y],
        ], dtype=np.float32)
        return DetectedDocument(
            corners=fallback_corners,
            confidence=0.20,
            aspect_ratio=w / max(1.0, float(h)),
            is_fallback=True,
        )

    # Extract coordinate indices of paper pixels
    y_idx, x_idx = np.where(paper_mask)
    pts = np.column_stack([x_idx, y_idx])

    # Extremal points of the polygon:
    # Top-Left: minimizes (x + y)
    # Bottom-Right: maximizes (x + y)
    # Top-Right: maximizes (x - y)
    # Bottom-Left: minimizes (x - y)
    tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
    br = pts[np.argmax(pts[:, 0] + pts[:, 1])]
    tr = pts[np.argmax(pts[:, 0] - pts[:, 1])]
    bl = pts[np.argmin(pts[:, 0] - pts[:, 1])]

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # Estimate aspect ratio
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    avg_w = (width_top + width_bottom) / 2.0
    avg_h = (height_left + height_right) / 2.0
    aspect_ratio = float(avg_w / max(1.0, avg_h))

    # Confidence based on closeness to expected aspect ratio (approx 1.47) and area coverage
    aspect_dev = abs(aspect_ratio - STANDARD_PAPER_ASPECT_RATIO) / STANDARD_PAPER_ASPECT_RATIO
    confidence = float(np.clip(1.0 - (aspect_dev * 1.2), 0.3, 0.98))

    return DetectedDocument(
        corners=corners,
        confidence=confidence,
        aspect_ratio=aspect_ratio,
        is_fallback=False,
    )


def rectify_document(
    image: Union[Image.Image, np.ndarray],
    corners: Optional[Union[np.ndarray, Sequence[Tuple[float, float]], DetectedDocument]] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> RectifiedECG:
    """Rectify a perspective-distorted document photograph into a canonical rectangular canvas.

    Args:
        image: Input smartphone ECG photograph (PIL Image or NumPy array).
        corners: Detected corners [TL, TR, BR, BL] or DetectedDocument. If None,
                 automatically runs detect_document_corners().
        target_size: Optional (width, height) in pixels. If None, standardizes to
                     1654 x 1122 px (canonical 280 mm x 190 mm @ 150 DPI).

    Returns:
        RectifiedECG instance containing the unwarped planar image.
    """
    if isinstance(image, Image.Image):
        src_img = image.copy()
    else:
        src_img = Image.fromarray(image)

    # 1. Resolve corners
    if corners is None:
        doc = detect_document_corners(src_img)
        corner_pts = doc.corners
    elif isinstance(corners, DetectedDocument):
        doc = corners
        corner_pts = doc.corners
    else:
        corner_arr = np.array(corners, dtype=np.float32)
        doc = DetectedDocument(
            corners=corner_arr,
            confidence=0.90,
            aspect_ratio=float(np.linalg.norm(corner_arr[1] - corner_arr[0]) / max(1.0, np.linalg.norm(corner_arr[3] - corner_arr[0]))),
            is_fallback=False,
        )
        corner_pts = corner_arr

    # 2. Determine target dimensions
    if target_size is not None:
        target_w, target_h = target_size
    else:
        # Standard 150 DPI resolution: 1654 x 1122
        target_w, target_h = 1654, 1122

    target_corners = np.array([
        [0.0, 0.0],
        [float(target_w), 0.0],
        [float(target_w), float(target_h)],
        [0.0, float(target_h)],
    ], dtype=np.float64)

    # 3. Calculate 8 coefficients mapping target rectangle back to source photo coordinates
    coeffs = solve_homography_coefficients(
        src_pts=[tuple(pt) for pt in corner_pts],
        dst_pts=[tuple(pt) for pt in target_corners],
    )

    # 4. Perspective warp into canonical rectilinear canvas
    rectified_img = src_img.transform(
        (target_w, target_h),
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
    )

    # Compute 3x3 homography matrix mapping corner_pts -> target_corners
    matrix: List[List[float]] = []
    for (x, y), (xp, yp) in zip(corner_pts, target_corners):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -xp * x, -xp * y, -xp])
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -yp * x, -yp * y, -yp])

    A = np.array(matrix, dtype=np.float64)
    _, _, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape((3, 3))
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return RectifiedECG(
        image=rectified_img,
        detected_document=doc,
        target_size=(target_w, target_h),
        homography_matrix=H,
    )


class DocumentRectifier:
    """High-level document detector and perspective unwarping rectifier."""

    def __init__(self, target_size: Tuple[int, int] = (1654, 1122)) -> None:
        self.target_size = target_size

    def __call__(self, image: Union[Image.Image, np.ndarray]) -> RectifiedECG:
        """Process image through document detection and perspective unwarping."""
        return rectify_document(image, target_size=self.target_size)
