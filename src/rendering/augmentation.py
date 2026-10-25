"""src/rendering/augmentation.py - Real-World Smartphone Capture & Distortion Simulator.

This module models physical artifacts encountered when a clinician or patient
photographs an ECG recording with a smartphone camera:
1. 3D Camera Pose & Perspective Tilt (yaw, pitch, roll, 8-parameter homography).
2. Ground-Truth Corner Tracking for exact evaluation of downstream rectification.
3. Non-Uniform Illumination & Radial Falloff (vignetting, flash hotspot).
4. Cast Shadows with Soft Penumbra (hand / phone occlusion).
5. Localized Specular Reflection / Glare.
6. Cylindrical Paper Curvature (Page Curl).
7. Optical Defocus Blur, Camera Shake, Gaussian Sensor Noise, and JPEG Compression.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from src.rendering.renderer import ECGImage

logger = logging.getLogger(__name__)


@dataclass
class DistortionConfig:
    """Master configuration for real-world smartphone photo distortion simulation."""

    # Perspective & 3D Pose
    enable_perspective: bool = True
    yaw_deg_range: Tuple[float, float] = (-14.0, 14.0)
    pitch_deg_range: Tuple[float, float] = (-14.0, 14.0)
    roll_deg_range: Tuple[float, float] = (-7.0, 7.0)
    distance_factor: float = 1.6  # Camera distance relative to max dimension

    # Paper Curvature
    enable_page_curl: bool = True
    curl_amplitude_range: Tuple[float, float] = (3.0, 8.0)  # Pixels of displacement

    # Illumination & Shadows
    enable_illumination: bool = True
    vignette_strength_range: Tuple[float, float] = (0.15, 0.35)
    enable_shadows: bool = True
    shadow_intensity_range: Tuple[float, float] = (0.35, 0.65)
    enable_glare: bool = True
    glare_intensity_range: Tuple[float, float] = (0.2, 0.5)

    # Optics, Sensor & Compression
    enable_blur: bool = True
    blur_radius_range: Tuple[float, float] = (0.5, 1.5)
    enable_noise: bool = True
    noise_sigma_range: Tuple[float, float] = (2.0, 5.0)
    enable_jpeg: bool = True
    jpeg_quality_range: Tuple[int, int] = (50, 75)

    # Background Surface (e.g. clinic desk behind the paper)
    background_color: str = "#262626"
    pad_margin_factor: float = 0.15

    @classmethod
    def from_preset(cls, preset_name: str) -> "DistortionConfig":
        """Instantiate a DistortionConfig from a named severity preset."""
        presets = {
            "clean": cls(
                enable_perspective=False,
                enable_page_curl=False,
                enable_illumination=False,
                enable_shadows=False,
                enable_glare=False,
                enable_blur=False,
                enable_noise=False,
                enable_jpeg=False,
            ),
            "mild": cls(
                enable_perspective=True,
                yaw_deg_range=(-6.0, 6.0),
                pitch_deg_range=(-6.0, 6.0),
                roll_deg_range=(-3.0, 3.0),
                enable_page_curl=True,
                curl_amplitude_range=(1.0, 4.0),
                enable_illumination=True,
                vignette_strength_range=(0.05, 0.18),
                enable_shadows=True,
                shadow_intensity_range=(0.2, 0.4),
                enable_glare=False,
                enable_blur=True,
                blur_radius_range=(0.3, 0.8),
                enable_noise=True,
                noise_sigma_range=(1.0, 3.0),
                enable_jpeg=True,
                jpeg_quality_range=(70, 85),
            ),
            "moderate": cls(
                enable_perspective=True,
                yaw_deg_range=(-14.0, 14.0),
                pitch_deg_range=(-14.0, 14.0),
                roll_deg_range=(-7.0, 7.0),
                enable_page_curl=True,
                curl_amplitude_range=(3.0, 8.0),
                enable_illumination=True,
                vignette_strength_range=(0.15, 0.35),
                enable_shadows=True,
                shadow_intensity_range=(0.35, 0.65),
                enable_glare=True,
                glare_intensity_range=(0.2, 0.5),
                enable_blur=True,
                blur_radius_range=(0.5, 1.5),
                enable_noise=True,
                noise_sigma_range=(2.0, 5.0),
                enable_jpeg=True,
                jpeg_quality_range=(50, 75),
            ),
            "severe": cls(
                enable_perspective=True,
                yaw_deg_range=(-20.0, 20.0),
                pitch_deg_range=(-20.0, 20.0),
                roll_deg_range=(-12.0, 12.0),
                enable_page_curl=True,
                curl_amplitude_range=(6.0, 14.0),
                enable_illumination=True,
                vignette_strength_range=(0.25, 0.50),
                enable_shadows=True,
                shadow_intensity_range=(0.50, 0.75),
                enable_glare=True,
                glare_intensity_range=(0.4, 0.8),
                enable_blur=True,
                blur_radius_range=(1.0, 2.5),
                enable_noise=True,
                noise_sigma_range=(4.0, 8.0),
                enable_jpeg=True,
                jpeg_quality_range=(35, 60),
            ),
        }
        if preset_name not in presets:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}")
        return presets[preset_name]


def load_augmentation_config(
    config_path: Optional[Union[str, Path]] = None, preset: Optional[str] = None
) -> DistortionConfig:
    """Load DistortionConfig from YAML file or named preset."""
    import yaml

    from src.utils.paths import CONFIGS_DIR

    if preset is not None:
        return DistortionConfig.from_preset(preset)

    target_path = Path(config_path) if config_path else CONFIGS_DIR / "augmentation.yaml"
    if not target_path.exists():
        logger.warning(
            f"Augmentation config not found at {target_path}; returning default moderate config."
        )
        return DistortionConfig.from_preset("moderate")

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    def_preset = data.get("default_preset", "moderate")
    presets = data.get("presets", {})
    if def_preset in presets:
        p_data = presets[def_preset]
        bg_data = data.get("background", {})
        return DistortionConfig(
            enable_perspective=bool(p_data.get("enable_perspective", True)),
            yaw_deg_range=tuple(p_data.get("yaw_deg_range", (-14.0, 14.0))),
            pitch_deg_range=tuple(p_data.get("pitch_deg_range", (-14.0, 14.0))),
            roll_deg_range=tuple(p_data.get("roll_deg_range", (-7.0, 7.0))),
            enable_page_curl=bool(p_data.get("enable_page_curl", True)),
            curl_amplitude_range=tuple(p_data.get("curl_amplitude_range", (3.0, 8.0))),
            enable_illumination=bool(p_data.get("enable_illumination", True)),
            vignette_strength_range=tuple(p_data.get("vignette_strength_range", (0.15, 0.35))),
            enable_shadows=bool(p_data.get("enable_shadows", True)),
            shadow_intensity_range=tuple(p_data.get("shadow_intensity_range", (0.35, 0.65))),
            enable_glare=bool(p_data.get("enable_glare", True)),
            glare_intensity_range=tuple(p_data.get("glare_intensity_range", (0.2, 0.5))),
            enable_blur=bool(p_data.get("enable_blur", True)),
            blur_radius_range=tuple(p_data.get("blur_radius_range", (0.5, 1.5))),
            enable_noise=bool(p_data.get("enable_noise", True)),
            noise_sigma_range=tuple(p_data.get("noise_sigma_range", (2.0, 5.0))),
            enable_jpeg=bool(p_data.get("enable_jpeg", True)),
            jpeg_quality_range=tuple(p_data.get("jpeg_quality_range", (50, 75))),
            background_color=str(bg_data.get("color", "#262626")),
            pad_margin_factor=float(bg_data.get("pad_margin_factor", 0.15)),
        )

    return DistortionConfig.from_preset("moderate")


@dataclass
class SimulatedECGPhoto:
    """Container for a simulated smartphone photograph with ground-truth geometry."""

    image: Image.Image
    corners_canonical: np.ndarray  # Shape (4, 2): [TL, TR, BR, BL] in canonical paper pixels
    corners_transformed: np.ndarray  # Shape (4, 2): [TL, TR, BR, BL] in output photo coordinates
    homography_matrix: np.ndarray  # Shape (3, 3): Projective matrix canonical -> photo
    metadata: Dict[str, Any]

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
        """Convert image to uint8 RGB NumPy array of shape (H, W, 3)."""
        return np.array(self.image, dtype=np.uint8)

    def to_tensor(self, channels_first: bool = True, normalize_0_1: bool = True) -> torch.Tensor:
        """Convert image to PyTorch tensor."""
        arr = self.to_numpy()
        tensor = torch.from_numpy(arr).float() / 255.0 if normalize_0_1 else torch.from_numpy(arr)
        if channels_first:
            tensor = tensor.permute(2, 0, 1)
        return tensor.contiguous()

    def save(self, path: Union[str, Path], quality: int = 95) -> Path:
        """Save photo to disk, creating parent directories if needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(p, quality=quality)
        return p


def solve_homography_coefficients(
    src_pts: Sequence[Tuple[float, float]],
    dst_pts: Sequence[Tuple[float, float]],
) -> List[float]:
    """Calculate 8-parameter coefficients for Pillow's Image.Transform.PERSPECTIVE.

    Maps pixels in the target output frame (dst_pts) back to coordinates in
    the input source image (src_pts):
        x = (a*x' + b*y' + c) / (g*x' + h*y' + 1)
        y = (d*x' + e*y' + f) / (g*x' + h*y' + 1)

    Returns:
        List of 8 float coefficients [a, b, c, d, e, f, g, h].
    """
    matrix: List[List[float]] = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        matrix.append([xp, yp, 1.0, 0.0, 0.0, 0.0, -x * xp, -x * yp, -x])
        matrix.append([0.0, 0.0, 0.0, xp, yp, 1.0, -y * xp, -y * yp, -y])

    A = np.array(matrix, dtype=np.float64)
    # Singular Value Decomposition to solve A * h = 0
    _, _, Vh = np.linalg.svd(A)
    h = Vh[-1]
    if abs(h[-1]) > 1e-12:
        h = h / h[-1]
    return h[:8].tolist()


def compute_3d_perspective_homography(
    width: int,
    height: int,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    distance_factor: float = 1.6,
    out_width: Optional[int] = None,
    out_height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mathematical 3D pinhole camera rotation and projective homography.

    Args:
        width: Input document pixel width.
        height: Input document pixel height.
        yaw_deg: Rotation angle around vertical axis (degrees).
        pitch_deg: Rotation angle around horizontal axis (degrees).
        roll_deg: In-plane camera tilt (degrees).
        distance_factor: Camera distance ratio relative to max image dimension.
        out_width: Desired output canvas width (defaults to padded width).
        out_height: Desired output canvas height (defaults to padded height).

    Returns:
        Tuple of:
            - H: 3x3 homography matrix mapping canonical corners to output frame
            - dst_corners: Array of shape (4, 2) with transformed corner coordinates
            - canonical_corners: Array of shape (4, 2) [TL, TR, BR, BL]
    """
    w, h = float(width), float(height)
    canonical = np.array(
        [
            [0.0, 0.0],  # Top-Left
            [w, 0.0],  # Top-Right
            [w, h],  # Bottom-Right
            [0.0, h],  # Bottom-Left
        ],
        dtype=np.float64,
    )

    # Convert angles to radians
    rx = np.radians(pitch_deg)
    ry = np.radians(yaw_deg)
    rz = np.radians(roll_deg)

    # 3D Rotation matrices
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx), np.cos(rx)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(ry), 0.0, np.sin(ry)],
            [0.0, 1.0, 0.0],
            [-np.sin(ry), 0.0, np.cos(ry)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0.0],
            [np.sin(rz), np.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R = Rz @ Ry @ Rx

    # Focal distance and camera center
    max_dim = max(w, h)
    f = distance_factor * max_dim
    d = f

    # Camera Intrinsic matrix K and translation t
    # Center of canonical plane
    cx, cy = w / 2.0, h / 2.0
    pts_centered = canonical.copy()
    pts_centered[:, 0] -= cx
    pts_centered[:, 1] -= cy

    # Project 4 corners in 3D: P_3d = R @ [x, y, 0]^T + [0, 0, d]^T
    transformed_pts: List[Tuple[float, float]] = []
    for pt in pts_centered:
        v = np.array([pt[0], pt[1], 0.0], dtype=np.float64)
        rot_v = R @ v
        z = rot_v[2] + d
        if z <= 0.1:
            z = 0.1
        xp = f * (rot_v[0] / z)
        yp = f * (rot_v[1] / z)
        transformed_pts.append((xp, yp))

    pts_arr = np.array(transformed_pts, dtype=np.float64)

    # Compute bounding canvas and offset so all corners sit inside with padding
    min_x, max_x = np.min(pts_arr[:, 0]), np.max(pts_arr[:, 0])
    min_y, max_y = np.min(pts_arr[:, 1]), np.max(pts_arr[:, 1])

    span_x = max_x - min_x
    span_y = max_y - min_y

    target_w = out_width if out_width is not None else int(round(span_x * 1.35))
    target_h = out_height if out_height is not None else int(round(span_y * 1.35))

    out_cx = target_w / 2.0
    out_cy = target_h / 2.0

    dst_corners = np.zeros_like(pts_arr)
    dst_corners[:, 0] = pts_arr[:, 0] + out_cx
    dst_corners[:, 1] = pts_arr[:, 1] + out_cy

    # Compute 3x3 homography canonical -> dst_corners via SVD
    matrix: List[List[float]] = []
    for (x, y), (xp, yp) in zip(canonical, dst_corners):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -xp * x, -xp * y, -xp])
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -yp * x, -yp * y, -yp])

    A = np.array(matrix, dtype=np.float64)
    _, _, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape((3, 3))
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H, dst_corners.astype(np.float32), canonical.astype(np.float32)


def apply_page_curl(
    image: Image.Image,
    amplitude_px: float = 5.0,
    axis: str = "horizontal",
) -> Image.Image:
    """Apply subtle cylindrical paper curvature (page curl) using coordinate warping."""
    if abs(amplitude_px) < 0.5:
        return image

    arr = np.array(image)
    h, w = arr.shape[:2]

    # Create coordinate grids
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)

    if axis == "horizontal":
        # Displace Y along X sinusoid
        disp = amplitude_px * np.sin(np.pi * x_idx / w)
        y_idx = np.clip(y_idx + disp, 0, h - 1)
    else:
        # Displace X along Y sinusoid
        disp = amplitude_px * np.sin(np.pi * y_idx / h)
        x_idx = np.clip(x_idx + disp, 0, w - 1)

    curled = np.zeros_like(arr)
    for c in range(arr.shape[2]):
        curled[:, :, c] = ndimage.map_coordinates(
            arr[:, :, c], [y_idx, x_idx], order=1, mode="nearest"
        )

    return Image.fromarray(curled)


def apply_illumination_gradient(
    image: Image.Image,
    vignette_strength: float = 0.25,
    center: Optional[Tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Apply radial illumination gradient (vignetting / flash hotspot falloff)."""
    if vignette_strength <= 0.01:
        return image

    w, h = image.size
    rand = rng or np.random.default_rng()
    cx = center[0] if center is not None else w * rand.uniform(0.4, 0.6)
    cy = center[1] if center is not None else h * rand.uniform(0.4, 0.6)

    # Compute radial distance map normalized by max radius
    y_coords, x_coords = np.ogrid[:h, :w]
    max_radius = np.sqrt(max(cx, w - cx) ** 2 + max(cy, h - cy) ** 2)
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (max_radius + 1e-6)

    # Illumination factor: 1.0 at center, falling off to (1.0 - vignette_strength) at corners
    vignette_map = 1.0 - (vignette_strength * (dist**1.5))
    vignette_map = np.clip(vignette_map, 0.0, 1.0).astype(np.float32)

    arr = np.array(image, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] *= vignette_map

    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def apply_cast_shadow(
    image: Image.Image,
    shadow_intensity: float = 0.45,
    polygon: Optional[List[Tuple[float, float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Synthesize a realistic cast shadow (hand/phone body) with a soft penumbra."""
    if shadow_intensity <= 0.05:
        return image

    w, h = image.size
    rand = rng or np.random.default_rng()

    # Generate a plausible cast shadow polygon across one corner/edge if none provided
    if polygon is None:
        corner = rand.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
        if corner == "top_left":
            poly = [(0, 0), (rand.uniform(0.3, 0.7) * w, 0), (0, rand.uniform(0.4, 0.8) * h)]
        elif corner == "top_right":
            poly = [(w, 0), (w, rand.uniform(0.4, 0.8) * h), (rand.uniform(0.3, 0.7) * w, 0)]
        elif corner == "bottom_left":
            poly = [(0, h), (0, rand.uniform(0.2, 0.6) * h), (rand.uniform(0.3, 0.7) * w, h)]
        else:
            poly = [(w, h), (rand.uniform(0.3, 0.7) * w, h), (w, rand.uniform(0.2, 0.6) * h)]
    else:
        poly = polygon

    # Render binary shadow mask
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(poly, fill=255)

    # Soft Gaussian blur on mask to produce penumbra transition (15-30 px radius)
    blur_rad = int(round(max(w, h) * 0.035))
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_rad))

    mask_arr = np.array(soft_mask, dtype=np.float32) / 255.0
    # Shadow factor: 1.0 where mask=0, (1.0 - shadow_intensity) where mask=1
    shadow_mult = 1.0 - (shadow_intensity * mask_arr)

    arr = np.array(image, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] *= shadow_mult

    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def apply_specular_glare(
    image: Image.Image,
    glare_intensity: float = 0.35,
    center: Optional[Tuple[float, float]] = None,
    radius: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Simulate localized specular glare from overhead clinic light or flash."""
    if glare_intensity <= 0.05:
        return image

    w, h = image.size
    rand = rng or np.random.default_rng()

    gx = center[0] if center is not None else rand.uniform(0.25 * w, 0.75 * w)
    gy = center[1] if center is not None else rand.uniform(0.25 * h, 0.75 * h)
    r = radius if radius is not None else rand.uniform(0.12 * min(w, h), 0.28 * min(w, h))

    y_coords, x_coords = np.ogrid[:h, :w]
    dist_sq = (x_coords - gx) ** 2 + (y_coords - gy) ** 2
    # Gaussian bell curve for reflection hotspot
    glare_map = glare_intensity * 255.0 * np.exp(-dist_sq / (2.0 * (r**2)))

    arr = np.array(image, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] += glare_map

    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def apply_optical_blur(image: Image.Image, radius_px: float = 1.0) -> Image.Image:
    """Apply camera optical defocus blur."""
    if radius_px <= 0.1:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius_px))


def apply_sensor_noise(
    image: Image.Image,
    sigma: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Apply zero-mean additive Gaussian sensor noise."""
    if sigma <= 0.2:
        return image

    rand = rng or np.random.default_rng()
    arr = np.array(image, dtype=np.float32)
    noise = rand.normal(0.0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def apply_jpeg_compression(image: Image.Image, quality: int = 65) -> Image.Image:
    """Simulate smartphone JPEG compression and quantization artifacts."""
    q = int(np.clip(quality, 20, 95))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=q)
    buffer.seek(0)
    return Image.open(buffer).copy()


class SmartphoneECGSimulator:
    """Production Smartphone Photograph & Distortion Simulation Engine."""

    def __init__(
        self,
        config: Optional[DistortionConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or DistortionConfig.from_preset("moderate")
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def simulate(
        self,
        image_or_array: Union[Image.Image, ECGImage, np.ndarray],
        custom_angles: Optional[Tuple[float, float, float]] = None,
    ) -> SimulatedECGPhoto:
        """Simulate realistic smartphone photo capture on an ECG recording.

        Args:
            image_or_array: Input pristine ECG recording (PIL Image, ECGImage, or array).
            custom_angles: Optional (yaw_deg, pitch_deg, roll_deg) angle override.

        Returns:
            SimulatedECGPhoto instance with transformed image and ground-truth corner coordinates.
        """
        # Step 0: Standardize input to PIL Image
        if isinstance(image_or_array, ECGImage):
            src_img = image_or_array.pil_image.copy()
        elif isinstance(image_or_array, np.ndarray):
            src_img = Image.fromarray(image_or_array)
        elif isinstance(image_or_array, Image.Image):
            src_img = image_or_array.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image_or_array)}")

        w, h = src_img.size
        rand = self._rng

        # Step 1: Subtle Paper Curvature (Page Curl)
        if self.config.enable_page_curl:
            amp = rand.uniform(*self.config.curl_amplitude_range)
            axis = str(rand.choice(["horizontal", "vertical"]))
            src_img = apply_page_curl(src_img, amplitude_px=amp, axis=axis)

        # Step 2: 3D Camera Pose & Perspective Transformation
        if custom_angles is not None:
            yaw, pitch, roll = custom_angles
        elif self.config.enable_perspective:
            yaw = rand.uniform(*self.config.yaw_deg_range)
            pitch = rand.uniform(*self.config.pitch_deg_range)
            roll = rand.uniform(*self.config.roll_deg_range)
        else:
            yaw, pitch, roll = 0.0, 0.0, 0.0

        H, dst_corners, canonical_corners = compute_3d_perspective_homography(
            width=w,
            height=h,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            distance_factor=self.config.distance_factor,
        )

        # Calculate bounding canvas size
        out_w = int(np.ceil(np.max(dst_corners[:, 0]) + (w * self.config.pad_margin_factor)))
        out_h = int(np.ceil(np.max(dst_corners[:, 1]) + (h * self.config.pad_margin_factor)))

        # Solve 8 coefficients mapping output pixels back to input
        coeffs = solve_homography_coefficients(
            src_pts=[tuple(pt) for pt in canonical_corners],
            dst_pts=[tuple(pt) for pt in dst_corners],
        )

        # Render background canvas
        bg_color = self.config.background_color
        warped_paper = src_img.transform(
            (out_w, out_h),
            Image.Transform.PERSPECTIVE,
            coeffs,
            Image.Resampling.BILINEAR,
            fillcolor=None,
        )

        # Create paper mask to alpha-composite paper onto desk background
        mask = Image.new("L", (w, h), 255)
        warped_mask = mask.transform(
            (out_w, out_h),
            Image.Transform.PERSPECTIVE,
            coeffs,
            Image.Resampling.BILINEAR,
            fillcolor=0,
        )

        final_photo = Image.new("RGB", (out_w, out_h), bg_color)
        final_photo.paste(warped_paper, (0, 0), warped_mask)

        # Step 3: Illumination & Lighting Variations
        if self.config.enable_illumination:
            vignette = rand.uniform(*self.config.vignette_strength_range)
            final_photo = apply_illumination_gradient(
                final_photo, vignette_strength=vignette, rng=rand
            )

        # Step 4: Cast Shadow Simulation
        if self.config.enable_shadows:
            shadow_int = rand.uniform(*self.config.shadow_intensity_range)
            final_photo = apply_cast_shadow(final_photo, shadow_intensity=shadow_int, rng=rand)

        # Step 5: Specular Glare / Reflection Hotspots
        if self.config.enable_glare:
            glare_int = rand.uniform(*self.config.glare_intensity_range)
            final_photo = apply_specular_glare(final_photo, glare_intensity=glare_int, rng=rand)

        # Step 6: Optical Defocus Blur & Shake
        if self.config.enable_blur:
            blur_rad = rand.uniform(*self.config.blur_radius_range)
            final_photo = apply_optical_blur(final_photo, radius_px=blur_rad)

        # Step 7: Gaussian Sensor Noise
        if self.config.enable_noise:
            noise_sig = rand.uniform(*self.config.noise_sigma_range)
            final_photo = apply_sensor_noise(final_photo, sigma=noise_sig, rng=rand)

        # Step 8: JPEG Compression Quantization
        if self.config.enable_jpeg:
            jpeg_q = int(rand.integers(*self.config.jpeg_quality_range))
            final_photo = apply_jpeg_compression(final_photo, quality=jpeg_q)

        meta: Dict[str, Any] = {
            "camera_yaw_deg": float(yaw),
            "camera_pitch_deg": float(pitch),
            "camera_roll_deg": float(roll),
            "distance_factor": float(self.config.distance_factor),
            "original_width": w,
            "original_height": h,
            "photo_width": out_w,
            "photo_height": out_h,
            "seed": self.seed,
        }

        return SimulatedECGPhoto(
            image=final_photo,
            corners_canonical=canonical_corners,
            corners_transformed=dst_corners,
            homography_matrix=H,
            metadata=meta,
        )
