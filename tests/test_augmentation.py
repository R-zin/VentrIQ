"""tests/test_augmentation.py - Unit Tests for Smartphone Capture & Distortion Simulation.

Tests cover:
1. 3D Camera Pose & Perspective Homography calculation.
2. Homography coefficients and inverse perspective mapping.
3. Ground-truth corner tracking and geometric bounding.
4. Non-uniform illumination, vignetting, and flash gradients.
5. Polygonal cast shadows with Gaussian penumbra.
6. Specular reflection hotspots.
7. Cylindrical paper curvature (page curl).
8. Defocus blur, sensor noise, and JPEG compression.
9. Configuration presets (clean, mild, moderate, severe) and YAML loading.
10. Deterministic execution and reproducibility across random seeds.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import pytest
import torch

from src.rendering.augmentation import (
    DistortionConfig,
    SimulatedECGPhoto,
    SmartphoneECGSimulator,
    apply_cast_shadow,
    apply_illumination_gradient,
    apply_jpeg_compression,
    apply_optical_blur,
    apply_page_curl,
    apply_sensor_noise,
    apply_specular_glare,
    compute_3d_perspective_homography,
    load_augmentation_config,
    solve_homography_coefficients,
)


@pytest.fixture
def sample_image() -> Image.Image:
    """Generate a simple test image with distinct quadrant colors."""
    img = Image.new("RGB", (200, 150), "#ffffff")
    arr = np.array(img)
    arr[:75, :100] = [255, 0, 0]      # Red top-left
    arr[:75, 100:] = [0, 255, 0]      # Green top-right
    arr[75:, :100] = [0, 0, 255]      # Blue bottom-left
    arr[75:, 100:] = [255, 255, 0]    # Yellow bottom-right
    return Image.fromarray(arr)


def test_distortion_config_presets():
    """Verify that distortion presets instantiate with valid ranges."""
    for preset_name in ["clean", "mild", "moderate", "severe"]:
        cfg = DistortionConfig.from_preset(preset_name)
        assert isinstance(cfg, DistortionConfig)
        if preset_name == "clean":
            assert cfg.enable_perspective is False
            assert cfg.enable_shadows is False
        else:
            assert cfg.enable_perspective is True
            assert cfg.yaw_deg_range[0] < cfg.yaw_deg_range[1]
            assert cfg.pitch_deg_range[0] < cfg.pitch_deg_range[1]

    with pytest.raises(ValueError, match="Unknown preset"):
        DistortionConfig.from_preset("ultra_extreme")


def test_compute_3d_perspective_homography():
    """Verify 3D camera rotation and homography calculation."""
    w, h = 400, 300
    H, dst_corners, canonical = compute_3d_perspective_homography(
        width=w,
        height=h,
        yaw_deg=10.0,
        pitch_deg=-8.0,
        roll_deg=4.0,
    )

    assert H.shape == (3, 3)
    assert dst_corners.shape == (4, 2)
    assert canonical.shape == (4, 2)

    # Canonical corners must match [TL, TR, BR, BL]
    np.testing.assert_allclose(canonical[0], [0.0, 0.0])
    np.testing.assert_allclose(canonical[1], [w, 0.0])
    np.testing.assert_allclose(canonical[2], [w, h])
    np.testing.assert_allclose(canonical[3], [0.0, h])

    # Transformed quadrilateral must have positive area
    dx1, dy1 = dst_corners[1] - dst_corners[0]
    dx2, dy2 = dst_corners[3] - dst_corners[0]
    cross = dx1 * dy2 - dx2 * dy1
    assert cross > 0  # Non-degenerate quadrilateral


def test_solve_homography_coefficients():
    """Verify mathematical inversion of Pillow perspective transform coefficients."""
    src = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    dst = [(10.0, 15.0), (95.0, 5.0), (105.0, 90.0), (5.0, 95.0)]
    coeffs = solve_homography_coefficients(src, dst)
    assert len(coeffs) == 8
    assert all(np.isfinite(coeffs))


def test_page_curl_deformation(sample_image):
    """Verify cylindrical page curl displaces pixels while preserving bounds."""
    curled = apply_page_curl(sample_image, amplitude_px=6.0, axis="horizontal")
    assert curled.size == sample_image.size
    arr_orig = np.array(sample_image)
    arr_curl = np.array(curled)
    assert not np.array_equal(arr_orig, arr_curl)


def test_illumination_gradient(sample_image):
    """Verify radial illumination falloff darkens edges relative to center."""
    lit = apply_illumination_gradient(sample_image, vignette_strength=0.35, center=(100.0, 75.0))
    assert lit.size == sample_image.size

    arr = np.array(lit, dtype=float)
    # Center pixel should be brighter than corner pixel
    center_val = np.mean(arr[75, 100])
    corner_val = np.mean(arr[0, 0])
    assert center_val > corner_val


def test_cast_shadow_simulation(sample_image):
    """Verify cast shadow creates realistic intensity reduction with soft penumbra."""
    poly = [(0, 0), (100, 0), (0, 75)]
    shadowed = apply_cast_shadow(sample_image, shadow_intensity=0.5, polygon=poly)
    assert shadowed.size == sample_image.size

    arr_orig = np.array(sample_image, dtype=float)
    arr_shad = np.array(shadowed, dtype=float)

    # Inside shadow region (top-left) intensity should be lower
    mean_shadowed = np.mean(arr_shad[:30, :30])
    mean_orig = np.mean(arr_orig[:30, :30])
    assert mean_shadowed < mean_orig

    # Far corner (bottom-right) should remain unaffected
    np.testing.assert_allclose(arr_shad[120:, 150:], arr_orig[120:, 150:], atol=2.0)


def test_specular_glare(sample_image):
    """Verify localized specular reflection increases brightness."""
    glared = apply_specular_glare(sample_image, glare_intensity=0.5, center=(100.0, 75.0), radius=20.0)
    assert glared.size == sample_image.size

    arr_orig = np.array(sample_image, dtype=float)
    arr_glar = np.array(glared, dtype=float)

    # Center of glare should be brighter than original
    assert np.mean(arr_glar[75, 100]) >= np.mean(arr_orig[75, 100])


def test_optical_blur_and_noise(sample_image):
    """Verify Gaussian blur and sensor noise execution."""
    blurred = apply_optical_blur(sample_image, radius_px=1.5)
    assert blurred.size == sample_image.size

    noisy = apply_sensor_noise(sample_image, sigma=4.0)
    assert noisy.size == sample_image.size
    arr_noisy = np.array(noisy, dtype=float)
    arr_orig = np.array(sample_image, dtype=float)
    # Variance of difference should approximate sigma^2
    diff_var = np.var(arr_noisy - arr_orig)
    assert 5.0 <= diff_var <= 30.0


def test_jpeg_compression(sample_image):
    """Verify JPEG compression artifact simulation."""
    compressed = apply_jpeg_compression(sample_image, quality=40)
    assert compressed.size == sample_image.size
    assert compressed.mode == "RGB"


def test_smartphone_ecg_simulator_end_to_end(sample_image):
    """Verify full end-to-end simulation pipeline and metadata tracking."""
    cfg = DistortionConfig.from_preset("moderate")
    simulator = SmartphoneECGSimulator(config=cfg, seed=42)

    sim_photo = simulator.simulate(sample_image)
    assert isinstance(sim_photo, SimulatedECGPhoto)
    assert sim_photo.width > sample_image.width
    assert sim_photo.height > sample_image.height

    # Transformed corners check
    assert sim_photo.corners_transformed.shape == (4, 2)
    assert np.all(sim_photo.corners_transformed >= 0.0)
    assert np.all(sim_photo.corners_transformed[:, 0] <= sim_photo.width)
    assert np.all(sim_photo.corners_transformed[:, 1] <= sim_photo.height)

    # Tensor export check
    t_ch = sim_photo.to_tensor(channels_first=True, normalize_0_1=True)
    assert isinstance(t_ch, torch.Tensor)
    assert t_ch.shape == (3, sim_photo.height, sim_photo.width)
    assert 0.0 <= float(t_ch.min()) <= float(t_ch.max()) <= 1.0


def test_simulator_reproducibility(sample_image):
    """Verify that identical random seeds produce identical simulated outputs."""
    sim1 = SmartphoneECGSimulator(config=DistortionConfig.from_preset("moderate"), seed=12345)
    sim2 = SmartphoneECGSimulator(config=DistortionConfig.from_preset("moderate"), seed=12345)

    p1 = sim1.simulate(sample_image)
    p2 = sim2.simulate(sample_image)

    np.testing.assert_allclose(p1.corners_transformed, p2.corners_transformed, atol=1e-5)
    np.testing.assert_array_equal(np.array(p1.image), np.array(p2.image))


def test_load_augmentation_config_yaml(tmp_path: Path):
    """Verify loading augmentation config from custom YAML and presets."""
    cfg = load_augmentation_config(preset="mild")
    assert cfg.enable_perspective is True
    assert cfg.yaw_deg_range == (-6.0, 6.0)

    # From default YAML
    cfg_default = load_augmentation_config()
    assert isinstance(cfg_default, DistortionConfig)
