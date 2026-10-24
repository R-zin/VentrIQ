"""tests/test_stress_and_robustness.py - Stress, robustness, and numerical boundary tests.

Tests extreme perspective tilts, boundary lighting conditions, empty/saturated inputs,
multi-worker PyTorch DataLoader concurrency, and end-to-end pipeline integrity.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import DatasetManifest, VentriQTorchDataset
from src.data.echonext import (
    EXPECTED_TIME_STEPS,
    NUM_LEADS,
    EchoNextDataset,
    create_mock_echonext_dataset,
)
from src.preprocessing.image import (
    ECGImageEnhancer,
    estimate_background_illumination,
    normalize_illumination,
)
from src.preprocessing.perspective import (
    DocumentRectifier,
    detect_document_corners,
)
from src.preprocessing.quality import ECGQualityEvaluator
from src.preprocessing.signal import ECGSignalPreprocessor
from src.preprocessing.transforms import (
    MobileNetV3Preprocessor,
    extract_12_lead_crops,
    image_to_mobilenet_tensor,
)
from src.rendering.augmentation import (
    DistortionConfig,
    SmartphoneECGSimulator,
    compute_3d_perspective_homography,
)
from src.rendering.layout import ECGLayoutConfig
from src.rendering.renderer import ECGPaperRenderer


def test_extreme_perspective_homography_stability() -> None:
    """Verify homography computation stability under extreme tilt angles."""
    w, h = 800, 600
    extreme_angles = [
        (35.0, 0.0, 0.0),
        (0.0, 40.0, 0.0),
        (0.0, 0.0, 30.0),
        (-35.0, -35.0, 25.0),
        (45.0, -40.0, -20.0),
    ]

    for yaw, pitch, roll in extreme_angles:
        H, dst_corners, canonical_corners = compute_3d_perspective_homography(
            w, h, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, distance_factor=1.8
        )
        assert H.shape == (3, 3)
        assert np.isfinite(H).all(), (
            f"Homography matrix has non-finite values at ({yaw}, {pitch}, {roll})"
        )
        det = float(np.linalg.det(H))
        assert abs(det) > 1e-6, f"Singular homography matrix at ({yaw}, {pitch}, {roll})"

        # Verify destination corners are finite
        assert np.isfinite(dst_corners).all()
        assert dst_corners.shape == (4, 2)


def test_corner_detector_on_degenerate_images() -> None:
    """Verify corner detection handles pure black, pure white, and extreme noise without crashing."""
    w, h = 640, 480

    # 1. Solid black image (e.g. phone camera covered)
    black_img = Image.new("RGB", (w, h), (0, 0, 0))
    detected_black = detect_document_corners(black_img)
    assert detected_black.corners.shape == (4, 2)
    assert np.isfinite(detected_black.corners).all()

    # 2. Solid white image (e.g. blinding specular glare)
    white_img = Image.new("RGB", (w, h), (255, 255, 255))
    detected_white = detect_document_corners(white_img)
    assert detected_white.corners.shape == (4, 2)
    assert np.isfinite(detected_white.corners).all()

    # 3. Random noise image
    noise_arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr, mode="RGB")
    detected_noise = detect_document_corners(noise_img)
    assert detected_noise.corners.shape == (4, 2)
    assert np.isfinite(detected_noise.corners).all()


def test_illumination_normalization_boundary_conditions() -> None:
    """Verify illumination correction on high dynamic range, inverted, and extreme occlusions."""
    h, w = 300, 400

    # 1. Inverted contrast image (dark background, light traces)
    inverted_arr = np.zeros((h, w, 3), dtype=np.uint8)
    inverted_arr[145:155, :] = 255  # white horizontal line
    inverted_img = Image.fromarray(inverted_arr, mode="RGB")

    gray_inv = np.mean(inverted_arr, axis=2).astype(np.float32)
    bg_inv = estimate_background_illumination(gray_inv, sigma=20.0)
    assert bg_inv.shape == (h, w)
    assert np.isfinite(bg_inv).all()

    norm_inv = normalize_illumination(inverted_img, sigma=20.0)
    assert norm_inv.size == (w, h)
    norm_arr = np.array(norm_inv)
    assert np.isfinite(norm_arr).all()
    assert norm_arr.min() >= 0 and norm_arr.max() <= 255

    # 2. Heavy shadow occluding 90% of image
    shadow_arr = np.full((h, w, 3), 220, dtype=np.uint8)
    shadow_arr[:, : int(w * 0.9)] = 30  # deep shadow zone
    shadow_img = Image.fromarray(shadow_arr, mode="RGB")

    enhancer = ECGImageEnhancer(enable_clahe=True)
    enhanced = enhancer(shadow_img)
    assert enhanced.size == (w, h)
    enh_arr = np.array(enhanced)
    assert np.isfinite(enh_arr).all()


def test_renderer_with_extreme_voltages_and_flatline() -> None:
    """Verify waveform renderer is numerically stable with extreme voltages and flatlines."""
    renderer = ECGPaperRenderer(
        ECGLayoutConfig(paper_width_mm=100.0, paper_height_mm=70.0, dpi=100)
    )

    # 1. Pure zero flatline
    flat_ecg = np.zeros((NUM_LEADS, EXPECTED_TIME_STEPS), dtype=np.float32)
    flat_img = renderer.render(flat_ecg)
    assert flat_img.size[0] > 0 and flat_img.size[1] > 0
    assert flat_img.metadata["layout_format"] == "3x4_1r"
    assert flat_img.metadata["dpi"] == 100

    # 2. Saturated rail voltages (+/- 50 mV)
    sat_ecg = np.full((NUM_LEADS, EXPECTED_TIME_STEPS), 50.0, dtype=np.float32)
    sat_ecg[::2, :] = -50.0
    sat_img = renderer.render(sat_ecg)
    assert sat_img.size[0] > 0 and sat_img.size[1] > 0


def test_mobilenet_transforms_with_irregular_aspect_ratios() -> None:
    """Verify vision transforms handle non-square aspect ratios and produce bounded tensors."""
    # Panoramic wide aspect ratio (1200 x 300)
    wide_img = Image.new("RGB", (1200, 300), color="#f5e6d3")
    tensor_wide = image_to_mobilenet_tensor(wide_img, target_size=(224, 224))
    assert tensor_wide.shape == (3, 224, 224)
    assert tensor_wide.dtype == torch.float32
    assert torch.isfinite(tensor_wide).all()

    # Tall vertical aspect ratio (300 x 1200)
    tall_img = Image.new("RGB", (300, 1200), color="#ffffff")
    crops = extract_12_lead_crops(tall_img, patch_size=(224, 112))
    assert crops.shape == (12, 3, 112, 224)
    assert crops.dtype == torch.float32
    assert torch.isfinite(crops).all()


def test_multiworker_dataloader_concurrency(tmp_path: pytest.TempPathFactory) -> None:
    """Verify PyTorch DataLoader multi-worker retrieval with zero deadlocks or corruption."""
    echonext_dir = tmp_path / "echonext_mock"
    create_mock_echonext_dataset(echonext_dir, num_train=20, num_val=6, num_test=6, seed=42)

    manifest = DatasetManifest.build_from_echonext(echonext_dir)
    dataset = VentriQTorchDataset(root=echonext_dir, records=manifest.records)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        drop_last=False,
    )

    total_samples = 0
    for epoch in range(2):
        epoch_samples = 0
        for batch in loader:
            waveforms = batch["waveform"]
            targets = batch["lvef"]
            assert waveforms.shape[0] <= 8
            assert waveforms.shape[1] == NUM_LEADS
            assert waveforms.shape[2] == EXPECTED_TIME_STEPS
            assert waveforms.dtype == torch.float32
            assert torch.isfinite(waveforms).all()
            assert targets.shape[0] == waveforms.shape[0]
            epoch_samples += waveforms.shape[0]
        assert epoch_samples == 32
        total_samples += epoch_samples

    assert total_samples == 64


def test_end_to_end_pipeline_integration(tmp_path: pytest.TempPathFactory) -> None:
    """Test full pipeline: Signal -> Quality -> Filter -> Render -> Distort -> Rectify -> CLAHE -> MobileNet."""
    echonext_dir = tmp_path / "echonext_pipeline"
    create_mock_echonext_dataset(echonext_dir, num_train=6, num_val=2, num_test=2, seed=42)
    raw_dataset = EchoNextDataset(echonext_dir)
    raw_ecg = raw_dataset.get_waveform(0, channels_first=True)

    # 1. Quality & Preprocessing
    qc = ECGQualityEvaluator()
    qc_rep = qc.evaluate(raw_ecg)
    assert qc_rep.is_valid

    preprocessor = ECGSignalPreprocessor(apply_filter=True, apply_normalization=True)
    processed = preprocessor.process(raw_ecg)
    assert processed.final is not None
    assert processed.final.shape in {
        (NUM_LEADS, EXPECTED_TIME_STEPS),
        (EXPECTED_TIME_STEPS, NUM_LEADS),
    }

    # 2. Render to Paper ECG
    renderer = ECGPaperRenderer(
        ECGLayoutConfig(paper_width_mm=140.0, paper_height_mm=95.0, dpi=100)
    )
    rendered = renderer.render(processed.final)
    assert rendered.pil_image.size[0] > 0

    # 3. Smartphone Capture Simulation
    dist_config = DistortionConfig.from_preset("moderate")
    simulator = SmartphoneECGSimulator(dist_config, seed=42)
    simulated = simulator.simulate(rendered.pil_image)
    assert simulated.image.size[0] > 0
    assert simulated.corners_transformed.shape == (4, 2)
    assert simulated.corners_canonical.shape == (4, 2)

    # 4. Rectification
    rectifier = DocumentRectifier(target_size=(560, 380))
    rectified = rectifier(simulated.image)
    assert rectified.image.size == (560, 380)

    # 5. Illumination Normalization & CLAHE
    enhancer = ECGImageEnhancer(enable_clahe=True)
    enhanced = enhancer(rectified.image)
    assert enhanced.size == (560, 380)

    # 6. MobileNetV3 Tensor Formatting
    tensor_preprocessor = MobileNetV3Preprocessor(
        target_size=(224, 224),
        extract_lead_crops=False,
        apply_rectification=False,
        apply_enhancement=False,
    )
    tensor = tensor_preprocessor(enhanced)
    assert tensor.shape == (3, 224, 224)
    assert torch.isfinite(tensor).all()

    crop_preprocessor = MobileNetV3Preprocessor(
        target_size=(224, 224),
        extract_lead_crops=True,
        apply_rectification=False,
        apply_enhancement=False,
    )
    crops = crop_preprocessor(enhanced)
    assert crops.shape == (12, 3, 112, 224)
    assert torch.isfinite(crops).all()
