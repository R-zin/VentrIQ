"""tests/test_renderer.py - Comprehensive Unit Tests for Phase 4 ECG Paper Renderer.

Tests cover:
1. Layout geometry, coordinate conversions, and clinical standards.
2. 3x4 + 1 Rhythm layout segmentation and lead mapping.
3. Minor (1 mm) and Major (5 mm) grid generation and spacing.
4. Calibration pulse geometry (1.0 mV x 0.20 s).
5. Waveform rendering into PIL Image, NumPy array, and PyTorch tensors.
6. Multi-resolution DPI scaling (100, 150, 200 DPI).
7. Metadata propagation from ECGRecord and clinical dictionaries.
8. PyTorch DataLoader integration with VentriQTorchDataset.
9. Numerical safety (NaNs, Infs, orientation auto-detection, error handling).
10. YAML configuration loading and style presets.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ECGRecord, VentriQTorchDataset
from src.data.echonext import (
    EXPECTED_TIME_STEPS,
    NUM_LEADS,
    SAMPLING_RATE_HZ,
    STANDARD_12_LEADS,
    create_mock_echonext_dataset,
)
from src.rendering import (
    CalibrationPulseConfig,
    ECGImage,
    ECGLayout,
    ECGLayoutConfig,
    ECGPaperRenderer,
    GridConfig,
    load_rendering_config,
)


@pytest.fixture
def synthetic_ecg() -> np.ndarray:
    """Generate a clean synthetic 12-lead ECG array of shape (2500, 12)."""
    t = np.linspace(0, 10.0, EXPECTED_TIME_STEPS, endpoint=False)
    waveforms = np.zeros((EXPECTED_TIME_STEPS, NUM_LEADS), dtype=np.float32)
    for lead in range(NUM_LEADS):
        qrs = np.sin(2 * np.pi * 1.2 * t) ** 11
        noise = 0.05 * np.cos(2 * np.pi * 0.2 * t)
        waveforms[:, lead] = (qrs + noise).astype(np.float32)
    return waveforms


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Generate a temporary mock EchoNext dataset."""
    return create_mock_echonext_dataset(
        output_dir=tmp_path / "mock_echonext",
        num_train=8,
        num_val=4,
        num_test=4,
        seed=123,
    )


# ---------------------------------------------------------------------------
# 1. Layout Configuration & Physical Geometry Tests
# ---------------------------------------------------------------------------


def test_layout_config_defaults_and_validation():
    """Verify standard clinical defaults and parameter validation."""
    cfg = ECGLayoutConfig()
    assert cfg.dpi == 150
    assert cfg.paper_speed_mm_s == 25.0
    assert cfg.voltage_gain_mm_mv == 10.0
    assert cfg.paper_width_mm == 280.0
    assert cfg.paper_height_mm == 190.0
    assert cfg.layout_format == "3x4_1r"
    assert cfg.rhythm_lead_name == "II"

    # Validation errors
    with pytest.raises(ValueError, match="DPI must be positive"):
        ECGLayoutConfig(dpi=0)

    with pytest.raises(ValueError, match="Paper dimensions must be positive"):
        ECGLayoutConfig(paper_width_mm=-10.0)

    with pytest.raises(ValueError, match="Unsupported layout format"):
        ECGLayoutConfig(layout_format="unsupported_format")

    with pytest.raises(ValueError, match="Invalid rhythm lead name"):
        ECGLayoutConfig(rhythm_lead_name="LeadX")


def test_layout_mm_to_px_conversions():
    """Verify physical millimeter to pixel transformations at various DPIs."""
    for dpi in [100, 150, 200, 300]:
        cfg = ECGLayoutConfig(dpi=dpi, paper_width_mm=254.0, paper_height_mm=254.0)
        layout = ECGLayout(cfg)
        # 254 mm is exactly 10 inches -> 10 * DPI pixels
        assert layout.canvas_width_px == 10 * dpi
        assert layout.canvas_height_px == 10 * dpi
        assert pytest.approx(layout.mm_to_px(25.4), rel=1e-3) == dpi
        assert pytest.approx(layout.px_to_mm(dpi), rel=1e-3) == 25.4


def test_grid_presets():
    """Verify pre-defined grid styling presets."""
    for preset in ["standard_pink", "classic_red", "subtle_gray", "clinical_green", "no_grid"]:
        g = GridConfig.from_preset(preset)
        assert isinstance(g, GridConfig)
        if preset == "no_grid":
            assert g.show_grid is False
        else:
            assert g.show_grid is True

    with pytest.raises(ValueError, match="Unknown grid preset"):
        GridConfig.from_preset("neon_yellow")


# ---------------------------------------------------------------------------
# 2. 3x4 + 1 Rhythm Strip Layout Segmentation Tests
# ---------------------------------------------------------------------------


def test_standard_3x4_1r_lead_segments():
    """Verify that standard 3x4 + 1 rhythm strip layout produces exact lead mapping."""
    cfg = ECGLayoutConfig(layout_format="3x4_1r", rhythm_lead_name="II")
    layout = ECGLayout(cfg)
    segments = layout.get_lead_segments()

    # 3 rows of 4 leads = 12 segments + 1 continuous rhythm strip = 13 segments
    assert len(segments) == 13

    # Check 3x4 lead ordering:
    # Row 0: I,   aVR, V1, V4
    # Row 1: II,  aVL, V2, V5
    # Row 2: III, aVF, V3, V6
    expected_matrix = [
        ["I",   "aVR", "V1", "V4"],
        ["II",  "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]

    for row_idx, row_leads in enumerate(expected_matrix):
        for col_idx, expected_lead in enumerate(row_leads):
            # Find matching segment
            matching = [s for s in segments if s.row == row_idx and s.col == col_idx]
            assert len(matching) == 1
            seg = matching[0]
            assert seg.lead_name == expected_lead
            assert seg.start_time_sec == pytest.approx(col_idx * 2.5)
            assert seg.duration_sec == pytest.approx(2.5)
            assert seg.end_time_sec == pytest.approx((col_idx + 1) * 2.5)
            assert seg.width_mm == pytest.approx(2.5 * 25.0)  # 62.5 mm

    # Check Row 3: Rhythm strip
    rhythm_segs = [s for s in segments if s.row == 3]
    assert len(rhythm_segs) == 1
    r_seg = rhythm_segs[0]
    assert r_seg.lead_name == "II"
    assert r_seg.start_time_sec == 0.0
    assert r_seg.duration_sec == 10.0
    assert r_seg.width_mm == pytest.approx(250.0)  # 10s * 25 mm/s


def test_alternative_layout_formats():
    """Verify alternative layout configurations (3x4, 12x1, 6x2)."""
    # 3x4 without rhythm strip: 12 segments
    cfg_3x4 = ECGLayoutConfig(layout_format="3x4")
    segs_3x4 = ECGLayout(cfg_3x4).get_lead_segments()
    assert len(segs_3x4) == 12

    # 12x1: 12 rows, each 10 seconds
    cfg_12x1 = ECGLayoutConfig(layout_format="12x1")
    segs_12x1 = ECGLayout(cfg_12x1).get_lead_segments()
    assert len(segs_12x1) == 12
    for s in segs_12x1:
        assert s.duration_sec == 10.0

    # 6x2: 6 rows, 2 columns (5 seconds each)
    cfg_6x2 = ECGLayoutConfig(layout_format="6x2")
    segs_6x2 = ECGLayout(cfg_6x2).get_lead_segments()
    assert len(segs_6x2) == 12
    for s in segs_6x2:
        assert s.duration_sec == 5.0


# ---------------------------------------------------------------------------
# 3. Grid & Calibration Pulse Geometry Tests
# ---------------------------------------------------------------------------


def test_grid_generation_coordinates():
    """Verify minor (1 mm) and major (5 mm) grid tick calculation."""
    cfg = ECGLayoutConfig(paper_width_mm=100.0, paper_height_mm=50.0, dpi=254)
    # At 254 DPI: 1 mm = 10 px exactly (25.4 mm = 254 px)
    layout = ECGLayout(cfg)
    x_min, y_min, x_maj, y_maj = layout.get_grid_lines_px()

    # 100 mm width has 101 ticks at 1 mm
    assert len(x_min) == 101
    assert len(y_min) == 51

    # 5 mm major grid lines
    assert len(x_maj) == 21
    assert len(y_maj) == 11

    # Spacing between ticks
    np.testing.assert_allclose(np.diff(x_min), 10.0, atol=1e-3)
    np.testing.assert_allclose(np.diff(x_maj), 50.0, atol=1e-3)


def test_calibration_pulse_geometry():
    """Verify standard 1.0 mV (10 mm) x 0.20 s (5 mm) calibration pulse."""
    cfg = ECGLayoutConfig(dpi=254)
    layout = ECGLayout(cfg)

    # At 254 DPI: 1 mm = 10 px
    pts = layout.get_calibration_pulse_points_px(row_idx=0)
    assert pts is not None
    assert pts.shape == (6, 2)

    # Height: 10 mm = 100 px step upward (negative y in image space)
    base_y = pts[0, 1]
    peak_y = pts[2, 1]
    step_height_px = base_y - peak_y
    assert pytest.approx(step_height_px, abs=1.0) == 100.0

    # Plateau width: 0.20 s * 25 mm/s = 5 mm = 50 px
    plateau_width_px = pts[3, 0] - pts[2, 0]
    assert pytest.approx(plateau_width_px, abs=1.0) == 50.0


# ---------------------------------------------------------------------------
# 4. ECG Paper Waveform Rendering Tests
# ---------------------------------------------------------------------------


def test_renderer_output_shapes_and_types(synthetic_ecg):
    """Verify that renderer produces valid PIL images, NumPy arrays, and PyTorch tensors."""
    cfg = ECGLayoutConfig(dpi=150, paper_width_mm=280.0, paper_height_mm=190.0)
    renderer = ECGPaperRenderer(config=cfg)

    ecg_img = renderer.render(synthetic_ecg)
    assert isinstance(ecg_img, ECGImage)

    # Dimensions
    expected_w = int(round(280.0 / 25.4 * 150))
    expected_h = int(round(190.0 / 25.4 * 150))
    assert ecg_img.width == expected_w
    assert ecg_img.height == expected_h
    assert ecg_img.size == (expected_w, expected_h)
    assert isinstance(ecg_img.pil_image, Image.Image)

    # NumPy array export: (H, W, 3) uint8
    arr = ecg_img.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (expected_h, expected_w, 3)
    assert arr.dtype == np.uint8

    # PyTorch tensor export: (3, H, W) float32 in [0, 1]
    t_ch_first = ecg_img.to_tensor(channels_first=True, normalize_0_1=True)
    assert isinstance(t_ch_first, torch.Tensor)
    assert t_ch_first.shape == (3, expected_h, expected_w)
    assert t_ch_first.dtype == torch.float32
    assert 0.0 <= float(t_ch_first.min()) <= float(t_ch_first.max()) <= 1.0

    # PyTorch tensor export without normalization: uint8
    t_raw = ecg_img.to_tensor(channels_first=False, normalize_0_1=False)
    assert t_raw.shape == (expected_h, expected_w, 3)
    assert t_raw.dtype == torch.uint8


def test_renderer_metadata_propagation(synthetic_ecg):
    """Verify that ECGRecord and metadata dictionaries propagate through rendering."""
    record = ECGRecord(
        record_id="ECHONEXT_train_000042",
        patient_key="P00042",
        split="train",
        waveform_ref=("train", 42),
        lvef_target=38.5,
        lvef_lte_45_label=1,
        shd_label=1,
    )
    renderer = ECGPaperRenderer()
    ecg_img = renderer.render(synthetic_ecg, record=record)

    meta = ecg_img.metadata
    assert meta["record_id"] == "ECHONEXT_train_000042"
    assert meta["patient_key"] == "P00042"
    assert meta["split"] == "train"
    assert meta["lvef_value"] == 38.5
    assert meta["lvef_lte_45_flag"] == 1
    assert meta["image_width"] == ecg_img.width
    assert meta["image_height"] == ecg_img.height
    assert meta["dpi"] == 150


def test_renderer_supersampling(synthetic_ecg):
    """Verify that 2x supersampling anti-aliasing executes and preserves dimensions."""
    cfg = ECGLayoutConfig(dpi=100, paper_width_mm=200.0, paper_height_mm=140.0)
    renderer = ECGPaperRenderer(config=cfg, supersample=2)
    ecg_img = renderer.render(synthetic_ecg)

    expected_w = int(round(200.0 / 25.4 * 100))
    expected_h = int(round(140.0 / 25.4 * 100))
    assert ecg_img.size == (expected_w, expected_h)


def test_renderer_input_orientation_and_safety(synthetic_ecg):
    """Verify automatic handling of transposed arrays, torch tensors, and NaNs."""
    renderer = ECGPaperRenderer()

    # Transposed input: (12, 2500)
    transposed = synthetic_ecg.T
    img1 = renderer.render(transposed)
    assert img1.size == (renderer.layout.canvas_width_px, renderer.layout.canvas_height_px)

    # PyTorch tensor input
    tensor_input = torch.from_numpy(synthetic_ecg)
    img2 = renderer.render(tensor_input)
    assert img2.size == img1.size

    # Input containing NaNs/Infs: should not crash, replaced safely with 0.0
    corrupted = synthetic_ecg.copy()
    corrupted[100:150, :] = np.nan
    corrupted[500:520, :] = np.inf
    img3 = renderer.render(corrupted)
    assert img3.size == img1.size

    # Invalid input dimensions
    with pytest.raises(ValueError, match="Waveform must be 2D"):
        renderer.render(np.zeros((10, 10, 10), dtype=np.float32))

    with pytest.raises(ValueError, match="invalid lead dimension"):
        renderer.render(np.zeros((2500, 8), dtype=np.float32))


def test_renderer_save_and_reload(synthetic_ecg, tmp_path: Path):
    """Verify file persistence, directory auto-creation, and round-trip reloading."""
    renderer = ECGPaperRenderer()
    ecg_img = renderer.render(synthetic_ecg)

    save_path = tmp_path / "deep" / "nested" / "test_ecg.png"
    result_path = ecg_img.save(save_path)
    assert result_path.exists()

    # Re-open with PIL
    with Image.open(result_path) as loaded:
        assert loaded.size == ecg_img.size
        assert loaded.mode == "RGB"


# ---------------------------------------------------------------------------
# 5. PyTorch Dataset & DataLoader Integration Tests
# ---------------------------------------------------------------------------


def test_ventriq_torch_dataset_with_image_renderer_transform(temp_dataset_dir: Path):
    """Verify that ECGPaperRenderer integrates as a PyTorch Dataset transform."""
    renderer = ECGPaperRenderer(config=ECGLayoutConfig(dpi=100))
    dataset = VentriQTorchDataset(root=temp_dataset_dir, transform=renderer)

    assert len(dataset) == 16  # 8 train + 4 val + 4 test
    sample = dataset[0]

    waveform_tensor = sample["waveform"]
    assert isinstance(waveform_tensor, torch.Tensor)
    # Tensor shape is (3, H, W)
    assert waveform_tensor.ndim == 3
    assert waveform_tensor.shape[0] == 3
    assert waveform_tensor.dtype == torch.float32
    assert 0.0 <= float(waveform_tensor.min()) <= float(waveform_tensor.max()) <= 1.0


def test_dataloader_batch_collation_with_renderer(temp_dataset_dir: Path):
    """Verify that PyTorch DataLoader batches rendered image tensors into (B, 3, H, W)."""
    renderer = ECGPaperRenderer(config=ECGLayoutConfig(dpi=100))
    dataset = VentriQTorchDataset(root=temp_dataset_dir, split="train", transform=renderer)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    waveforms = batch["waveform"]
    # Shape must be (B, 3, H, W)
    assert waveforms.ndim == 4
    assert waveforms.shape[0] == 4
    assert waveforms.shape[1] == 3
    assert waveforms.dtype == torch.float32

    # LVEF targets
    assert batch["lvef"].shape == (4,)
    assert batch["lvef_lte_45"].shape == (4,)


# ---------------------------------------------------------------------------
# 6. YAML Configuration Loading & Fallback Tests
# ---------------------------------------------------------------------------


def test_load_rendering_config_from_yaml(tmp_path: Path):
    """Verify that load_rendering_config parses custom and default YAML configs."""
    custom_yaml = tmp_path / "custom_rendering.yaml"
    custom_yaml.write_text("""
paper:
  dpi: 200
  paper_speed_mm_s: 50.0
  voltage_gain_mm_mv: 20.0
  width_mm: 300.0
  height_mm: 200.0

layout:
  format: "3x4_1r"
  rhythm_lead: "V1"
  row_height_mm: 42.0

signal:
  input_units: "mv"
  amplitude_scale_mm_per_unit: 10.0

grid:
  style: "classic_red"
  minor_color: "#ffaaaa"
  major_color: "#ee5555"

calibration:
  show_pulse: false

typography:
  show_lead_labels: true
  label_font_size_pt: 11
""")

    cfg = load_rendering_config(custom_yaml)
    assert cfg.dpi == 200
    assert cfg.paper_speed_mm_s == 50.0
    assert cfg.voltage_gain_mm_mv == 20.0
    assert cfg.paper_width_mm == 300.0
    assert cfg.paper_height_mm == 200.0
    assert cfg.rhythm_lead_name == "V1"
    assert cfg.row_height_mm == 42.0
    assert cfg.input_units == "mv"
    assert cfg.calibration.show_pulse is False
    assert cfg.label_font_size_pt == 11

    # Fallback on non-existent path
    fallback_cfg = load_rendering_config(tmp_path / "non_existent.yaml")
    assert isinstance(fallback_cfg, ECGLayoutConfig)
