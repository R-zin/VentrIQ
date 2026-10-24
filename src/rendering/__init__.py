"""src/rendering - VentriQ Waveform-to-Image Paper Rendering Module (Phase 4).

This package provides high-fidelity, clinical-grade conversion of digital
12-lead ECG waveforms into photorealistic ECG paper recordings.
"""

from __future__ import annotations

from src.rendering.augmentation import (
    DistortionConfig,
    SimulatedECGPhoto,
    SmartphoneECGSimulator,
    compute_3d_perspective_homography,
    load_augmentation_config,
)
from src.rendering.layout import (
    DEFAULT_MAJOR_GRID_MM,
    DEFAULT_MINOR_GRID_MM,
    DEFAULT_PAPER_SPEED_MM_PER_SEC,
    DEFAULT_VOLTAGE_GAIN_MM_PER_MV,
    LEAD_NAME_TO_INDEX,
    CalibrationPulseConfig,
    ECGLayout,
    ECGLayoutConfig,
    GridConfig,
    LeadSegment,
    load_rendering_config,
)
from src.rendering.renderer import (
    ECGImage,
    ECGPaperRenderer,
)

__all__ = [
    "CalibrationPulseConfig",
    "DistortionConfig",
    "ECGImage",
    "ECGLayout",
    "ECGLayoutConfig",
    "ECGPaperRenderer",
    "GridConfig",
    "LeadSegment",
    "SimulatedECGPhoto",
    "SmartphoneECGSimulator",
    "compute_3d_perspective_homography",
    "load_augmentation_config",
    "DEFAULT_MAJOR_GRID_MM",
    "DEFAULT_MINOR_GRID_MM",
    "DEFAULT_PAPER_SPEED_MM_PER_SEC",
    "DEFAULT_VOLTAGE_GAIN_MM_PER_MV",
    "LEAD_NAME_TO_INDEX",
    "load_rendering_config",
]
