"""src/vision - EchoNext waveform-to-image conversion and MobileNetV3 SHD classification.

Pipeline:
    EchoNext .npy waveforms
        -> ECGImageConverter / convert_echonext_to_images  (images + image_manifest.csv)
        -> ECGImageDataset + build_mobilenet_v3 + train_classifier  (best.pt checkpoint)
        -> SHDPredictor  (probability of SHD for a user's ECG photo or scan)
"""

from __future__ import annotations

from src.vision.canonical import (
    CanonicalizationConfig,
    ECGImageCanonicalizer,
    canonical_size_for_dpi,
)
from src.vision.conversion import (
    ECGImageConverter,
    ImageConversionConfig,
    RenderStyle,
    convert_echonext_to_images,
    load_conversion_config,
    sample_render_style,
    select_records,
)
from src.vision.dataset import DEFAULT_IMAGE_SIZE, ECGImageDataset, build_image_transform
from src.vision.inference import SHDPrediction, SHDPredictor
from src.vision.model import build_mobilenet_v3
from src.vision.training import (
    TrainingConfig,
    average_precision,
    binary_metrics,
    roc_auc,
    train_classifier,
    youden_threshold,
)

__all__ = [
    "CanonicalizationConfig",
    "DEFAULT_IMAGE_SIZE",
    "ECGImageCanonicalizer",
    "ECGImageConverter",
    "ECGImageDataset",
    "ImageConversionConfig",
    "RenderStyle",
    "SHDPrediction",
    "SHDPredictor",
    "TrainingConfig",
    "average_precision",
    "binary_metrics",
    "build_image_transform",
    "build_mobilenet_v3",
    "canonical_size_for_dpi",
    "convert_echonext_to_images",
    "load_conversion_config",
    "roc_auc",
    "sample_render_style",
    "select_records",
    "train_classifier",
    "youden_threshold",
]
