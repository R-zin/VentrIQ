"""src/vision/canonical.py - Shared canonical image view for training and inference.

A CNN only generalizes to user-supplied ECG photographs if the images it sees at
inference time pass through exactly the same preprocessing as the images it was
trained on. ``ECGImageCanonicalizer`` is that single shared step:

    any ECG image (clean render, scan, or smartphone photo)
        -> optional document detection + perspective rectification
        -> resize to the canonical paper canvas (280 mm x 190 mm aspect ratio)
        -> optional illumination normalization + CLAHE contrast enhancement

The EchoNext-to-image converter (``src/vision/conversion.py``) applies it offline
before images are written to disk, and ``SHDPredictor`` (``src/vision/inference.py``)
applies it to every image a user submits, using the settings stored in the model
checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Tuple, Union

import numpy as np
from PIL import Image

from src.preprocessing.image import ECGImageEnhancer
from src.preprocessing.perspective import detect_document_corners, rectify_document

RectifyMode = Literal["auto", "always", "never"]

# Canonical ECG paper geometry used by the Phase 4 renderer (landscape 280 mm x 190 mm).
PAPER_WIDTH_MM: float = 280.0
PAPER_HEIGHT_MM: float = 190.0
MM_PER_INCH: float = 25.4


def canonical_size_for_dpi(dpi: int) -> Tuple[int, int]:
    """Return the (width, height) pixel size of the canonical paper canvas at ``dpi``."""
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    width = int(round(PAPER_WIDTH_MM / MM_PER_INCH * dpi))
    height = int(round(PAPER_HEIGHT_MM / MM_PER_INCH * dpi))
    return width, height


@dataclass
class CanonicalizationConfig:
    """Settings for the canonical image view (persisted in manifests and checkpoints)."""

    canonical_size: Tuple[int, int] = (800, 543)  # 280 x 190 mm paper aspect ratio
    rectify: RectifyMode = "auto"
    enhance: bool = True
    illumination_sigma: float = 15.0  # tuned for ~800 px wide canvases
    clahe_clip_limit: float = 2.5

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["canonical_size"] = list(self.canonical_size)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalizationConfig":
        d = dict(data)
        if "canonical_size" in d:
            w, h = d["canonical_size"]
            d["canonical_size"] = (int(w), int(h))
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class ECGImageCanonicalizer:
    """Bring an arbitrary ECG image into the canonical view the CNN is trained on."""

    def __init__(self, config: CanonicalizationConfig | None = None) -> None:
        self.config = config or CanonicalizationConfig()
        if self.config.rectify not in ("auto", "always", "never"):
            raise ValueError(f"Unknown rectify mode: {self.config.rectify}")
        self.enhancer = (
            ECGImageEnhancer(
                illumination_sigma=self.config.illumination_sigma,
                clahe_clip_limit=self.config.clahe_clip_limit,
            )
            if self.config.enhance
            else None
        )

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        rectify: RectifyMode | None = None,
    ) -> Image.Image:
        """Return the canonical RGB view of ``image``.

        Args:
            image: PIL image or uint8 array (H, W[, C]).
            rectify: Per-call override of the configured rectify mode.
                ``"auto"`` rectifies only when a paper boundary is confidently detected
                (detection failures fall back to using the full frame instead of
                cropping with a guessed margin); ``"always"`` uses the detector's result
                even when it falls back; ``"never"`` skips rectification (use for
                images that are already flat, full-frame ECG pages).
        """
        img = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        if img.mode != "RGB":
            img = img.convert("RGB")

        mode = rectify or self.config.rectify
        if mode != "never":
            doc = detect_document_corners(img)
            if mode == "always" or not doc.is_fallback:
                img = rectify_document(
                    img, corners=doc, target_size=self.config.canonical_size
                ).image

        size = (int(self.config.canonical_size[0]), int(self.config.canonical_size[1]))
        if img.size != size:
            img = img.resize(size, Image.Resampling.LANCZOS)

        if self.enhancer is not None:
            img = self.enhancer(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
        return img
