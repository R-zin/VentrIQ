"""src/vision/conversion.py - EchoNext NumPy waveforms -> labelled ECG images for CNN training.

This module turns the memory-mapped EchoNext waveform arrays (``[N, 1, 2500, 12]``
z-scored 12-lead ECGs at 250 Hz) into an image classification dataset that a
MobileNetV3 can be trained on to predict structural heart disease (SHD) from a
picture of a paper ECG.

For every record (and every requested variant of it) the converter:

1. Slices the waveform from the per-split ``.npy`` memory map.
2. Samples a *render style* (grid color preset, trace thickness, amplitude scale) so
   the model does not overfit to one ECG cart's printout look.
3. Renders the standard 3x4 + lead II rhythm strip paper layout (Phase 4 renderer).
   No record/patient identifiers are printed on the page, so the CNN cannot learn
   shortcuts from header text, and no PHI ends up in the images.
4. Optionally simulates a smartphone photograph of the printout (Phase 5 simulator:
   perspective tilt, page curl, shadows, glare, blur, noise, JPEG).
5. Applies the shared ``ECGImageCanonicalizer`` (rectification + resize + contrast
   enhancement), i.e. exactly what ``SHDPredictor`` will apply to user images.
6. Writes ``{output_dir}/{split}/{record_id}_v{variant}.{ext}`` and one row in
   ``{output_dir}/image_manifest.csv`` linking the image to its patient, official
   split and labels (``shd_flag``, ``lvef_value``, ``lvef_lte_45_flag``).

Every random choice is derived from ``(seed, record_id, variant)``, so conversions are
reproducible, independent of worker count, and resumable (existing files are skipped).
Official EchoNext splits are preserved, so there is no patient leakage between the
train/val/test image sets.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from src.data.dataset import DatasetManifest, ECGRecord
from src.data.echonext import EXPECTED_TIME_STEPS, NUM_LEADS
from src.rendering.augmentation import DistortionConfig, SmartphoneECGSimulator
from src.rendering.layout import ECGLayoutConfig, GridConfig
from src.rendering.renderer import ECGPaperRenderer
from src.vision.canonical import (
    CanonicalizationConfig,
    ECGImageCanonicalizer,
    canonical_size_for_dpi,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "image_manifest.csv"
CONFIG_FILENAME = "conversion_config.json"

GRID_STYLES: Tuple[str, ...] = (
    "standard_pink",
    "classic_red",
    "subtle_gray",
    "clinical_green",
    "no_grid",
)
PHOTO_PRESETS: Tuple[str, ...] = ("none", "clean", "mild", "moderate", "severe")

LABEL_COLUMNS: Dict[str, str] = {
    "shd": "shd_flag",
    "lvef_lte_45": "lvef_lte_45_flag",
}


@dataclass
class ImageConversionConfig:
    """Configuration of the EchoNext waveform-to-image conversion."""

    # Paper is rendered at ``dpi`` (100 DPI -> 1102 x 748 px), optionally photographed,
    # then canonicalized and stored at ``canonical_size`` (~170 KB per JPEG at q90).
    dpi: int = 100
    canonical_size: Tuple[int, int] = (800, 543)
    image_format: str = "jpg"  # 'png' or 'jpg'
    jpeg_quality: int = 90
    variants_per_record: int = 1
    seed: int = 42

    # Render-style domain randomization (sampled independently per image)
    grid_style_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "standard_pink": 0.45,
            "classic_red": 0.25,
            "subtle_gray": 0.15,
            "clinical_green": 0.05,
            "no_grid": 0.10,
        }
    )
    line_width_range: Tuple[float, float] = (1.0, 2.0)
    # EchoNext waveforms are z-scored, not millivolts. Jittering the mm-per-unit scale
    # makes the model robust to the (unknown) gain of real printouts.
    amplitude_scale_range: Tuple[float, float] = (4.5, 8.0)

    # Smartphone-photo simulation mix: preset name -> probability
    photo_preset_weights: Dict[str, float] = field(
        default_factory=lambda: {"none": 0.5, "mild": 0.3, "moderate": 0.2}
    )

    # Canonical (model-facing) view; also used verbatim at inference time
    enhance: bool = True

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        self.canonical_size = (int(self.canonical_size[0]), int(self.canonical_size[1]))
        if min(self.canonical_size) < 32:
            raise ValueError("canonical_size must be at least 32 x 32 px")
        if self.image_format.lower() not in ("png", "jpg", "jpeg"):
            raise ValueError(f"Unsupported image_format: {self.image_format}")
        if self.variants_per_record < 1:
            raise ValueError("variants_per_record must be >= 1")
        for name in self.grid_style_weights:
            if name not in GRID_STYLES:
                raise ValueError(f"Unknown grid style '{name}'. Available: {GRID_STYLES}")
        for name in self.photo_preset_weights:
            if name not in PHOTO_PRESETS:
                raise ValueError(f"Unknown photo preset '{name}'. Available: {PHOTO_PRESETS}")
        for weights in (self.grid_style_weights, self.photo_preset_weights):
            if not weights or sum(weights.values()) <= 0 or min(weights.values()) < 0:
                raise ValueError("Sampling weights must be non-negative and sum to > 0")
        for lo, hi in (self.line_width_range, self.amplitude_scale_range):
            if lo <= 0 or hi < lo:
                raise ValueError("Ranges must satisfy 0 < low <= high")

    @property
    def extension(self) -> str:
        return "jpg" if self.image_format.lower() in ("jpg", "jpeg") else "png"

    @property
    def canonicalization(self) -> CanonicalizationConfig:
        # Illumination-estimation scale follows the canvas width (30 px @ 1654 px wide).
        sigma = 30.0 * self.canonical_size[0] / canonical_size_for_dpi(150)[0]
        return CanonicalizationConfig(
            canonical_size=self.canonical_size,
            rectify="auto",
            enhance=self.enhance,
            illumination_sigma=round(sigma, 2),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["canonical_size"] = list(self.canonical_size)
        d["line_width_range"] = list(self.line_width_range)
        d["amplitude_scale_range"] = list(self.amplitude_scale_range)
        d["canonicalization"] = self.canonicalization.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageConversionConfig":
        d = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for key in ("canonical_size", "line_width_range", "amplitude_scale_range"):
            if key in d:
                d[key] = tuple(float(x) for x in d[key])
        return cls(**d)


@dataclass(frozen=True)
class RenderStyle:
    """The randomized appearance parameters chosen for a single image."""

    grid_style: str
    line_width_px: float
    amplitude_scale_mm_per_unit: float
    photo_preset: str
    photo_seed: int


def _record_seed(seed: int, record_id: str, variant: int) -> np.random.SeedSequence:
    """Deterministic, worker-independent seed for one (record, variant)."""
    return np.random.SeedSequence([int(seed), zlib.crc32(record_id.encode("utf-8")), int(variant)])


def _weighted_choice(rng: np.random.Generator, weights: Dict[str, float]) -> str:
    names = list(weights.keys())
    p = np.asarray([weights[n] for n in names], dtype=np.float64)
    return str(names[int(rng.choice(len(names), p=p / p.sum()))])


def sample_render_style(config: ImageConversionConfig, record_id: str, variant: int) -> RenderStyle:
    """Sample the appearance of one image deterministically from ``(seed, record_id, variant)``."""
    rng = np.random.default_rng(_record_seed(config.seed, record_id, variant))
    return RenderStyle(
        grid_style=_weighted_choice(rng, config.grid_style_weights),
        line_width_px=float(rng.uniform(*config.line_width_range)),
        amplitude_scale_mm_per_unit=float(rng.uniform(*config.amplitude_scale_range)),
        photo_preset=_weighted_choice(rng, config.photo_preset_weights),
        photo_seed=int(rng.integers(0, 2**31 - 1)),
    )


def _base_layout_config(dpi: int) -> ECGLayoutConfig:
    # Header shows only a neutral title (never record or patient identifiers).
    return ECGLayoutConfig(dpi=dpi, header_text="12-Lead ECG")


class ECGImageConverter:
    """Convert individual 12-lead waveforms into canonical, model-ready ECG images."""

    def __init__(self, config: Optional[ImageConversionConfig] = None) -> None:
        self.config = config or ImageConversionConfig()
        self.base_layout = _base_layout_config(self.config.dpi)
        self.canonicalizer = ECGImageCanonicalizer(self.config.canonicalization)
        # One renderer per grid style so the background grid cache stays valid.
        self._renderers: Dict[str, ECGPaperRenderer] = {}

    def _renderer_for(self, style: RenderStyle) -> ECGPaperRenderer:
        renderer = self._renderers.get(style.grid_style)
        if renderer is None:
            layout = replace(self.base_layout, grid=GridConfig.from_preset(style.grid_style))
            renderer = ECGPaperRenderer(config=layout, supersample=1, enable_grid_cache=True)
            self._renderers[style.grid_style] = renderer
        # Trace width and amplitude are read at draw time and do not affect the cached
        # grid or the lead geometry, so they can be swapped per image.
        renderer.config = replace(
            renderer.config,
            waveform_line_width_px=style.line_width_px,
            amplitude_scale_mm_per_unit=style.amplitude_scale_mm_per_unit,
        )
        return renderer

    def convert(
        self,
        waveform: np.ndarray,
        style: RenderStyle,
    ) -> Image.Image:
        """Render ``waveform`` (2500x12 or 12x2500) with ``style`` and return the canonical image."""
        paper = self._renderer_for(style).render(waveform).pil_image

        if style.photo_preset == "none":
            # Already a flat, full-frame page: no rectification needed.
            return self.canonicalizer(paper, rectify="never")

        simulator = SmartphoneECGSimulator(
            DistortionConfig.from_preset(style.photo_preset), seed=style.photo_seed
        )
        photo = simulator.simulate(paper).image
        # Same detection + rectification path that user photos go through at inference.
        return self.canonicalizer(photo, rectify="auto")

    def convert_record(
        self, waveform: np.ndarray, record_id: str, variant: int = 0
    ) -> Tuple[Image.Image, RenderStyle]:
        style = sample_render_style(self.config, record_id, variant)
        return self.convert(waveform, style), style


# ---------------------------------------------------------------------------
# Waveform access (memory-mapped, per process)
# ---------------------------------------------------------------------------


class _WaveformSource:
    """Minimal per-process cache of EchoNext per-split memory maps."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self._mmaps: Dict[str, np.ndarray] = {}

    def _locate(self, split: str) -> Path:
        for name in (
            f"EchoNext_{split}_waveforms.npy",
            f"echonext_{split}_waveforms.npy",
            f"{split}_waveforms.npy",
        ):
            p = self.root / name
            if p.exists():
                return p
        raise FileNotFoundError(f"No waveform .npy file for split '{split}' in {self.root}")

    def get(self, split: str, local_index: int) -> np.ndarray:
        mmap = self._mmaps.get(split)
        if mmap is None:
            mmap = np.load(self._locate(split), mmap_mode="r")
            self._mmaps[split] = mmap
        wf = np.squeeze(np.asarray(mmap[int(local_index)], dtype=np.float32))
        if wf.shape == (NUM_LEADS, EXPECTED_TIME_STEPS):
            wf = wf.T
        if wf.shape != (EXPECTED_TIME_STEPS, NUM_LEADS):
            raise ValueError(f"Unexpected waveform shape {wf.shape} for {split}[{local_index}]")
        return wf


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

_WORKER: Dict[str, Any] = {}


def _init_worker(data_dir: str, config_dict: Dict[str, Any], output_dir: str) -> None:
    _WORKER["source"] = _WaveformSource(data_dir)
    _WORKER["converter"] = ECGImageConverter(ImageConversionConfig.from_dict(config_dict))
    _WORKER["output_dir"] = Path(output_dir)


def _relative_image_path(record: ECGRecord, variant: int, ext: str) -> str:
    return f"{record.split}/{record.record_id}_v{variant}.{ext}"


def _process_task(task: Tuple[Dict[str, Any], int, bool]) -> Dict[str, Any]:
    record_dict, variant, overwrite = task
    record = ECGRecord.from_dict(record_dict)
    converter: ECGImageConverter = _WORKER["converter"]
    cfg = converter.config
    rel_path = _relative_image_path(record, variant, cfg.extension)
    out_path: Path = _WORKER["output_dir"] / rel_path
    style = sample_render_style(cfg, record.record_id, variant)

    status = "skipped"
    if overwrite or not out_path.exists():
        split_key, local_idx = record.waveform_ref
        waveform = _WORKER["source"].get(split_key, local_idx)
        image = converter.convert(waveform, style)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.stem + ".tmp" + out_path.suffix)
        if cfg.extension == "jpg":
            image.save(tmp_path, format="JPEG", quality=cfg.jpeg_quality)
        else:
            image.save(tmp_path, format="PNG", optimize=False)
        tmp_path.replace(out_path)  # atomic: no half-written images after interruption
        status = "written"

    width, height = cfg.canonicalization.canonical_size
    return {
        "image_path": rel_path,
        "record_id": record.record_id,
        "patient_key": record.patient_key,
        "split": record.split,
        "variant": variant,
        "shd_flag": record.shd_label,
        "lvef_value": record.lvef_target,
        "lvef_lte_45_flag": record.lvef_lte_45_label,
        "grid_style": style.grid_style,
        "line_width_px": round(style.line_width_px, 4),
        "amplitude_scale_mm_per_unit": round(style.amplitude_scale_mm_per_unit, 4),
        "photo_preset": style.photo_preset,
        "image_width": width,
        "image_height": height,
        "_status": status,
    }


def select_records(
    records: Sequence[ECGRecord],
    splits: Iterable[str] = ("train", "val", "test"),
    label: Optional[str] = "shd",
    max_per_split: Optional[int] = None,
    seed: int = 42,
) -> List[ECGRecord]:
    """Filter records to the requested splits / labelled records, optionally subsampling."""
    if label is not None and label not in LABEL_COLUMNS:
        raise ValueError(f"Unknown label '{label}'. Available: {list(LABEL_COLUMNS)}")
    selected: List[ECGRecord] = []
    rng = np.random.default_rng(seed)
    for split in splits:
        recs = [r for r in records if r.split == split]
        if label == "shd":
            recs = [r for r in recs if r.shd_label is not None]
        elif label == "lvef_lte_45":
            recs = [r for r in recs if r.lvef_lte_45_label is not None]
        if max_per_split is not None and 0 < max_per_split < len(recs):
            idx = np.sort(rng.choice(len(recs), size=max_per_split, replace=False))
            recs = [recs[i] for i in idx]
        selected.extend(recs)
    return selected


def convert_echonext_to_images(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[ImageConversionConfig] = None,
    records: Optional[Sequence[ECGRecord]] = None,
    splits: Iterable[str] = ("train", "val", "test"),
    label: Optional[str] = "shd",
    max_per_split: Optional[int] = None,
    num_workers: int = 0,
    overwrite: bool = False,
    progress_every: int = 500,
) -> pd.DataFrame:
    """Convert EchoNext waveforms to an image dataset + manifest for CNN training.

    Args:
        data_dir: EchoNext directory (metadata CSV + ``EchoNext_{split}_waveforms.npy``).
        output_dir: Destination root; images go to ``{output_dir}/{split}/``.
        config: Conversion settings (render randomization, photo mix, resolution...).
        records: Pre-built ``ECGRecord`` list (e.g. from a Phase 3 manifest). If None,
            records are built directly from ``data_dir``.
        splits: Official EchoNext splits to convert.
        label: Only keep records with this label present ('shd', 'lvef_lte_45' or None).
        max_per_split: Optional random subsample size per split (quick experiments).
        num_workers: Number of worker processes (0 = convert in this process).
        overwrite: Re-render images that already exist (otherwise resume/skip them).
        progress_every: Log progress every N images.

    Returns:
        The image manifest as a DataFrame (also written to ``image_manifest.csv``).
    """
    cfg = config or ImageConversionConfig()
    data_dir = Path(data_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if records is None:
        records = DatasetManifest.build_from_echonext(data_dir, seed=cfg.seed).records
    selected = select_records(records, splits, label, max_per_split, cfg.seed)
    tasks = [(r.to_dict(), v, overwrite) for r in selected for v in range(cfg.variants_per_record)]
    logger.info(
        "Converting %d records x %d variant(s) = %d images into %s (workers=%d)",
        len(selected),
        cfg.variants_per_record,
        len(tasks),
        out_root,
        num_workers,
    )

    init_args = (str(data_dir), cfg.to_dict(), str(out_root))
    rows: List[Dict[str, Any]] = []

    def _collect(results: Iterable[Dict[str, Any]]) -> None:
        for i, row in enumerate(results, start=1):
            rows.append(row)
            if progress_every and i % progress_every == 0:
                logger.info("  %d / %d images done", i, len(tasks))

    if num_workers and num_workers > 0:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=init_args,
        ) as pool:
            chunksize = max(1, min(64, len(tasks) // (num_workers * 8) or 1))
            _collect(pool.map(_process_task, tasks, chunksize=chunksize))
    else:
        _init_worker(*init_args)
        try:
            _collect(_process_task(t) for t in tasks)
        finally:
            _WORKER.clear()

    manifest = pd.DataFrame(rows)
    written = int((manifest["_status"] == "written").sum()) if len(manifest) else 0
    if len(manifest):
        manifest = manifest.drop(columns=["_status"])
    manifest.to_csv(out_root / MANIFEST_FILENAME, index=False)
    with open(out_root / CONFIG_FILENAME, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    logger.info(
        "Conversion finished: %d images in manifest (%d newly written, %d reused).",
        len(manifest),
        written,
        len(manifest) - written,
    )
    return manifest


def load_conversion_config(image_dir: Union[str, Path]) -> ImageConversionConfig:
    """Load the ``conversion_config.json`` written next to an image manifest."""
    path = Path(image_dir) / CONFIG_FILENAME
    with open(path, "r", encoding="utf-8") as f:
        return ImageConversionConfig.from_dict(json.load(f))
