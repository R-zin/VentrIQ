"""src/rendering/renderer.py - Photorealistic ECG Paper Waveform Renderer.

This module converts digital 12-lead ECG waveforms (numpy arrays or PyTorch tensors)
into clinical-grade, standardized ECG paper recordings with millimeter grid,
standard 3x4 lead layout, continuous Lead II rhythm strip, calibration pulses,
and clinical lead typography.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.data.dataset import ECGRecord
from src.data.echonext import (
    NUM_LEADS,
)
from src.rendering.layout import (
    ECGLayout,
    ECGLayoutConfig,
)

logger = logging.getLogger(__name__)


def _resolve_system_font(size_pt: int, bold: bool = True) -> Any:
    """Attempt to load a clean sans-serif system font, falling back to PIL default font."""
    font_candidates = [
        # Linux / Ubuntu standard fonts (CI runner)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
        if bold
        else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # macOS standard fonts
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
        "Helvetica.ttf",
    ]

    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size_pt)
        except Exception:
            continue

    # Resizable default font in modern Pillow
    try:
        return ImageFont.load_default(size=size_pt)
    except Exception:
        return ImageFont.load_default()


@dataclass
class ECGImage:
    """Container for a rendered ECG paper recording with export and conversion utilities."""

    pil_image: Image.Image
    layout_config: ECGLayoutConfig
    metadata: Dict[str, Any]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.pil_image.width

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.pil_image.height

    @property
    def size(self) -> Tuple[int, int]:
        """(width, height) in pixels."""
        return self.pil_image.size

    def to_numpy(self) -> np.ndarray:
        """Convert image to uint8 RGB NumPy array of shape (H, W, 3)."""
        return np.array(self.pil_image, dtype=np.uint8)

    def to_tensor(
        self,
        channels_first: bool = True,
        normalize_0_1: bool = True,
    ) -> torch.Tensor:
        """Convert image to PyTorch tensor.

        Args:
            channels_first: If True, returns shape (3, H, W). Otherwise (H, W, 3).
            normalize_0_1: If True, values are scaled to float32 in [0.0, 1.0].
                           Otherwise uint8 tensor in [0, 255].
        """
        arr = self.to_numpy()
        if normalize_0_1:
            tensor = torch.from_numpy(arr).float() / 255.0
        else:
            tensor = torch.from_numpy(arr)

        if channels_first:
            # (H, W, 3) -> (3, H, W)
            tensor = tensor.permute(2, 0, 1)

        return tensor.contiguous()

    def save(
        self,
        path: Union[str, Path],
        format: Optional[str] = None,
        quality: int = 95,
        dpi: Optional[Tuple[int, int]] = None,
    ) -> Path:
        """Save the rendered image to disk, creating parent directories if necessary."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img_dpi = dpi or (self.layout_config.dpi, self.layout_config.dpi)
        self.pil_image.save(save_path, format=format, quality=quality, dpi=img_dpi)
        return save_path

    def _repr_png_(self) -> bytes:
        """IPython / Jupyter Notebook rich display helper."""
        buffer = io.BytesIO()
        self.pil_image.save(buffer, format="PNG")
        return buffer.getvalue()


class ECGPaperRenderer:
    """Photorealistic ECG Paper Waveform Renderer for 12-lead electrocardiograms."""

    def __init__(
        self,
        config: Optional[ECGLayoutConfig] = None,
        supersample: int = 1,
        enable_grid_cache: bool = True,
    ) -> None:
        """Initialize the ECG renderer.

        Args:
            config: ECGLayoutConfig defining geometry, speed, gain, grid, and DPI.
            supersample: Supersampling anti-aliasing factor (1 for direct fast drawing,
                         2 for 2x super-sampled anti-aliasing). Default 1.
            enable_grid_cache: If True, caches the rendered millimeter background grid
                               for rapid batch generation.
        """
        self.config = config or ECGLayoutConfig()
        self.layout = ECGLayout(self.config)
        self.supersample = max(1, int(supersample))
        self.enable_grid_cache = enable_grid_cache

        # Pre-cached background grid image
        self._cached_grid_image: Optional[Image.Image] = None

    def _create_background_grid(self) -> Image.Image:
        """Render the complete millimeter ECG paper grid."""
        scale = self.supersample
        w_px = self.layout.canvas_width_px * scale
        h_px = self.layout.canvas_height_px * scale

        bg_color = self.config.grid.background_color
        grid_img = Image.new("RGB", (w_px, h_px), bg_color)
        draw = ImageDraw.Draw(grid_img)

        if not self.config.grid.show_grid:
            return grid_img

        x_minor, y_minor, x_major, y_major = self.layout.get_grid_lines_px()

        minor_col = self.config.grid.minor_color
        major_col = self.config.grid.major_color
        minor_w = max(1, int(round(self.config.grid.minor_line_width_px * scale)))
        major_w = max(1, int(round(self.config.grid.major_line_width_px * scale)))

        # Minor vertical grid lines (1 mm)
        for x in x_minor:
            xs = int(round(x * scale))
            draw.line([(xs, 0), (xs, h_px)], fill=minor_col, width=minor_w)

        # Minor horizontal grid lines (1 mm)
        for y in y_minor:
            ys = int(round(y * scale))
            draw.line([(0, ys), (w_px, ys)], fill=minor_col, width=minor_w)

        # Major vertical grid lines (5 mm)
        for x in x_major:
            xs = int(round(x * scale))
            draw.line([(xs, 0), (xs, h_px)], fill=major_col, width=major_w)

        # Major horizontal grid lines (5 mm)
        for y in y_major:
            ys = int(round(y * scale))
            draw.line([(0, ys), (w_px, ys)], fill=major_col, width=major_w)

        return grid_img

    def get_background_grid(self) -> Image.Image:
        """Get background grid image (retrieved from cache or newly rendered)."""
        if self.enable_grid_cache:
            if self._cached_grid_image is None:
                self._cached_grid_image = self._create_background_grid()
            return self._cached_grid_image.copy()
        return self._create_background_grid()

    def _prepare_waveform_array(self, waveform: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Ensure waveform is a 2D float32 numpy array of shape (samples, 12)."""
        if isinstance(waveform, torch.Tensor):
            arr = waveform.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.array(waveform, dtype=np.float32)

        arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise ValueError(f"Waveform must be 2D, got shape {arr.shape}")

        # Check orientation: (12, N) vs (N, 12)
        if arr.shape[0] == NUM_LEADS and arr.shape[1] != NUM_LEADS:
            arr = arr.T
        elif arr.shape[1] != NUM_LEADS:
            raise ValueError(f"Waveform has invalid lead dimension {arr.shape}. Expected 12 leads.")

        # Numerical safety: replace any NaNs or Infs with 0.0
        if not np.isfinite(arr).all():
            logger.warning("Non-finite values encountered in waveform array; replacing with 0.0")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        return arr

    def render(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        record: Optional[Union[ECGRecord, Dict[str, Any]]] = None,
        header_text: Optional[str] = None,
        footer_text: Optional[str] = None,
    ) -> ECGImage:
        """Render a 12-lead ECG waveform onto standardized ECG paper.

        Args:
            waveform: ECG array of shape (2500, 12) or (12, 2500).
            record: Optional ECGRecord or dict containing patient/diagnostic metadata.
            header_text: Optional custom header override.
            footer_text: Optional custom footer override.

        Returns:
            ECGImage instance containing the rendered PIL Image and metadata.
        """
        arr = self._prepare_waveform_array(waveform)
        scale = self.supersample
        fs = self.config.sampling_rate_hz

        # Step 1: Base background grid
        base_img = self.get_background_grid()
        draw = ImageDraw.Draw(base_img)

        # Prepare fonts
        font_main = _resolve_system_font(
            int(round(self.config.label_font_size_pt * scale)), bold=True
        )
        font_sub = _resolve_system_font(
            int(round((self.config.label_font_size_pt - 1) * scale)), bold=False
        )

        # Step 2: Draw Calibration Pulses
        cal = self.config.calibration
        if cal.show_pulse:
            cal_color = self.config.waveform_color
            cal_width = max(1, int(round(self.config.waveform_line_width_px * scale)))
            for row_idx in cal.rows_to_display:
                pts_px = self.layout.get_calibration_pulse_points_px(row_idx)
                if pts_px is not None:
                    scaled_pts = [(float(x * scale), float(y * scale)) for x, y in pts_px]
                    draw.line(scaled_pts, fill=cal_color, width=cal_width, joint="curve")

        # Step 3: Draw Lead Waveforms & Annotations
        segments = self.layout.get_lead_segments()
        wave_color = self.config.waveform_color
        wave_width = max(1, int(round(self.config.waveform_line_width_px * scale)))

        # Amplitude conversion factor: mm per data unit
        if self.config.input_units == "zscore":
            amp_scale_mm = self.config.amplitude_scale_mm_per_unit
        else:
            amp_scale_mm = self.config.voltage_gain_mm_mv

        for seg in segments:
            # 3.1: Draw Lead Label
            if self.config.show_lead_labels:
                lbl_x = seg.label_x_mm * self.layout.px_per_mm * scale
                lbl_y = seg.label_y_mm * self.layout.px_per_mm * scale
                draw.text(
                    (lbl_x, lbl_y), seg.label_text, fill=self.config.label_color, font=font_main
                )

            # 3.2: Extract waveform slice
            start_sample = int(round(seg.start_time_sec * fs))
            num_samples = int(round(seg.duration_sec * fs))
            end_sample = min(start_sample + num_samples, arr.shape[0])

            lead_signal = arr[start_sample:end_sample, seg.lead_index]
            n_pts = len(lead_signal)
            if n_pts < 2:
                continue

            # Time to X mapping (mm)
            # t ranges from 0 to seg.duration_sec
            t_rel_sec = np.linspace(0.0, seg.duration_sec, n_pts, endpoint=False)
            x_mm = seg.start_x_mm + t_rel_sec * self.config.paper_speed_mm_s

            # Voltage to Y mapping (mm): Positive voltage deflects UP
            y_mm = seg.baseline_y_mm - (lead_signal * amp_scale_mm)

            # Convert to scaled pixels
            x_px = x_mm * self.layout.px_per_mm * scale
            y_px = y_mm * self.layout.px_per_mm * scale

            # Assemble polyline
            points = [(float(x), float(y)) for x, y in zip(x_px, y_px)]
            draw.line(points, fill=wave_color, width=wave_width, joint="curve")

        # Step 4: Draw Clinical Header & Footer Banners
        rec_id = None
        pat_key = None
        split = None
        lvef_val = None
        lvef_flag = None

        if record is not None:
            if isinstance(record, ECGRecord):
                rec_id = record.record_id
                pat_key = record.patient_key
                split = record.split
                lvef_val = record.lvef_target
                lvef_flag = record.lvef_lte_45_label
            elif isinstance(record, dict):
                rec_id = record.get("record_id")
                pat_key = record.get("patient_key")
                split = record.get("split")
                raw_lvef = record.get("lvef") if "lvef" in record else record.get("lvef_value")
                if raw_lvef is not None:
                    if hasattr(raw_lvef, "item"):
                        raw_lvef = raw_lvef.item()
                    lvef_val = float(raw_lvef)

                raw_flag = (
                    record.get("lvef_lte_45")
                    if "lvef_lte_45" in record
                    else record.get("lvef_lte_45_flag")
                )
                if raw_flag is not None:
                    if hasattr(raw_flag, "item"):
                        raw_flag = raw_flag.item()
                    lvef_flag = int(raw_flag) if raw_flag >= 0 else None

        if self.config.show_header:
            left_header = header_text or self.config.header_text
            if left_header is None:
                header_parts = []
                if rec_id:
                    header_parts.append(f"Record: {rec_id}")
                if pat_key:
                    header_parts.append(f"Patient: {pat_key}")
                if split:
                    header_parts.append(f"Split: {split}")
                if not header_parts:
                    header_parts.append("VentriQ Clinical ECG Recording")
                left_header = "  |  ".join(header_parts)

            right_header = "12-Lead Standard (3x4 + II Rhythm)  |  250 Hz"
            hdr_y = int(round(3.0 * self.layout.px_per_mm * scale))
            hdr_x_left = int(round(self.config.margin_left_mm * self.layout.px_per_mm * scale))
            hdr_x_right = int(
                round(
                    (self.config.paper_width_mm - self.config.margin_right_mm - 60.0)
                    * self.layout.px_per_mm
                    * scale
                )
            )

            draw.text((hdr_x_left, hdr_y), left_header, fill="#374151", font=font_main)
            draw.text((hdr_x_right, hdr_y), right_header, fill="#4b5563", font=font_sub)

        if self.config.show_footer:
            ftr = footer_text or self.config.footer_text
            if ftr is None:
                speed = int(self.config.paper_speed_mm_s)
                gain = int(self.config.voltage_gain_mm_mv)
                ftr = f"{speed} mm/s    {gain} mm/mV    0.5 - 40 Hz    Standard 12-Lead Calibration: 1.0 mV (10 mm) / 0.20 s (5 mm)"

            ftr_y = int(
                round(
                    (self.config.paper_height_mm - self.config.margin_bottom_mm + 3.0)
                    * self.layout.px_per_mm
                    * scale
                )
            )
            ftr_x = int(round(self.config.margin_left_mm * self.layout.px_per_mm * scale))
            draw.text((ftr_x, ftr_y), ftr, fill="#4b5563", font=font_sub)

        # Step 5: Downsample if supersampling enabled
        final_w = self.layout.canvas_width_px
        final_h = self.layout.canvas_height_px
        if scale > 1:
            final_img = base_img.resize((final_w, final_h), Image.Resampling.LANCZOS)
        else:
            final_img = base_img

        meta: Dict[str, Any] = {
            "record_id": rec_id,
            "patient_key": pat_key,
            "split": split,
            "lvef_value": float(lvef_val)
            if lvef_val is not None and not np.isnan(lvef_val)
            else None,
            "lvef_lte_45_flag": int(lvef_flag) if lvef_flag is not None else None,
            "image_width": final_w,
            "image_height": final_h,
            "dpi": self.config.dpi,
            "paper_speed_mm_s": self.config.paper_speed_mm_s,
            "voltage_gain_mm_mv": self.config.voltage_gain_mm_mv,
            "layout_format": self.config.layout_format,
            "created_at": datetime.now().isoformat(),
        }

        return ECGImage(pil_image=final_img, layout_config=self.config, metadata=meta)

    def __call__(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """PyTorch transform interface: returns normalized tensor (3, H, W)."""
        ecg_img = self.render(waveform)
        return ecg_img.to_tensor(channels_first=True, normalize_0_1=True)
