"""src/rendering/layout.py - Physical Geometry, Grid Coordinates & 12-Lead ECG Layout.

This module provides the geometric definitions, clinical cardiology standards,
grid specifications, and coordinate transformations for rendering digital 12-lead
ECG waveforms onto standardized, photorealistic ECG recording paper.

Standard Clinical Conventions Followed:
- Paper Speed: 25.0 mm/s (1 mm = 0.04 s = 40 ms; 5 mm = 0.20 s = 200 ms)
- Voltage Gain: 10.0 mm/mV (1 mm = 0.1 mV; 5 mm = 0.5 mV; 10 mm = 1.0 mV)
- Standard 3x4 + 1 Rhythm Strip Layout:
    Row 0: Lead I (0-2.5s), aVR (2.5-5.0s), V1 (5.0-7.5s), V4 (7.5-10.0s)
    Row 1: Lead II (0-2.5s), aVL (2.5-5.0s), V2 (5.0-7.5s), V5 (7.5-10.0s)
    Row 2: Lead III (0-2.5s), aVF (2.5-5.0s), V3 (5.0-7.5s), V6 (7.5-10.0s)
    Row 3: Lead II continuous 10-second rhythm strip (0.0-10.0s)
- Calibration Pulse: 1.0 mV (10 mm) height x 0.2 s (5 mm) width standard step pulse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.data.echonext import (
    EXPECTED_TIME_STEPS,
    NUM_LEADS,
    RECORDING_DURATION_SEC,
    SAMPLING_RATE_HZ,
    STANDARD_12_LEADS,
)

logger = logging.getLogger(__name__)

# Millimeters per inch constant for DPI conversion
MM_PER_INCH: float = 25.4

# Clinical Cardiology Defaults
DEFAULT_PAPER_SPEED_MM_PER_SEC: float = 25.0
DEFAULT_VOLTAGE_GAIN_MM_PER_MV: float = 10.0
DEFAULT_MINOR_GRID_MM: float = 1.0
DEFAULT_MAJOR_GRID_MM: float = 5.0

# Standard lead index mapping
LEAD_NAME_TO_INDEX: Dict[str, int] = {lead: idx for idx, lead in enumerate(STANDARD_12_LEADS)}


@dataclass(frozen=True)
class GridConfig:
    """Configuration for the millimeter ECG background grid."""

    minor_grid_mm: float = DEFAULT_MINOR_GRID_MM
    major_grid_mm: float = DEFAULT_MAJOR_GRID_MM
    minor_color: str = "#fed7d7"  # Soft pink minor grid line
    major_color: str = "#f87171"  # Darker pink/red major grid line
    background_color: str = "#ffffff"  # Clean medical paper white or ivory (#fdfbf7)
    minor_line_width_px: float = 0.5
    major_line_width_px: float = 1.0
    show_grid: bool = True
    style: str = "standard_pink"  # 'standard_pink', 'classic_red', 'subtle_gray', 'green', 'clean_white'

    @classmethod
    def from_preset(cls, preset_name: str) -> "GridConfig":
        """Instantiate GridConfig from a pre-defined style preset."""
        presets = {
            "standard_pink": cls(
                minor_color="#fecaca",
                major_color="#f87171",
                background_color="#ffffff",
                minor_line_width_px=0.6,
                major_line_width_px=1.2,
                style="standard_pink",
            ),
            "classic_red": cls(
                minor_color="#ffcdd2",
                major_color="#e57373",
                background_color="#fffcf9",
                minor_line_width_px=0.6,
                major_line_width_px=1.3,
                style="classic_red",
            ),
            "subtle_gray": cls(
                minor_color="#eeeeee",
                major_color="#cccccc",
                background_color="#ffffff",
                minor_line_width_px=0.5,
                major_line_width_px=1.0,
                style="subtle_gray",
            ),
            "clinical_green": cls(
                minor_color="#d1fae5",
                major_color="#6ee7b7",
                background_color="#ffffff",
                minor_line_width_px=0.6,
                major_line_width_px=1.2,
                style="clinical_green",
            ),
            "no_grid": cls(
                minor_color="#ffffff",
                major_color="#ffffff",
                background_color="#ffffff",
                show_grid=False,
                style="no_grid",
            ),
        }
        if preset_name not in presets:
            raise ValueError(f"Unknown grid preset: {preset_name}. Available: {list(presets.keys())}")
        return presets[preset_name]


@dataclass(frozen=True)
class CalibrationPulseConfig:
    """Configuration for standard clinical voltage calibration pulses."""

    show_pulse: bool = True
    pulse_height_mv: float = 1.0  # Standard 1.0 mV height (10 mm)
    pulse_width_sec: float = 0.2   # Standard 0.2 s width (5 mm, 1 large box)
    lead_in_sec: float = 0.08      # Baseline before step up (2 mm)
    lead_out_sec: float = 0.08     # Baseline after step down (2 mm)
    rows_to_display: Tuple[int, ...] = (0, 1, 2, 3)  # Display pulse on all rows


@dataclass(frozen=True)
class LeadSegment:
    """Geometric and temporal specification of an individual lead box on paper."""

    lead_name: str
    lead_index: int
    row: int
    col: int
    start_time_sec: float
    duration_sec: float
    start_x_mm: float
    width_mm: float
    baseline_y_mm: float
    height_mm: float
    label_text: str
    label_x_mm: float
    label_y_mm: float

    @property
    def end_time_sec(self) -> float:
        return self.start_time_sec + self.duration_sec

    @property
    def end_x_mm(self) -> float:
        return self.start_x_mm + self.width_mm


@dataclass
class ECGLayoutConfig:
    """Master configuration for ECG paper geometry, layout, and rendering parameters."""

    dpi: int = 150
    paper_speed_mm_s: float = DEFAULT_PAPER_SPEED_MM_PER_SEC
    voltage_gain_mm_mv: float = DEFAULT_VOLTAGE_GAIN_MM_PER_MV
    sampling_rate_hz: int = SAMPLING_RATE_HZ
    duration_sec: float = RECORDING_DURATION_SEC

    # Paper Dimensions (in millimeters)
    # Default 280 mm x 190 mm fits comfortably on standard A4 / US Letter landscape thermal paper
    paper_width_mm: float = 280.0
    paper_height_mm: float = 190.0
    margin_left_mm: float = 12.0
    margin_right_mm: float = 12.0
    margin_top_mm: float = 15.0
    margin_bottom_mm: float = 15.0

    # Layout Format
    layout_format: str = "3x4_1r"  # '3x4_1r', '3x4', '12x1', '6x2'
    rhythm_lead_name: str = "II"
    row_height_mm: float = 38.0

    # Signal Amplitude Scaling
    # EchoNext waveforms are z-score standardized.
    # amplitude_scale_mm_per_unit maps 1 unit (1 standard deviation) to physical millimeters on paper.
    # At 6.0 mm/unit, +/- 2.5 SD deflections stay comfortably within 38 mm row heights without clipping.
    amplitude_scale_mm_per_unit: float = 6.0
    input_units: str = "zscore"  # 'zscore' or 'mv'

    # Trace Appearance
    waveform_color: str = "#111827"  # Very dark slate / charcoal
    waveform_line_width_px: float = 1.3
    anti_aliased: bool = True

    # Grid & Calibration
    grid: GridConfig = field(default_factory=GridConfig)
    calibration: CalibrationPulseConfig = field(default_factory=CalibrationPulseConfig)

    # Lead Labels & Typography
    show_lead_labels: bool = True
    label_color: str = "#1f2937"
    label_font_size_pt: int = 9
    show_header: bool = True
    show_footer: bool = True
    header_text: Optional[str] = None
    footer_text: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate layout parameters."""
        if self.dpi <= 0:
            raise ValueError(f"DPI must be positive, got {self.dpi}")
        if self.paper_width_mm <= 0 or self.paper_height_mm <= 0:
            raise ValueError("Paper dimensions must be positive.")
        if self.paper_speed_mm_s <= 0 or self.voltage_gain_mm_mv <= 0:
            raise ValueError("Speed and gain must be positive.")
        if self.layout_format not in {"3x4_1r", "3x4", "12x1", "6x2"}:
            raise ValueError(f"Unsupported layout format: {self.layout_format}")
        if self.rhythm_lead_name not in LEAD_NAME_TO_INDEX:
            raise ValueError(f"Invalid rhythm lead name: {self.rhythm_lead_name}")


class ECGLayout:
    """Translates physical clinical millimeter geometry into pixel coordinate systems."""

    def __init__(self, config: Optional[ECGLayoutConfig] = None) -> None:
        self.config = config or ECGLayoutConfig()
        self._px_per_mm = self.config.dpi / MM_PER_INCH
        self._segments: Optional[List[LeadSegment]] = None

    @property
    def px_per_mm(self) -> float:
        """Pixels per physical millimeter."""
        return self._px_per_mm

    def mm_to_px(self, mm_val: float) -> float:
        """Convert millimeters to exact float pixels."""
        return mm_val * self._px_per_mm

    def px_to_mm(self, px_val: float) -> float:
        """Convert pixels to physical millimeters."""
        return px_val / self._px_per_mm

    @property
    def canvas_width_px(self) -> int:
        """Total image width in pixels."""
        return int(round(self.mm_to_px(self.config.paper_width_mm)))

    @property
    def canvas_height_px(self) -> int:
        """Total image height in pixels."""
        return int(round(self.mm_to_px(self.config.paper_height_mm)))

    @property
    def printable_width_mm(self) -> float:
        """Printable width excluding left and right margins."""
        return self.config.paper_width_mm - (self.config.margin_left_mm + self.config.margin_right_mm)

    @property
    def printable_height_mm(self) -> float:
        """Printable height excluding top and bottom margins."""
        return self.config.paper_height_mm - (self.config.margin_top_mm + self.config.margin_bottom_mm)

    def get_lead_segments(self) -> List[LeadSegment]:
        """Compute the complete set of lead segment geometries for the chosen layout."""
        if self._segments is not None:
            return self._segments

        fmt = self.config.layout_format
        if fmt == "3x4_1r":
            segments = self._build_3x4_1r_segments()
        elif fmt == "3x4":
            segments = self._build_3x4_segments()
        elif fmt == "12x1":
            segments = self._build_12x1_segments()
        elif fmt == "6x2":
            segments = self._build_6x2_segments()
        else:
            raise ValueError(f"Unsupported layout format: {fmt}")

        self._segments = segments
        return self._segments

    def _build_3x4_1r_segments(self) -> List[LeadSegment]:
        """Build standard 3 rows of 4 leads (2.5s each) + 1 bottom rhythm lead (10s)."""
        # Clinical 3x4 lead mapping:
        # Col 0 (0.0 - 2.5s): I,   II,  III
        # Col 1 (2.5 - 5.0s): aVR, aVL, aVF
        # Col 2 (5.0 - 7.5s): V1,  V2,  V3
        # Col 3 (7.5 - 10.0s): V4,  V5,  V6
        grid_matrix = [
            ["I",   "aVR", "V1", "V4"],  # Row 0
            ["II",  "aVL", "V2", "V5"],  # Row 1
            ["III", "aVF", "V3", "V6"],  # Row 2
        ]

        segments: List[LeadSegment] = []
        # Signal width across 10 seconds at 25 mm/s is 250 mm
        col_duration_sec = 2.5
        col_width_mm = col_duration_sec * self.config.paper_speed_mm_s  # 62.5 mm

        # Calculate horizontal start offset
        start_x_mm = self.config.margin_left_mm
        row_height = self.config.row_height_mm

        # 3 Rows of 4 leads
        for row_idx, row_leads in enumerate(grid_matrix):
            baseline_y_mm = self.config.margin_top_mm + (row_idx + 0.5) * row_height

            for col_idx, lead_name in enumerate(row_leads):
                lead_idx = LEAD_NAME_TO_INDEX[lead_name]
                t_start = col_idx * col_duration_sec
                seg_x = start_x_mm + col_idx * col_width_mm

                label_x = seg_x + 2.0
                label_y = baseline_y_mm - (row_height * 0.35)

                segments.append(
                    LeadSegment(
                        lead_name=lead_name,
                        lead_index=lead_idx,
                        row=row_idx,
                        col=col_idx,
                        start_time_sec=t_start,
                        duration_sec=col_duration_sec,
                        start_x_mm=seg_x,
                        width_mm=col_width_mm,
                        baseline_y_mm=baseline_y_mm,
                        height_mm=row_height,
                        label_text=lead_name,
                        label_x_mm=label_x,
                        label_y_mm=label_y,
                    )
                )

        # Row 3: Continuous rhythm strip
        rhythm_lead = self.config.rhythm_lead_name
        rhythm_idx = LEAD_NAME_TO_INDEX[rhythm_lead]
        rhythm_row = 3
        rhythm_baseline_y = self.config.margin_top_mm + (rhythm_row + 0.5) * row_height
        total_signal_width_mm = self.config.duration_sec * self.config.paper_speed_mm_s  # 250 mm

        segments.append(
            LeadSegment(
                lead_name=rhythm_lead,
                lead_index=rhythm_idx,
                row=rhythm_row,
                col=0,
                start_time_sec=0.0,
                duration_sec=self.config.duration_sec,
                start_x_mm=start_x_mm,
                width_mm=total_signal_width_mm,
                baseline_y_mm=rhythm_baseline_y,
                height_mm=row_height,
                label_text=f"{rhythm_lead} (Rhythm)",
                label_x_mm=start_x_mm + 2.0,
                label_y_mm=rhythm_baseline_y - (row_height * 0.35),
            )
        )

        return segments

    def _build_3x4_segments(self) -> List[LeadSegment]:
        """Build standard 3 rows of 4 leads without rhythm strip."""
        grid_matrix = [
            ["I",   "aVR", "V1", "V4"],
            ["II",  "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
        ]
        segments: List[LeadSegment] = []
        col_duration_sec = 2.5
        col_width_mm = col_duration_sec * self.config.paper_speed_mm_s
        start_x_mm = self.config.margin_left_mm
        row_height = self.printable_height_mm / 3.0

        for row_idx, row_leads in enumerate(grid_matrix):
            baseline_y_mm = self.config.margin_top_mm + (row_idx + 0.5) * row_height
            for col_idx, lead_name in enumerate(row_leads):
                lead_idx = LEAD_NAME_TO_INDEX[lead_name]
                t_start = col_idx * col_duration_sec
                seg_x = start_x_mm + col_idx * col_width_mm
                segments.append(
                    LeadSegment(
                        lead_name=lead_name,
                        lead_index=lead_idx,
                        row=row_idx,
                        col=col_idx,
                        start_time_sec=t_start,
                        duration_sec=col_duration_sec,
                        start_x_mm=seg_x,
                        width_mm=col_width_mm,
                        baseline_y_mm=baseline_y_mm,
                        height_mm=row_height,
                        label_text=lead_name,
                        label_x_mm=seg_x + 2.0,
                        label_y_mm=baseline_y_mm - (row_height * 0.35),
                    )
                )
        return segments

    def _build_12x1_segments(self) -> List[LeadSegment]:
        """Build stacked layout of 12 full rows (1 row per lead)."""
        segments: List[LeadSegment] = []
        start_x_mm = self.config.margin_left_mm
        total_width_mm = self.config.duration_sec * self.config.paper_speed_mm_s
        row_height = self.printable_height_mm / 12.0

        for row_idx, lead_name in enumerate(STANDARD_12_LEADS):
            lead_idx = LEAD_NAME_TO_INDEX[lead_name]
            baseline_y_mm = self.config.margin_top_mm + (row_idx + 0.5) * row_height
            segments.append(
                LeadSegment(
                    lead_name=lead_name,
                    lead_index=lead_idx,
                    row=row_idx,
                    col=0,
                    start_time_sec=0.0,
                    duration_sec=self.config.duration_sec,
                    start_x_mm=start_x_mm,
                    width_mm=total_width_mm,
                    baseline_y_mm=baseline_y_mm,
                    height_mm=row_height,
                    label_text=lead_name,
                    label_x_mm=start_x_mm + 2.0,
                    label_y_mm=baseline_y_mm - (row_height * 0.35),
                )
            )
        return segments

    def _build_6x2_segments(self) -> List[LeadSegment]:
        """Build 6 rows of 2 columns (5 seconds each)."""
        pairs = [
            ("I", "aVR"),
            ("II", "aVL"),
            ("III", "aVF"),
            ("V1", "V4"),
            ("V2", "V5"),
            ("V3", "V6"),
        ]
        segments: List[LeadSegment] = []
        col_duration_sec = 5.0
        col_width_mm = col_duration_sec * self.config.paper_speed_mm_s
        start_x_mm = self.config.margin_left_mm
        row_height = self.printable_height_mm / 6.0

        for row_idx, (l1, l2) in enumerate(pairs):
            baseline_y_mm = self.config.margin_top_mm + (row_idx + 0.5) * row_height
            for col_idx, lead_name in enumerate([l1, l2]):
                lead_idx = LEAD_NAME_TO_INDEX[lead_name]
                t_start = col_idx * col_duration_sec
                seg_x = start_x_mm + col_idx * col_width_mm
                segments.append(
                    LeadSegment(
                        lead_name=lead_name,
                        lead_index=lead_idx,
                        row=row_idx,
                        col=col_idx,
                        start_time_sec=t_start,
                        duration_sec=col_duration_sec,
                        start_x_mm=seg_x,
                        width_mm=col_width_mm,
                        baseline_y_mm=baseline_y_mm,
                        height_mm=row_height,
                        label_text=lead_name,
                        label_x_mm=seg_x + 2.0,
                        label_y_mm=baseline_y_mm - (row_height * 0.35),
                    )
                )
        return segments

    def get_grid_lines_px(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute pixel coordinates of minor and major grid lines across the canvas.

        Returns:
            Tuple of 4 1D arrays:
                - x_minor_px: horizontal pixel coordinates for 1 mm vertical lines
                - y_minor_px: vertical pixel coordinates for 1 mm horizontal lines
                - x_major_px: horizontal pixel coordinates for 5 mm vertical lines
                - y_major_px: vertical pixel coordinates for 5 mm horizontal lines
        """
        minor_step = self.config.grid.minor_grid_mm
        major_step = self.config.grid.major_grid_mm

        w_mm = self.config.paper_width_mm
        h_mm = self.config.paper_height_mm

        # Grid line positions in millimeters
        # Snap ticks from 0 to dimension
        x_ticks_minor_mm = np.arange(0.0, w_mm + 0.001, minor_step)
        y_ticks_minor_mm = np.arange(0.0, h_mm + 0.001, minor_step)

        x_ticks_major_mm = np.arange(0.0, w_mm + 0.001, major_step)
        y_ticks_major_mm = np.arange(0.0, h_mm + 0.001, major_step)

        x_minor_px = np.round(self.mm_to_px(x_ticks_minor_mm)).astype(np.float32)
        y_minor_px = np.round(self.mm_to_px(y_ticks_minor_mm)).astype(np.float32)

        x_major_px = np.round(self.mm_to_px(x_ticks_major_mm)).astype(np.float32)
        y_major_px = np.round(self.mm_to_px(y_ticks_major_mm)).astype(np.float32)

        return x_minor_px, y_minor_px, x_major_px, y_major_px

    def get_calibration_pulse_points_px(self, row_idx: int) -> Optional[np.ndarray]:
        """Generate pixel polyline coordinates for the standard 1 mV calibration pulse.

        Standard clinical calibration geometry:
        - Lead-in: 0.08 s (2.0 mm) at baseline
        - Vertical up-step: 1.0 mV (10.0 mm height)
        - Plateau: 0.20 s (5.0 mm width)
        - Vertical down-step: back to baseline
        - Lead-out: 0.08 s (2.0 mm) at baseline

        Returns:
            Array of shape (N, 2) of (x_px, y_px) coordinates, or None if row not enabled.
        """
        cal = self.config.calibration
        if not cal.show_pulse or row_idx not in cal.rows_to_display:
            return None

        # Position calibration pulse in the margin area immediately preceding the first column
        row_height = self.config.row_height_mm
        baseline_y_mm = self.config.margin_top_mm + (row_idx + 0.5) * row_height

        pulse_h_mm = cal.pulse_height_mv * self.config.voltage_gain_mm_mv
        pulse_w_mm = cal.pulse_width_sec * self.config.paper_speed_mm_s
        lead_in_mm = cal.lead_in_sec * self.config.paper_speed_mm_s
        lead_out_mm = cal.lead_out_sec * self.config.paper_speed_mm_s

        total_pulse_w_mm = lead_in_mm + pulse_w_mm + lead_out_mm
        # Sit flush against the left margin
        start_x_mm = max(1.0, self.config.margin_left_mm - total_pulse_w_mm - 1.0)

        # Pulse profile in (x_mm, y_mm):
        # Note: In standard display, positive voltage moves UP (decreasing y_mm)
        p1 = (start_x_mm, baseline_y_mm)
        p2 = (start_x_mm + lead_in_mm, baseline_y_mm)
        p3 = (start_x_mm + lead_in_mm, baseline_y_mm - pulse_h_mm)
        p4 = (start_x_mm + lead_in_mm + pulse_w_mm, baseline_y_mm - pulse_h_mm)
        p5 = (start_x_mm + lead_in_mm + pulse_w_mm, baseline_y_mm)
        p6 = (start_x_mm + total_pulse_w_mm, baseline_y_mm)

        pts_mm = np.array([p1, p2, p3, p4, p5, p6], dtype=np.float32)
        pts_px = np.zeros_like(pts_mm)
        pts_px[:, 0] = self.mm_to_px(pts_mm[:, 0])
        pts_px[:, 1] = self.mm_to_px(pts_mm[:, 1])

        return pts_px


def load_rendering_config(config_path: Optional[Union[str, Path]] = None) -> ECGLayoutConfig:
    """Load ECGLayoutConfig from a YAML configuration file.

    Args:
        config_path: Path to YAML file. If None, looks for configs/rendering.yaml.

    Returns:
        Configured ECGLayoutConfig instance.
    """
    import yaml
    from src.utils.paths import CONFIGS_DIR

    target_path = Path(config_path) if config_path else CONFIGS_DIR / "rendering.yaml"
    if not target_path.exists():
        logger.warning(f"Rendering config not found at {target_path}; falling back to default config.")
        return ECGLayoutConfig()

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    paper = data.get("paper", {})
    layout = data.get("layout", {})
    sig = data.get("signal", {})
    trace = data.get("trace", {})
    grid_data = data.get("grid", {})
    cal_data = data.get("calibration", {})
    typo = data.get("typography", {})

    # Build GridConfig
    grid_cfg = GridConfig(
        minor_grid_mm=float(grid_data.get("minor_grid_mm", DEFAULT_MINOR_GRID_MM)),
        major_grid_mm=float(grid_data.get("major_grid_mm", DEFAULT_MAJOR_GRID_MM)),
        minor_color=str(grid_data.get("minor_color", "#fed7d7")),
        major_color=str(grid_data.get("major_color", "#f87171")),
        background_color=str(grid_data.get("background_color", "#ffffff")),
        minor_line_width_px=float(grid_data.get("minor_line_width_px", 0.6)),
        major_line_width_px=float(grid_data.get("major_line_width_px", 1.2)),
        show_grid=bool(grid_data.get("show_grid", True)),
        style=str(grid_data.get("style", "standard_pink")),
    )

    # Build CalibrationPulseConfig
    cal_cfg = CalibrationPulseConfig(
        show_pulse=bool(cal_data.get("show_pulse", True)),
        pulse_height_mv=float(cal_data.get("pulse_height_mv", 1.0)),
        pulse_width_sec=float(cal_data.get("pulse_width_sec", 0.2)),
        lead_in_sec=float(cal_data.get("lead_in_sec", 0.08)),
        lead_out_sec=float(cal_data.get("lead_out_sec", 0.08)),
        rows_to_display=tuple(cal_data.get("rows_to_display", [0, 1, 2, 3])),
    )

    return ECGLayoutConfig(
        dpi=int(paper.get("dpi", 150)),
        paper_speed_mm_s=float(paper.get("paper_speed_mm_s", DEFAULT_PAPER_SPEED_MM_PER_SEC)),
        voltage_gain_mm_mv=float(paper.get("voltage_gain_mm_mv", DEFAULT_VOLTAGE_GAIN_MM_PER_MV)),
        paper_width_mm=float(paper.get("width_mm", 280.0)),
        paper_height_mm=float(paper.get("height_mm", 190.0)),
        margin_left_mm=float(paper.get("margin_left_mm", 12.0)),
        margin_right_mm=float(paper.get("margin_right_mm", 12.0)),
        margin_top_mm=float(paper.get("margin_top_mm", 15.0)),
        margin_bottom_mm=float(paper.get("margin_bottom_mm", 15.0)),
        layout_format=str(layout.get("format", "3x4_1r")),
        rhythm_lead_name=str(layout.get("rhythm_lead", "II")),
        row_height_mm=float(layout.get("row_height_mm", 38.0)),
        sampling_rate_hz=int(sig.get("sampling_rate_hz", SAMPLING_RATE_HZ)),
        duration_sec=float(sig.get("duration_sec", RECORDING_DURATION_SEC)),
        input_units=str(sig.get("input_units", "zscore")),
        amplitude_scale_mm_per_unit=float(sig.get("amplitude_scale_mm_per_unit", 6.0)),
        waveform_color=str(trace.get("color", "#111827")),
        waveform_line_width_px=float(trace.get("line_width_px", 1.3)),
        anti_aliased=bool(trace.get("anti_aliased", True)),
        grid=grid_cfg,
        calibration=cal_cfg,
        show_lead_labels=bool(typo.get("show_lead_labels", True)),
        label_color=str(typo.get("label_color", "#1f2937")),
        label_font_size_pt=int(typo.get("label_font_size_pt", 9)),
        show_header=bool(typo.get("show_header", True)),
        show_footer=bool(typo.get("show_footer", True)),
    )

