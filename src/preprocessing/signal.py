from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml
from scipy import signal as scipy_signal

from src.data.echonext import (
    SAMPLING_RATE_HZ,
)
from src.preprocessing.quality import ECGQualityEvaluator, QualityReport

logger = logging.getLogger(__name__)


@dataclass
class ProcessedECG:
    """Represents multi-stage ECG signal representations and quality metrics.

    Maintains clean separation between:
    - original: Raw released waveform directly from dataset.
    - quality_checked: Standardized array verified by QualityReport.
    - filtered: Optional filtered signal (None if filtering bypassed).
    - normalized: Optional normalized signal (None if normalization bypassed).
    - final: Active output signal intended for downstream consumption.
    - quality_report: Complete quality diagnostics.
    - config: Preprocessing parameters applied.
    """

    original: np.ndarray
    quality_report: QualityReport
    filtered: Optional[np.ndarray] = None
    normalized: Optional[np.ndarray] = None
    final: Optional[np.ndarray] = None
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.final is None:
            if self.normalized is not None:
                self.final = self.normalized
            elif self.filtered is not None:
                self.final = self.filtered
            else:
                self.final = self.original


class ECGSignalPreprocessor:
    """Safe, reproducible 12-lead ECG signal preprocessor.

    Adheres to the critical domain constraint: EchoNext waveforms are already
    median-filtered and z-score standardized. Destructive filtering or
    automatic re-normalization is disabled by default and only applied when
    explicitly configured.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        sampling_rate_hz: Optional[int] = None,
        apply_filter: Optional[bool] = None,
        lowcut_hz: Optional[float] = None,
        highcut_hz: Optional[float] = None,
        filter_order: Optional[int] = None,
        apply_notch: Optional[bool] = None,
        notch_frequencies_hz: Optional[List[float]] = None,
        apply_normalization: Optional[bool] = None,
        normalization_method: Optional[str] = None,
        epsilon: Optional[float] = None,
        evaluator: Optional[ECGQualityEvaluator] = None,
    ) -> None:
        # Load configuration file if provided
        self.config: Dict[str, Any] = {}
        if config_path is not None:
            cpath = Path(config_path)
            if cpath.exists():
                with open(cpath, "r") as f:
                    self.config = yaml.safe_load(f) or {}

        # Resolve parameters: explicit kwargs take precedence over config file
        filt_cfg = self.config.get("filtering", {})
        norm_cfg = self.config.get("normalization", {})
        sig_cfg = self.config.get("signal", {})

        self.fs = (
            sampling_rate_hz
            if sampling_rate_hz is not None
            else sig_cfg.get("sampling_rate_hz", SAMPLING_RATE_HZ)
        )
        self.apply_filter = (
            apply_filter if apply_filter is not None else filt_cfg.get("apply_filter", False)
        )
        self.lowcut = lowcut_hz if lowcut_hz is not None else filt_cfg.get("lowcut_hz", 0.5)
        self.highcut = highcut_hz if highcut_hz is not None else filt_cfg.get("highcut_hz", 40.0)
        self.filter_order = filter_order if filter_order is not None else filt_cfg.get("order", 4)
        self.filter_type = filt_cfg.get("filter_type", "butterworth")

        self.apply_notch = (
            apply_notch if apply_notch is not None else filt_cfg.get("apply_notch", False)
        )
        self.notch_freqs = (
            notch_frequencies_hz
            if notch_frequencies_hz is not None
            else filt_cfg.get("notch_frequencies_hz", [50.0, 60.0])
        )
        self.notch_q = filt_cfg.get("notch_quality_factor", 30.0)

        self.apply_normalization = (
            apply_normalization
            if apply_normalization is not None
            else norm_cfg.get("apply_normalization", False)
        )
        self.norm_method = (
            normalization_method
            if normalization_method is not None
            else norm_cfg.get("method", "lead_zscore")
        )
        self.epsilon = epsilon if epsilon is not None else norm_cfg.get("epsilon", 1e-8)

        # Quality evaluator
        self.evaluator = evaluator or ECGQualityEvaluator(sampling_rate_hz=self.fs)

    def _design_bandpass_sos(self) -> np.ndarray:
        """Design zero-phase bandpass filter using Second-Order Sections (SOS)."""
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        # Guard boundaries
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))

        sos = scipy_signal.butter(self.filter_order, [low, high], btype="bandpass", output="sos")
        return sos

    def apply_zero_phase_bandpass(
        self, signal_arr: np.ndarray, sos: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply forward-backward bandpass filter along the time axis (axis 0).

        Zero-phase filtering guarantees 0 ms time shift in cardiac peak locations.
        """
        if sos is None:
            sos = self._design_bandpass_sos()

        n_samples = signal_arr.shape[0]
        # Calculate safe padlen for short signals
        # scipy default padlen is 3 * (2 * n_sections + 1)
        n_sections = sos.shape[0]
        default_padlen = 3 * (2 * n_sections + 1)
        padlen = min(default_padlen, n_samples - 1) if n_samples > 1 else None

        if padlen is None or padlen <= 0 or n_samples <= default_padlen:
            # Signal too short for forward-backward filter padding, return as is
            logger.warning(
                "Signal length (%d) too short for safe forward-backward filtering. Skipping.",
                n_samples,
            )
            return signal_arr.copy()

        filtered = scipy_signal.sosfiltfilt(sos, signal_arr, axis=0, padlen=padlen)
        return filtered.astype(np.float32)

    def apply_notch_filters(self, signal_arr: np.ndarray) -> np.ndarray:
        """Apply narrow-band notch filters to remove powerline interference."""
        out = signal_arr.copy()
        nyquist = 0.5 * self.fs

        for f0 in self.notch_freqs:
            if f0 >= nyquist:
                continue
            w0 = f0 / nyquist
            b, a = scipy_signal.iirnotch(w0, self.notch_q)
            # Apply zero-phase filtfilt along time axis
            n_samples = out.shape[0]
            padlen = min(15, n_samples - 1)
            if padlen > 0:
                out = scipy_signal.filtfilt(b, a, out, axis=0, padlen=padlen)

        return out.astype(np.float32)

    def normalize(
        self,
        signal_arr: np.ndarray,
        method: Optional[str] = None,
        train_mean: Optional[float] = None,
        train_std: Optional[float] = None,
    ) -> np.ndarray:
        """Normalize signal without corrupting clinical relative lead amplitudes.

        Args:
            signal_arr: Array of shape (samples, leads).
            method: Normalization mode ('lead_zscore', 'global_zscore', 'minmax', 'robust').
            train_mean: Pre-calculated training mean (for global_zscore).
            train_std: Pre-calculated training std (for global_zscore).
        """
        norm_type = method or self.norm_method
        arr = signal_arr.copy()

        if norm_type == "lead_zscore":
            # Per-lead zero-mean and unit variance
            mean = np.mean(arr, axis=0, keepdims=True)
            std = np.std(arr, axis=0, keepdims=True)
            return ((arr - mean) / (std + self.epsilon)).astype(np.float32)

        elif norm_type == "global_zscore":
            # Dataset-wide or fixed statistics (avoids validation/test leakage)
            g_mean = train_mean if train_mean is not None else float(np.mean(arr))
            g_std = train_std if train_std is not None else float(np.std(arr))
            return ((arr - g_mean) / (g_std + self.epsilon)).astype(np.float32)

        elif norm_type == "minmax":
            # Scale to [-1, 1] per lead
            lead_min = np.min(arr, axis=0, keepdims=True)
            lead_max = np.max(arr, axis=0, keepdims=True)
            scale = lead_max - lead_min
            scale[scale < self.epsilon] = 1.0
            norm = 2.0 * ((arr - lead_min) / scale) - 1.0
            return norm.astype(np.float32)

        elif norm_type == "robust":
            # Median / IQR per lead (resistant to artifact spikes)
            median = np.median(arr, axis=0, keepdims=True)
            q75, q25 = np.percentile(arr, [75, 25], axis=0, keepdims=True)
            iqr = q75 - q25
            iqr[iqr < self.epsilon] = 1.0
            return ((arr - median) / iqr).astype(np.float32)

        return arr.astype(np.float32)

    def process(
        self,
        ecg: Any,
        channels_first: bool = False,
        train_mean: Optional[float] = None,
        train_std: Optional[float] = None,
    ) -> ProcessedECG:
        """Execute full preprocessing pipeline on a single 12-lead ECG.

        Args:
            ecg: Input array of shape (2500, 12) or (12, 2500).
            channels_first: If True, returns final arrays oriented as (12, 2500).
            train_mean: Optional training mean for global z-score normalization.
            train_std: Optional training std for global z-score normalization.

        Returns:
            ProcessedECG holding raw, quality-checked, optional filtered,
            and optional normalized representations.
        """
        # Step 1: Quality evaluation and numerical sanity
        quality_rep = self.evaluator.evaluate(ecg)

        if not quality_rep.is_valid:
            # Handle invalid array gracefully
            empty_arr = np.array([], dtype=np.float32)
            return ProcessedECG(
                original=empty_arr,
                quality_report=quality_rep,
                filtered=None,
                normalized=None,
                final=empty_arr,
                config=self.get_config(),
            )

        # Standardize orientation to (samples, leads)
        arr = np.asarray(ecg, dtype=np.float32)
        if arr.shape == (self.evaluator.num_leads, self.evaluator.expected_steps):
            arr = arr.T
        original_waveform = arr.copy()

        # Step 2: Optional filtering (bypassed if apply_filter is False)
        filtered_waveform: Optional[np.ndarray] = None
        current_waveform = original_waveform

        if self.apply_filter:
            current_waveform = self.apply_zero_phase_bandpass(current_waveform)
            if self.apply_notch:
                current_waveform = self.apply_notch_filters(current_waveform)
            filtered_waveform = current_waveform.copy()

        # Step 3: Optional normalization (bypassed if apply_normalization is False)
        normalized_waveform: Optional[np.ndarray] = None
        if self.apply_normalization:
            current_waveform = self.normalize(
                current_waveform,
                train_mean=train_mean,
                train_std=train_std,
            )
            normalized_waveform = current_waveform.copy()

        final_waveform = current_waveform.copy()

        # Re-transpose if channels_first requested
        if channels_first:
            original_waveform = original_waveform.T
            if filtered_waveform is not None:
                filtered_waveform = filtered_waveform.T
            if normalized_waveform is not None:
                normalized_waveform = normalized_waveform.T
            final_waveform = final_waveform.T

        return ProcessedECG(
            original=original_waveform,
            quality_report=quality_rep,
            filtered=filtered_waveform,
            normalized=normalized_waveform,
            final=final_waveform,
            config=self.get_config(),
        )

    def get_config(self) -> Dict[str, Any]:
        """Return active preprocessing settings."""
        return {
            "sampling_rate_hz": self.fs,
            "apply_filter": self.apply_filter,
            "filter_type": self.filter_type,
            "lowcut_hz": self.lowcut,
            "highcut_hz": self.highcut,
            "filter_order": self.filter_order,
            "apply_notch": self.apply_notch,
            "notch_frequencies_hz": self.notch_freqs,
            "apply_normalization": self.apply_normalization,
            "normalization_method": self.norm_method,
        }


# =========================================================================
# Backward-compatibility alias preserving original signal_preprocessing class
# =========================================================================
class signal_preprocessing:
    """Backward-compatible wrapper for legacy signal_preprocessing class.

    Preserves existing API signatures while correcting previous bugs
    (e.g., bandpass filter specification and operator precedence).
    """

    def __init__(self) -> None:
        self.preprocessor = ECGSignalPreprocessor()

    def bandpass_filter(
        self, ecg: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0, fs: int = 250
    ) -> np.ndarray:
        """Filter ECG with zero-phase bandpass filter."""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = scipy_signal.butter(4, [low, high], btype="bandpass")
        # Detect time axis: axis 0 if shape is (2500, 12), axis -1 if shape is (12, 2500)
        axis = 0 if (ecg.ndim == 2 and ecg.shape[0] > ecg.shape[1]) else -1
        padlen = min(27, ecg.shape[axis] - 1) if ecg.shape[axis] > 1 else None
        return scipy_signal.filtfilt(b, a, ecg, axis=axis, padlen=padlen)

    def normalize(self, ecg: np.ndarray) -> np.ndarray:
        """Normalize ECG per lead with proper parenthesis operator precedence."""
        axis = 0 if (ecg.ndim == 2 and ecg.shape[0] > ecg.shape[1]) else -1
        mean = np.mean(ecg, axis=axis, keepdims=True)
        std = np.std(ecg, axis=axis, keepdims=True)
        return (ecg - mean) / (std + 1e-8)

    def preprocess_ecg(self, ecg: Any) -> Optional[np.ndarray]:
        """Validate, filter, and normalize ECG."""
        if ecg is None:
            return None
        ecg_arr = np.asarray(ecg, dtype=np.float32)
        if not np.isfinite(ecg_arr).all() or ecg_arr.size == 0:
            return None
        filtered = self.bandpass_filter(ecg_arr, lowcut=0.5, highcut=40.0, fs=250)
        normalized = self.normalize(filtered)
        return normalized
