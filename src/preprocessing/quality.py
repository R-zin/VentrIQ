from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal as scipy_signal

from src.data.echonext import (
    EXPECTED_TIME_STEPS,
    NUM_LEADS,
    SAMPLING_RATE_HZ,
    STANDARD_12_LEADS,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Detailed quality assessment report for a single 12-lead ECG recording."""

    is_valid: bool = True
    is_usable: bool = True
    has_nans: bool = False
    has_infs: bool = False
    is_empty: bool = False
    shape: Tuple[int, ...] = field(default_factory=tuple)
    flatline_leads: List[str] = field(default_factory=list)
    clipped_leads: List[str] = field(default_factory=list)
    noisy_leads: List[str] = field(default_factory=list)
    lead_snr_db: Dict[str, float] = field(default_factory=dict)
    overall_snr_db: float = 0.0
    rejection_reasons: List[str] = field(default_factory=list)
    lead_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert quality report to dictionary."""
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d


class ECGQualityEvaluator:
    """Evaluates 12-lead ECG signal quality, detecting lead faults, noise, and clipping.

    Ensures that bad data is flagged with structured metadata rather than
    silently discarded or causing unhandled runtime failures.
    """

    def __init__(
        self,
        sampling_rate_hz: int = SAMPLING_RATE_HZ,
        expected_time_steps: int = EXPECTED_TIME_STEPS,
        num_leads: int = NUM_LEADS,
        lead_names: Optional[List[str]] = None,
        flatline_variance_threshold: float = 1.0e-5,
        flatline_window_sec: float = 0.5,
        clipping_consecutive_samples: int = 8,
        min_acceptable_snr_db: float = -3.0,
        max_allowed_faulty_leads: int = 2,
    ) -> None:
        self.fs = sampling_rate_hz
        self.expected_steps = expected_time_steps
        self.num_leads = num_leads
        self.lead_names = lead_names or STANDARD_12_LEADS
        self.flatline_var_thresh = flatline_variance_threshold
        self.window_samples = max(2, int(flatline_window_sec * self.fs))
        self.clipping_consecutive = clipping_consecutive_samples
        self.min_snr_db = min_acceptable_snr_db
        self.max_faulty_leads = max_allowed_faulty_leads

    def validate_shape_and_values(
        self, ecg: Any
    ) -> Tuple[bool, List[str], Optional[np.ndarray]]:
        """Validate array dimensions, types, and finiteness.

        Returns:
            Tuple of (is_valid, rejection_reasons, normalized_array_or_None).
            Normalized array is oriented as (samples, leads).
        """
        rejection_reasons: List[str] = []

        if ecg is None:
            return False, ["Input is None"], None

        try:
            arr = np.asarray(ecg, dtype=np.float32)
        except Exception as e:
            return False, [f"Failed to convert input to float32 array: {e}"], None

        if arr.size == 0:
            return False, ["Input array is empty (size 0)"], None

        if arr.ndim != 2:
            return (
                False,
                [f"Expected 2D array (samples, leads), got {arr.ndim}D array with shape {arr.shape}"],
                None,
            )

        # Standardize orientation to (samples, leads) = (2500, 12)
        if arr.shape == (self.num_leads, self.expected_steps):
            arr = arr.T
        elif arr.shape != (self.expected_steps, self.num_leads):
            rejection_reasons.append(
                f"Unexpected dimensions: shape is {arr.shape}, expected "
                f"({self.expected_steps}, {self.num_leads}) or ({self.num_leads}, {self.expected_steps})"
            )
            return False, rejection_reasons, None

        if np.isnan(arr).any():
            rejection_reasons.append("Array contains NaN (missing) values")

        if np.isinf(arr).any():
            rejection_reasons.append("Array contains infinite (+/-inf) values")

        is_valid = len(rejection_reasons) == 0
        return is_valid, rejection_reasons, arr

    def detect_flatlines(self, ecg: np.ndarray) -> List[str]:
        """Detect leads that are flatlined or disconnected."""
        flatline_leads: List[str] = []
        n_samples, n_leads = ecg.shape

        for lead_idx in range(n_leads):
            signal_lead = ecg[:, lead_idx]
            lead_name = (
                self.lead_names[lead_idx]
                if lead_idx < len(self.lead_names)
                else f"Lead_{lead_idx}"
            )

            # Check full-lead variance
            lead_var = float(np.var(signal_lead))
            if lead_var < self.flatline_var_thresh:
                flatline_leads.append(lead_name)
                continue

            # Check windowed rolling variance for partial flatlines (e.g. loose electrode)
            # Use stride to efficiently check across recording
            num_windows = n_samples // self.window_samples
            if num_windows > 0:
                reshaped = signal_lead[: num_windows * self.window_samples].reshape(
                    num_windows, self.window_samples
                )
                window_vars = np.var(reshaped, axis=1)
                # If more than 60% of windows are flatline, flag lead
                if np.mean(window_vars < self.flatline_var_thresh) > 0.60:
                    flatline_leads.append(lead_name)

        return flatline_leads

    def detect_clipping(self, ecg: np.ndarray) -> List[str]:
        """Detect leads exhibiting saturation or rail clipping."""
        clipped_leads: List[str] = []
        n_samples, n_leads = ecg.shape

        for lead_idx in range(n_leads):
            signal_lead = ecg[:, lead_idx]
            lead_name = (
                self.lead_names[lead_idx]
                if lead_idx < len(self.lead_names)
                else f"Lead_{lead_idx}"
            )

            # Check if values sit consecutively at the extreme min or max
            lead_min = float(np.min(signal_lead))
            lead_max = float(np.max(signal_lead))
            lead_range = lead_max - lead_min

            if lead_range < 1e-6:
                # Flatline handled separately
                continue

            # Test for repeated exact values at edges
            diffs = np.diff(signal_lead)
            zero_diffs = np.isclose(diffs, 0.0, atol=1e-5)

            # Find consecutive flat runs
            is_at_extremes = np.isclose(signal_lead[:-1], lead_min, atol=1e-3 * lead_range) | \
                             np.isclose(signal_lead[:-1], lead_max, atol=1e-3 * lead_range)
            clipped_samples = zero_diffs & is_at_extremes

            # Run-length check
            if np.any(clipped_samples):
                # Count longest sequence of clipped samples
                run_length = 0
                max_run = 0
                for is_clipped in clipped_samples:
                    if is_clipped:
                        run_length += 1
                        if run_length > max_run:
                            max_run = run_length
                    else:
                        run_length = 0

                if max_run >= self.clipping_consecutive:
                    clipped_leads.append(lead_name)

        return clipped_leads

    def compute_lead_snr(self, ecg: np.ndarray) -> Tuple[Dict[str, float], float, List[str]]:
        """Estimate Signal-to-Noise Ratio (SNR) per lead and overall.

        Compares in-band cardiac power (0.5 Hz - 40 Hz) against out-of-band/high-frequency noise.
        """
        lead_snr: Dict[str, float] = {}
        noisy_leads: List[str] = []
        n_samples, n_leads = ecg.shape

        for lead_idx in range(n_leads):
            signal_lead = ecg[:, lead_idx]
            lead_name = (
                self.lead_names[lead_idx]
                if lead_idx < len(self.lead_names)
                else f"Lead_{lead_idx}"
            )

            # Detrend for spectral analysis
            lead_detrended = scipy_signal.detrend(signal_lead)

            # Compute power spectrum via Welch
            nperseg = min(n_samples, int(2.0 * self.fs))
            freqs, psd = scipy_signal.welch(lead_detrended, fs=self.fs, nperseg=nperseg)

            # Cardiac band: 0.5 - 40 Hz (primary QRS, T-wave energy)
            cardiac_mask = (freqs >= 0.5) & (freqs <= 40.0)
            # High-frequency noise band: 45 - Nyquist
            noise_mask = (freqs >= 45.0)

            power_signal = float(np.sum(psd[cardiac_mask]))
            power_noise = float(np.sum(psd[noise_mask])) + 1e-12

            ratio = max(power_signal / power_noise, 1e-6)
            snr_db = float(10.0 * np.log10(ratio))
            lead_snr[lead_name] = round(snr_db, 2)

            if snr_db < self.min_snr_db:
                noisy_leads.append(lead_name)

        overall_snr = float(np.mean(list(lead_snr.values()))) if lead_snr else 0.0
        return lead_snr, round(overall_snr, 2), noisy_leads

    def compute_lead_metrics(self, ecg: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute summary statistical metrics per lead."""
        metrics: Dict[str, Dict[str, float]] = {}
        n_samples, n_leads = ecg.shape

        for lead_idx in range(n_leads):
            sig = ecg[:, lead_idx]
            lead_name = (
                self.lead_names[lead_idx]
                if lead_idx < len(self.lead_names)
                else f"Lead_{lead_idx}"
            )
            metrics[lead_name] = {
                "mean": round(float(np.mean(sig)), 4),
                "std": round(float(np.std(sig)), 4),
                "min": round(float(np.min(sig)), 4),
                "max": round(float(np.max(sig)), 4),
                "variance": round(float(np.var(sig)), 6),
            }
        return metrics

    def evaluate(
        self, ecg: Any, lead_names: Optional[List[str]] = None
    ) -> QualityReport:
        """Perform comprehensive quality evaluation of an ECG array.

        Args:
            ecg: Input array of shape (2500, 12) or (12, 2500).
            lead_names: Optional custom list of 12 lead names.

        Returns:
            QualityReport with validation status and lead-level quality flags.
        """
        if lead_names is not None:
            self.lead_names = lead_names

        is_valid, rejection_reasons, arr = self.validate_shape_and_values(ecg)

        if not is_valid or arr is None:
            report = QualityReport(
                is_valid=False,
                is_usable=False,
                has_nans=bool(ecg is not None and np.isnan(np.asarray(ecg)).any())
                if isinstance(ecg, (np.ndarray, list))
                else False,
                has_infs=bool(ecg is not None and np.isinf(np.asarray(ecg)).any())
                if isinstance(ecg, (np.ndarray, list))
                else False,
                is_empty=bool(ecg is not None and getattr(ecg, "size", 0) == 0)
                if isinstance(ecg, (np.ndarray, list))
                else (ecg is None),
                shape=getattr(ecg, "shape", ()) if hasattr(ecg, "shape") else (),
                rejection_reasons=rejection_reasons,
            )
            return report

        # Detect faults
        flatline_leads = self.detect_flatlines(arr)
        clipped_leads = self.detect_clipping(arr)
        lead_snr, overall_snr, noisy_leads = self.compute_lead_snr(arr)
        lead_metrics = self.compute_lead_metrics(arr)

        # Determine usability based on maximum allowable faulty leads
        total_faulty = len(set(flatline_leads).union(clipped_leads))
        is_usable = True

        if total_faulty > self.max_faulty_leads:
            is_usable = False
            rejection_reasons.append(
                f"Too many faulty leads ({total_faulty} > max allowed {self.max_faulty_leads})"
            )

        if len(flatline_leads) >= self.num_leads:
            is_usable = False
            rejection_reasons.append("Entire ECG recording is flatlined")

        report = QualityReport(
            is_valid=True,
            is_usable=is_usable,
            has_nans=False,
            has_infs=False,
            is_empty=False,
            shape=arr.shape,
            flatline_leads=flatline_leads,
            clipped_leads=clipped_leads,
            noisy_leads=noisy_leads,
            lead_snr_db=lead_snr,
            overall_snr_db=overall_snr,
            rejection_reasons=rejection_reasons,
            lead_metrics=lead_metrics,
        )

        return report
