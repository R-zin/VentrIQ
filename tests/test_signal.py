from __future__ import annotations

import numpy as np
import pytest

from src.data.echonext import EXPECTED_TIME_STEPS, NUM_LEADS, SAMPLING_RATE_HZ
from src.preprocessing.signal import (
    ECGSignalPreprocessor,
    ProcessedECG,
    signal_preprocessing,
)


@pytest.fixture
def synthetic_ecg() -> np.ndarray:
    """Generate a clean synthetic 12-lead ECG array of shape (2500, 12)."""
    t = np.linspace(0, 10.0, EXPECTED_TIME_STEPS, endpoint=False)
    ecg = np.zeros((EXPECTED_TIME_STEPS, NUM_LEADS), dtype=np.float32)
    hr = 60.0  # 1 Hz fundamental
    for lead in range(NUM_LEADS):
        # QRS-like peaks + baseline
        qrs = np.sin(2 * np.pi * hr / 60.0 * t) ** 15
        ecg[:, lead] = (qrs + 0.05 * np.sin(2 * np.pi * 0.2 * t)).astype(np.float32)
    return ecg


def test_preprocessor_default_preserves_waveform(synthetic_ecg):
    """Test that default preprocessor keeps original EchoNext waveform intact."""
    preprocessor = ECGSignalPreprocessor()
    assert preprocessor.apply_filter is False
    assert preprocessor.apply_normalization is False

    res = preprocessor.process(synthetic_ecg)
    assert isinstance(res, ProcessedECG)
    assert res.quality_report.is_valid is True
    assert res.quality_report.is_usable is True
    # Verify original and final are identical when filtering and normalization are bypassed
    np.testing.assert_allclose(res.original, res.final)
    assert res.filtered is None
    assert res.normalized is None


def test_preprocessor_with_filtering(synthetic_ecg):
    """Test configurable zero-phase filtering and ensure 0 ms peak delay."""
    preprocessor = ECGSignalPreprocessor(apply_filter=True, lowcut_hz=0.5, highcut_hz=40.0)
    res = preprocessor.process(synthetic_ecg)

    assert res.filtered is not None
    assert res.filtered.shape == (EXPECTED_TIME_STEPS, NUM_LEADS)

    # Check zero-phase property: peak positions must match within 1 sample (0 ms shift)
    lead0_orig = res.original[:, 0]
    lead0_filt = res.filtered[:, 0]

    peak_orig = int(np.argmax(lead0_orig[:250]))
    peak_filt = int(np.argmax(lead0_filt[:250]))
    assert abs(peak_orig - peak_filt) <= 1


def test_preprocessor_with_notch(synthetic_ecg):
    """Test notch filtering for 50 Hz powerline interference."""
    t = np.linspace(0, 10.0, EXPECTED_TIME_STEPS, endpoint=False)
    # Add strong 50 Hz hum
    noisy_ecg = synthetic_ecg + (0.5 * np.sin(2 * np.pi * 50.0 * t))[:, None]

    preprocessor = ECGSignalPreprocessor(
        apply_filter=True,
        apply_notch=True,
        notch_frequencies_hz=[50.0],
    )
    res = preprocessor.process(noisy_ecg)
    assert res.filtered is not None

    # Power at 50 Hz must be substantially attenuated
    fft_orig = np.abs(np.fft.rfft(noisy_ecg[:, 0]))
    fft_filt = np.abs(np.fft.rfft(res.filtered[:, 0]))
    freqs = np.fft.rfftfreq(EXPECTED_TIME_STEPS, 1.0 / SAMPLING_RATE_HZ)
    idx_50 = np.argmin(np.abs(freqs - 50.0))

    assert fft_filt[idx_50] < 0.2 * fft_orig[idx_50]


def test_normalization_methods(synthetic_ecg):
    """Test lead_zscore, minmax, and robust normalization modes."""
    # Test lead_zscore
    prep_zscore = ECGSignalPreprocessor(
        apply_normalization=True, normalization_method="lead_zscore"
    )
    res_z = prep_zscore.process(synthetic_ecg)
    assert res_z.normalized is not None
    # Each lead should have approximately mean 0 and std 1
    means = np.mean(res_z.normalized, axis=0)
    stds = np.std(res_z.normalized, axis=0)
    np.testing.assert_allclose(means, np.zeros(NUM_LEADS), atol=1e-3)
    np.testing.assert_allclose(stds, np.ones(NUM_LEADS), atol=1e-3)

    # Test minmax
    prep_minmax = ECGSignalPreprocessor(apply_normalization=True, normalization_method="minmax")
    res_m = prep_minmax.process(synthetic_ecg)
    assert res_m.normalized is not None
    assert np.all(res_m.normalized >= -1.0 - 1e-4)
    assert np.all(res_m.normalized <= 1.0 + 1e-4)


def test_short_signals_safety():
    """Test that preprocessor handles short signals without raising padding errors."""
    short_ecg = np.random.randn(30, NUM_LEADS).astype(np.float32)
    preprocessor = ECGSignalPreprocessor(apply_filter=True)
    # Should not raise exception
    filtered = preprocessor.apply_zero_phase_bandpass(short_ecg)
    assert filtered.shape == (30, NUM_LEADS)


def test_invalid_input_handling():
    """Test explicit handling of None, NaNs, Infs, and incorrect dimensions."""
    preprocessor = ECGSignalPreprocessor()

    # None
    res_none = preprocessor.process(None)
    assert res_none.quality_report.is_valid is False
    assert res_none.final.size == 0

    # NaNs
    nan_ecg = np.zeros((EXPECTED_TIME_STEPS, NUM_LEADS), dtype=np.float32)
    nan_ecg[10, 2] = np.nan
    res_nan = preprocessor.process(nan_ecg)
    assert res_nan.quality_report.is_valid is False
    assert "Array contains NaN (missing) values" in res_nan.quality_report.rejection_reasons

    # Infs
    inf_ecg = np.zeros((EXPECTED_TIME_STEPS, NUM_LEADS), dtype=np.float32)
    inf_ecg[10, 2] = np.inf
    res_inf = preprocessor.process(inf_ecg)
    assert res_inf.quality_report.is_valid is False
    assert "Array contains infinite (+/-inf) values" in res_inf.quality_report.rejection_reasons

    # Wrong dimensions
    bad_dim = np.zeros((100, 5), dtype=np.float32)
    res_dim = preprocessor.process(bad_dim)
    assert res_dim.quality_report.is_valid is False


def test_channels_first_option(synthetic_ecg):
    """Test channels-first output orientation (12, 2500)."""
    preprocessor = ECGSignalPreprocessor(apply_filter=True, apply_normalization=True)
    res = preprocessor.process(synthetic_ecg, channels_first=True)
    assert res.final.shape == (NUM_LEADS, EXPECTED_TIME_STEPS)
    assert res.original.shape == (NUM_LEADS, EXPECTED_TIME_STEPS)


def test_legacy_signal_preprocessing_compatibility(synthetic_ecg):
    """Test that legacy signal_preprocessing class works with bug fixes applied."""
    legacy = signal_preprocessing()
    # Filter
    filtered = legacy.bandpass_filter(synthetic_ecg.T, lowcut=0.5, highcut=40.0, fs=250)
    assert filtered.shape == (NUM_LEADS, EXPECTED_TIME_STEPS)

    # Normalize
    normalized = legacy.normalize(filtered)
    assert normalized.shape == (NUM_LEADS, EXPECTED_TIME_STEPS)

    # Preprocess
    res = legacy.preprocess_ecg(synthetic_ecg)
    assert res is not None
    assert res.shape == (EXPECTED_TIME_STEPS, NUM_LEADS)

    # Preprocess invalid
    assert legacy.preprocess_ecg(None) is None
    bad = synthetic_ecg.copy()
    bad[5, 5] = np.nan
    assert legacy.preprocess_ecg(bad) is None
