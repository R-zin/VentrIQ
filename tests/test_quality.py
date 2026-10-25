from __future__ import annotations

import numpy as np
import pytest

from src.data.echonext import EXPECTED_TIME_STEPS, NUM_LEADS
from src.preprocessing.quality import ECGQualityEvaluator


@pytest.fixture
def clean_ecg() -> np.ndarray:
    """Generate clean synthetic 12-lead ECG."""
    t = np.linspace(0, 10.0, EXPECTED_TIME_STEPS, endpoint=False)
    ecg = np.zeros((EXPECTED_TIME_STEPS, NUM_LEADS), dtype=np.float32)
    for lead in range(NUM_LEADS):
        qrs = np.sin(2 * np.pi * 1.2 * t + lead * 0.1) ** 11
        noise = np.random.normal(0, 0.02, EXPECTED_TIME_STEPS).astype(np.float32)
        ecg[:, lead] = qrs + noise
    return ecg


def test_quality_clean_ecg(clean_ecg):
    """Test that a clean ECG passes all quality checks."""
    evaluator = ECGQualityEvaluator()
    report = evaluator.evaluate(clean_ecg)

    assert report.is_valid is True
    assert report.is_usable is True
    assert len(report.flatline_leads) == 0
    assert len(report.clipped_leads) == 0
    assert report.overall_snr_db > 5.0
    assert len(report.rejection_reasons) == 0


def test_flatline_detection(clean_ecg):
    """Test detection of an artificially flatlined lead."""
    evaluator = ECGQualityEvaluator()
    ecg_with_flatline = clean_ecg.copy()
    # Flatline Lead V1 (index 6)
    ecg_with_flatline[:, 6] = 0.0

    report = evaluator.evaluate(ecg_with_flatline)
    assert report.is_valid is True
    assert "V1" in report.flatline_leads
    # 1 flatline lead is within max allowed (2), so still usable
    assert report.is_usable is True


def test_clipping_detection(clean_ecg):
    """Test detection of lead saturation/clipping."""
    evaluator = ECGQualityEvaluator()
    ecg_with_clipping = clean_ecg.copy()
    # Clip Lead II (index 1) at maximum value for 20 consecutive samples
    max_val = float(np.max(ecg_with_clipping[:, 1]))
    ecg_with_clipping[100:125, 1] = max_val

    report = evaluator.evaluate(ecg_with_clipping)
    assert report.is_valid is True
    assert "II" in report.clipped_leads


def test_excessive_faults_unusable(clean_ecg):
    """Test that an ECG with too many faulty leads is flagged as unusable."""
    evaluator = ECGQualityEvaluator(max_allowed_faulty_leads=2)
    bad_ecg = clean_ecg.copy()
    # Flatline 3 leads: Lead I, II, III
    bad_ecg[:, 0] = 0.0
    bad_ecg[:, 1] = 0.0
    bad_ecg[:, 2] = 0.0

    report = evaluator.evaluate(bad_ecg)
    assert report.is_valid is True
    assert report.is_usable is False
    assert len(report.flatline_leads) == 3
    assert any("Too many faulty leads" in r for r in report.rejection_reasons)


def test_entire_recording_flatline():
    """Test that an all-zero ECG recording is flagged as unusable."""
    evaluator = ECGQualityEvaluator()
    flat_ecg = np.zeros((EXPECTED_TIME_STEPS, NUM_LEADS), dtype=np.float32)

    report = evaluator.evaluate(flat_ecg)
    assert report.is_valid is True
    assert report.is_usable is False
    assert len(report.flatline_leads) == NUM_LEADS
    assert "Entire ECG recording is flatlined" in report.rejection_reasons


def test_nan_and_inf_handling():
    """Test detection of NaNs and infinite values."""
    evaluator = ECGQualityEvaluator()

    # NaNs
    arr_nan = np.ones((EXPECTED_TIME_STEPS, NUM_LEADS), dtype=np.float32)
    arr_nan[50, 3] = np.nan
    rep_nan = evaluator.evaluate(arr_nan)
    assert rep_nan.is_valid is False
    assert rep_nan.has_nans is True

    # Infs
    arr_inf = np.ones((EXPECTED_TIME_STEPS, NUM_LEADS), dtype=np.float32)
    arr_inf[50, 3] = np.inf
    rep_inf = evaluator.evaluate(arr_inf)
    assert rep_inf.is_valid is False
    assert rep_inf.has_infs is True


def test_transposed_input_handling(clean_ecg):
    """Test that evaluator seamlessly accepts (12, 2500) transposed orientation."""
    evaluator = ECGQualityEvaluator()
    transposed = clean_ecg.T  # (12, 2500)
    report = evaluator.evaluate(transposed)
    assert report.is_valid is True
    assert report.shape == (EXPECTED_TIME_STEPS, NUM_LEADS)


def test_report_serialization(clean_ecg):
    """Test converting QualityReport to a JSON-serializable dictionary."""
    evaluator = ECGQualityEvaluator()
    report = evaluator.evaluate(clean_ecg)
    d = report.to_dict()

    assert isinstance(d, dict)
    assert "is_valid" in d
    assert "is_usable" in d
    assert "flatline_leads" in d
    assert "lead_snr_db" in d
    assert isinstance(d["shape"], list)
