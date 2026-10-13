from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.echonext import (
    EXPECTED_TIME_STEPS,
    LVEF_MODERATE_THRESHOLD,
    NUM_LEADS,
    SAMPLING_RATE_HZ,
    STANDARD_12_LEADS,
    EchoNextDataset,
    EchonextDataset,
    create_mock_echonext_dataset,
)


@pytest.fixture(scope="module")
def mock_dataset_dir():
    """Create a temporary mock EchoNext dataset directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = create_mock_echonext_dataset(
            output_dir=tmpdir,
            num_train=30,
            num_val=10,
            num_test=10,
            seed=42,
        )
        yield Path(path)


def test_mock_dataset_creation(mock_dataset_dir):
    """Test that create_mock_echonext_dataset outputs all required files."""
    assert (mock_dataset_dir / "EchoNext_metadata_100k.csv").exists()
    assert (mock_dataset_dir / "EchoNext_train_waveforms.npy").exists()
    assert (mock_dataset_dir / "EchoNext_val_waveforms.npy").exists()
    assert (mock_dataset_dir / "EchoNext_test_waveforms.npy").exists()

    train_mmap = np.load(mock_dataset_dir / "EchoNext_train_waveforms.npy", mmap_mode="r")
    assert train_mmap.shape == (30, 1, EXPECTED_TIME_STEPS, NUM_LEADS)


def test_dataset_initialization(mock_dataset_dir):
    """Test EchoNextDataset loading and length."""
    ds = EchoNextDataset(mock_dataset_dir)
    assert len(ds) == 50  # 30 train + 10 val + 10 test
    assert ds.patient_col == "patient_key"
    assert "split" in ds.metadata.columns


def test_mmap_waveform_retrieval(mock_dataset_dir):
    """Test memory-mapped waveform loading with standard and transposed layouts."""
    ds = EchoNextDataset(mock_dataset_dir)
    wf = ds.get_waveform(0)
    assert isinstance(wf, np.ndarray)
    assert wf.shape == (EXPECTED_TIME_STEPS, NUM_LEADS)
    assert wf.dtype == np.float32
    assert np.isfinite(wf).all()

    # Channels-first retrieval
    wf_cf = ds.get_waveform(0, channels_first=True)
    assert wf_cf.shape == (NUM_LEADS, EXPECTED_TIME_STEPS)


def test_target_extraction(mock_dataset_dir):
    """Test extracting continuous LVEF and binary LVEF flag."""
    ds = EchoNextDataset(mock_dataset_dir)
    lvef = ds.get_lvef(0)
    assert isinstance(lvef, float)
    assert 10.0 <= lvef <= 90.0

    flag = ds.get_lvef_flag(0)
    assert flag in (0, 1)
    if lvef <= LVEF_MODERATE_THRESHOLD:
        assert flag == 1
    else:
        assert flag == 0


def test_split_filtering(mock_dataset_dir):
    """Test filtering dataset to specific splits."""
    ds_train = EchoNextDataset(mock_dataset_dir, split="train")
    ds_val = EchoNextDataset(mock_dataset_dir, split="val")
    ds_test = EchoNextDataset(mock_dataset_dir, split="test")

    assert len(ds_train) == 30
    assert len(ds_val) == 10
    assert len(ds_test) == 10

    # Ensure waveform retrieval works smoothly with filtered splits
    wf = ds_val.get_waveform(0)
    assert wf.shape == (EXPECTED_TIME_STEPS, NUM_LEADS)


def test_patient_isolation_verification(mock_dataset_dir):
    """Test verifying absence of patient leakage across splits."""
    ds = EchoNextDataset(mock_dataset_dir)
    res = ds.verify_patient_isolation()

    assert res["has_split_column"] is True
    assert res["is_leak_free"] is True
    assert res["total_overlapping_patients"] == 0


def test_patient_isolation_detects_leak():
    """Test that patient overlap verification catches injected leakage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create metadata with an intentional patient leak between train and val
        df = pd.DataFrame([
            {"patient_key": "PAT001", "split": "train", "lvef_value": 55.0},
            {"patient_key": "PAT002", "split": "train", "lvef_value": 60.0},
            {"patient_key": "PAT001", "split": "val", "lvef_value": 55.0},  # LEAK!
            {"patient_key": "PAT003", "split": "test", "lvef_value": 40.0},
        ])
        df.to_csv(tmp_path / "EchoNext_metadata_100k.csv", index=False)
        ds = EchoNextDataset(tmp_path)
        res = ds.verify_patient_isolation()

        assert res["is_leak_free"] is False
        assert res["total_overlapping_patients"] == 1
        assert res["overlaps"]["train_vs_val"] == 1


def test_waveform_alignment_verification(mock_dataset_dir):
    """Test waveform to metadata alignment checker."""
    ds = EchoNextDataset(mock_dataset_dir)
    res = ds.verify_waveform_alignment()
    assert res["all_aligned"] is True
    assert res["splits"]["train"]["aligned"] is True
    assert res["splits"]["train"]["metadata_rows"] == 30
    assert res["splits"]["train"]["waveform_rows"] == 30


def test_summary_statistics(mock_dataset_dir):
    """Test exploration summary generation."""
    ds = EchoNextDataset(mock_dataset_dir)
    summary = ds.summarize()

    assert summary["total_records"] == 50
    assert summary["sampling_frequency_hz"] == SAMPLING_RATE_HZ
    assert summary["duration_sec"] == 10.0
    assert len(summary["lead_names"]) == NUM_LEADS
    assert summary["lead_names"] == STANDARD_12_LEADS
    assert "lvef_statistics" in summary
    assert summary["lvef_statistics"]["count"] == 50


def test_error_handling(mock_dataset_dir):
    """Test proper exception handling for invalid indices and non-existent folders."""
    ds = EchoNextDataset(mock_dataset_dir)

    with pytest.raises(IndexError):
        ds.get_waveform(9999)

    with pytest.raises(IndexError):
        ds.get_lvef(-1)

    with pytest.raises(FileNotFoundError):
        EchoNextDataset("/non/existent/directory/path/12345")


def test_backward_compatibility(mock_dataset_dir):
    """Test that original class name and methods function as expected."""
    legacy_ds = EchonextDataset(mock_dataset_dir)
    assert len(legacy_ds) == 50

    wf = legacy_ds.get_train_ecg(0)
    assert wf.shape == (EXPECTED_TIME_STEPS, NUM_LEADS)

    lvef = legacy_ds.get_lvef(0)
    assert isinstance(lvef, float)

    train_p, val_p, test_p = legacy_ds.train_spliting_patients(legacy_ds.metadata, seed=123)
    assert len(train_p) > 0
    assert len(val_p) > 0
    assert len(test_p) > 0
    # Verify no patient overlap in generated custom split
    train_pats = set(train_p["patient_key"])
    val_pats = set(val_p["patient_key"])
    test_pats = set(test_p["patient_key"])
    assert len(train_pats.intersection(val_pats)) == 0
    assert len(train_pats.intersection(test_pats)) == 0
    assert len(val_pats.intersection(test_pats)) == 0
