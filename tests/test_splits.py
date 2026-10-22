from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import ECGRecord
from src.data.splits import (
    PatientGroupKFold,
    PatientLeakageError,
    SplitVerificationReport,
    StratifiedPatientKFold,
    create_patient_kfold_splits,
    verify_patient_isolation,
)


@pytest.fixture
def mock_clean_split_df() -> pd.DataFrame:
    """Fixture with clean, patient-isolated splits (no leakage)."""
    return pd.DataFrame({
        "patient_key": [
            "P001", "P001", "P002", "P003",  # train: 4 records, 3 patients
            "P004", "P005",                  # val: 2 records, 2 patients
            "P006", "P007", "P007",          # test: 3 records, 2 patients
        ],
        "split": [
            "train", "train", "train", "train",
            "val", "val",
            "test", "test", "test",
        ],
        "lvef_lte_45_flag": [0, 0, 1, 0, 1, 0, 1, 0, 0],
    })


@pytest.fixture
def mock_leaking_split_df() -> pd.DataFrame:
    """Fixture with deliberate patient leakage across train and val/test splits."""
    return pd.DataFrame({
        "patient_key": [
            "P001", "P002", "P003",  # train
            "P002", "P004",          # val: P002 leaked from train!
            "P001", "P005",          # test: P001 leaked from train!
        ],
        "split": [
            "train", "train", "train",
            "val", "val",
            "test", "test",
        ],
    })


def test_verify_patient_isolation_clean(mock_clean_split_df):
    """Test verification passes with 0 leakage on properly partitioned data."""
    report = verify_patient_isolation(mock_clean_split_df, raise_on_leak=True)
    assert isinstance(report, SplitVerificationReport)
    assert report.is_valid is True
    assert report.leakage_detected is False
    assert report.total_records == 9
    assert report.total_unique_patients == 7
    assert report.record_counts_per_split["train"] == 4
    assert report.patient_counts_per_split["train"] == 3
    assert report.record_counts_per_split["val"] == 2
    assert report.record_counts_per_split["test"] == 3
    assert len(report.overlap_counts) == 0
    assert "PASSED (Zero Leakage)" in report.summary()


def test_verify_patient_isolation_detects_leakage(mock_leaking_split_df):
    """Test detection of patient leakage and raising of PatientLeakageError."""
    with pytest.raises(PatientLeakageError) as exc_info:
        verify_patient_isolation(mock_leaking_split_df, raise_on_leak=True)
    assert "Patient leakage detected" in str(exc_info.value)

    # When raise_on_leak=False, returns detailed audit report
    report = verify_patient_isolation(mock_leaking_split_df, raise_on_leak=False)
    assert report.is_valid is False
    assert report.leakage_detected is True
    assert "train_vs_val" in report.overlapping_patients
    assert "P002" in report.overlapping_patients["train_vs_val"]
    assert "train_vs_test" in report.overlapping_patients
    assert "P001" in report.overlapping_patients["train_vs_test"]
    assert "FAILED (Patient Leakage Detected)" in report.summary()


def test_verify_patient_isolation_on_ecg_records():
    """Test verification on Sequence of ECGRecord objects."""
    records = [
        ECGRecord("rec1", "P1", "train", ("train", 0)),
        ECGRecord("rec2", "P2", "train", ("train", 1)),
        ECGRecord("rec3", "P3", "val", ("val", 0)),
        ECGRecord("rec4", "P4", "test", ("test", 0)),
    ]
    report = verify_patient_isolation(records)
    assert report.is_valid is True
    assert report.total_records == 4
    assert report.total_unique_patients == 4


def test_patient_group_kfold_zero_leakage():
    """Verify that PatientGroupKFold guarantees 0 patient overlap across all folds."""
    np.random.seed(42)
    # 50 records belonging to 15 patients
    patients = [f"P{i:03d}" for i in np.random.choice(15, size=50)]
    X = np.arange(50)

    n_splits = 4
    splitter = PatientGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    assert splitter.get_n_splits() == n_splits

    val_records_seen = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, groups=patients)):
        # 1. Train and val sets must be disjoint
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(train_idx) + len(val_idx) == 50

        # 2. Patient sets must be strictly disjoint (Zero Leakage)
        train_pats = set(patients[i] for i in train_idx)
        val_pats = set(patients[i] for i in val_idx)
        assert len(train_pats.intersection(val_pats)) == 0, f"Patient leakage in fold {fold}!"

        val_records_seen.extend(val_idx.tolist())

    # 3. Every sample must be in validation set of exactly one fold
    assert sorted(val_records_seen) == list(range(50))


def test_patient_group_kfold_validation_errors():
    """Test input validations for PatientGroupKFold."""
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        PatientGroupKFold(n_splits=1)

    splitter = PatientGroupKFold(n_splits=5)
    # Fewer unique patients than n_splits
    X = [0, 1, 2, 3]
    groups = ["P1", "P1", "P2", "P2"]
    with pytest.raises(ValueError, match="cannot be smaller than n_splits"):
        list(splitter.split(X, groups=groups))

    # Length mismatch
    with pytest.raises(ValueError, match="Length of groups"):
        list(splitter.split(X, groups=["P1", "P2"]))


def test_stratified_patient_kfold():
    """Verify StratifiedPatientKFold preserves patient grouping with zero leakage."""
    np.random.seed(42)
    patients = [f"P{i:02d}" for i in range(20)]
    # Duplicate some patients (multiple ECGs per patient)
    all_patients = patients + patients[:10]  # 30 records
    labels = [1 if int(p[1:]) % 2 == 0 else 0 for p in all_patients]

    splitter = StratifiedPatientKFold(n_splits=3, shuffle=True, random_state=42)
    val_records_seen = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(all_patients, y=labels, groups=all_patients)):
        train_pats = set(all_patients[i] for i in train_idx)
        val_pats = set(all_patients[i] for i in val_idx)
        assert len(train_pats.intersection(val_pats)) == 0, f"Patient leak in fold {fold}"
        val_records_seen.extend(val_idx.tolist())

    assert sorted(val_records_seen) == list(range(len(all_patients)))


def test_create_patient_kfold_splits_helper(mock_clean_split_df):
    """Test create_patient_kfold_splits helper function."""
    splits = create_patient_kfold_splits(mock_clean_split_df, n_splits=3, seed=123)
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        train_pats = set(mock_clean_split_df.iloc[train_idx]["patient_key"])
        val_pats = set(mock_clean_split_df.iloc[val_idx]["patient_key"])
        assert len(train_pats.intersection(val_pats)) == 0
