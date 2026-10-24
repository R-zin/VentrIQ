"""src/data/splits.py - Patient-Level Split Verification & Preservation.

This module provides tools to verify zero patient leakage across dataset partitions
and to generate rigorous, leakage-free patient-grouped cross-validation folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import logging
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PatientLeakageError(ValueError):
    """Exception raised when patient identifiers overlap across dataset splits."""
    pass


@dataclass
class SplitVerificationReport:
    """Detailed audit report on patient isolation across dataset splits."""

    is_valid: bool
    leakage_detected: bool
    patient_counts_per_split: Dict[str, int]
    record_counts_per_split: Dict[str, int]
    overlapping_patients: Dict[str, List[str]] = field(default_factory=dict)
    overlap_counts: Dict[str, int] = field(default_factory=dict)
    total_records: int = 0
    total_unique_patients: int = 0

    def summary(self) -> str:
        """Return human-readable summary string of split verification."""
        status = "PASSED (Zero Leakage)" if self.is_valid else "FAILED (Patient Leakage Detected)"
        lines = [
            f"=== VentriQ Split Verification: {status} ===",
            f"Total Records: {self.total_records:,}",
            f"Total Unique Patients: {self.total_unique_patients:,}",
            "Splits Breakdown:",
        ]
        for split, rec_cnt in self.record_counts_per_split.items():
            pat_cnt = self.patient_counts_per_split.get(split, 0)
            lines.append(f"  - {split}: {rec_cnt:,} records, {pat_cnt:,} patients")

        if self.leakage_detected:
            lines.append("Leakage Details:")
            for pair, count in self.overlap_counts.items():
                pats = self.overlapping_patients.get(pair, [])
                sample = ", ".join(pats[:5])
                suffix = "..." if len(pats) > 5 else ""
                lines.append(f"  - {pair}: {count:,} shared patients ({sample}{suffix})")
        else:
            lines.append("Integrity: All partitions are strictly patient-disjoint.")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON logging."""
        return {
            "is_valid": self.is_valid,
            "leakage_detected": self.leakage_detected,
            "total_records": self.total_records,
            "total_unique_patients": self.total_unique_patients,
            "record_counts_per_split": self.record_counts_per_split,
            "patient_counts_per_split": self.patient_counts_per_split,
            "overlap_counts": self.overlap_counts,
            "overlapping_patients": self.overlapping_patients,
        }


def _extract_patient_and_split(
    data: Any,
    patient_col: str = "patient_key",
    split_col: str = "split",
) -> Tuple[List[str], List[str]]:
    """Extract parallel lists of patient IDs and split names from diverse data inputs."""
    # Case 1: pandas DataFrame
    if isinstance(data, pd.DataFrame):
        p_col = patient_col if patient_col in data.columns else "patient_id"
        if p_col not in data.columns:
            raise KeyError(f"Patient column '{patient_col}' not found in DataFrame.")
        if split_col not in data.columns:
            raise KeyError(f"Split column '{split_col}' not found in DataFrame.")
        patients = data[p_col].astype(str).tolist()
        splits = data[split_col].astype(str).tolist()
        return patients, splits

    # Case 2: EchoNextDataset instance
    if hasattr(data, "metadata") and isinstance(data.metadata, pd.DataFrame):
        return _extract_patient_and_split(data.metadata, getattr(data, "patient_col", patient_col), split_col)

    # Case 3: Sequence of ECGRecord or dict-like objects
    if isinstance(data, Sequence) and len(data) > 0:
        first = data[0]
        if hasattr(first, "patient_key") and hasattr(first, "split"):
            patients = [str(getattr(item, "patient_key")) for item in data]
            splits = [str(getattr(item, "split")) for item in data]
            return patients, splits
        if isinstance(first, dict) and patient_col in first and split_col in first:
            patients = [str(item[patient_col]) for item in data]
            splits = [str(item[split_col]) for item in data]
            return patients, splits

    # Case 4: Mapping of split name to list of patient IDs
    if isinstance(data, dict):
        patients = []
        splits = []
        for sp, pat_list in data.items():
            for p in pat_list:
                patients.append(str(p))
                splits.append(str(sp))
        return patients, splits

    raise TypeError(
        f"Unsupported data type for split extraction: {type(data)}. "
        "Expected pd.DataFrame, EchoNextDataset, Sequence[ECGRecord], or Dict[str, Sequence]."
    )


def verify_patient_isolation(
    data: Any,
    patient_col: str = "patient_key",
    split_col: str = "split",
    raise_on_leak: bool = True,
) -> SplitVerificationReport:
    """Verify that patient identifiers do not cross split partitions.

    Args:
        data: DataFrame, EchoNextDataset, Sequence of ECGRecords, or dict of split -> patients.
        patient_col: Name of patient identifier column/field.
        split_col: Name of split column/field.
        raise_on_leak: If True, raises PatientLeakageError if leakage is found.

    Returns:
        SplitVerificationReport with detailed breakdown.

    Raises:
        PatientLeakageError: If raise_on_leak is True and patient overlap is detected.
    """
    patients, splits = _extract_patient_and_split(data, patient_col=patient_col, split_col=split_col)

    split_to_patients: Dict[str, Set[str]] = {}
    split_to_record_count: Dict[str, int] = {}

    for p, s in zip(patients, splits):
        if s not in split_to_patients:
            split_to_patients[s] = set()
            split_to_record_count[s] = 0
        split_to_patients[s].add(p)
        split_to_record_count[s] += 1

    unique_splits = list(split_to_patients.keys())
    patient_counts = {s: len(pats) for s, pats in split_to_patients.items()}

    leakage_detected = False
    overlapping_patients: Dict[str, List[str]] = {}
    overlap_counts: Dict[str, int] = {}

    # Check pairwise split intersections
    for s1, s2 in itertools.combinations(unique_splits, 2):
        overlap = split_to_patients[s1].intersection(split_to_patients[s2])
        if overlap:
            leakage_detected = True
            pair_key = f"{s1}_vs_{s2}"
            sorted_overlap = sorted(list(overlap))
            overlapping_patients[pair_key] = sorted_overlap
            overlap_counts[pair_key] = len(sorted_overlap)

    all_patients = set(patients)
    report = SplitVerificationReport(
        is_valid=not leakage_detected,
        leakage_detected=leakage_detected,
        patient_counts_per_split=patient_counts,
        record_counts_per_split=split_to_record_count,
        overlapping_patients=overlapping_patients,
        overlap_counts=overlap_counts,
        total_records=len(patients),
        total_unique_patients=len(all_patients),
    )

    if leakage_detected and raise_on_leak:
        err_msg = (
            f"Patient leakage detected between dataset splits! "
            f"Overlapping splits: {list(overlap_counts.keys())}. "
            f"Total leaking pairs: {sum(overlap_counts.values())}. "
            f"Run verify_patient_isolation(..., raise_on_leak=False) to inspect details."
        )
        logger.error(err_msg)
        raise PatientLeakageError(err_msg)

    return report


class PatientGroupKFold:
    """K-Fold cross-validator that guarantees zero patient leakage.

    All records belonging to the same patient identifier are placed exclusively
    into the same fold. The greedy partition algorithm balances the total record
    count per fold as evenly as possible.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        """Initialize PatientGroupKFold.

        Args:
            n_splits: Number of folds (must be >= 2).
            shuffle: Whether to shuffle patient groups before assignment.
            random_state: Random seed for deterministic reproducibility.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Sequence[Any]] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and validation sets.

        Args:
            X: Array-like dataset or collection of records.
            y: Optional targets (unused in standard grouping).
            groups: Group labels (patient identifiers). If None, inferred from X.

        Yields:
            Tuple of (train_indices, val_indices) as numpy int64 arrays.
        """
        n_samples = len(X)
        if groups is None:
            if isinstance(X, pd.DataFrame):
                p_col = "patient_key" if "patient_key" in X.columns else "patient_id"
                if p_col not in X.columns:
                    raise ValueError(f"Could not infer patient groups from DataFrame columns: {list(X.columns)}")
                groups = X[p_col].tolist()
            elif isinstance(X, Sequence) and len(X) > 0 and hasattr(X[0], "patient_key"):
                groups = [getattr(rec, "patient_key") for rec in X]
            else:
                raise ValueError("Patient 'groups' must be provided if X does not expose 'patient_key'.")

        if len(groups) != n_samples:
            raise ValueError(f"Length of groups ({len(groups)}) must match length of X ({n_samples}).")

        # Map each unique patient to its sample indices
        patient_to_indices: Dict[Any, List[int]] = {}
        for idx, p in enumerate(groups):
            if p not in patient_to_indices:
                patient_to_indices[p] = []
            patient_to_indices[p].append(idx)

        unique_patients = list(patient_to_indices.keys())
        if len(unique_patients) < self.n_splits:
            raise ValueError(
                f"Number of unique patients ({len(unique_patients)}) cannot be smaller "
                f"than n_splits ({self.n_splits})."
            )

        rng = np.random.RandomState(self.random_state)
        if self.shuffle:
            # Deterministically permute unique patients
            perm = rng.permutation(len(unique_patients))
            unique_patients = [unique_patients[i] for i in perm]

        # Sort patients by group size descending for greedy bin packing balance
        unique_patients.sort(key=lambda p: len(patient_to_indices[p]), reverse=True)

        # Assign patients to folds using greedy minimum-weight bin packing
        fold_sample_counts = [0] * self.n_splits
        fold_patient_indices: List[List[int]] = [[] for _ in range(self.n_splits)]

        for p in unique_patients:
            best_fold = int(np.argmin(fold_sample_counts))
            indices = patient_to_indices[p]
            fold_patient_indices[best_fold].extend(indices)
            fold_sample_counts[best_fold] += len(indices)

        # Generate (train_idx, val_idx)
        all_indices = np.arange(n_samples)
        for fold_idx in range(self.n_splits):
            val_idx = np.array(sorted(fold_patient_indices[fold_idx]), dtype=np.int64)
            val_set = set(val_idx.tolist())
            train_idx = np.array([i for i in range(n_samples) if i not in val_set], dtype=np.int64)
            yield train_idx, val_idx


class StratifiedPatientKFold:
    """Stratified K-Fold cross-validator with strict zero patient leakage.

    Partitions patient groups such that patient-level binary class distribution
    (e.g., LVEF <= 45% or SHD) is preserved across folds while ensuring all records
    for any given patient reside strictly in one fold.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Sequence[Any],
        groups: Optional[Sequence[Any]] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified patient-disjoint folds."""
        n_samples = len(X)
        if groups is None:
            if isinstance(X, pd.DataFrame):
                p_col = "patient_key" if "patient_key" in X.columns else "patient_id"
                groups = X[p_col].tolist()
            elif isinstance(X, Sequence) and len(X) > 0 and hasattr(X[0], "patient_key"):
                groups = [getattr(rec, "patient_key") for rec in X]
            else:
                raise ValueError("groups must be provided if X does not expose 'patient_key'.")

        if len(groups) != n_samples or len(y) != n_samples:
            raise ValueError("Lengths of X, y, and groups must all match.")

        # Aggregate patient labels: if patient has any positive sample, label is 1, else 0
        patient_to_indices: Dict[Any, List[int]] = {}
        patient_to_labels: Dict[Any, List[Any]] = {}
        for idx, (p, label) in enumerate(zip(groups, y)):
            if p not in patient_to_indices:
                patient_to_indices[p] = []
                patient_to_labels[p] = []
            patient_to_indices[p].append(idx)
            if label is not None and not (isinstance(label, float) and np.isnan(label)):
                patient_to_labels[p].append(label)

        # Resolve primary patient label (mode or max)
        patient_class: Dict[Any, int] = {}
        for p, labs in patient_to_labels.items():
            if labs:
                patient_class[p] = int(max(labs))
            else:
                patient_class[p] = 0

        # Group patients by class
        class_to_patients: Dict[int, List[Any]] = {}
        for p, cls_label in patient_class.items():
            if cls_label not in class_to_patients:
                class_to_patients[cls_label] = []
            class_to_patients[cls_label].append(p)

        rng = np.random.RandomState(self.random_state)
        fold_patient_indices: List[List[int]] = [[] for _ in range(self.n_splits)]
        fold_sample_counts = [0] * self.n_splits

        # Distribute patients from each class separately across folds
        for cls_label, patients in sorted(class_to_patients.items()):
            if self.shuffle:
                perm = rng.permutation(len(patients))
                patients = [patients[i] for i in perm]
            patients.sort(key=lambda p: len(patient_to_indices[p]), reverse=True)

            for p in patients:
                best_fold = int(np.argmin(fold_sample_counts))
                indices = patient_to_indices[p]
                fold_patient_indices[best_fold].extend(indices)
                fold_sample_counts[best_fold] += len(indices)

        for fold_idx in range(self.n_splits):
            val_idx = np.array(sorted(fold_patient_indices[fold_idx]), dtype=np.int64)
            val_set = set(val_idx.tolist())
            train_idx = np.array([i for i in range(n_samples) if i not in val_set], dtype=np.int64)
            yield train_idx, val_idx


def create_patient_kfold_splits(
    data: Any,
    n_splits: int = 5,
    stratify_col: Optional[str] = None,
    patient_col: str = "patient_key",
    shuffle: bool = True,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Helper function to generate k-fold train/validation index pairs.

    Args:
        data: DataFrame or Sequence of records.
        n_splits: Number of folds (default: 5).
        stratify_col: Optional column name for stratified splitting.
        patient_col: Column name identifying patients.
        shuffle: Whether to shuffle before partitioning.
        seed: Random seed.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    if stratify_col is not None and isinstance(data, pd.DataFrame):
        strat_splitter = StratifiedPatientKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=seed
        )
        y = data[stratify_col].tolist()
        groups = data[patient_col].tolist()
        return list(strat_splitter.split(data, y=y, groups=groups))

    group_splitter = PatientGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    return list(group_splitter.split(data))
