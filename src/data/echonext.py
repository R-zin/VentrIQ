from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard 12-lead ECG lead configuration (Columbia / GE MUSE standard layout)
STANDARD_12_LEADS: List[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

# Clinical and signal constants for EchoNext
SAMPLING_RATE_HZ: int = 250
RECORDING_DURATION_SEC: float = 10.0
EXPECTED_TIME_STEPS: int = 2500  # 10s * 250 Hz
NUM_LEADS: int = 12

# Clinical diagnostic thresholds for LVEF
LVEF_MODERATE_THRESHOLD: float = 45.0  # LVEF <= 45% indicates moderate-to-severe systolic dysfunction
LVEF_SEVERE_THRESHOLD: float = 35.0    # LVEF <= 35% indicates severe systolic dysfunction

# Official metadata schema definitions
PATIENT_KEY_CANDIDATES = ["patient_key", "patient_id"]
SPLIT_COLUMN = "split"
LVEF_VALUE_COLUMN = "lvef_value"
LVEF_LTE_45_COLUMN = "lvef_lte_45_flag"
SHD_MODERATE_COLUMN = "shd_moderate_or_greater_flag"
OFFICIAL_SPLITS = ["train", "val", "test", "no_split"]


class EchoNextDataset:
    """Dataset loader and interface for the EchoNext 12-lead ECG dataset.

    The EchoNext dataset pairs standard 12-lead ECG waveforms (sampled at 250 Hz
    for 10 seconds) with echocardiography-confirmed measurements (LVEF) and
    structural heart disease labels.

    Waveforms in the official release are packaged as large memory-mapped NumPy
    arrays per split (shape: [N, 1, 2500, 12]). This loader ensures on-demand,
    zero-copy access without loading complete datasets into RAM.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: Optional[str] = None,
        mmap_mode: str = "r",
        validate_files: bool = False,
    ) -> None:
        """Initialize the EchoNext dataset.

        Args:
            root: Directory containing EchoNext metadata and waveform files.
            split: Optional split filter ('train', 'val', 'test', 'no_split').
            mmap_mode: NumPy memory-mapping mode ('r' for read-only).
            validate_files: If True, validates presence and shapes of waveform files.
        """
        self.root = Path(root)
        self.split_filter = split
        self.mmap_mode = mmap_mode

        if not self.root.exists():
            raise FileNotFoundError(f"EchoNext root directory does not exist: {self.root}")

        # Locate metadata CSV (supports official EchoNext_metadata_100k.csv or lowercase)
        self.metadata_path = self._locate_file([
            "EchoNext_metadata_100k.csv",
            "echonext_metadata_100k.csv",
            "metadata.csv",
        ])
        if self.metadata_path is None:
            raise FileNotFoundError(
                f"EchoNext metadata CSV not found in {self.root}. "
                "Expected 'EchoNext_metadata_100k.csv' or 'echonext_metadata_100k.csv'."
            )

        self._full_metadata = pd.read_csv(self.metadata_path)

        # Detect patient key column
        self.patient_col = self._detect_patient_column(self._full_metadata)

        # Check if official split column exists
        if SPLIT_COLUMN in self._full_metadata.columns:
            # Maintain index mapping to split-specific row positions
            self._full_metadata["_split_local_index"] = (
                self._full_metadata.groupby(SPLIT_COLUMN).cumcount()
            )
        else:
            self._full_metadata["_split_local_index"] = np.arange(len(self._full_metadata))

        # Filter by split if requested
        if self.split_filter is not None:
            if SPLIT_COLUMN not in self._full_metadata.columns:
                raise ValueError(
                    f"Cannot filter by split '{split}': metadata has no '{SPLIT_COLUMN}' column."
                )
            self.metadata = (
                self._full_metadata[self._full_metadata[SPLIT_COLUMN] == self.split_filter]
                .copy()
                .reset_index(drop=True)
            )
        else:
            self.metadata = self._full_metadata.copy()

        # Cache for opened memory-mapped waveform files: split_name -> np.ndarray (mmap)
        self._waveform_mmaps: Dict[str, Optional[np.ndarray]] = {}

        if validate_files:
            self.verify_waveform_alignment()

    def _locate_file(self, candidates: List[str]) -> Optional[Path]:
        """Search for a filename among candidates in root directory."""
        for candidate in candidates:
            path = self.root / candidate
            if path.exists():
                return path
        return None

    def _detect_patient_column(self, df: pd.DataFrame) -> str:
        """Identify patient key identifier column."""
        for candidate in PATIENT_KEY_CANDIDATES:
            if candidate in df.columns:
                return candidate
        # Fallback to first column ending with 'id' or 'key'
        for col in df.columns:
            if col.lower().endswith(("_id", "_key", "id")):
                return col
        return "patient_key"

    def _get_waveform_mmap(self, split: str) -> Optional[np.ndarray]:
        """Lazy-load memory-mapped array for a given split."""
        if split in self._waveform_mmaps:
            return self._waveform_mmaps[split]

        candidate_names = [
            f"EchoNext_{split}_waveforms.npy",
            f"echonext_{split}_waveforms.npy",
            f"{split}_waveforms.npy",
        ]
        path = self._locate_file(candidate_names)
        if path is not None:
            mmap = np.load(path, mmap_mode=self.mmap_mode)
            self._waveform_mmaps[split] = mmap
            return mmap

        self._waveform_mmaps[split] = None
        return None

    def __len__(self) -> int:
        """Return the number of records in the active metadata view."""
        return len(self.metadata)

    def get_waveform(self, index: int, channels_first: bool = False) -> np.ndarray:
        """Retrieve a single 12-lead ECG waveform via memory-mapping.

        Args:
            index: Row index in the active dataset view.
            channels_first: If True, returns shape (12, 2500).
                            If False (default), returns shape (2500, 12).

        Returns:
            np.ndarray of shape (2500, 12) or (12, 2500) as float32.
        """
        if index < 0 or index >= len(self.metadata):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.metadata)}")

        row = self.metadata.iloc[index]

        # Case 1: Individual file path specified in metadata (legacy or custom format)
        if "ecg_file" in row and pd.notna(row["ecg_file"]):
            file_path = self.root / str(row["ecg_file"])
            if not file_path.exists():
                raise FileNotFoundError(f"Waveform file not found: {file_path}")
            raw = np.load(file_path, mmap_mode=self.mmap_mode)
            waveform = np.array(raw, dtype=np.float32)

        # Case 2: Monolithic official EchoNext per-split .npy array
        else:
            split = row.get(SPLIT_COLUMN, self.split_filter or "train")
            local_idx = row.get("_split_local_index", index)

            mmap = self._get_waveform_mmap(str(split))
            if mmap is None:
                # Try single unified waveforms file
                mmap = self._get_waveform_mmap("all") or self._get_waveform_mmap("")

            if mmap is None:
                raise FileNotFoundError(
                    f"No waveform .npy file found for split '{split}' in {self.root}."
                )

            # Slices memory map on-demand: official shape is [N, 1, 2500, 12] or [N, 2500, 12]
            sample = mmap[int(local_idx)]
            waveform = np.array(sample, dtype=np.float32)

        # Standardize shape to (2500, 12)
        waveform = np.squeeze(waveform)

        if waveform.shape == (NUM_LEADS, EXPECTED_TIME_STEPS):
            # Transpose (12, 2500) -> (2500, 12)
            waveform = waveform.T
        elif waveform.shape != (EXPECTED_TIME_STEPS, NUM_LEADS):
            raise ValueError(
                f"Unexpected waveform shape {waveform.shape} for record {index}. "
                f"Expected ({EXPECTED_TIME_STEPS}, {NUM_LEADS}) or ({NUM_LEADS}, {EXPECTED_TIME_STEPS})."
            )

        if channels_first:
            waveform = waveform.T

        return waveform

    def get_train_ecg(self, index: int) -> np.ndarray:
        """Backward-compatible helper to retrieve a waveform."""
        return self.get_waveform(index)

    def get_lvef(self, index: int) -> float:
        """Retrieve LVEF measurement (%) for a given record.

        Returns NaN if unmeasured or missing.
        """
        if index < 0 or index >= len(self.metadata):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.metadata)}")
        val = self.metadata.iloc[index].get(LVEF_VALUE_COLUMN, np.nan)
        return float(val) if pd.notna(val) else np.nan

    def get_lvef_flag(self, index: int) -> Optional[int]:
        """Retrieve reduced-LVEF flag (LVEF <= 45%)."""
        if index < 0 or index >= len(self.metadata):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.metadata)}")
        val = self.metadata.iloc[index].get(LVEF_LTE_45_COLUMN, None)
        if pd.isna(val) or val is None:
            # Fall back to calculating from lvef_value if available
            lvef = self.get_lvef(index)
            return int(lvef <= LVEF_MODERATE_THRESHOLD) if pd.notna(lvef) else None
        return int(val)

    def get_metadata_record(self, index: int) -> Dict[str, Any]:
        """Return full metadata row as a dictionary."""
        if index < 0 or index >= len(self.metadata):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.metadata)}")
        record = self.metadata.iloc[index].to_dict()
        record.pop("_split_local_index", None)
        return record

    def verify_patient_isolation(self) -> Dict[str, Any]:
        """Verify patient-level separation between official splits.

        Checks whether patients in val and test appear in the training split.
        Returns detailed overlap statistics.
        """
        if SPLIT_COLUMN not in self._full_metadata.columns:
            return {
                "has_split_column": False,
                "message": f"Column '{SPLIT_COLUMN}' not present in metadata.",
            }

        splits_present = self._full_metadata[SPLIT_COLUMN].dropna().unique()
        patients_per_split: Dict[str, set] = {}
        ecgs_per_split: Dict[str, int] = {}

        for sp in splits_present:
            subset = self._full_metadata[self._full_metadata[SPLIT_COLUMN] == sp]
            patients_per_split[str(sp)] = set(subset[self.patient_col].dropna().unique())
            ecgs_per_split[str(sp)] = len(subset)

        overlaps: Dict[str, int] = {}
        total_overlap = 0

        # Check pairwise overlaps across key evaluation splits
        eval_splits = [s for s in ["train", "val", "test"] if s in patients_per_split]
        for i in range(len(eval_splits)):
            for j in range(i + 1, len(eval_splits)):
                s1, s2 = eval_splits[i], eval_splits[j]
                intersection = patients_per_split[s1].intersection(patients_per_split[s2])
                overlap_count = len(intersection)
                overlaps[f"{s1}_vs_{s2}"] = overlap_count
                total_overlap += overlap_count

        is_leak_free = (total_overlap == 0)

        return {
            "has_split_column": True,
            "patient_column": self.patient_col,
            "is_leak_free": is_leak_free,
            "ecgs_per_split": ecgs_per_split,
            "unique_patients_per_split": {k: len(v) for k, v in patients_per_split.items()},
            "overlaps": overlaps,
            "total_overlapping_patients": total_overlap,
        }

    def verify_waveform_alignment(self) -> Dict[str, Any]:
        """Verify alignment between metadata split rows and waveform arrays."""
        results: Dict[str, Any] = {"splits": {}, "all_aligned": True}

        if SPLIT_COLUMN not in self._full_metadata.columns:
            results["all_aligned"] = False
            results["error"] = f"No '{SPLIT_COLUMN}' column found."
            return results

        for sp in ["train", "val", "test", "no_split"]:
            count = (self._full_metadata[SPLIT_COLUMN] == sp).sum()
            if count == 0:
                continue

            mmap = self._get_waveform_mmap(sp)
            if mmap is None:
                results["splits"][sp] = {
                    "metadata_rows": count,
                    "waveform_rows": None,
                    "aligned": False,
                    "note": f"Waveform file for split '{sp}' not found.",
                }
                results["all_aligned"] = False
            else:
                waveform_rows = mmap.shape[0]
                aligned = bool(count == waveform_rows)
                results["splits"][sp] = {
                    "metadata_rows": int(count),
                    "waveform_rows": int(waveform_rows),
                    "shape": list(mmap.shape),
                    "aligned": aligned,
                }
                if not aligned:
                    results["all_aligned"] = False

        results["all_aligned"] = bool(results["all_aligned"])
        return results

    def summarize(self) -> Dict[str, Any]:
        """Compute exploration statistics across the dataset."""
        df = self.metadata
        total_ecgs = len(df)
        patient_col = self.patient_col
        unique_patients = int(df[patient_col].nunique()) if patient_col in df.columns else total_ecgs

        # Split counts
        split_counts: Dict[str, int] = {}
        if SPLIT_COLUMN in df.columns:
            split_counts = df[SPLIT_COLUMN].value_counts(dropna=False).to_dict()

        # Target statistics (LVEF)
        lvef_stats: Dict[str, Any] = {}
        if LVEF_VALUE_COLUMN in df.columns:
            lvef_series = df[LVEF_VALUE_COLUMN].dropna()
            if len(lvef_series) > 0:
                lvef_stats = {
                    "count": int(len(lvef_series)),
                    "missing": int(df[LVEF_VALUE_COLUMN].isna().sum()),
                    "missing_pct": float(df[LVEF_VALUE_COLUMN].isna().mean() * 100),
                    "mean": float(lvef_series.mean()),
                    "std": float(lvef_series.std()),
                    "median": float(lvef_series.median()),
                    "min": float(lvef_series.min()),
                    "max": float(lvef_series.max()),
                    "q25": float(lvef_series.quantile(0.25)),
                    "q75": float(lvef_series.quantile(0.75)),
                    "severe_reduced_count": int((lvef_series <= LVEF_SEVERE_THRESHOLD).sum()),
                    "moderate_reduced_count": int(
                        ((lvef_series > LVEF_SEVERE_THRESHOLD) & (lvef_series <= LVEF_MODERATE_THRESHOLD)).sum()
                    ),
                    "preserved_count": int((lvef_series > LVEF_MODERATE_THRESHOLD).sum()),
                }

        # Binary label statistics
        binary_flags: Dict[str, Any] = {}
        for flag_col in [LVEF_LTE_45_COLUMN, SHD_MODERATE_COLUMN]:
            if flag_col in df.columns:
                val_counts = df[flag_col].value_counts(dropna=False).to_dict()
                pos_count = int((df[flag_col] == 1).sum())
                binary_flags[flag_col] = {
                    "positive_count": pos_count,
                    "positive_prevalence_pct": float(df[flag_col].mean() * 100) if len(df) else 0.0,
                    "counts": {str(k): int(v) for k, v in val_counts.items()},
                }

        # Missing values per column
        missing_stats = df.isna().sum().to_dict()

        return {
            "total_records": total_ecgs,
            "unique_patients": unique_patients,
            "ecgs_per_patient_mean": float(total_ecgs / max(1, unique_patients)),
            "split_distribution": {str(k): int(v) for k, v in split_counts.items()},
            "lvef_statistics": lvef_stats,
            "binary_flags": binary_flags,
            "missing_values": {str(k): int(v) for k, v in missing_stats.items()},
            "sampling_frequency_hz": SAMPLING_RATE_HZ,
            "duration_sec": RECORDING_DURATION_SEC,
            "lead_names": STANDARD_12_LEADS,
        }

    def train_spliting_patients(
        self,
        df: pd.DataFrame,
        seed: int = 42,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform patient-level train/validation/test split.

        Preserved for backward compatibility, updated to guarantee zero
        patient leakage and deterministic reproducibility.
        """
        patient_col = self._detect_patient_column(df)
        patients = np.array(df[patient_col].dropna().unique())
        rng = np.random.default_rng(seed)
        shuffled = patients.copy()
        rng.shuffle(shuffled)

        num = len(shuffled)
        n_train = int(train_ratio * num)
        n_val = int((train_ratio + val_ratio) * num)

        train_patients = set(shuffled[:n_train])
        val_patients = set(shuffled[n_train:n_val])
        test_patients = set(shuffled[n_val:])

        train_p = df[df[patient_col].isin(train_patients)].copy().reset_index(drop=True)
        val_p = df[df[patient_col].isin(val_patients)].copy().reset_index(drop=True)
        test_p = df[df[patient_col].isin(test_patients)].copy().reset_index(drop=True)

        return train_p, val_p, test_p


# Preserve original class name capitalization as alias
EchonextDataset = EchoNextDataset


def create_mock_echonext_dataset(
    output_dir: Union[str, Path],
    num_train: int = 70,
    num_val: int = 15,
    num_test: int = 15,
    seed: int = 42,
) -> Path:
    """Generate a lightweight, synthetic EchoNext dataset for testing and CI.

    Creates valid metadata CSV and memory-mappable .npy waveform files matching
    the official EchoNext schema, shapes, sampling rate, and lead layout.

    Args:
        output_dir: Directory to populate with mock dataset files.
        num_train: Number of records for the train split.
        num_val: Number of records for the val split.
        num_test: Number of records for the test split.
        seed: Random seed for reproducibility.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    split_configs = [
        ("train", num_train, 0.8),   # Train has some repeat patients
        ("val", num_val, 1.0),       # Val has strictly 1 ECG per patient
        ("test", num_test, 1.0),     # Test has strictly 1 ECG per patient
    ]

    all_rows: List[Dict[str, Any]] = []
    patient_id_counter = 1000

    for split_name, count, unique_patient_ratio in split_configs:
        num_patients = max(1, int(count * unique_patient_ratio))
        patient_pool = [f"P{patient_id_counter + i:05d}" for i in range(num_patients)]
        patient_id_counter += num_patients + 50  # Prevent patient overlap across splits

        # Generate realistic synthetic 12-lead ECG waveforms
        # Shape: (count, 1, 2500, 12)
        waveforms = np.zeros((count, 1, EXPECTED_TIME_STEPS, NUM_LEADS), dtype=np.float32)
        t = np.linspace(0, RECORDING_DURATION_SEC, EXPECTED_TIME_STEPS, endpoint=False)

        for i in range(count):
            pat_key = patient_pool[i % len(patient_pool)]
            # Realistic synthetic LVEF: Gaussian centered around 56% with some reduced values
            lvef = float(np.clip(rng.normal(loc=55.0, scale=12.0), 15.0, 75.0))
            lvef_lte_45 = int(lvef <= LVEF_MODERATE_THRESHOLD)
            shd_flag = int(lvef_lte_45 or (rng.random() < 0.15))

            # Heart rate between 50 and 110 bpm
            hr = float(rng.uniform(55.0, 95.0))
            f_heart = hr / 60.0

            # Construct synthetic QRS-like waveforms per lead
            for lead_idx in range(NUM_LEADS):
                # Lead-dependent amplitude factor and phase
                lead_gain = 0.5 + 0.5 * np.cos(lead_idx)
                # Baseline oscillation + QRS impulses
                base = 0.1 * np.sin(2 * np.pi * 0.25 * t)  # baseline wander
                qrs = lead_gain * np.sin(2 * np.pi * f_heart * t) ** 9
                noise = rng.normal(0, 0.05, EXPECTED_TIME_STEPS).astype(np.float32)
                # EchoNext waveforms are normalized (z-score like)
                sig = (base + qrs + noise).astype(np.float32)
                waveforms[i, 0, :, lead_idx] = sig

            row: Dict[str, Any] = {
                "patient_key": pat_key,
                "acquisition_year": int(rng.integers(2010, 2022)),
                "split": split_name,
                "sex": "male" if rng.random() > 0.48 else "female",
                "age_at_ecg": float(np.clip(rng.normal(62, 14), 18, 90)),
                "ventricular_rate": hr,
                "atrial_rate": hr + rng.choice([0.0, 0.0, rng.normal(0, 2)]),
                "pr_interval": float(np.clip(rng.normal(165, 25), 100, 300)),
                "qrs_duration": float(np.clip(rng.normal(95, 18), 60, 180)),
                "qt_corrected": float(np.clip(rng.normal(430, 30), 340, 550)),
                "lvef_value": round(lvef, 1),
                "lvef_lte_45_flag": lvef_lte_45,
                "lvwt_gte_13_flag": int(rng.random() < 0.2),
                "shd_moderate_or_greater_flag": shd_flag,
                "aortic_stenosis_moderate_or_greater_flag": int(rng.random() < 0.05),
                "aortic_regurgitation_moderate_or_greater_flag": int(rng.random() < 0.04),
                "mitral_regurgitation_moderate_or_greater_flag": int(rng.random() < 0.08),
                "tricuspid_regurgitation_moderate_or_greater_flag": int(rng.random() < 0.06),
                "pulmonary_regurgitation_moderate_or_greater_flag": int(rng.random() < 0.02),
                "rv_systolic_dysfunction_moderate_or_greater_flag": int(rng.random() < 0.05),
                "pericardial_effusion_moderate_large_flag": int(rng.random() < 0.03),
                "pasp_gte_45_flag": int(rng.random() < 0.12),
                "tr_max_gte_32_flag": int(rng.random() < 0.10),
            }
            all_rows.append(row)

        # Save split waveform .npy
        waveform_file = out / f"EchoNext_{split_name}_waveforms.npy"
        np.save(waveform_file, waveforms)

    # Save metadata CSV
    meta_df = pd.DataFrame(all_rows)
    meta_df.to_csv(out / "EchoNext_metadata_100k.csv", index=False)
    # Also save lowercase alias for maximum compatibility
    meta_df.to_csv(out / "echonext_metadata_100k.csv", index=False)

    logger.info("Created mock EchoNext dataset at %s with %d total records.", out, len(meta_df))
    return out
