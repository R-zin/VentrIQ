"""src/data/dataset.py - Canonical Record Representation, Manifest & PyTorch Datasets.

This module implements Phase 3 of the VentriQ pipeline:
1. ECGRecord: Canonical, immutable schema for 12-lead ECG records.
2. DatasetManifest: Complete traceability, exclusion auditing, and file checksums.
3. VentriQTorchDataset: Zero-copy, multi-worker safe PyTorch dataset slicing waveforms
   on-the-fly from memory-mapped EchoNext arrays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.echonext import (
    EXPECTED_TIME_STEPS,
    LVEF_LTE_45_COLUMN,
    LVEF_MODERATE_THRESHOLD,
    LVEF_VALUE_COLUMN,
    NUM_LEADS,
    OFFICIAL_SPLITS,
    PATIENT_KEY_CANDIDATES,
    SHD_MODERATE_COLUMN,
    SPLIT_COLUMN,
    STANDARD_12_LEADS,
    EchoNextDataset,
)
from src.data.splits import verify_patient_isolation

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 checksum of a file efficiently in chunks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum computation: {path}")

    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


@dataclass(frozen=True)
class ECGRecord:
    """Canonical, immutable representation of a single 12-lead ECG recording.

    Guarantees strict traceability from global record ID down to local memory-mapped
    array positions and clinical diagnostic targets.
    """

    record_id: str
    patient_key: str
    split: str
    waveform_ref: Tuple[str, int]  # (split_file_key, local_index)
    lvef_target: Optional[float] = None
    lvef_lte_45_label: Optional[int] = None
    shd_label: Optional[int] = None
    quality_flags: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate integrity and basic boundary constraints."""
        if not self.record_id:
            raise ValueError("record_id must be a non-empty string.")
        if not self.patient_key:
            raise ValueError("patient_key must be a non-empty string.")
        if not isinstance(self.waveform_ref, tuple) or len(self.waveform_ref) != 2:
            raise ValueError(f"waveform_ref must be a 2-tuple (split, index), got {self.waveform_ref}")

        # Validate LVEF range if target is provided
        if self.lvef_target is not None and not np.isnan(self.lvef_target):
            if not (0.0 <= self.lvef_target <= 100.0):
                logger.warning(
                    "LVEF target %s for record %s is outside standard physiological range [0, 100]",
                    self.lvef_target,
                    self.record_id,
                )

        # Validate binary labels
        if self.lvef_lte_45_label is not None and self.lvef_lte_45_label not in (0, 1):
            raise ValueError(f"lvef_lte_45_label must be 0 or 1, got {self.lvef_lte_45_label}")
        if self.shd_label is not None and self.shd_label not in (0, 1):
            raise ValueError(f"shd_label must be 0 or 1, got {self.shd_label}")

    @property
    def has_valid_lvef(self) -> bool:
        """Check whether record contains a valid, non-NaN continuous LVEF target."""
        return self.lvef_target is not None and bool(np.isfinite(self.lvef_target))

    @property
    def has_valid_shd(self) -> bool:
        """Check whether record contains a valid structural heart disease label."""
        return self.shd_label is not None

    @property
    def is_usable(self) -> bool:
        """Check whether record passed signal quality control checks."""
        if self.quality_flags is None:
            return True
        return bool(self.quality_flags.get("is_usable", True))

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        d = asdict(self)
        # Ensure tuple is preserved as list in dict representation
        d["waveform_ref"] = list(self.waveform_ref)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ECGRecord:
        """Construct ECGRecord from dictionary."""
        d = dict(data)
        if isinstance(d.get("waveform_ref"), (list, tuple)):
            d["waveform_ref"] = (str(d["waveform_ref"][0]), int(d["waveform_ref"][1]))
        return cls(**d)


@dataclass
class DatasetManifest:
    """Audit-logged manifest detailing dataset composition, file hashes, and exclusions."""

    records: List[ECGRecord]
    source_checksums: Dict[str, str] = field(default_factory=dict)
    retained_counts_per_split: Dict[str, int] = field(default_factory=dict)
    excluded_counts_per_split: Dict[str, int] = field(default_factory=dict)
    exclusion_breakdown: Dict[str, int] = field(default_factory=dict)
    patient_counts_per_split: Dict[str, int] = field(default_factory=dict)
    target_statistics: Dict[str, Any] = field(default_factory=dict)
    environment_metadata: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42

    def to_dict(self, include_records: bool = True) -> Dict[str, Any]:
        """Serialize manifest to dictionary."""
        data: Dict[str, Any] = {
            "source_checksums": self.source_checksums,
            "retained_counts_per_split": self.retained_counts_per_split,
            "excluded_counts_per_split": self.excluded_counts_per_split,
            "exclusion_breakdown": self.exclusion_breakdown,
            "patient_counts_per_split": self.patient_counts_per_split,
            "target_statistics": self.target_statistics,
            "environment_metadata": self.environment_metadata,
            "random_seed": self.random_seed,
            "total_retained_records": len(self.records),
        }
        if include_records:
            data["records"] = [r.to_dict() for r in self.records]
        return data

    def save(self, output_path: Union[str, Path], include_records: bool = True) -> Path:
        """Save manifest to a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.to_dict(include_records=include_records), f, indent=2)
        logger.info("Saved dataset manifest to %s (%d records)", out, len(self.records))
        return out

    @classmethod
    def load(cls, manifest_path: Union[str, Path]) -> DatasetManifest:
        """Load manifest from JSON file."""
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {path}")

        with open(path, "r") as f:
            raw = json.load(f)

        records = [ECGRecord.from_dict(r) for r in raw.get("records", [])]
        return cls(
            records=records,
            source_checksums=raw.get("source_checksums", {}),
            retained_counts_per_split=raw.get("retained_counts_per_split", {}),
            excluded_counts_per_split=raw.get("excluded_counts_per_split", {}),
            exclusion_breakdown=raw.get("exclusion_breakdown", {}),
            patient_counts_per_split=raw.get("patient_counts_per_split", {}),
            target_statistics=raw.get("target_statistics", {}),
            environment_metadata=raw.get("environment_metadata", {}),
            random_seed=raw.get("random_seed", 42),
        )

    @classmethod
    def build_from_echonext(
        cls,
        root: Union[str, Path],
        quality_evaluator: Optional[Any] = None,
        exclude_missing_lvef: bool = False,
        require_usable_quality: bool = False,
        seed: int = 42,
    ) -> DatasetManifest:
        """Build a canonical audit-logged manifest directly from an EchoNext directory."""
        dataset = EchoNextDataset(root)
        df = dataset.metadata.copy()
        patient_col = dataset.patient_col

        # 1. Compute SHA256 checksums of source files
        source_checksums: Dict[str, str] = {}
        if dataset.metadata_path and dataset.metadata_path.exists():
            source_checksums[dataset.metadata_path.name] = compute_file_sha256(dataset.metadata_path)

        for sp in ["train", "val", "test"]:
            npy_path = dataset._locate_file([
                f"EchoNext_{sp}_waveforms.npy",
                f"echonext_{sp}_waveforms.npy",
                f"{sp}_waveforms.npy",
            ])
            if npy_path and npy_path.exists():
                source_checksums[npy_path.name] = compute_file_sha256(npy_path)

        # 2. Iterate through records and apply exclusion rules with audit logging
        retained_records: List[ECGRecord] = []
        retained_counts: Dict[str, int] = {}
        excluded_counts: Dict[str, int] = {}
        exclusion_breakdown: Dict[str, int] = {
            "missing_lvef": 0,
            "unusable_signal_quality": 0,
            "missing_waveform": 0,
        }

        # Track per-split cumcount for waveform mapping
        split_col = SPLIT_COLUMN if SPLIT_COLUMN in df.columns else "split"
        if split_col in df.columns:
            df["_split_local_index"] = df.groupby(split_col).cumcount()
        else:
            df["_split_local_index"] = np.arange(len(df))

        for idx, row in df.iterrows():
            split_val = str(row.get(split_col, "all"))
            local_idx = int(row.get("_split_local_index", idx))
            patient_id = str(row.get(patient_col, f"P_{idx:06d}"))
            record_id = f"ECHONEXT_{split_val}_{idx:06d}"

            # Extract targets
            lvef_raw = row.get(LVEF_VALUE_COLUMN, np.nan)
            lvef_target = float(lvef_raw) if pd.notna(lvef_raw) else None

            lvef_lte_45_raw = row.get(LVEF_LTE_45_COLUMN, None)
            if pd.notna(lvef_lte_45_raw) and lvef_lte_45_raw is not None:
                lvef_lte_45_label = int(lvef_lte_45_raw)
            elif lvef_target is not None:
                lvef_lte_45_label = int(lvef_target <= LVEF_MODERATE_THRESHOLD)
            else:
                lvef_lte_45_label = None

            shd_raw = row.get(SHD_MODERATE_COLUMN, None)
            shd_label = int(shd_raw) if pd.notna(shd_raw) and shd_raw is not None else None

            # Optional quality evaluation
            quality_flags: Optional[Dict[str, Any]] = None
            if quality_evaluator is not None:
                try:
                    wf = dataset.get_waveform(idx)
                    qc_report = quality_evaluator.evaluate(wf)
                    quality_flags = qc_report.to_dict()
                except Exception as exc:
                    logger.warning("Quality evaluation failed for record %s: %s", record_id, exc)
                    quality_flags = {"is_valid": False, "is_usable": False, "error": str(exc)}

            # Evaluate exclusions
            is_excluded = False
            if exclude_missing_lvef and lvef_target is None:
                is_excluded = True
                exclusion_breakdown["missing_lvef"] += 1
            elif require_usable_quality and quality_flags is not None and not quality_flags.get("is_usable", True):
                is_excluded = True
                exclusion_breakdown["unusable_signal_quality"] += 1

            if is_excluded:
                excluded_counts[split_val] = excluded_counts.get(split_val, 0) + 1
                continue

            # Record is retained
            retained_counts[split_val] = retained_counts.get(split_val, 0) + 1
            record = ECGRecord(
                record_id=record_id,
                patient_key=patient_id,
                split=split_val,
                waveform_ref=(split_val, local_idx),
                lvef_target=lvef_target,
                lvef_lte_45_label=lvef_lte_45_label,
                shd_label=shd_label,
                quality_flags=quality_flags,
            )
            retained_records.append(record)

        # 3. Patient counts per split
        patient_counts: Dict[str, int] = {}
        for sp in set(r.split for r in retained_records):
            pats = set(r.patient_key for r in retained_records if r.split == sp)
            patient_counts[sp] = len(pats)

        # 4. Target distribution statistics
        valid_lvefs = [r.lvef_target for r in retained_records if r.has_valid_lvef]
        lvef_lte_45_flags = [r.lvef_lte_45_label for r in retained_records if r.lvef_lte_45_label is not None]
        shd_flags = [r.shd_label for r in retained_records if r.shd_label is not None]

        target_stats: Dict[str, Any] = {
            "lvef_count": len(valid_lvefs),
            "lvef_mean": float(np.mean(valid_lvefs)) if valid_lvefs else None,
            "lvef_std": float(np.std(valid_lvefs)) if valid_lvefs else None,
            "lvef_min": float(np.min(valid_lvefs)) if valid_lvefs else None,
            "lvef_median": float(np.median(valid_lvefs)) if valid_lvefs else None,
            "lvef_max": float(np.max(valid_lvefs)) if valid_lvefs else None,
            "lvef_lte_45_rate": float(np.mean(lvef_lte_45_flags)) if lvef_lte_45_flags else None,
            "shd_rate": float(np.mean(shd_flags)) if shd_flags else None,
        }

        # 5. Environment & version metadata
        env_meta = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "torch_version": torch.__version__,
        }

        return cls(
            records=retained_records,
            source_checksums=source_checksums,
            retained_counts_per_split=retained_counts,
            excluded_counts_per_split=excluded_counts,
            exclusion_breakdown=exclusion_breakdown,
            patient_counts_per_split=patient_counts,
            target_statistics=target_stats,
            environment_metadata=env_meta,
            random_seed=seed,
        )


def ventriq_worker_init_fn(worker_id: int) -> None:
    """Deterministic worker initialization function for PyTorch DataLoader.

    Ensures independent, reproducible random seeds across all PyTorch multi-process workers
    and safe per-process state isolation.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


class VentriQTorchDataset(Dataset):
    """Zero-copy PyTorch Dataset for 12-lead ECG waveforms.

    Slices waveforms directly on-the-fly from memory-mapped NumPy arrays, ensuring
    minimal memory overhead, multi-worker safety, and strict alignment with clinical targets.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        records: Optional[Sequence[ECGRecord]] = None,
        split: Optional[str] = None,
        transform: Optional[Union[Callable, Any]] = None,
        target_transform: Optional[Callable] = None,
        channels_first: bool = True,
        exclude_missing_lvef: bool = False,
        require_usable_quality: bool = False,
        mmap_mode: str = "r",
    ) -> None:
        """Initialize VentriQTorchDataset.

        Args:
            root: Root directory of EchoNext dataset (used if records is None or to load mmaps).
            records: Pre-filtered list of ECGRecord objects.
            split: Split name to filter by ('train', 'val', 'test').
            transform: Optional transform callable or ECGSignalPreprocessor instance.
            target_transform: Optional callable applied to target dictionary.
            channels_first: If True (default), output tensor is (12, 2500) for CNNs.
                            If False, output tensor is (2500, 12).
            exclude_missing_lvef: If True, filters out records with unmeasured LVEF.
            require_usable_quality: If True, filters out records flagged unusable by QC.
            mmap_mode: Memory-map mode (default 'r' for read-only).
        """
        self.root = Path(root) if root is not None else None
        self.split_filter = split
        self.transform = transform
        self.target_transform = target_transform
        self.channels_first = channels_first
        self.mmap_mode = mmap_mode

        # Cache of memory-mapped arrays: split_name -> np.ndarray
        self._waveform_mmaps: Dict[str, Optional[np.ndarray]] = {}

        if records is not None:
            filtered_records = list(records)
            if self.split_filter is not None:
                filtered_records = [r for r in filtered_records if r.split == self.split_filter]
        elif self.root is not None:
            # Build records from raw EchoNext directory
            manifest = DatasetManifest.build_from_echonext(
                self.root,
                exclude_missing_lvef=exclude_missing_lvef,
                require_usable_quality=require_usable_quality,
            )
            filtered_records = manifest.records
            if self.split_filter is not None:
                filtered_records = [r for r in filtered_records if r.split == self.split_filter]
        else:
            raise ValueError("Either 'root' or 'records' must be provided to initialize VentriQTorchDataset.")

        # Apply record-level filtering options
        if exclude_missing_lvef:
            filtered_records = [r for r in filtered_records if r.has_valid_lvef]
        if require_usable_quality:
            filtered_records = [r for r in filtered_records if r.is_usable]

        self.records: List[ECGRecord] = filtered_records

    def __len__(self) -> int:
        """Return the number of records in the active dataset view."""
        return len(self.records)

    def _locate_mmap_file(self, split: str) -> Optional[Path]:
        """Locate memory-mapped waveform file in root directory."""
        if self.root is None:
            return None

        candidates = [
            f"EchoNext_{split}_waveforms.npy",
            f"echonext_{split}_waveforms.npy",
            f"{split}_waveforms.npy",
        ]
        for name in candidates:
            p = self.root / name
            if p.exists():
                return p
        return None

    def _get_waveform_mmap(self, split: str) -> np.ndarray:
        """Retrieve memory-mapped array for the given split (process/worker-safe lazy loading)."""
        if split in self._waveform_mmaps and self._waveform_mmaps[split] is not None:
            return self._waveform_mmaps[split]

        path = self._locate_mmap_file(split)
        if path is None:
            # Fall back to unified file or check split == 'all'
            path = self._locate_mmap_file("all")

        if path is None or not path.exists():
            raise FileNotFoundError(
                f"Memory-mapped waveform file for split '{split}' not found in {self.root}."
            )

        mmap = np.load(path, mmap_mode=self.mmap_mode)
        self._waveform_mmaps[split] = mmap
        return mmap

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Fetch a single record by index.

        Returns:
            Dictionary containing:
                - 'waveform': torch.FloatTensor of shape (12, 2500) or (2500, 12)
                - 'record_id': str
                - 'patient_key': str
                - 'split': str
                - 'lvef': torch.FloatTensor (scalar, NaN if missing)
                - 'lvef_lte_45': torch.LongTensor (scalar, -1 if missing)
                - 'shd': torch.LongTensor (scalar, -1 if missing)
                - 'is_usable': bool
        """
        if index < 0 or index >= len(self.records):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.records)}")

        record = self.records[index]
        split_key, local_idx = record.waveform_ref

        # Slice memory map directly (zero unnecessary RAM copy until sliced)
        mmap = self._get_waveform_mmap(split_key)
        raw_slice = mmap[int(local_idx)]

        # Convert slice to float32 ndarray (2500, 12)
        waveform = np.array(raw_slice, dtype=np.float32, copy=True)
        waveform = np.squeeze(waveform)

        if waveform.shape == (NUM_LEADS, EXPECTED_TIME_STEPS):
            waveform = waveform.T
        elif waveform.shape != (EXPECTED_TIME_STEPS, NUM_LEADS):
            raise ValueError(
                f"Unexpected waveform shape {waveform.shape} for record {record.record_id}. "
                f"Expected ({EXPECTED_TIME_STEPS}, {NUM_LEADS})."
            )

        # Apply optional transform (e.g. ECGSignalPreprocessor or callable)
        if self.transform is not None:
            if hasattr(self.transform, "process"):
                # ECGSignalPreprocessor instance
                res = self.transform.process(waveform)
                waveform = res.final
            elif callable(self.transform):
                waveform = self.transform(waveform)

        # Transpose to channels_first if requested: (2500, 12) -> (12, 2500)
        if self.channels_first:
            waveform = waveform.T

        # Convert to torch tensor
        waveform_tensor = torch.from_numpy(waveform).to(dtype=torch.float32)

        # Build targets
        lvef_val = record.lvef_target if record.has_valid_lvef else float("nan")
        lvef_tensor = torch.tensor(lvef_val, dtype=torch.float32)

        lvef_lte_45_val = record.lvef_lte_45_label if record.lvef_lte_45_label is not None else -1
        lvef_lte_45_tensor = torch.tensor(lvef_lte_45_val, dtype=torch.long)

        shd_val = record.shd_label if record.shd_label is not None else -1
        shd_tensor = torch.tensor(shd_val, dtype=torch.long)

        sample = {
            "waveform": waveform_tensor,
            "record_id": record.record_id,
            "patient_key": record.patient_key,
            "split": record.split,
            "lvef": lvef_tensor,
            "lvef_lte_45": lvef_lte_45_tensor,
            "shd": shd_tensor,
            "is_usable": record.is_usable,
        }

        if self.target_transform is not None:
            sample = self.target_transform(sample)

        return sample

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        split: Optional[str] = None,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Union[Callable, Any]] = None,
        channels_first: bool = True,
        exclude_missing_lvef: bool = False,
        require_usable_quality: bool = False,
    ) -> VentriQTorchDataset:
        """Construct VentriQTorchDataset directly from a saved dataset manifest."""
        manifest = DatasetManifest.load(manifest_path)
        data_root = root or Path(manifest_path).parent.parent / "data" / "raw" / "echonext"
        return cls(
            root=data_root,
            records=manifest.records,
            split=split,
            transform=transform,
            channels_first=channels_first,
            exclude_missing_lvef=exclude_missing_lvef,
            require_usable_quality=require_usable_quality,
        )

    @classmethod
    def from_echonext(
        cls,
        dataset: EchoNextDataset,
        transform: Optional[Union[Callable, Any]] = None,
        channels_first: bool = True,
        exclude_missing_lvef: bool = False,
        require_usable_quality: bool = False,
    ) -> VentriQTorchDataset:
        """Construct VentriQTorchDataset from an existing EchoNextDataset instance."""
        manifest = DatasetManifest.build_from_echonext(
            root=dataset.root,
            exclude_missing_lvef=exclude_missing_lvef,
            require_usable_quality=require_usable_quality,
        )
        return cls(
            root=dataset.root,
            records=manifest.records,
            split=dataset.split_filter,
            transform=transform,
            channels_first=channels_first,
            exclude_missing_lvef=exclude_missing_lvef,
            require_usable_quality=require_usable_quality,
        )
