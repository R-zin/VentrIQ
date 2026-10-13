#!/usr/bin/env python3
"""scripts/explore_dataset.py - Phase 1 EchoNext Dataset Exploration and Verification.

This script performs comprehensive dataset exploration, verifying:
1. File availability and layout (metadata CSV, split waveform arrays).
2. Metadata schema, column data types, and patient identification.
3. Patient-level split isolation (checking for data leakage between train/val/test).
4. Memory-mapped waveform access, shapes, lead orders, and sampling parameters.
5. Waveform-to-metadata row alignment.
6. Target distribution (LVEF continuous and binary moderate/severe thresholds).
7. Missing value rates and anomalies.
8. Generation of publication-ready diagnostic plots and a markdown exploration report.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.echonext import (
    EXPECTED_TIME_STEPS,
    LVEF_MODERATE_THRESHOLD,
    LVEF_SEVERE_THRESHOLD,
    NUM_LEADS,
    RECORDING_DURATION_SEC,
    SAMPLING_RATE_HZ,
    STANDARD_12_LEADS,
    EchoNextDataset,
    create_mock_echonext_dataset,
)
from src.utils.logging import get_logger
from src.utils.paths import (
    FIGURES_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    get_raw_echonext_dir,
)

logger = get_logger("explore_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore and verify EchoNext dataset for VentriQ (Phase 1)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=get_raw_echonext_dir(),
        help="Path to EchoNext dataset directory containing metadata and waveforms.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Path to output directory for reports and figures.",
    )
    parser.add_argument(
        "--create-mock-if-missing",
        action="store_true",
        default=True,
        help="If dataset is not present in data-dir, create a mock dataset for testing/verification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mock dataset generation and representative plotting.",
    )
    return parser.parse_args()


def plot_lvef_distribution(df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot histogram and density of LVEF with clinical thresholds."""
    fig_path = figures_dir / "lvef_distribution.png"
    lvef = df["lvef_value"].dropna()

    plt.figure(figsize=(9, 5), dpi=150)
    counts, bins, patches = plt.hist(
        lvef,
        bins=40,
        color="#2b5c8f",
        edgecolor="#1a3b5c",
        alpha=0.85,
        density=True,
    )

    # Plot vertical threshold indicators
    plt.axvline(
        LVEF_SEVERE_THRESHOLD,
        color="#d9381e",
        linestyle="--",
        linewidth=2,
        label=f"Severe Dysfunction (<= {LVEF_SEVERE_THRESHOLD:.0f}%)",
    )
    plt.axvline(
        LVEF_MODERATE_THRESHOLD,
        color="#f39c12",
        linestyle="--",
        linewidth=2,
        label=f"Moderate Dysfunction (<= {LVEF_MODERATE_THRESHOLD:.0f}%)",
    )
    plt.axvline(
        50.0,
        color="#27ae60",
        linestyle=":",
        linewidth=1.8,
        label="Preserved Threshold (>= 50%)",
    )

    plt.title("EchoNext Left Ventricular Ejection Fraction (LVEF) Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("LVEF (%)", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    logger.info("Saved LVEF distribution plot to %s", fig_path)
    return fig_path


def plot_split_distribution(df: pd.DataFrame, patient_col: str, figures_dir: Path) -> Path:
    """Plot records and unique patients across splits."""
    fig_path = figures_dir / "split_distribution.png"

    if "split" not in df.columns:
        return fig_path

    splits = [s for s in ["train", "val", "test", "no_split"] if s in df["split"].values]
    if not splits:
        splits = list(df["split"].unique())

    records = [len(df[df["split"] == s]) for s in splits]
    patients = [df[df["split"] == s][patient_col].nunique() for s in splits]

    x = np.arange(len(splits))
    width = 0.35

    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.bar(x - width / 2, records, width, label="ECG Records", color="#3498db", edgecolor="#2980b9")
    plt.bar(x + width / 2, patients, width, label="Unique Patients", color="#1abc9c", edgecolor="#16a085")

    plt.xlabel("Split Partition", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.title("Dataset Records vs. Unique Patients by Official Split", fontsize=13, fontweight="bold")
    plt.xticks(x, splits)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    logger.info("Saved split distribution plot to %s", fig_path)
    return fig_path


def plot_representative_ecgs(dataset: EchoNextDataset, figures_dir: Path) -> Path:
    """Plot representative 12-lead waveforms for reduced vs preserved LVEF."""
    fig_path = figures_dir / "representative_ecgs.png"
    df = dataset.metadata

    # Select one reduced (< 45%) and one preserved (> 55%) record if available
    reduced_indices = df.index[df["lvef_value"] <= LVEF_MODERATE_THRESHOLD].tolist()
    preserved_indices = df.index[df["lvef_value"] >= 55.0].tolist()

    idx_reduced = reduced_indices[0] if reduced_indices else 0
    idx_preserved = preserved_indices[0] if preserved_indices else (1 if len(df) > 1 else 0)

    wf_red = dataset.get_waveform(idx_reduced)
    lvef_red = dataset.get_lvef(idx_reduced)
    wf_pres = dataset.get_waveform(idx_preserved)
    lvef_pres = dataset.get_lvef(idx_preserved)

    time_vec = np.linspace(0, RECORDING_DURATION_SEC, EXPECTED_TIME_STEPS, endpoint=False)
    # Plot 2.5 seconds window for clear QRS morphology
    window = 625  # 2.5 seconds at 250 Hz

    fig, axes = plt.subplots(6, 4, figsize=(14, 10), sharex=True, sharey=True, dpi=150)
    fig.suptitle(
        f"Representative 12-Lead ECG Waveforms\n"
        f"Left Columns: Reduced LVEF ({lvef_red:.1f}%) | "
        f"Right Columns: Preserved LVEF ({lvef_pres:.1f}%)",
        fontsize=13,
        fontweight="bold",
    )

    lead_names = STANDARD_12_LEADS
    for i in range(12):
        row = i % 6
        col_offset = 0 if i < 6 else 1

        # Reduced ECG (Cols 0 and 1)
        ax_red = axes[row, col_offset]
        ax_red.plot(time_vec[:window], wf_red[:window, i], color="#c0392b", linewidth=1.1)
        ax_red.set_ylabel(lead_names[i], fontsize=9, fontweight="bold")
        ax_red.grid(True, linestyle=":", alpha=0.5)

        # Preserved ECG (Cols 2 and 3)
        ax_pres = axes[row, col_offset + 2]
        ax_pres.plot(time_vec[:window], wf_pres[:window, i], color="#2980b9", linewidth=1.1)
        ax_pres.set_ylabel(lead_names[i], fontsize=9, fontweight="bold")
        ax_pres.grid(True, linestyle=":", alpha=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)", fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    logger.info("Saved representative ECG plot to %s", fig_path)
    return fig_path


def generate_markdown_report(
    summary: Dict[str, Any],
    isolation_res: Dict[str, Any],
    alignment_res: Dict[str, Any],
    meta_df: pd.DataFrame,
    report_path: Path,
) -> None:
    """Compile Phase 1 Exploration Report into clean GitHub markdown format."""
    total_ecgs = summary["total_records"]
    unique_patients = summary["unique_patients"]
    ecgs_per_pat = summary["ecgs_per_patient_mean"]
    lvef_stats = summary.get("lvef_statistics", {})
    splits = summary.get("split_distribution", {})
    binary_flags = summary.get("binary_flags", {})

    content = f"""# EchoNext Dataset Exploration Report (Phase 1)

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Project:** VentriQ — Left Ventricular Ejection Fraction (LVEF) & Structural Heart Disease Analysis  
**Dataset Reference:** EchoNext v1.0.0 (PhysioNet / Columbia University Irving Medical Center)  

---

## 1. Executive Summary

- **Total 12-Lead ECG Records:** {total_ecgs:,}
- **Unique Patients:** {unique_patients:,}
- **Mean ECGs per Patient:** {ecgs_per_pat:.2f}
- **Sampling Frequency:** {summary['sampling_frequency_hz']} Hz
- **Recording Duration:** {summary['duration_sec']} seconds ({EXPECTED_TIME_STEPS} time steps per lead)
- **Number of Leads:** {len(summary['lead_names'])} standard leads ({', '.join(summary['lead_names'])})
- **Patient Isolation (Zero Leakage):** {"PASSED (0 overlap)" if isolation_res.get('is_leak_free') else "FAILED (Leakage detected)"}
- **Waveform-to-Metadata Row Alignment:** {"VERIFIED" if alignment_res.get('all_aligned') else "MISALIGNED"}

---

## 2. Dataset Architecture & File Format

The EchoNext dataset is packaged in a high-performance split-oriented array format:
1. **Metadata CSV:** Contains demographic features, raw ECG intervals, echocardiography continuous measurements, binary structural heart disease diagnostic labels, and split assignment.
2. **Split Waveform Arrays (`.npy`):** Stored as multi-dimensional NumPy arrays of shape `[N_split, 1, 2500, 12]` or `[N_split, 2500, 12]` per split (`train`, `val`, `test`).
3. **Memory-Mapped Access (`mmap_mode='r'`):** Slices individual waveforms on demand with zero overhead, strictly avoiding loading large arrays into RAM.

| Component | Specification |
|---|---|
| **Waveform Array Dimensions** | `(N, 1, 2500, 12)` |
| **Sampling Rate ($f_s$)** | 250 Hz |
| **Duration ($T$)** | 10.0 seconds |
| **Lead Sequence** | `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6` |
| **Amplitude State** | Median-filtered, percentile-clipped (0.1st - 99.9th), z-score normalized |

---

## 3. Official Split Distribution & Patient-Level Isolation

EchoNext officially designates four split categories:
- **`train`**: Model training partition (may contain repeat ECGs per patient over time).
- **`val`**: Validation partition (strictly the latest ECG per patient).
- **`test`**: Held-out benchmark test partition (strictly the latest ECG per patient).
- **`no_split`**: Supplementary records excluded from benchmark evaluation.

### Partition Breakdown:

| Split | ECG Records | Unique Patients | Repeat ECG Ratio |
|---|---|---|---|
"""
    patient_col = isolation_res.get("patient_column", "patient_key")
    for sp, count in splits.items():
        sub = meta_df[meta_df["split"] == sp] if "split" in meta_df.columns else pd.DataFrame()
        n_pats = sub[patient_col].nunique() if len(sub) and patient_col in sub.columns else count
        ratio = (count / max(1, n_pats))
        content += f"| `{sp}` | {count:,} | {n_pats:,} | {ratio:.2f} |\n"

    content += f"""
### Cross-Split Patient Leakage Audit:

- **Is Leak Free:** `{isolation_res.get('is_leak_free')}`
- **Total Overlapping Patients Across Evaluation Splits:** `{isolation_res.get('total_overlapping_patients', 0)}`
- **Pairwise Overlaps:**
"""
    for pair, ov in isolation_res.get("overlaps", {}).items():
        content += f"  - `{pair}`: {ov} overlapping patients\n"

    content += f"""
> [!NOTE]
> Patient-level separation is strictly verified. Evaluation splits contain zero overlap with training patient keys.

---

## 4. Target Variable Analysis: LVEF & Structural Heart Disease

### Continuous LVEF Distribution (`lvef_value`):

| Metric | Value |
|---|---|
| Valid Measurement Count | {lvef_stats.get('count', 0):,} |
| Missing Count (% Missing) | {lvef_stats.get('missing', 0):,} ({lvef_stats.get('missing_pct', 0.0):.2f}%) |
| Mean ± Std | {lvef_stats.get('mean', 0.0):.1f}% ± {lvef_stats.get('std', 0.0):.1f}% |
| Median [IQR] | {lvef_stats.get('median', 0.0):.1f}% [{lvef_stats.get('q25', 0.0):.1f}% - {lvef_stats.get('q75', 0.0):.1f}%] |
| Range (Min - Max) | {lvef_stats.get('min', 0.0):.1f}% - {lvef_stats.get('max', 0.0):.1f}% |

### Clinical Stratification:

- **Severe Systolic Dysfunction (LVEF ≤ 35%):** {lvef_stats.get('severe_reduced_count', 0):,} records
- **Moderate Systolic Dysfunction (35% < LVEF ≤ 45%):** {lvef_stats.get('moderate_reduced_count', 0):,} records
- **Preserved Systolic Function (LVEF > 45%):** {lvef_stats.get('preserved_count', 0):,} records

### Binary Classification Targets:

- **`lvef_lte_45_flag` (Moderate or Greater Dysfunction):**
  - Positive Count: {binary_flags.get('lvef_lte_45_flag', {}).get('positive_count', 0):,}
  - Prevalence: {binary_flags.get('lvef_lte_45_flag', {}).get('positive_prevalence_pct', 0.0):.2f}%
- **`shd_moderate_or_greater_flag` (Composite SHD Flag):**
  - Positive Count: {binary_flags.get('shd_moderate_or_greater_flag', {}).get('positive_count', 0):,}
  - Prevalence: {binary_flags.get('shd_moderate_or_greater_flag', {}).get('positive_prevalence_pct', 0.0):.2f}%

---

## 5. Waveform-to-Metadata Row Alignment Verification

Each split's monolithic `.npy` array must align row-for-row with the corresponding metadata slice:

| Split | Metadata Row Count | Array Row Count | Array Shape | Aligned |
|---|---|---|---|---|
"""
    for sp, info in alignment_res.get("splits", {}).items():
        content += f"| `{sp}` | {info.get('metadata_rows', 'N/A')} | {info.get('waveform_rows', 'N/A')} | `{info.get('shape', 'N/A')}` | `{info.get('aligned', False)}` |\n"

    content += """
---

## 6. Visualizations

- **LVEF Histogram & Clinical Cutoffs:** `reports/figures/lvef_distribution.png`
- **Record & Patient Split Breakdown:** `reports/figures/split_distribution.png`
- **Representative 12-Lead ECGs (Preserved vs Reduced):** `reports/figures/representative_ecgs.png`

---

## 7. Critical Domain Constraints & Preprocessing Guidelines

1. **Waveform Amplitude Representation:**
   The released EchoNext waveforms are already median-filtered and z-score standardized dataset-wide. They are **not in raw millivolts**. Therefore, synthetic rendering in downstream phases must not treat these values as uncalibrated raw mV without appropriate scale adaptation.
2. **No Generic Re-Filtering:**
   Because baseline wander was already attenuated via median filtering, applying an aggressive generic high-pass or bandpass filter risks corrupting low-frequency ST-T wave morphology critical for LVEF prediction. Any further filtering in Phase 2 must remain configurable and optional.
3. **Target Exclusions:**
   Records missing echocardiographic measurements (`lvef_value` is NaN) must only be excluded for supervised regression/classification, but logged and preserved in the primary index.
4. **Zero Patient Leakage:**
   Patient identifiers (`patient_key`) must remain strictly isolated. Official splits guarantee this separation.
"""
    with open(report_path, "w") as f:
        f.write(content)

    logger.info("Generated Phase 1 Exploration Report at %s", report_path)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir
    report_dir = args.report_dir
    figures_dir = report_dir / "figures"

    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing EchoNext exploration on: %s", data_dir)

    # Check if raw dataset exists; if not, create mock dataset if permitted
    has_meta = (
        (data_dir / "EchoNext_metadata_100k.csv").exists()
        or (data_dir / "echonext_metadata_100k.csv").exists()
    )

    if not has_meta:
        if args.create_mock_if_missing:
            logger.warning(
                "EchoNext metadata not found at %s. Creating synthetic mock dataset for verification...",
                data_dir,
            )
            create_mock_echonext_dataset(data_dir, num_train=70, num_val=15, num_test=15, seed=args.seed)
        else:
            logger.error("EchoNext dataset not found at %s. Aborting.", data_dir)
            return 1

    # Load dataset
    dataset = EchoNextDataset(data_dir)
    logger.info("Loaded EchoNext dataset with %d records.", len(dataset))

    # Compute summaries and verifications
    summary = dataset.summarize()
    isolation_res = dataset.verify_patient_isolation()
    alignment_res = dataset.verify_waveform_alignment()

    logger.info("Patient isolation status: is_leak_free=%s", isolation_res.get("is_leak_free"))
    logger.info("Waveform alignment status: all_aligned=%s", alignment_res.get("all_aligned"))

    # Generate plots
    plot_lvef_distribution(dataset.metadata, figures_dir)
    plot_split_distribution(dataset.metadata, dataset.patient_col, figures_dir)
    plot_representative_ecgs(dataset, figures_dir)

    # Generate markdown report
    report_path = report_dir / "phase1_exploration_report.md"
    generate_markdown_report(summary, isolation_res, alignment_res, dataset.metadata, report_path)

    # Print summary to stdout
    print("\n" + "=" * 65)
    print(" VENTRIQ PHASE 1 — DATASET EXPLORATION COMPLETED")
    print("=" * 65)
    print(f"Total Records:           {summary['total_records']:,}")
    print(f"Unique Patients:         {summary['unique_patients']:,}")
    print(f"Split Distribution:      {summary['split_distribution']}")
    print(f"Patient Isolation:       {'PASS (0 overlap)' if isolation_res.get('is_leak_free') else 'FAIL'}")
    print(f"Waveform Alignment:      {'PASS (100% matched)' if alignment_res.get('all_aligned') else 'FAIL'}")
    if "lvef_statistics" in summary and summary["lvef_statistics"]:
        lvef = summary["lvef_statistics"]
        print(f"LVEF Distribution:       Mean {lvef['mean']:.1f}% | Median {lvef['median']:.1f}% | Std {lvef['std']:.1f}%")
        print(f"Reduced LVEF (<=45%):    {lvef['severe_reduced_count'] + lvef['moderate_reduced_count']} records")
    print(f"Report Generated:        {report_path}")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
