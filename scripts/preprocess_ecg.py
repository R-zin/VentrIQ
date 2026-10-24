#!/usr/bin/env python3
"""scripts/preprocess_ecg.py - Phase 2 ECG Signal Preprocessing & Quality Control CLI.

This script executes Phase 2 preprocessing and quality evaluation:
1. Evaluates signal quality (flatlines, clipping, SNR, numerical sanity).
2. Applies optional zero-phase forward-backward filtering.
3. Applies optional morphology-preserving normalization.
4. Generates before-and-after comparison figures.
5. Produces a comprehensive Phase 2 quality-control summary report.
6. Strictly preserves raw files and memory-maps inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.echonext import (
    EXPECTED_TIME_STEPS,
    RECORDING_DURATION_SEC,
    SAMPLING_RATE_HZ,
    STANDARD_12_LEADS,
    EchoNextDataset,
    create_mock_echonext_dataset,
)
from src.preprocessing.signal import ECGSignalPreprocessor
from src.utils.logging import get_logger
from src.utils.paths import (
    CONFIGS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    get_raw_echonext_dir,
)

logger = get_logger("preprocess_ecg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VentriQ Phase 2: ECG Preprocessing & Quality Control"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=get_raw_echonext_dir(),
        help="Path to EchoNext dataset directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "preprocessing.yaml",
        help="Path to preprocessing YAML configuration.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Path to output report directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum records to process for QC audit.",
    )
    parser.add_argument(
        "--apply-filter",
        action="store_true",
        default=False,
        help="Enable optional experimental bandpass and notch filtering.",
    )
    parser.add_argument(
        "--apply-norm",
        action="store_true",
        default=False,
        help="Enable optional normalization.",
    )
    return parser.parse_args()


def plot_filtering_comparison(
    raw_sig: np.ndarray,
    filt_sig: np.ndarray,
    norm_sig: Optional[np.ndarray],
    figures_dir: Path,
) -> Path:
    """Generate high-resolution 12-lead before-and-after filtering comparison plot."""
    out_path = figures_dir / "filtering_comparison.png"
    time_s = np.linspace(0, RECORDING_DURATION_SEC, EXPECTED_TIME_STEPS, endpoint=False)
    # 3-second zoom window for clear wave morphology
    zoom_samples = int(3.0 * SAMPLING_RATE_HZ)

    fig, axes = plt.subplots(6, 2, figsize=(14, 11), sharex=True, dpi=150)
    fig.suptitle(
        "VentriQ Phase 2: ECG Signal Processing Comparison (First 3 Seconds)\n"
        "Original Released Waveform (Blue) vs. Zero-Phase Filtered (Red)",
        fontsize=13,
        fontweight="bold",
    )

    for i in range(12):
        row = i % 6
        col = i // 6
        ax = axes[row, col]

        ax.plot(
            time_s[:zoom_samples],
            raw_sig[:zoom_samples, i],
            color="#2980b9",
            alpha=0.8,
            linewidth=1.2,
            label="Original (Released)",
        )
        ax.plot(
            time_s[:zoom_samples],
            filt_sig[:zoom_samples, i],
            color="#c0392b",
            linestyle="--",
            alpha=0.9,
            linewidth=1.1,
            label="Zero-Phase Filtered",
        )
        ax.set_ylabel(STANDARD_12_LEADS[i], fontweight="bold", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Time (seconds)", fontsize=9)
    axes[-1, 1].set_xlabel("Time (seconds)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    logger.info("Saved filtering comparison plot to %s", out_path)
    return out_path


def plot_lead_quality_distribution(
    fault_counts: Dict[str, Dict[str, int]],
    total_audited: int,
    figures_dir: Path,
) -> Path:
    """Plot distribution of lead faults (flatline, clipping, low SNR) across leads."""
    out_path = figures_dir / "lead_quality_distribution.png"

    leads = STANDARD_12_LEADS
    flat_counts = [fault_counts["flatline"].get(lead_name, 0) for lead_name in leads]
    clip_counts = [fault_counts["clipping"].get(lead_name, 0) for lead_name in leads]
    noise_counts = [fault_counts["noisy"].get(lead_name, 0) for lead_name in leads]

    x = np.arange(len(leads))
    width = 0.25

    plt.figure(figsize=(10, 4.5), dpi=150)
    plt.bar(x - width, flat_counts, width, label="Flatlined", color="#e74c3c")
    plt.bar(x, clip_counts, width, label="Clipped/Railed", color="#e67e22")
    plt.bar(x + width, noise_counts, width, label="Low SNR (< -3 dB)", color="#f1c40f")

    plt.title(
        f"Lead Quality Fault Distribution Across {total_audited} Inspected ECGs",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("ECG Lead", fontsize=10)
    plt.ylabel("Flagged Count", fontsize=10)
    plt.xticks(x, leads)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    logger.info("Saved lead quality distribution plot to %s", out_path)
    return out_path


def generate_phase2_report(
    total_audited: int,
    usable_count: int,
    unusable_count: int,
    fault_counts: Dict[str, Dict[str, int]],
    snr_stats: Dict[str, float],
    config_applied: Dict[str, Any],
    report_path: Path,
) -> None:
    """Compile Phase 2 Quality Control Report in markdown format."""
    content = f"""# VentriQ — Phase 2 Quality Control & Signal Preprocessing Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Scope:** Signal Sanity, Quality Evaluation, Configurable Zero-Phase Filtering & Normalization
**Dataset Reference:** EchoNext v1.0.0

---

## 1. Executive Summary

- **Total Inspected ECG Records:** {total_audited:,}
- **Clinically Usable Records:** {usable_count:,} ({usable_count / max(1, total_audited) * 100:.1f}%)
- **Flagged / Unusable Records:** {unusable_count:,} ({unusable_count / max(1, total_audited) * 100:.1f}%)
- **Mean Overall SNR:** {snr_stats.get("mean_overall_snr_db", 0.0):.2f} dB (Min: {snr_stats.get("min_overall_snr_db", 0.0):.2f} dB, Max: {snr_stats.get("max_overall_snr_db", 0.0):.2f} dB)
- **Active Preprocessing Mode:**
  - Filtering Enabled: `{config_applied.get("apply_filter", False)}`
  - Filter Type: `{config_applied.get("filter_type", "butterworth")}` ({config_applied.get("lowcut_hz")} - {config_applied.get("highcut_hz")} Hz, Order {config_applied.get("filter_order")})
  - Notch Filter: `{config_applied.get("apply_notch", False)}`
  - Normalization Enabled: `{config_applied.get("apply_normalization", False)}` ({config_applied.get("normalization_method", "none")})

---

## 2. Multi-Stage Signal Representation Strategy

To satisfy strict biomedical signal processing integrity, the pipeline maintains 4 distinct representations:

| Representation Stage | Purpose | Processing Applied |
|---|---|---|
| **1. Original Released Waveform** | Immutable baseline directly from dataset | Unmodified memory-mapped array |
| **2. Quality-Checked Waveform** | Annotated with structured quality report | Sanity-validated (finiteness, dimensions) |
| **3. Optional Filtered Waveform** | Research experiment baseline wander removal | Zero-phase forward-backward Butterworth SOS (0 ms delay) |
| **4. Optional Model-Normalized Waveform** | Input format for downstream neural networks | Lead-wise or global z-score / minmax |

> [!NOTE]
> EchoNext released waveforms are **already median-filtered and z-score standardized**. Automatic re-filtering is disabled by default to avoid corrupting low-frequency ST-T wave morphology.

---

## 3. Lead Quality & Anomaly Analysis

### Lead Fault Frequencies Across {total_audited} Records:

| Lead | Flatline Count | Clipped / Railed Count | Low SNR Count (< -3 dB) | Mean SNR (dB) |
|---|---|---|---|---|
"""
    for lead in STANDARD_12_LEADS:
        fl = fault_counts["flatline"].get(lead, 0)
        cl = fault_counts["clipping"].get(lead, 0)
        ns = fault_counts["noisy"].get(lead, 0)
        mean_snr = snr_stats.get(f"snr_{lead}", 0.0)
        content += f"| `{lead}` | {fl} | {cl} | {ns} | {mean_snr:.2f} |\n"

    content += """
### Fault Detection Definitions:
- **Flatline Lead:** Lead variance $< 1.0 \\times 10^{-5}$ across recording or $>60\\%$ of $0.5\\text{ s}$ windows.
- **Clipped Lead:** $\\ge 8$ consecutive samples equal to the extremal min or max boundary.
- **Low SNR Lead:** Cardiac band ($0.5\\text{--}40\\text{ Hz}$) to high-frequency noise floor ratio $< -3.0\\text{ dB}$.
- **Record Unusable Criteria:** $>2$ faulty leads in a single recording or complete flatline across all leads.

---

## 4. Verification & Diagnostics Artifacts

- **Filtering Comparison (Zero-Phase Alignment):** `reports/figures/filtering_comparison.png`
- **Lead Quality Distribution:** `reports/figures/lead_quality_distribution.png`
- **Structured JSON Summary:** `data/processed/quality_summary.json`

---

## 5. Next Steps for Phase 3 (Dataset Preparation)
1. Use `QualityReport` flags to index clean subsets for supervised LVEF training.
2. Ensure excluded records remain traceable in the dataset manifest.
3. Keep training, validation, and test normalization statistics isolated.
"""
    with open(report_path, "w") as f:
        f.write(content)
    logger.info("Generated Phase 2 Quality Report at %s", report_path)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir
    config_path = args.config
    report_dir = args.report_dir
    figures_dir = report_dir / "figures"
    processed_dir = PROCESSED_DATA_DIR

    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 2 Preprocessing & Quality Audit on: %s", data_dir)

    # Ensure dataset exists; create mock if needed
    if (
        not (data_dir / "EchoNext_metadata_100k.csv").exists()
        and not (data_dir / "echonext_metadata_100k.csv").exists()
    ):
        logger.warning(
            "EchoNext metadata not found at %s. Creating synthetic mock dataset...", data_dir
        )
        create_mock_echonext_dataset(data_dir, num_train=70, num_val=15, num_test=15, seed=42)

    dataset = EchoNextDataset(data_dir)
    total_records = len(dataset)
    audit_limit = min(args.limit, total_records)
    logger.info(
        "Dataset loaded with %d records. Auditing %d records...", total_records, audit_limit
    )

    # Initialize Preprocessor
    preprocessor = ECGSignalPreprocessor(
        config_path=config_path,
        apply_filter=args.apply_filter,
        apply_normalization=args.apply_norm,
    )

    # For comparison plot, create a demo preprocessor that filters and normalizes
    demo_preprocessor = ECGSignalPreprocessor(
        config_path=config_path,
        apply_filter=True,
        apply_normalization=True,
    )

    usable_count = 0
    unusable_count = 0
    fault_counts = {"flatline": {}, "clipping": {}, "noisy": {}}
    snr_list = []
    lead_snr_accum = {lead_name: [] for lead_name in STANDARD_12_LEADS}

    sample_raw: Optional[np.ndarray] = None
    sample_filt: Optional[np.ndarray] = None
    sample_norm: Optional[np.ndarray] = None

    for idx in range(audit_limit):
        raw_waveform = dataset.get_waveform(idx)
        processed = preprocessor.process(raw_waveform)
        rep = processed.quality_report

        if rep.is_usable:
            usable_count += 1
        else:
            unusable_count += 1

        for fl in rep.flatline_leads:
            fault_counts["flatline"][fl] = fault_counts["flatline"].get(fl, 0) + 1
        for cl in rep.clipped_leads:
            fault_counts["clipping"][cl] = fault_counts["clipping"].get(cl, 0) + 1
        for ns in rep.noisy_leads:
            fault_counts["noisy"][ns] = fault_counts["noisy"].get(ns, 0) + 1

        snr_list.append(rep.overall_snr_db)
        for lead, snr_val in rep.lead_snr_db.items():
            if lead in lead_snr_accum:
                lead_snr_accum[lead].append(snr_val)

        if idx == 0:
            demo_res = demo_preprocessor.process(raw_waveform)
            sample_raw = demo_res.original
            sample_filt = demo_res.filtered
            sample_norm = demo_res.normalized

    # Compute aggregate SNR metrics
    snr_stats: Dict[str, float] = {
        "mean_overall_snr_db": float(np.mean(snr_list)) if snr_list else 0.0,
        "min_overall_snr_db": float(np.min(snr_list)) if snr_list else 0.0,
        "max_overall_snr_db": float(np.max(snr_list)) if snr_list else 0.0,
    }
    for lead in STANDARD_12_LEADS:
        vals = lead_snr_accum[lead]
        snr_stats[f"snr_{lead}"] = float(np.mean(vals)) if vals else 0.0

    # Save comparison plot
    if sample_raw is not None and sample_filt is not None:
        plot_filtering_comparison(sample_raw, sample_filt, sample_norm, figures_dir)

    # Save fault distribution plot
    plot_lead_quality_distribution(fault_counts, audit_limit, figures_dir)

    # Save Phase 2 Report
    report_path = report_dir / "phase2_quality_report.md"
    generate_phase2_report(
        audit_limit,
        usable_count,
        unusable_count,
        fault_counts,
        snr_stats,
        preprocessor.get_config(),
        report_path,
    )

    # Save structured summary JSON
    summary_json_path = processed_dir / "quality_summary.json"
    summary_data = {
        "audited_records": audit_limit,
        "usable_records": usable_count,
        "unusable_records": unusable_count,
        "fault_counts": fault_counts,
        "snr_statistics": snr_stats,
        "preprocessor_config": preprocessor.get_config(),
        "timestamp": datetime.now().isoformat(),
    }
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved quality summary JSON to %s", summary_json_path)

    print("\n" + "=" * 65)
    print(" VENTRIQ PHASE 2 — PREPROCESSING & QC AUDIT COMPLETED")
    print("=" * 65)
    print(f"Audited Records:         {audit_limit:,}")
    print(f"Usable Records:          {usable_count:,} ({usable_count / audit_limit * 100:.1f}%)")
    print(
        f"Unusable Records:        {unusable_count:,} ({unusable_count / audit_limit * 100:.1f}%)"
    )
    print(f"Mean Overall SNR:        {snr_stats['mean_overall_snr_db']:.2f} dB")
    print(f"Quality Report:          {report_path}")
    print(f"Summary JSON:            {summary_json_path}")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
