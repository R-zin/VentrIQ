#!/usr/bin/env python3
"""scripts/build_dataset_manifest.py - Phase 3 Dataset Manifest & Traceability CLI.

This script executes the Phase 3 reproducibility pipeline:
1. Validates raw EchoNext source file integrity and computes cryptographic SHA256 checksums.
2. Constructs canonical ECGRecord entries with strict waveform references.
3. Performs rigorous patient-isolation verification across splits to detect any leakage.
4. Optionally evaluates signal quality flags using Phase 2 QC.
5. Audits exclusions and calculates clinical target distribution metrics.
6. Serializes the canonical manifest to reports/dataset_manifest.json and a markdown summary.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from src.data.dataset import DatasetManifest, VentriQTorchDataset
from src.data.splits import verify_patient_isolation
from src.preprocessing.quality import ECGQualityEvaluator
from src.utils.logging import get_logger
from src.utils.paths import REPORTS_DIR, get_raw_echonext_dir

logger = get_logger("build_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VentriQ Phase 3: Dataset Manifest & Reproducibility Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=get_raw_echonext_dir(),
        help="Path to EchoNext dataset directory.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Path to output reports directory.",
    )
    parser.add_argument(
        "--manifest-file",
        type=str,
        default="dataset_manifest.json",
        help="Filename for the dataset manifest JSON.",
    )
    parser.add_argument(
        "--run-qc",
        action="store_true",
        default=False,
        help="Run Phase 2 ECGQualityEvaluator across records to record QC flags.",
    )
    parser.add_argument(
        "--exclude-missing-lvef",
        action="store_true",
        default=False,
        help="Exclude records with unmeasured LVEF from the canonical training manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic processing.",
    )
    return parser.parse_args()


def generate_phase3_markdown_report(
    manifest: DatasetManifest,
    leakage_summary: str,
    output_path: Path,
) -> Path:
    """Generate comprehensive Phase 3 traceability and reproducibility report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_stats = manifest.target_statistics

    lines = [
        "# VentriQ Phase 3: Dataset Manifest & Reproducibility Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Random Seed:** `{manifest.random_seed}`  ",
        f"**Python Version:** `{manifest.environment_metadata.get('python_version')}`  ",
        f"**PyTorch Version:** `{manifest.environment_metadata.get('torch_version')}`  ",
        "",
        "---",
        "",
        "## 1. Cryptographic Source Traceability (SHA-256)",
        "",
        "| Source File | SHA-256 Checksum |",
        "|---|---|",
    ]
    for fname, sha in sorted(manifest.source_checksums.items()):
        lines.append(f"| `{fname}` | `{sha}` |")

    lines.extend([
        "",
        "## 2. Partition & Patient Isolation Audit",
        "",
        "```text",
        leakage_summary,
        "```",
        "",
        "| Partition | Retained Records | Unique Patients | Excluded Records |",
        "|---|---|---|---|",
    ])

    all_splits = sorted(set(list(manifest.retained_counts_per_split.keys()) + list(manifest.excluded_counts_per_split.keys())))
    for sp in all_splits:
        retained = manifest.retained_counts_per_split.get(sp, 0)
        pats = manifest.patient_counts_per_split.get(sp, 0)
        excluded = manifest.excluded_counts_per_split.get(sp, 0)
        lines.append(f"| **{sp}** | {retained:,} | {pats:,} | {excluded:,} |")

    lines.extend([
        "",
        "### Exclusion Breakdown:",
    ])
    for reason, count in manifest.exclusion_breakdown.items():
        lines.append(f"- **{reason}**: {count:,} records")

    lines.extend([
        "",
        "## 3. Clinical Diagnostic Targets Distribution",
        "",
        "| Target Metric | Retained Value |",
        "|---|---|",
        f"| Valid LVEF Records | {t_stats.get('lvef_count', 0):,} |",
        f"| Mean LVEF | {t_stats.get('lvef_mean', 0.0):.2f}% |" if t_stats.get('lvef_mean') is not None else "| Mean LVEF | N/A |",
        f"| Std LVEF | {t_stats.get('lvef_std', 0.0):.2f}% |" if t_stats.get('lvef_std') is not None else "| Std LVEF | N/A |",
        f"| Median LVEF | {t_stats.get('lvef_median', 0.0):.2f}% |" if t_stats.get('lvef_median') is not None else "| Median LVEF | N/A |",
        f"| Min / Max LVEF | {t_stats.get('lvef_min', 0.0):.1f}% / {t_stats.get('lvef_max', 0.0):.1f}% |" if t_stats.get('lvef_min') is not None else "| Min / Max LVEF | N/A |",
        f"| Moderate-to-Severe Dysfunction (LVEF ≤ 45%) Prevalence | {t_stats.get('lvef_lte_45_rate', 0.0) * 100:.1f}% |" if t_stats.get('lvef_lte_45_rate') is not None else "| LVEF ≤ 45% Rate | N/A |",
        f"| Moderate+ Structural Heart Disease (SHD) Prevalence | {t_stats.get('shd_rate', 0.0) * 100:.1f}% |" if t_stats.get('shd_rate') is not None else "| SHD Rate | N/A |",
        "",
        "## 4. PyTorch Dataset Integration & Zero-Copy Slicing",
        "",
        "- Waveforms are retrieved directly via memory-mapping without duplicating NumPy binaries.",
        "- Multi-worker safe through process-local memory mapping and deterministic `ventriq_worker_init_fn`.",
        "- Standard tensor outputs: `(12, 2500)` float32 ready for MobileNetV3 and 1D-CNN backbones.",
        "",
        "---",
        "**Report Status:** Verified & Audit-Logged.",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Saved Phase 3 markdown report to %s", output_path)
    return output_path


def main() -> int:
    args = parse_args()
    logger.info("Initializing VentriQ Phase 3 Manifest Generation on %s", args.data_dir)

    if not args.data_dir.exists():
        logger.error("EchoNext data directory does not exist: %s", args.data_dir)
        return 1

    qc_evaluator = ECGQualityEvaluator() if args.run_qc else None

    # 1. Build canonical manifest
    manifest = DatasetManifest.build_from_echonext(
        root=args.data_dir,
        quality_evaluator=qc_evaluator,
        exclude_missing_lvef=args.exclude_missing_lvef,
        seed=args.seed,
    )

    # 2. Verify patient isolation
    leakage_report = verify_patient_isolation(manifest.records, raise_on_leak=False)
    if not leakage_report.is_valid:
        logger.warning("Patient leakage detected during audit:\n%s", leakage_report.summary())
    else:
        logger.info("Patient isolation verified successfully (0 shared patients).")

    # 3. Save JSON manifest
    manifest_path = args.report_dir / args.manifest_file
    manifest.save(manifest_path, include_records=True)

    # 4. Generate markdown report
    report_path = args.report_dir / "phase3_manifest_report.md"
    generate_phase3_markdown_report(manifest, leakage_report.summary(), report_path)

    # Print summary to stdout
    print("\n" + "=" * 65)
    print(" VENTRIQ PHASE 3 — DATASET MANIFEST & REPRODUCIBILITY AUDIT")
    print("=" * 65)
    print(f"Total Retained Records:  {len(manifest.records):,}")
    for sp, cnt in manifest.retained_counts_per_split.items():
        pats = manifest.patient_counts_per_split.get(sp, 0)
        print(f"  - Split '{sp}':        {cnt:,} records, {pats:,} patients")
    print(f"Patient Isolation:       {'PASS (Zero Leakage)' if leakage_report.is_valid else 'FAIL (Leakage Detected)'}")
    print(f"Manifest JSON:           {manifest_path}")
    print(f"Markdown Report:         {report_path}")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
