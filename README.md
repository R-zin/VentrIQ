# VentriQ: Phased ECG Preprocessing Pipeline

[![CI Pipeline](https://github.com/R-zin/VentrIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/R-zin/VentrIQ/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

VentriQ is a biomedical research pipeline designed to estimate **Left Ventricular Ejection Fraction (LVEF)** and identify ECG patterns associated with **Structural Heart Disease (SHD)** from smartphone photographs of standard 12-lead ECGs.

The project targets deployment with lightweight computer vision models (such as MobileNetV3) for point-of-care smartphone screening, utilizing the **EchoNext** dataset (PhysioNet / Columbia University Irving Medical Center).

---

## Current Status

### Phase 1 — Dataset Exploration & Alignment (Complete)
- **Memory-Mapped Loading (`mmap_mode='r'`):** Reads multi-gigabyte EchoNext monolithic waveform arrays on demand without loading whole datasets into RAM.
- **Patient-Level Isolation Audit:** Verifies zero cross-split patient leakage between `train`, `val`, and `test` partitions.
- **Waveform-to-Metadata Row Alignment:** Validates 1-to-1 correspondence between metadata rows and split array slices.
- **Target Definitions:** Handles continuous LVEF (`lvef_value`) and official clinical binary thresholds:
  - Moderate-to-severe systolic dysfunction: $\text{LVEF} \le 45\%$ (`lvef_lte_45_flag`)
  - Severe systolic dysfunction: $\text{LVEF} \le 35\%$
  - Structural heart disease composite flag: `shd_moderate_or_greater_flag`
- **Lead Layout:** Standard clinical 12 leads (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`), $f_s = 250$ Hz, 10.0 seconds ($N=2500$ samples).

### Phase 2 — Signal Preprocessing & Quality Control (Complete)
- **Signal Sanity Validation:** Explicitly checks dimensions, data types, NaNs, infinities, and empty inputs.
- **Comprehensive Quality Control (`src/preprocessing/quality.py`):** Automatically evaluates and records quality flags:
  - Flatline / disconnected leads (variance $< 1.0 \times 10^{-5}$)
  - Saturation / rail clipping ($\ge 8$ consecutive extremal samples)
  - Powerline hum ($50\text{ Hz} / 60\text{ Hz}$) and low SNR detection ($< -3.0\text{ dB}$)
  - Multi-lead unsuitability threshold ($>2$ faulty leads)
- **Zero-Phase Forward-Backward Filtering (`src/preprocessing/signal.py`):**
  - Butterworth second-order sections (`sosfiltfilt`) ensuring **0 ms phase delay** at cardiac peaks.
  - Safe boundary padding handling for short signals without runtime crashes.
  - Optional and configurable via [`configs/preprocessing.yaml`](configs/preprocessing.yaml) to preserve EchoNext's released preprocessed morphology by default.
- **Morphology-Preserving Normalization:** Optional lead z-score, global training set z-score (preventing test set leakage), min-max, and robust IQR scaling.
- **Multi-Stage Representation:** Preserves distinct states: `original`, `quality_checked`, `filtered`, `normalized`, and `final`.

> [!NOTE]
> For detailed engineering plans, tasks, and completion criteria for subsequent phases (Dataset Preparation, Waveform Renderer, and Smartphone Vision Simulation), see [**`docs/PHASE_PLANS.md`**](docs/PHASE_PLANS.md).

---

## Repository Structure

```text
VentriQ/
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions automated test & exploration CI
├── configs/
│   └── preprocessing.yaml             # Preprocessing & quality control configuration
├── docs/
│   └── PHASE_PLANS.md                 # Engineering specifications for Phases 2 - 5
├── data/
│   ├── raw/
│   │   └── echonext/                  # Raw EchoNext dataset (or synthetic mock)
│   ├── processed/
│   │   └── quality_summary.json       # Phase 2 aggregate quality statistics
│   └── synthetic/
├── notebooks/
│   ├── 01_explore_echonext.ipynb       # Phase 1 dataset exploration & schema verification
│   ├── 02_validate_preprocessing.ipynb # Phase 2 signal quality & filtering verification
│   └── explore_echonext.ipynb
├── reports/
│   ├── phase1_exploration_report.md   # Generated Phase 1 summary report
│   ├── phase2_quality_report.md       # Generated Phase 2 quality-control report
│   └── figures/                       # Diagnostic plots (LVEF, filtering, lead faults)
├── scripts/
│   ├── explore_dataset.py             # Phase 1 dataset exploration CLI
│   └── preprocess_ecg.py              # Phase 2 signal preprocessing & QC CLI
├── src/
│   ├── data/
│   │   ├── echonext.py                # EchoNext dataset loader and mmap interface
│   │   └── signal.py                  # Legacy backward-compatible signal module
│   ├── preprocessing/
│   │   ├── quality.py                 # ECGQualityEvaluator and QualityReport
│   │   └── signal.py                  # ECGSignalPreprocessor and ProcessedECG
│   └── utils/
│       ├── logging.py                 # Standardized logger
│       └── paths.py                   # Project path resolution utilities
├── tests/
│   ├── test_echonext.py               # Phase 1 unit tests (11 tests)
│   ├── test_quality.py                # Phase 2 quality evaluator tests (8 tests)
│   └── test_signal.py                 # Phase 2 signal preprocessor tests (8 tests)
├── pyproject.toml                     # Project metadata and dependencies
└── uv.lock                            # Deterministic lockfile
```

---

## Getting Started

### 1. Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, reproducible dependency management:

```bash
# Clone the repository
git clone https://github.com/R-zin/VentrIQ.git
cd VentriQ

# Install dependencies and setup virtual environment
uv sync
```

### 2. Running Unit Tests

Execute the comprehensive test suite (27 unit tests):

```bash
uv run pytest tests/ -v
```

### 3. Running Preprocessing & Exploration Scripts

Run the Phase 1 exploration CLI:
```bash
uv run python scripts/explore_dataset.py --report-dir reports/
```

Run the Phase 2 preprocessing & QC audit CLI:
```bash
uv run python scripts/preprocess_ecg.py --report-dir reports/
```

### 4. Interactive Jupyter Notebooks

- Phase 1 Exploration:
  ```bash
  uv run jupyter notebook notebooks/01_explore_echonext.ipynb
  ```
- Phase 2 Preprocessing & Quality Verification:
  ```bash
  uv run jupyter notebook notebooks/02_validate_preprocessing.ipynb
  ```

---

## Deliverables Summary

- **Phase 1 Exploration Report:** [`reports/phase1_exploration_report.md`](reports/phase1_exploration_report.md)
- **Phase 2 Quality Report:** [`reports/phase2_quality_report.md`](reports/phase2_quality_report.md)
- **Unit Tests:** 27 passing tests (100% pass rate) covering dataset loading, memory-mapping, patient isolation, zero-phase filtering, fault detection, and edge cases.
- **Continuous Integration:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) validates all tests and runs automated reporting on pushes and PRs.
