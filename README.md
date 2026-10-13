# VentriQ: Phased ECG Preprocessing Pipeline

[![CI Pipeline](https://github.com/R-zin/VentrIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/R-zin/VentrIQ/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

VentriQ is a biomedical research pipeline designed to estimate **Left Ventricular Ejection Fraction (LVEF)** and identify ECG patterns associated with **Structural Heart Disease (SHD)** from smartphone photographs of standard 12-lead ECGs.

The project targets deployment with lightweight computer vision models (such as MobileNetV3) for point-of-care smartphone screening, utilizing the **EchoNext** dataset (PhysioNet / Columbia University Irving Medical Center).

---

## Current Status: Phase 1 — Dataset Exploration & Verification

Phase 1 provides a safe, memory-efficient data loading, exploration, and verification infrastructure:

- **Memory-Mapped Loading (`mmap_mode='r'`):** Reads multi-gigabyte EchoNext monolithic waveform arrays on demand without loading whole datasets into RAM.
- **Patient-Level Isolation Audit:** Verifies zero cross-split patient leakage between `train`, `val`, and `test` partitions.
- **Waveform-to-Metadata Row Alignment:** Validates 1-to-1 correspondence between metadata rows and split array slices.
- **Target Definitions:** Handles continuous LVEF (`lvef_value`) and official clinical binary thresholds:
  - Moderate-to-severe systolic dysfunction: $\text{LVEF} \le 45\%$ (`lvef_lte_45_flag`)
  - Severe systolic dysfunction: $\text{LVEF} \le 35\%$
  - Structural heart disease composite flag: `shd_moderate_or_greater_flag`
- **Lead Layout:** Standard clinical 12 leads (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`), $f_s = 250$ Hz, 10.0 seconds ($N=2500$ samples).

> [!NOTE]
> For detailed engineering plans, tasks, and completion criteria for subsequent phases (Signal Preprocessing, Dataset Preparation, Waveform Renderer, and Smartphone Vision Simulation), see [**`docs/PHASE_PLANS.md`**](docs/PHASE_PLANS.md).

---

## Repository Structure

```text
VentriQ/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions automated test & exploration CI
├── docs/
│   └── PHASE_PLANS.md           # Engineering specifications for Phases 2 - 5
├── data/
│   ├── raw/
│   │   └── echonext/            # Raw EchoNext dataset (or synthetic mock)
│   ├── processed/
│   └── synthetic/
├── notebooks/
│   ├── 01_explore_echonext.ipynb # Interactive exploration and visualization
│   └── explore_echonext.ipynb
├── reports/
│   ├── phase1_exploration_report.md # Generated Phase 1 summary report
│   └── figures/                 # Diagnostic plots (LVEF distribution, waveforms)
├── scripts/
│   └── explore_dataset.py       # Phase 1 dataset exploration CLI
├── src/
│   ├── data/
│   │   └── echonext.py          # EchoNext dataset loader and mmap interface
│   └── utils/
│       ├── logging.py           # Standardized logger
│       └── paths.py             # Project path resolution utilities
├── tests/
│   └── test_echonext.py         # Comprehensive unit tests
├── pyproject.toml               # Project metadata and dependencies
└── uv.lock                      # Deterministic lockfile
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

Execute the test suite with `pytest`:

```bash
uv run pytest tests/ -v
```

### 3. Running Dataset Exploration

Run the automated exploration script (creates a mock dataset if raw EchoNext data is not locally present):

```bash
uv run python scripts/explore_dataset.py --report-dir reports/
```

### 4. Exploring in Jupyter

Open the Phase 1 exploration notebook:

```bash
uv run jupyter notebook notebooks/01_explore_echonext.ipynb
```

---

## Phase 1 Deliverables Summary

- **Exploration Report:** See [`reports/phase1_exploration_report.md`](reports/phase1_exploration_report.md)
- **Unit Tests:** 11 passing tests covering memory-mapping, patient isolation audits, target extraction, and error handling.
- **Continuous Integration:** `.github/workflows/ci.yml` validates tests and runs exploration reporting on pushes and pull requests.
