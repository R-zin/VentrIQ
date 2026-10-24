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

### Phase 3 — Dataset Manifest & Split Reproducibility (Complete)
- **Canonical Record Schema (`ECGRecord`):** Immutable dataclass (`frozen=True`) linking global record IDs (`ECHONEXT_{split}_{index:06d}`), patient keys, split partitions, memory-mapped waveform slice coordinates `(split, local_index)`, continuous LVEF targets, binary $\le 45\%$ labels, and structural heart disease flags.
- **Audit Logging & Cryptographic Traceability (`DatasetManifest`):** Computes SHA-256 hashes of all raw source binaries and metadata files, recording retained record counts, exclusion breakdown (`missing_lvef`, `unusable_signal_quality`), target distributions, software environment versions, and random seed in [`reports/dataset_manifest.json`](reports/dataset_manifest.json).
- **Patient-Level Split Verification & Preservation (`src/data/splits.py`):**
  - `verify_patient_isolation`: Audits dataset splits and raises `PatientLeakageError` if any patient identity is shared between partitions.
  - `PatientGroupKFold` & `StratifiedPatientKFold`: Greedy-balanced patient-group cross-validation ensuring zero patient leakage across train and validation folds.
- **Zero-Copy PyTorch Integration (`VentriQTorchDataset`):**
  - Inherits from `torch.utils.data.Dataset`.
  - Slices waveforms on-the-fly directly from memory-mapped `.npy` files with negligible memory footprint.
  - Multi-worker safe across DataLoader processes via deterministic `ventriq_worker_init_fn`.
  - Standardized output tensors formatted as `(12, 2500)` float32 ready for vision/1D-CNN backbones.
  - Seamlessly integrates with Phase 2 signal preprocessing and quality filtering pipelines.

### Phase 4 — Digital 12-Lead ECG Paper Image Renderer (Complete)
- **Cardiology Standard Paper Geometry (`src/rendering/layout.py`):**
  - Paper speed: $25.0\text{ mm/s}$ ($1\text{ mm} = 0.04\text{ s} = 40\text{ ms}$; $5\text{ mm} = 0.20\text{ s} = 200\text{ ms}$).
  - Voltage calibration: $10.0\text{ mm/mV}$ ($1\text{ mm} = 0.1\text{ mV}$; $5\text{ mm} = 0.5\text{ mV}$; $10\text{ mm} = 1.0\text{ mV}$).
  - Standard physical paper canvas ($280\text{ mm} \times 190\text{ mm}$) at configurable DPI ($150\text{--}300\text{ DPI}$).
- **Clinical 3x4 + 1 Rhythm Strip Layout:**
  - Row 0: Lead I ($0\text{--}2.5\text{ s}$), aVR ($2.5\text{--}5.0\text{ s}$), V1 ($5.0\text{--}7.5\text{ s}$), V4 ($7.5\text{--}10.0\text{ s}$)
  - Row 1: Lead II ($0\text{--}2.5\text{ s}$), aVL ($2.5\text{--}5.0\text{ s}$), V2 ($5.0\text{--}7.5\text{ s}$), V5 ($7.5\text{--}10.0\text{ s}$)
  - Row 2: Lead III ($0\text{--}2.5\text{ s}$), aVF ($2.5\text{--}5.0\text{ s}$), V3 ($5.0\text{--}7.5\text{ s}$), V6 ($7.5\text{--}10.0\text{ s}$)
  - Row 3: Continuous Lead II 10-second rhythm strip ($0.0\text{--}10.0\text{ s}$).
- **Millimeter Background Grid & Calibration:**
  - Precise minor ($1.0\text{ mm}$) and major ($5.0\text{ mm}$) grid generation with pre-cached high-performance rendering.
  - Multiple clinical styling presets (`standard_pink`, `classic_red`, `subtle_gray`, `clinical_green`, `no_grid`).
  - Standard clinical $1.0\text{ mV} (10\text{ mm}) \times 0.20\text{ s} (5\text{ mm})$ rectangular calibration pulses at each lead baseline.
- **Waveform Rendering Engine (`src/rendering/renderer.py`):**
  - Smooth anti-aliased trace rendering with optional 2x supersampling.
  - EchoNext amplitude adaptation: maps z-score standardized units to millimeters without false millivolt assumptions or inter-lead clipping.
  - Clinical typography with lead labels (`I`, `II`, `aVR`, etc.) and header/footer parameters.
- **End-to-End PyTorch Vision Pipeline Integration:**
  - Renderer acts directly as a PyTorch transform `VentriQTorchDataset(transform=renderer)` yielding `(3, H, W)` float32 image tensors for MobileNetV3 backbones.
- **Batch Image Generator & Traceability Manifest (`scripts/generate_ecg_images.py`):**
  - Generates synthetic ECG paper images into `data/synthetic/{split}/{record_id}.png`.
  - Produces complete `data/synthetic/image_manifest.csv` strictly linking each image to patient key, split, LVEF target, and binary classification labels.


### Phase 5 — Smartphone Photo Simulation & Vision Preprocessing (Complete)
- **Real-World Smartphone Capture Simulation (`src/rendering/augmentation.py`):**
  - **3D Camera Pose & Perspective Homography:** Simulates camera yaw ($\pm 15^\circ$), pitch ($\pm 15^\circ$), and roll ($\pm 10^\circ$) via 3D pinhole camera rotation and 8-parameter homography projection.
  - **Ground-Truth Corner Coordinate Tracking:** Records exact transformed corner coordinates for quantitative evaluation of downstream localization algorithms.
  - **Non-Uniform Illumination & Vignetting:** Radial lighting gradients and flash hotspots.
  - **Polygonal Cast Shadows:** Hand and smartphone body occlusions with realistic soft Gaussian penumbras.
  - **Specular Glare:** Localized high-intensity reflection hotspots simulating overhead clinic lighting.
  - **Paper Deformations & Optical Artifacts:** Cylindrical page curl, defocus blur, motion blur, additive Gaussian sensor noise, and JPEG compression quantization.
  - **Severity Presets (`configs/augmentation.yaml`):** `clean`, `mild`, `moderate`, and `severe` configurations.
- **Document Localization & Corner Detection (`src/preprocessing/perspective.py`):**
  - Robust Otsu bimodal segmentation and Sobel gradient magnitude edge detection.
  - Extremal convex quadrilateral fitting extracting ordered corners `[TL, TR, BR, BL]`.
  - Achieves $< 5\text{ px}$ mean corner localization error against simulation ground truth.
- **Four-Point Perspective Rectification:**
  - Inverts projective homography to unwarp tilted paper documents back into canonical rectilinear coordinates matching standard $280\text{ mm} \times 190\text{ mm}$ ECG paper.
- **Illumination Normalization & Contrast Enhancement (`src/preprocessing/image.py`):**
  - Background illumination estimation via morphological opening and large-kernel Gaussian filtering.
  - Division normalization ($I_{norm} = I / I_{bg} \times 245$) eliminating cast shadow gradients while preserving fine waveform traces.
  - Contrast Limited Adaptive Histogram Equalization (CLAHE) optimizing trace and grid visibility.
- **MobileNetV3 Edge Tensor Formatting (`src/preprocessing/transforms.py`):**
  - Standardized resizing ($224 \times 224$ or high-res $448 \times 448$) and float32 tensor conversion.
  - ImageNet channel standardization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  - Optional 12-lead multi-crop patch extraction yielding shape $(12, 3, 112, 224)$.
  - `MobileNetV3Preprocessor` callable pipeline integrating directly into PyTorch DataLoaders.
- **End-to-End CLI Pipeline & Evaluation (`scripts/validate_pipeline.py`):**
  - Evaluates Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
  - Generates multi-stage comparative figure and [`reports/phase5_vision_report.md`](reports/phase5_vision_report.md).

---

## Repository Structure

```text
VentriQ/
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions automated test & exploration CI
├── configs/
│   ├── augmentation.yaml              # Phase 5 smartphone distortion & capture presets
│   ├── preprocessing.yaml             # Phase 2 preprocessing & quality control configuration
│   └── rendering.yaml                 # Phase 4 ECG paper rendering & grid configuration
├── docs/
│   └── PHASE_PLANS.md                 # Engineering specifications for Phases 1 - 5
├── data/
│   ├── raw/
│   │   └── echonext/                  # Raw EchoNext dataset (or synthetic mock)
│   ├── processed/
│   │   └── quality_summary.json       # Phase 2 aggregate quality statistics
│   └── synthetic/
│       ├── train/                     # Rendered synthetic ECG paper recordings
│       ├── val/
│       ├── test/
│       └── image_manifest.csv         # Waveform-to-image clinical metadata manifest
├── notebooks/
│   ├── 01_explore_echonext.ipynb       # Phase 1 dataset exploration & schema verification
│   ├── 02_validate_preprocessing.ipynb # Phase 2 signal quality & filtering verification
│   ├── 03_validate_dataset_manifest.ipynb # Phase 3 manifest & PyTorch dataset verification
│   ├── 04_visualize_ecg_images.ipynb  # Phase 4 ECG paper rendering & layout verification
│   └── 05_validate_vision_pipeline.ipynb # Phase 5 smartphone simulation & vision pipeline
├── reports/
│   ├── dataset_manifest.json          # Phase 3 canonical dataset manifest
│   ├── phase1_exploration_report.md   # Generated Phase 1 summary report
│   ├── phase2_quality_report.md       # Generated Phase 2 quality-control report
│   ├── phase3_manifest_report.md      # Generated Phase 3 manifest & reproducibility report
│   ├── phase4_rendering_report.md     # Generated Phase 4 ECG paper rendering report
│   ├── phase5_vision_report.md        # Generated Phase 5 smartphone vision report
│   └── figures/                       # Diagnostic plots, rendered ECGs & vision stages
├── scripts/
│   ├── explore_dataset.py             # Phase 1 dataset exploration CLI
│   ├── preprocess_ecg.py              # Phase 2 signal preprocessing & QC CLI
│   ├── build_dataset_manifest.py      # Phase 3 dataset manifest & reproducibility CLI
│   ├── generate_ecg_images.py         # Phase 4 waveform-to-image paper renderer CLI
│   └── validate_pipeline.py           # Phase 5 end-to-end vision pipeline CLI
├── src/
│   ├── data/
│   │   ├── dataset.py                 # Phase 3 ECGRecord, DatasetManifest, VentriQTorchDataset
│   │   ├── echonext.py                # EchoNext dataset loader and mmap interface
│   │   ├── signal.py                  # Legacy backward-compatible signal module
│   │   └── splits.py                  # Phase 3 patient leakage audit & PatientGroupKFold
│   ├── preprocessing/
│   │   ├── image.py                   # Phase 5 shadow division & CLAHE contrast enhancement
│   │   ├── perspective.py             # Phase 5 corner detection & 4-point homography unwarping
│   │   ├── quality.py                 # Phase 2 ECGQualityEvaluator and QualityReport
│   │   ├── signal.py                  # Phase 2 ECGSignalPreprocessor and ProcessedECG
│   │   └── transforms.py              # Phase 5 MobileNetV3Preprocessor & vision transforms
│   ├── rendering/
│   │   ├── augmentation.py            # Phase 5 smartphone 3D pose, shadows, glare & noise
│   │   ├── layout.py                  # Phase 4 physical paper geometry, 3x4 layout & grid
│   │   └── renderer.py                # Phase 4 ECGPaperRenderer, ECGImage & PyTorch transform
│   └── utils/
│       ├── logging.py                 # Standardized logger
│       └── paths.py                   # Project path resolution utilities
├── tests/
│   ├── test_augmentation.py           # Phase 5 distortion simulation tests (12 tests)
│   ├── test_dataset.py                # Phase 3 PyTorch dataset & manifest tests (9 tests)
│   ├── test_echonext.py               # Phase 1 unit tests (11 tests)
│   ├── test_image_preprocessing.py    # Phase 5 illumination & transform tests (8 tests)
│   ├── test_perspective.py            # Phase 5 corner detection & rectification tests (6 tests)
│   ├── test_quality.py                # Phase 2 quality evaluator tests (8 tests)
│   ├── test_renderer.py               # Phase 4 ECG paper renderer tests (15 tests)
│   ├── test_signal.py                 # Phase 2 signal preprocessor tests (8 tests)
│   └── test_splits.py                 # Phase 3 patient isolation & K-Fold tests (7 tests)
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

### 2. Running Quality Gates & Tests

Execute the comprehensive test suite (94 unit, stress, and robustness tests with coverage enforcement):

```bash
# Run pytest with full coverage report and XML export
uv run pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=85

# Run Ruff linter and formatting checks
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests

# Run Mypy static type checking
uv run mypy src
```

### 3. Running Preprocessing, Exploration & Manifest Scripts

Run the Phase 1 exploration CLI:
```bash
uv run python scripts/explore_dataset.py --report-dir reports/
```

Run the Phase 2 preprocessing & QC audit CLI:
```bash
uv run python scripts/preprocess_ecg.py --report-dir reports/
```

Run the Phase 3 dataset manifest generation CLI:
```bash
uv run python scripts/build_dataset_manifest.py --report-dir reports/
```

Run the Phase 4 waveform-to-image paper renderer CLI:
```bash
uv run python scripts/generate_ecg_images.py --report-dir reports/ --num-samples 15
```

Run the Phase 5 smartphone vision simulation & rectification CLI:
```bash
uv run python scripts/validate_pipeline.py --report-dir reports/ --num-samples 5
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
- Phase 3 Manifest & PyTorch Dataset:
  ```bash
  uv run jupyter notebook notebooks/03_validate_dataset_manifest.ipynb
  ```
- Phase 4 ECG Paper Rendering & Layout:
  ```bash
  uv run jupyter notebook notebooks/04_visualize_ecg_images.ipynb
  ```
- Phase 5 Smartphone Simulation & Vision Pipeline:
  ```bash
  uv run jupyter notebook notebooks/05_validate_vision_pipeline.ipynb
  ```

---

## Deliverables Summary

- **Phase 1 Exploration Report:** [`reports/phase1_exploration_report.md`](reports/phase1_exploration_report.md)
- **Phase 2 Quality Report:** [`reports/phase2_quality_report.md`](reports/phase2_quality_report.md)
- **Phase 3 Manifest & Reproducibility Report:** [`reports/phase3_manifest_report.md`](reports/phase3_manifest_report.md)
- **Phase 4 ECG Paper Rendering Report:** [`reports/phase4_rendering_report.md`](reports/phase4_rendering_report.md)
- **Phase 5 Smartphone Vision Report:** [`reports/phase5_vision_report.md`](reports/phase5_vision_report.md)
- **Canonical Dataset Manifest:** [`reports/dataset_manifest.json`](reports/dataset_manifest.json)
- **Synthetic Image Manifest:** [`data/synthetic/image_manifest.csv`](data/synthetic/image_manifest.csv)
- **Unit & Stress Tests:** 94 passing tests (100% pass rate, 90.12% code coverage) covering dataset loading, memory-mapping, patient isolation, zero-phase filtering, fault detection, PyTorch zero-copy dataset slicing, multi-worker safety, GroupKFold, clinical ECG paper geometry, calibration pulses, waveform rendering, 3D perspective distortion, corner detection, homography rectification, shadow division normalization, MobileNetV3 tensors, boundary stress conditions, and multi-worker DataLoader concurrency.
- **Continuous Integration:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) executes a 4-stage parallel quality gate: Ruff code style linting, Mypy static typing, Pytest coverage enforcement (minimum 85%), and clinical pipeline end-to-end integration artifact verification.
