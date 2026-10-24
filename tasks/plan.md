# Implementation Plan: Phase 5 — Smartphone Photograph Simulation & Vision Preprocessing

## Overview

Phase 5 bridges the gap between synthetic digital ECG paper recordings (Phase 4) and lightweight edge vision models (MobileNetV3) for point-of-care Left Ventricular Ejection Fraction (LVEF) estimation and Structural Heart Disease (SHD) screening.

This module develops two complementary capabilities:
1. **Smartphone Capture Simulation Engine (`src/rendering/augmentation.py`):** Synthesizes physical real-world camera artifacts—3D camera tilt (yaw, pitch, roll homography), non-uniform lighting, hand/phone cast shadows, specular glares, page curl, defocus blur, sensor noise, and JPEG compression—with ground-truth corner tracking.
2. **Computer Vision Rectification Pipeline (`src/preprocessing/perspective.py`, `image.py`, `transforms.py`):** Automatically detects document boundaries, estimates four-point homographies, unwarps perspective back to canonical planar coordinates, normalizes illumination/contrast (CLAHE), and formats tensors ($B \times 3 \times H \times W$, float32) for MobileNetV3 backbones.

---

## Architecture Decisions

- **Lightweight Dependency Stack (NumPy, SciPy, Pillow, PyTorch):**
  - Implement perspective warping and homography via Pillow's native C-accelerated `Image.Transform.PERSPECTIVE` and PyTorch's `torch.nn.functional.grid_sample`.
  - Implement edge detection, corner localization, and filtering with `scipy.ndimage` and `numpy`.
  - Avoids heavyweight OpenCV runtime dependencies, maintaining full cross-platform CI portability across macOS and Linux without binary wheel conflicts.
- **Ground-Truth Corner Coordinate Tracking:**
  - The simulation engine explicitly exports transformed quadrilateral corner coordinates alongside distorted images, allowing exact mathematical evaluation of corner localization and rectification accuracy.
- **Multi-Stage Processing Pipeline:**
  - Strictly preserves intermediate stages: `distorted_photo` $\rightarrow$ `detected_corners` $\rightarrow$ `rectified_canvas` $\rightarrow$ `illumination_corrected` $\rightarrow$ `mobilenet_tensor`.
- **Configurable Artifact Severity Levels:**
  - Provide `mild`, `moderate`, and `severe` presets in `configs/augmentation.yaml` to support both realistic benign testing and high-stress vision augmentation.
- **Metric-Driven Quality Assurance:**
  - Automated evaluation of Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) between rectified recordings and canonical ground-truth paper.

---

## Dependency Graph

```text
Phase 4: ECGPaperRenderer (Canonical Paper Image)
    │
    ▼
Task 1: Camera 3D Perspective & Homography Warp (src/rendering/augmentation.py)
    │
    ▼
Task 2: Non-Uniform Illumination, Shadows & Glares (src/rendering/augmentation.py)
    │
    ▼
Task 3: Paper Deformations, Optical Blur & Sensor Noise (src/rendering/augmentation.py, configs/augmentation.yaml)
    │
    ├──────────────────────────────────────────────────────┐
    ▼                                                      ▼
[Checkpoint 1: Distortion Simulation Engine]         Phase 4 Waveform Ground Truth
    │                                                      │
    ▼                                                      │
Task 4: Document Boundary & Corner Localization            │
        (src/preprocessing/perspective.py)                 │
    │                                                      │
    ▼                                                      │
Task 5: Four-Point Perspective Unwarping & Rectification   │
        (src/preprocessing/perspective.py)                 │
    │                                                      │
    ├──────────────────────────────────────────────────────┘
    ▼
[Checkpoint 2: Perspective Rectification]
    │
    ▼
Task 6: Illumination Normalization & CLAHE Contrast (src/preprocessing/image.py)
    │
    ▼
Task 7: MobileNetV3 Tensor Formatting & Transforms (src/preprocessing/transforms.py)
    │
    ▼
[Checkpoint 3: Vision Preprocessing & Tensor Pipeline]
    │
    ▼
Task 8: End-to-End Validation CLI & PSNR/SSIM Metrics (scripts/validate_pipeline.py)
    │
    ▼
Task 9: Interactive Notebook, CI Workflow & Phase Plans (notebooks/05, CI, Docs)
    │
    ▼
[Checkpoint 4: Phase 5 Final Completion]
```

---

## Task List

### Phase 5.1: Smartphone Capture & Distortion Simulation

- [ ] **Task 1: Camera Geometric & Perspective Warp Simulation**
  - Implement 3D camera pose simulation (yaw $\pm 15^\circ$, pitch $\pm 15^\circ$, roll $\pm 10^\circ$).
  - Calculate 8-parameter homography matrix $H$ and inverse $H^{-1}$.
  - Warp planar ECG images and compute ground-truth transformed corner coordinates.
  - *Files:* `src/rendering/augmentation.py`, `tests/test_augmentation.py`

- [ ] **Task 2: Non-Uniform Illumination, Shadow Masks & Specular Glare**
  - Implement radial illumination falloff (vignetting / flash gradient).
  - Implement polygonal cast shadow simulation (hand/phone occlusion with soft gaussian penumbra).
  - Implement localized high-intensity specular reflection hotspots.
  - *Files:* `src/rendering/augmentation.py`, `tests/test_augmentation.py`

- [ ] **Task 3: Paper Deformations, Optical Blur & Sensor Noise**
  - Implement cylindrical page curl deformation along horizontal/vertical axes.
  - Implement optical defocus blur, camera shake/motion blur, and additive Gaussian sensor noise.
  - Implement JPEG compression artifact simulation with configurable quality factors.
  - Create `configs/augmentation.yaml` with `mild`, `moderate`, and `severe` presets.
  - *Files:* `src/rendering/augmentation.py`, `configs/augmentation.yaml`, `tests/test_augmentation.py`

### Checkpoint: Smartphone Distortion Simulation Engine
- [ ] Distortion engine produces realistic, visually inspected simulated smartphone photos.
- [ ] Ground-truth corner coordinates are accurately tracked for every distorted image.
- [ ] All distortion unit tests pass (`uv run pytest tests/test_augmentation.py -v`).

---

### Phase 5.2: Document Localization & Perspective Rectification

- [ ] **Task 4: Corner Detection & Document Boundary Localization**
  - Implement adaptive gradient magnitude and edge detection using `scipy.ndimage`.
  - Implement convex quadrilateral fitting and 4-corner localization (top-left, top-right, bottom-right, bottom-left).
  - Provide fallback to canvas boundaries with heuristic confidence scoring.
  - *Files:* `src/preprocessing/perspective.py`, `tests/test_perspective.py`

- [ ] **Task 5: Four-Point Perspective Unwarping & Rectification**
  - Implement 4-point homography estimation mapping detected corners to canonical canvas aspect ratio ($280\text{ mm} \times 190\text{ mm}$).
  - Implement inverse perspective warping to restore rectangular orientation.
  - Assert that rectified grid lines are aligned horizontally and vertically.
  - *Files:* `src/preprocessing/perspective.py`, `tests/test_perspective.py`

### Checkpoint: Perspective Rectification
- [ ] Perspective-distorted images are unwarped back into planar rectangles.
- [ ] Corner localization error on synthetic ground-truth corners is quantified.
- [ ] Unit tests pass (`uv run pytest tests/test_perspective.py -v`).

---

### Phase 5.3: Illumination Normalization & MobileNetV3 Tensor Formatting

- [ ] **Task 6: Illumination Correction & Contrast Normalization**
  - Implement background illumination estimation via morphological filtering / large Gaussian blur.
  - Implement division normalization to eliminate cast shadow gradients.
  - Implement contrast enhancement (CLAHE / adaptive equalization) to maximize ECG grid and waveform contrast.
  - *Files:* `src/preprocessing/image.py`, `tests/test_image_preprocessing.py`

- [ ] **Task 7: MobileNetV3 Tensor Sizing & PyTorch Vision Transforms**
  - Implement standardized tensor formatting ($B \times 3 \times H \times W$, float32).
  - Support configurable resolutions: standard full canvas ($224 \times 224$, $448 \times 448$) and 12-lead multi-crop patches.
  - Implement ImageNet channel normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  - Create callable `MobileNetV3Preprocessor` class integrating into PyTorch DataLoader pipelines.
  - *Files:* `src/preprocessing/transforms.py`, `src/preprocessing/__init__.py`, `tests/test_transforms.py`

### Checkpoint: Vision Preprocessing & Tensor Pipeline
- [ ] Illumination normalization successfully removes cast shadows and highlights trace waveforms.
- [ ] Output tensors match exact MobileNetV3 input shape and numerical bounds.
- [ ] PyTorch DataLoader loads batches of rectified, preprocessed vision tensors.

---

### Phase 5.4: End-to-End Validation, Notebook & CI Integration

- [ ] **Task 8: End-to-End CLI Pipeline & Quantitative Metrics (PSNR/SSIM)**
  - Implement `scripts/validate_pipeline.py` executing: Raw Waveform $\rightarrow$ Paper ECG $\rightarrow$ Smartphone Distortion $\rightarrow$ Vision Rectification $\rightarrow$ MobileNetV3 Tensor.
  - Compute PSNR and SSIM between rectified images and canonical ground-truth paper.
  - Output summary report `reports/phase5_vision_report.md` and diagnostic comparison figures in `reports/figures/`.
  - *Files:* `scripts/validate_pipeline.py`, `reports/phase5_vision_report.md`

- [ ] **Task 9: Interactive Notebook, CI Workflow & Documentation Update**
  - Create interactive Jupyter notebook `notebooks/05_validate_vision_pipeline.ipynb` demonstrating the end-to-end pipeline.
  - Update `.github/workflows/ci.yml` to run Phase 5 unit tests and CLI validation.
  - Update `docs/PHASE_PLANS.md` (mark Phase 5 as Complete) and `README.md`.
  - *Files:* `notebooks/05_validate_vision_pipeline.ipynb`, `.github/workflows/ci.yml`, `docs/PHASE_PLANS.md`, `README.md`

### Checkpoint: Phase 5 Final Completion
- [ ] Complete test suite passes (`uv run pytest tests/ -v`).
- [ ] End-to-end CLI validation script succeeds with quantitative report and figures.
- [ ] CI pipeline validates all phases (1, 2, 3, 4, & 5).

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Heavy external dependencies (e.g. OpenCV)** | High | Use pure NumPy, SciPy (`scipy.ndimage`), Pillow, and PyTorch for all geometric transforms, avoiding compiled wheel conflicts across macOS/Linux CI. |
| **Corner localization failure on extreme tilts (>25°)** | Medium | Provide a confidence score and fallback mechanism to default margins; enforce realistic camera tilt ranges ($\pm 15^\circ$) in standard presets. |
| **Information loss from strong shadows or specular glares** | Medium | Apply background division normalization and local adaptive histogram equalization (CLAHE) to restore waveform visibility in shadow boundaries. |
| **Computational overhead during real-time DataLoader batches** | Low | Implement efficient vector math; support offline batch generation for training while enabling on-the-fly inference. |

---

## Open Questions & Future Considerations

1. **Resolution for MobileNetV3 Backbones:**
   - MobileNetV3 standard resolution is $224 \times 224$. However, 12-lead ECG grids contain fine high-frequency details. Should we support dual options: direct $224 \times 224$ downsampling vs multi-lead crop patches ($12 \times 3 \times 112 \times 224$)? (Task 7 implements both).
2. **Real Smartphone Photo Benchmark Dataset:**
   - While Phase 5 simulates realistic distortions, testing against a small prospective set of real smartphone photographs will provide final clinical domain validation.
