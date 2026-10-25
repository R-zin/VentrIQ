# Tasks: Phase 5 — Smartphone Photograph Simulation & Vision Preprocessing

## Phase 5.1: Smartphone Capture & Distortion Simulation

### Task 1: Camera Geometric & Perspective Warp Simulation

**Description:**
Implement realistic 3D camera pose simulation and 2D projective transformation (homography) to model the perspective distortions introduced when a smartphone photographs a paper ECG at an angle (yaw, pitch, roll, camera distance).

**Acceptance criteria:**
- [x] Computes 3D rotation matrices for camera yaw ($\pm 15^\circ$), pitch ($\pm 15^\circ$), and roll ($\pm 10^\circ$).
- [x] Calculates the 8-parameter homography matrix $H$ and inverse $H^{-1}$.
- [x] Warps input planar ECG images using projective transformation without clipping essential signal content.
- [x] Computes and records the exact 4 ground-truth transformed corner coordinates `(top_left, top_right, bottom_right, bottom_left)` for downstream evaluation.

**Verification:**
- [x] Tests pass: `uv run pytest tests/test_augmentation.py -k "test_perspective" -v`
- [x] Manual check: Verify warped quadrilateral geometry and corner points on a sample image.

**Dependencies:** None

**Files likely touched:**
- `src/rendering/augmentation.py`
- `tests/test_augmentation.py`

**Estimated scope:** S (2 files)

---

### Task 2: Non-Uniform Illumination, Shadow Masks & Specular Glare

**Description:**
Synthesize physical lighting variations encountered during smartphone capture, including radial vignetting falloff, cast shadows (e.g., from the smartphone body or clinician's hands), and localized specular glares.

**Acceptance criteria:**
- [x] Implements radial illumination falloff (vignetting/flash gradient) with configurable center and intensity.
- [x] Simulates cast polygonal shadows with soft Gaussian penumbra edges (intensity reduction factor $0.3\text{--}0.7$).
- [x] Simulates localized high-intensity specular reflection hotspots with smooth exponential decay.
- [x] Maintains pixel values safely within valid image range $[0, 255]$ without integer overflow or clipping artifacts.

**Verification:**
- [x] Tests pass: `uv run pytest tests/test_augmentation.py -k "test_illumination" -v`
- [x] Manual check: Visual inspection of cast shadows and specular glares on rendered ECG paper.

**Dependencies:** Task 1

**Files likely touched:**
- `src/rendering/augmentation.py`
- `tests/test_augmentation.py`

**Estimated scope:** S (2 files)

---

### Task 3: Paper Deformations, Optical Blur & Sensor Noise

**Description:**
Model physical paper handling artifacts (cylindrical page curl) and camera optical/sensor imperfections (defocus blur, motion blur, Gaussian/Poisson noise, and JPEG compression quantization). Provide configuration presets in `configs/augmentation.yaml`.

**Acceptance criteria:**
- [x] Implements subtle cylindrical paper curvature (page curl) along horizontal or vertical axes.
- [x] Implements optical defocus blur and directional motion blur via Gaussian/linear convolution filters.
- [x] Implements additive zero-mean Gaussian sensor noise with configurable standard deviation.
- [x] Implements JPEG compression artifact simulation with configurable quality factors ($30\text{--}90$).
- [x] Creates `configs/augmentation.yaml` with `mild`, `moderate`, and `severe` distortion presets.
- [x] Encapsulates complete pipeline in `SmartphoneECGSimulator` class with reproducible random seeds.

**Verification:**
- [x] Tests pass: `uv run pytest tests/test_augmentation.py -v`
- [x] Manual check: Run simulation pipeline across synthetic cohort and verify visual realism.

**Dependencies:** Tasks 1, 2

**Files likely touched:**
- `src/rendering/augmentation.py`
- `configs/augmentation.yaml`
- `tests/test_augmentation.py`

**Estimated scope:** M (3 files)

---

## Checkpoint 1: Distortion Simulation Engine
- [x] All distortion simulation unit tests pass: `uv run pytest tests/test_augmentation.py -v`
- [x] Simulated images display combined perspective tilt, shadows, curl, blur, and noise.
- [x] Ground-truth corner coordinates are accurately exported with every simulated image.

---

## Phase 5.2: Document Localization & Perspective Rectification

### Task 4: Corner Detection & Document Boundary Localization

**Description:**
Build the document localization module that detects paper edges and extracts the four corner coordinates of a photographed ECG document against arbitrary backgrounds.

**Acceptance criteria:**
- [x] Implements gradient magnitude edge detection using `scipy.ndimage` Sobel operators.
- [x] Identifies document boundary contours and fits a convex bounding quadrilateral.
- [x] Extracts ordered 4 corner coordinates: `[top_left, top_right, bottom_right, bottom_left]`.
- [x] Provides fallback to canvas margins with heuristic confidence scoring when boundaries are ambiguous.

**Verification:**
- [x] Tests pass: `uv run pytest tests/test_perspective.py -k "test_corner_detection" -v`
- [x] Manual check: Measure mean Euclidean distance error against ground-truth corners from Task 1.

**Dependencies:** Task 1

**Files likely touched:**
- `src/preprocessing/perspective.py`
- `tests/test_perspective.py`

**Estimated scope:** S (2 files)

---

### Task 5: Four-Point Perspective Unwarping & Rectification

**Description:**
Implement the homography rectification engine that maps the four detected document corners back to a canonical rectangular canvas with standard ECG paper aspect ratio ($280\text{ mm} \times 190\text{ mm}$).

**Acceptance criteria:**
- [x] Calculates the 3x3 homography matrix mapping the arbitrary input quadrilateral to a rectilinear target rectangle.
- [x] Warps the image using inverse perspective transformation into canonical planar coordinates.
- [x] Verifies that rectified grid lines are aligned horizontally and vertically.
- [x] Preserves image dimensions and aspect ratio matching Phase 4 canonical layout standards.

**Verification:**
- [x] Tests pass: `uv run pytest tests/test_perspective.py -v`
- [x] Manual check: Verify that a rotated/tilted synthetic ECG paper is restored to an upright rectangle.

**Dependencies:** Task 4

**Files likely touched:**
- `src/preprocessing/perspective.py`
- `tests/test_perspective.py`

**Estimated scope:** S (2 files)

---

## Checkpoint 2: Perspective Rectification
- [x] Perspective-distorted images are unwarped back into rectilinear planar canvases.
- [x] Corner localization error on synthetic ground-truth images is $< 5.0\text{ px}$.
- [x] All perspective and rectification tests pass: `uv run pytest tests/test_perspective.py -v`.

---

## Phase 5.3: Illumination Normalization & MobileNetV3 Tensor Formatting

### Task 6: Illumination Correction & Contrast Normalization

**Description:**
Implement an illumination correction module that removes shadow gradients and enhances waveform contrast on rectified ECG paper recordings.

**Acceptance criteria:**
- [x] Estimates the background illumination field using large-kernel Gaussian filtering or morphological opening.
- [x] Applies division normalization ($I_{norm} = I / I_{bg}$) to cancel shadow gradients while preserving high-frequency waveform traces.
- [x] Implements local adaptive contrast enhancement (CLAHE) to optimize trace visibility across lead boxes.
- [x] Standardizes background to uniform medical white/light gray.

**Verification:**
- [x] Tests pass: `uv run pytest tests/test_image_preprocessing.py -k "test_illumination" -v`
- [x] Manual check: Visual before/after inspection showing shadow attenuation and trace contrast enhancement.

**Dependencies:** Task 5

**Files likely touched:**
- `src/preprocessing/image.py`
- `tests/test_image_preprocessing.py`

**Estimated scope:** S (2 files)

---

### Task 7: MobileNetV3 Tensor Sizing & PyTorch Vision Transforms

**Description:**
Build the vision transformation pipeline that resizes, normalizes, and packages rectified ECG images into PyTorch tensors formatted for MobileNetV3 backbone models.

**Acceptance criteria:**
- [x] Implements standardized image resizing to standard vision resolutions (default $224 \times 224$, high-res $448 \times 448$).
- [x] Supports 12-lead patch cropping ($12 \times 3 \times 112 \times 224$) extracting individual lead segments.
- [x] Converts images to float32 tensors with ImageNet channel standardization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- [x] Creates callable `MobileNetV3Preprocessor` class compatible as a transform on `VentriQTorchDataset`.
- [x] Enables PyTorch DataLoader batch collation producing tensors of shape `(B, 3, H, W)`.

**Verification:**
- [x] Tests pass: `uv run pytest tests/test_image_preprocessing.py -v`
- [x] Manual check: Load DataLoader batch and verify tensor shape, dtype, and numerical bounds.

**Dependencies:** Task 6

**Files likely touched:**
- `src/preprocessing/transforms.py`
- `src/preprocessing/__init__.py`
- `tests/test_image_preprocessing.py`

**Estimated scope:** M (3 files)

---

## Checkpoint 3: Vision Preprocessing & Tensor Pipeline
- [x] Illumination normalization removes cast shadows and enhances trace contrast.
- [x] Output tensors match MobileNetV3 shape `(B, 3, 224, 224)` and ImageNet normalization statistics.
- [x] PyTorch DataLoader loads batches of preprocessed vision tensors without bottlenecks.

---

## Phase 5.4: End-to-End Validation, Notebook & CI Integration

### Task 8: End-to-End CLI Pipeline & Quantitative Metrics (PSNR/SSIM)

**Description:**
Build the end-to-end validation CLI script that executes the complete pipeline (waveform $\rightarrow$ synthetic paper $\rightarrow$ smartphone distortion $\rightarrow$ vision rectification $\rightarrow$ model-ready tensor) and computes quantitative fidelity metrics.

**Acceptance criteria:**
- [x] Implements `scripts/validate_pipeline.py` with CLI arguments (`--data-dir`, `--num-samples`, `--severity`, `--report-dir`).
- [x] Evaluates Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) between rectified images and planar ground-truth paper.
- [x] Generates comparative figure showing: Original Paper | Distorted Photo | Rectified Paper | MobileNet Tensor.
- [x] Generates markdown report `reports/phase5_vision_report.md` recording metric distributions and verification status.

**Verification:**
- [x] Script runs successfully: `uv run python scripts/validate_pipeline.py --num-samples 5`
- [x] Output artifacts verified: `reports/phase5_vision_report.md` and comparison figures exist.

**Dependencies:** Tasks 3, 5, 7

**Files likely touched:**
- `scripts/validate_pipeline.py`
- `reports/phase5_vision_report.md`

**Estimated scope:** M (2 files)

---

### Task 9: Interactive Notebook, CI Workflow & Documentation Update

**Description:**
Create an interactive demonstration notebook, update the GitHub Actions CI pipeline, and finalize project documentation for Phase 5.

**Acceptance criteria:**
- [x] Creates interactive notebook `notebooks/05_validate_vision_pipeline.ipynb` with step-by-step visualizations.
- [x] Updates `.github/workflows/ci.yml` to run Phase 5 unit tests and CLI validation step.
- [x] Updates `docs/PHASE_PLANS.md` (marking Phase 5 as Complete) and `README.md` with Phase 5 architecture, CLI usage, and deliverables.
- [x] Ensures 100% test pass rate across all phases (Phases 1, 2, 3, 4, & 5).

**Verification:**
- [x] Complete test suite passes: `uv run pytest tests/ -v`
- [x] CI pipeline step locally verified: all reports and data artifacts exist.

**Dependencies:** Task 8

**Files likely touched:**
- `notebooks/05_validate_vision_pipeline.ipynb`
- `.github/workflows/ci.yml`
- `docs/PHASE_PLANS.md`
- `README.md`

**Estimated scope:** M (4 files)

---

## Checkpoint 4: Phase 5 Final Completion
- [x] All unit tests pass across the entire repository with zero regressions.
- [x] End-to-end simulation, distortion, rectification, and MobileNetV3 tensor formatting verified.
- [x] CI workflow passes with all pipeline artifacts verified.
- [x] Ready for feature branch commit, push, and Pull Request.
