# EchoNext Dataset Exploration Report (Phase 1)

**Generated:** 2026-10-13 12:46:50  
**Project:** VentriQ — Left Ventricular Ejection Fraction (LVEF) & Structural Heart Disease Analysis  
**Dataset Reference:** EchoNext v1.0.0 (PhysioNet / Columbia University Irving Medical Center)  

---

## 1. Executive Summary

- **Total 12-Lead ECG Records:** 100
- **Unique Patients:** 86
- **Mean ECGs per Patient:** 1.16
- **Sampling Frequency:** 250 Hz
- **Recording Duration:** 10.0 seconds (2500 time steps per lead)
- **Number of Leads:** 12 standard leads (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6)
- **Patient Isolation (Zero Leakage):** PASSED (0 overlap)
- **Waveform-to-Metadata Row Alignment:** VERIFIED

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
| `train` | 70 | 56 | 1.25 |
| `val` | 15 | 15 | 1.00 |
| `test` | 15 | 15 | 1.00 |

### Cross-Split Patient Leakage Audit:

- **Is Leak Free:** `True`
- **Total Overlapping Patients Across Evaluation Splits:** `0`
- **Pairwise Overlaps:**
  - `train_vs_val`: 0 overlapping patients
  - `train_vs_test`: 0 overlapping patients
  - `val_vs_test`: 0 overlapping patients

> [!NOTE]
> Patient-level separation is strictly verified. Evaluation splits contain zero overlap with training patient keys.

---

## 4. Target Variable Analysis: LVEF & Structural Heart Disease

### Continuous LVEF Distribution (`lvef_value`):

| Metric | Value |
|---|---|
| Valid Measurement Count | 100 |
| Missing Count (% Missing) | 0 (0.00%) |
| Mean ± Std | 53.2% ± 11.2% |
| Median [IQR] | 53.8% [44.4% - 60.0%] |
| Range (Min - Max) | 28.4% - 75.0% |

### Clinical Stratification:

- **Severe Systolic Dysfunction (LVEF ≤ 35%):** 6 records
- **Moderate Systolic Dysfunction (35% < LVEF ≤ 45%):** 20 records
- **Preserved Systolic Function (LVEF > 45%):** 74 records

### Binary Classification Targets:

- **`lvef_lte_45_flag` (Moderate or Greater Dysfunction):**
  - Positive Count: 26
  - Prevalence: 26.00%
- **`shd_moderate_or_greater_flag` (Composite SHD Flag):**
  - Positive Count: 36
  - Prevalence: 36.00%

---

## 5. Waveform-to-Metadata Row Alignment Verification

Each split's monolithic `.npy` array must align row-for-row with the corresponding metadata slice:

| Split | Metadata Row Count | Array Row Count | Array Shape | Aligned |
|---|---|---|---|---|
| `train` | 70 | 70 | `[70, 1, 2500, 12]` | `True` |
| `val` | 15 | 15 | `[15, 1, 2500, 12]` | `True` |
| `test` | 15 | 15 | `[15, 1, 2500, 12]` | `True` |

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
