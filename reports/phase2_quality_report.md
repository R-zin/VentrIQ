# VentriQ — Phase 2 Quality Control & Signal Preprocessing Report

**Generated:** 2026-10-19 21:13:37  
**Scope:** Signal Sanity, Quality Evaluation, Configurable Zero-Phase Filtering & Normalization  
**Dataset Reference:** EchoNext v1.0.0  

---

## 1. Executive Summary

- **Total Inspected ECG Records:** 100
- **Clinically Usable Records:** 100 (100.0%)
- **Flagged / Unusable Records:** 0 (0.0%)
- **Mean Overall SNR:** 12.18 dB (Min: 11.91 dB, Max: 12.47 dB)
- **Active Preprocessing Mode:**
  - Filtering Enabled: `False`
  - Filter Type: `butterworth` (0.5 - 40.0 Hz, Order 4)
  - Notch Filter: `False`
  - Normalization Enabled: `False` (lead_zscore)

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

### Lead Fault Frequencies Across 100 Records:

| Lead | Flatline Count | Clipped / Railed Count | Low SNR Count (< -3 dB) | Mean SNR (dB) |
|---|---|---|---|---|
| `I` | 0 | 0 | 0 | 20.67 |
| `II` | 0 | 0 | 0 | 18.41 |
| `III` | 0 | 0 | 0 | 10.40 |
| `aVR` | 0 | 0 | 0 | 0.64 |
| `aVL` | 0 | 0 | 0 | 6.60 |
| `aVF` | 0 | 0 | 0 | 16.86 |
| `V1` | 0 | 0 | 0 | 20.51 |
| `V2` | 0 | 0 | 0 | 19.55 |
| `V3` | 0 | 0 | 0 | 13.48 |
| `V4` | 0 | 0 | 0 | 1.43 |
| `V5` | 0 | 0 | 0 | 2.79 |
| `V6` | 0 | 0 | 0 | 14.80 |

### Fault Detection Definitions:
- **Flatline Lead:** Lead variance $< 1.0 \times 10^{-5}$ across recording or $>60\%$ of $0.5\text{ s}$ windows.
- **Clipped Lead:** $\ge 8$ consecutive samples equal to the extremal min or max boundary.
- **Low SNR Lead:** Cardiac band ($0.5\text{--}40\text{ Hz}$) to high-frequency noise floor ratio $< -3.0\text{ dB}$.
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
