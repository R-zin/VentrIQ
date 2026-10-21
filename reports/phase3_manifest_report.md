# VentriQ Phase 3: Dataset Manifest & Reproducibility Report

**Generated:** 2026-10-22 00:35:46  
**Random Seed:** `42`  
**Python Version:** `3.14.0`  
**PyTorch Version:** `2.14.1`  

---

## 1. Cryptographic Source Traceability (SHA-256)

| Source File | SHA-256 Checksum |
|---|---|
| `EchoNext_metadata_100k.csv` | `68f3622547b98fd322793454e08ac3807552f116b9181c619be6b2c4ef1d4de8` |
| `EchoNext_test_waveforms.npy` | `40d25698f2b13cfb505bf7f93b4a9bd7882af1bd51cdc0a3529ba824792bef4d` |
| `EchoNext_train_waveforms.npy` | `421ae6a7a7c40b3f08412d92e9b90a41f32d990ccc5cec86c68b7cb4a240cdcb` |
| `EchoNext_val_waveforms.npy` | `f2f4e53a00f3729a5a7452f2677f31bf08184d72edaa100d2399ccc75e4695a2` |

## 2. Partition & Patient Isolation Audit

```text
=== VentriQ Split Verification: PASSED (Zero Leakage) ===
Total Records: 100
Total Unique Patients: 86
Splits Breakdown:
  - train: 70 records, 56 patients
  - val: 15 records, 15 patients
  - test: 15 records, 15 patients
Integrity: All partitions are strictly patient-disjoint.
```

| Partition | Retained Records | Unique Patients | Excluded Records |
|---|---|---|---|
| **test** | 15 | 15 | 0 |
| **train** | 70 | 56 | 0 |
| **val** | 15 | 15 | 0 |

### Exclusion Breakdown:
- **missing_lvef**: 0 records
- **unusable_signal_quality**: 0 records
- **missing_waveform**: 0 records

## 3. Clinical Diagnostic Targets Distribution

| Target Metric | Retained Value |
|---|---|
| Valid LVEF Records | 100 |
| Mean LVEF | 53.18% |
| Std LVEF | 11.14% |
| Median LVEF | 53.80% |
| Min / Max LVEF | 28.4% / 75.0% |
| Moderate-to-Severe Dysfunction (LVEF ≤ 45%) Prevalence | 26.0% |
| Moderate+ Structural Heart Disease (SHD) Prevalence | 36.0% |

## 4. PyTorch Dataset Integration & Zero-Copy Slicing

- Waveforms are retrieved directly via memory-mapping without duplicating NumPy binaries.
- Multi-worker safe through process-local memory mapping and deterministic `ventriq_worker_init_fn`.
- Standard tensor outputs: `(12, 2500)` float32 ready for MobileNetV3 and 1D-CNN backbones.

---
**Report Status:** Verified & Audit-Logged.
