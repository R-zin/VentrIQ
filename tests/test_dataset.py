from __future__ import annotations

import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    DatasetManifest,
    ECGRecord,
    VentriQTorchDataset,
    compute_file_sha256,
    ventriq_worker_init_fn,
)
from src.data.echonext import (
    EXPECTED_TIME_STEPS,
    NUM_LEADS,
    EchoNextDataset,
    create_mock_echonext_dataset,
)
from src.preprocessing.signal import ECGSignalPreprocessor


@pytest.fixture(scope="module")
def mock_dataset_dir():
    """Create a temporary mock EchoNext dataset directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = create_mock_echonext_dataset(
            output_dir=tmpdir,
            num_train=20,
            num_val=8,
            num_test=8,
            seed=42,
        )
        yield Path(path)


def test_ecg_record_immutability_and_schema():
    """Test ECGRecord schema validation, properties, and strict immutability."""
    rec = ECGRecord(
        record_id="ECHONEXT_train_000001",
        patient_key="P100",
        split="train",
        waveform_ref=("train", 0),
        lvef_target=55.5,
        lvef_lte_45_label=0,
        shd_label=1,
        quality_flags={"is_valid": True, "is_usable": True},
    )

    assert rec.record_id == "ECHONEXT_train_000001"
    assert rec.has_valid_lvef is True
    assert rec.has_valid_shd is True
    assert rec.is_usable is True

    # Test frozen immutability
    with pytest.raises((FrozenInstanceError, AttributeError)):
        rec.lvef_target = 60.0  # type: ignore

    # Serialization round-trip
    rec_dict = rec.to_dict()
    assert rec_dict["waveform_ref"] == ["train", 0]
    rec_reconstructed = ECGRecord.from_dict(rec_dict)
    assert rec_reconstructed == rec


def test_ecg_record_validation():
    """Test validation errors on malformed record attributes."""
    with pytest.raises(ValueError, match="record_id must be a non-empty string"):
        ECGRecord("", "P1", "train", ("train", 0))

    with pytest.raises(ValueError, match="waveform_ref must be a 2-tuple"):
        ECGRecord("rec1", "P1", "train", "invalid_ref")  # type: ignore

    with pytest.raises(ValueError, match="lvef_lte_45_label must be 0 or 1"):
        ECGRecord("rec1", "P1", "train", ("train", 0), lvef_lte_45_label=2)


def test_compute_file_sha256(mock_dataset_dir):
    """Test SHA256 checksum computation."""
    meta_file = mock_dataset_dir / "EchoNext_metadata_100k.csv"
    checksum = compute_file_sha256(meta_file)
    assert isinstance(checksum, str)
    assert len(checksum) == 64

    with pytest.raises(FileNotFoundError):
        compute_file_sha256(mock_dataset_dir / "non_existent.npy")


def test_dataset_manifest_build_and_serialization(mock_dataset_dir):
    """Test building, saving, and loading a DatasetManifest."""
    manifest = DatasetManifest.build_from_echonext(
        root=mock_dataset_dir,
        seed=123,
    )

    assert len(manifest.records) == 36  # 20 + 8 + 8
    assert manifest.retained_counts_per_split["train"] == 20
    assert manifest.retained_counts_per_split["val"] == 8
    assert manifest.retained_counts_per_split["test"] == 8

    # Source checksums must include all 4 files
    assert "EchoNext_metadata_100k.csv" in manifest.source_checksums
    assert "EchoNext_train_waveforms.npy" in manifest.source_checksums

    # Serialization to JSON
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        manifest.save(tmp_path)
        assert tmp_path.exists()

        # Load back
        loaded = DatasetManifest.load(tmp_path)
        assert len(loaded.records) == len(manifest.records)
        assert loaded.source_checksums == manifest.source_checksums
        assert loaded.retained_counts_per_split == manifest.retained_counts_per_split
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_ventriq_torch_dataset_retrieval(mock_dataset_dir):
    """Test VentriQTorchDataset zero-copy slicing, tensor formats, and dimensions."""
    ds = VentriQTorchDataset(root=mock_dataset_dir, split="train", channels_first=True)
    assert len(ds) == 20

    sample = ds[0]
    assert isinstance(sample, dict)
    assert "waveform" in sample
    assert "record_id" in sample
    assert "lvef" in sample
    assert "lvef_lte_45" in sample
    assert "shd" in sample

    wf = sample["waveform"]
    assert isinstance(wf, torch.Tensor)
    assert wf.dtype == torch.float32
    # Channels first: (12, 2500)
    assert wf.shape == (NUM_LEADS, EXPECTED_TIME_STEPS)

    # Channels last retrieval: (2500, 12)
    ds_cl = VentriQTorchDataset(root=mock_dataset_dir, split="train", channels_first=False)
    sample_cl = ds_cl[0]
    assert sample_cl["waveform"].shape == (EXPECTED_TIME_STEPS, NUM_LEADS)


def test_ventriq_torch_dataset_with_dataloader(mock_dataset_dir):
    """Test PyTorch DataLoader multi-batch and multi-worker loading without memory bloat."""
    ds = VentriQTorchDataset(root=mock_dataset_dir, split="train", channels_first=True)

    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        worker_init_fn=ventriq_worker_init_fn,
    )

    batch_count = 0
    total_samples = 0
    for batch in loader:
        batch_count += 1
        waveforms = batch["waveform"]
        total_samples += len(waveforms)
        assert waveforms.shape == (4, NUM_LEADS, EXPECTED_TIME_STEPS)
        assert waveforms.dtype == torch.float32
        assert torch.isfinite(waveforms).all()
        assert len(batch["record_id"]) == 4

    assert batch_count == 5  # 20 samples / 4 batch size
    assert total_samples == 20


def test_ventriq_torch_dataset_with_preprocessor_transform(mock_dataset_dir):
    """Test VentriQTorchDataset pipeline integration with Phase 2 signal preprocessor."""
    preprocessor = ECGSignalPreprocessor(apply_filter=True, lowcut_hz=0.5, highcut_hz=40.0)
    ds = VentriQTorchDataset(
        root=mock_dataset_dir,
        split="val",
        transform=preprocessor,
        channels_first=True,
    )

    sample = ds[0]
    wf = sample["waveform"]
    assert wf.shape == (NUM_LEADS, EXPECTED_TIME_STEPS)
    assert torch.isfinite(wf).all()


def test_ventriq_torch_dataset_from_echonext(mock_dataset_dir):
    """Test constructing VentriQTorchDataset via from_echonext factory method."""
    echonext_ds = EchoNextDataset(mock_dataset_dir, split="test")
    ds = VentriQTorchDataset.from_echonext(echonext_ds, channels_first=True)
    assert len(ds) == 8
    sample = ds[0]
    assert sample["split"] == "test"
    assert sample["waveform"].shape == (NUM_LEADS, EXPECTED_TIME_STEPS)


def test_ventriq_torch_dataset_error_handling(mock_dataset_dir):
    """Test index out of bounds error handling."""
    ds = VentriQTorchDataset(root=mock_dataset_dir, split="val")
    with pytest.raises(IndexError):
        _ = ds[999]
    with pytest.raises(IndexError):
        _ = ds[-1]
