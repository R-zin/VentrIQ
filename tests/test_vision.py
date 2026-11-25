"""Tests for src/vision: EchoNext waveform-to-image conversion and MobileNetV3 SHD training."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import DatasetManifest
from src.data.echonext import create_mock_echonext_dataset
from src.vision import (
    CanonicalizationConfig,
    ECGImageCanonicalizer,
    ECGImageConverter,
    ECGImageDataset,
    ImageConversionConfig,
    SHDPredictor,
    TrainingConfig,
    average_precision,
    binary_metrics,
    build_image_transform,
    build_mobilenet_v3,
    canonical_size_for_dpi,
    convert_echonext_to_images,
    load_conversion_config,
    roc_auc,
    sample_render_style,
    select_records,
    train_classifier,
    youden_threshold,
)
from src.vision.conversion import CONFIG_FILENAME, MANIFEST_FILENAME, RenderStyle

SMALL = (200, 136)  # tiny canonical size keeps tests fast


def _fast_config(**overrides) -> ImageConversionConfig:
    params = dict(
        dpi=60,
        canonical_size=SMALL,
        photo_preset_weights={"none": 1.0},
        enhance=False,
    )
    params.update(overrides)
    return ImageConversionConfig(**params)


@pytest.fixture(scope="module")
def mock_root(tmp_path_factory) -> Path:
    return create_mock_echonext_dataset(
        tmp_path_factory.mktemp("echonext"), num_train=8, num_val=4, num_test=4, seed=3
    )


@pytest.fixture(scope="module")
def converted(mock_root, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("images")
    convert_echonext_to_images(mock_root, out, _fast_config(variants_per_record=2))
    return out


@pytest.fixture
def waveform(mock_root) -> np.ndarray:
    arr = np.load(mock_root / "EchoNext_train_waveforms.npy", mmap_mode="r")
    return np.asarray(arr[0, 0], dtype=np.float32)


# --- canonical view -------------------------------------------------------


def test_canonical_size_for_dpi():
    assert canonical_size_for_dpi(150) == (1654, 1122)
    assert canonical_size_for_dpi(100) == (1102, 748)
    with pytest.raises(ValueError):
        canonical_size_for_dpi(0)


def test_canonicalizer_resizes_and_returns_rgb():
    canon = ECGImageCanonicalizer(CanonicalizationConfig(canonical_size=SMALL, enhance=True))
    gray = Image.new("L", (400, 300), 200)
    out = canon(gray, rectify="never")
    assert out.size == SMALL and out.mode == "RGB"


def test_canonicalizer_rectifies_photo_on_dark_background():
    paper = Image.new("RGB", (300, 204), "white")
    photo = Image.new("RGB", (500, 400), "#202020")
    photo.paste(paper, (100, 98))
    canon = ECGImageCanonicalizer(CanonicalizationConfig(canonical_size=SMALL, enhance=False))
    out = np.asarray(canon(photo, rectify="auto"), dtype=np.float32)
    # After rectification the dark desk is gone: the page fills the frame.
    assert out.mean() > 200


def test_canonicalization_config_roundtrip():
    cfg = CanonicalizationConfig(canonical_size=(320, 217), rectify="never", enhance=False)
    assert CanonicalizationConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        ECGImageCanonicalizer(CanonicalizationConfig(rectify="sometimes"))  # type: ignore[arg-type]


# --- conversion config & style sampling -----------------------------------


def test_conversion_config_validation():
    with pytest.raises(ValueError):
        ImageConversionConfig(grid_style_weights={"plaid": 1.0})
    with pytest.raises(ValueError):
        ImageConversionConfig(photo_preset_weights={"underwater": 1.0})
    with pytest.raises(ValueError):
        ImageConversionConfig(variants_per_record=0)
    with pytest.raises(ValueError):
        ImageConversionConfig(image_format="gif")
    with pytest.raises(ValueError):
        ImageConversionConfig(amplitude_scale_range=(5.0, 2.0))


def test_conversion_config_roundtrip():
    cfg = _fast_config(variants_per_record=3, image_format="png")
    assert ImageConversionConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.extension == "png"
    assert cfg.canonicalization.canonical_size == SMALL


def test_render_style_is_deterministic_and_varied():
    cfg = ImageConversionConfig(seed=7)
    a = sample_render_style(cfg, "ECHONEXT_train_000001", 0)
    assert a == sample_render_style(cfg, "ECHONEXT_train_000001", 0)
    styles = {sample_render_style(cfg, f"R{i}", 0) for i in range(40)}
    assert len({s.grid_style for s in styles}) > 1
    assert len({s.photo_preset for s in styles}) > 1
    lo, hi = cfg.amplitude_scale_range
    assert all(lo <= s.amplitude_scale_mm_per_unit <= hi for s in styles)
    assert sample_render_style(cfg, "R0", 1) != sample_render_style(cfg, "R0", 0)


# --- single-image conversion ----------------------------------------------


@pytest.mark.parametrize("preset", ["none", "mild"])
def test_converter_outputs_canonical_image(waveform, preset):
    conv = ECGImageConverter(_fast_config())
    style = RenderStyle("classic_red", 1.5, 6.0, preset, photo_seed=1)
    img = conv.convert(waveform, style)
    assert img.size == SMALL and img.mode == "RGB"
    arr = np.asarray(img)
    assert arr.min() < 100  # dark trace pixels present
    assert arr.mean() > 150  # mostly paper


def test_converter_accepts_channels_first(waveform):
    conv = ECGImageConverter(_fast_config())
    style = RenderStyle("standard_pink", 1.2, 6.0, "none", 0)
    a = np.asarray(conv.convert(waveform, style))
    b = np.asarray(conv.convert(waveform.T, style))
    np.testing.assert_array_equal(a, b)


def test_amplitude_style_changes_rendering(waveform):
    conv = ECGImageConverter(_fast_config())
    a = conv.convert(waveform, RenderStyle("no_grid", 1.2, 3.0, "none", 0))
    b = conv.convert(waveform, RenderStyle("no_grid", 1.2, 9.0, "none", 0))
    assert not np.array_equal(np.asarray(a), np.asarray(b))


# --- batch conversion ------------------------------------------------------


def test_batch_conversion_manifest(converted, mock_root):
    df = pd.read_csv(converted / MANIFEST_FILENAME)
    assert len(df) == (8 + 4 + 4) * 2
    assert set(df["split"]) == {"train", "val", "test"}
    assert set(df["variant"]) == {0, 1}
    assert df["shd_flag"].isin([0, 1]).all()
    for rel in df["image_path"]:
        p = converted / rel
        assert p.is_file()
        assert Image.open(p).size == SMALL
    # Splits keep the official partitions -> no patient shared across splits.
    pats = {s: set(df.loc[df["split"] == s, "patient_key"]) for s in ("train", "val", "test")}
    assert not (pats["train"] & pats["val"]) and not (pats["train"] & pats["test"])
    assert (converted / CONFIG_FILENAME).is_file()
    assert load_conversion_config(converted).canonical_size == SMALL


def test_conversion_is_resumable(mock_root, tmp_path):
    cfg = _fast_config()
    df1 = convert_echonext_to_images(mock_root, tmp_path, cfg, splits=["val"])
    first = tmp_path / df1["image_path"].iloc[0]
    mtime = first.stat().st_mtime_ns
    df2 = convert_echonext_to_images(mock_root, tmp_path, cfg, splits=["val"])
    assert first.stat().st_mtime_ns == mtime
    pd.testing.assert_frame_equal(df1, df2)
    convert_echonext_to_images(mock_root, tmp_path, cfg, splits=["val"], overwrite=True)
    assert first.stat().st_mtime_ns != mtime


def test_parallel_conversion_matches_serial(mock_root, tmp_path):
    cfg = _fast_config(photo_preset_weights={"none": 0.5, "mild": 0.5}, image_format="png")
    serial = convert_echonext_to_images(mock_root, tmp_path / "a", cfg, splits=["test"])
    parallel = convert_echonext_to_images(
        mock_root, tmp_path / "b", cfg, splits=["test"], num_workers=2
    )
    pd.testing.assert_frame_equal(serial, parallel)
    for rel in serial["image_path"]:
        np.testing.assert_array_equal(
            np.asarray(Image.open(tmp_path / "a" / rel)),
            np.asarray(Image.open(tmp_path / "b" / rel)),
        )


def test_select_records_filters_and_subsamples(mock_root):
    records = DatasetManifest.build_from_echonext(mock_root).records
    sel = select_records(records, splits=["train"], max_per_split=3, seed=1)
    assert len(sel) == 3 and all(r.split == "train" for r in sel)
    assert select_records(records, splits=["train"], max_per_split=3, seed=1) == sel
    assert len(select_records(records, splits=["val", "test"], label=None)) == 8
    with pytest.raises(ValueError):
        select_records(records, label="age")


# --- image dataset ---------------------------------------------------------


def test_image_dataset_samples(converted):
    ds = ECGImageDataset(
        converted, split="train", transform=build_image_transform((96, 64), train=True)
    )
    assert len(ds) == 16
    sample = ds[0]
    assert sample["image"].shape == (3, 64, 96)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].item() in (0.0, 1.0)
    assert ds.pos_weight() > 0
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4)))
    assert batch["image"].shape == (4, 3, 64, 96)


def test_image_dataset_requires_root_for_dataframe(converted):
    df = pd.read_csv(converted / MANIFEST_FILENAME)
    with pytest.raises(ValueError):
        ECGImageDataset(df)
    with pytest.raises(ValueError):
        ECGImageDataset(converted, label="age")
    df.loc[0, "shd_flag"] = np.nan
    ds = ECGImageDataset(df, image_root=converted)
    assert len(ds) == len(df) - 1


# --- metrics ----------------------------------------------------------------


def test_roc_auc_and_average_precision():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert math.isnan(roc_auc(np.ones(3), np.arange(3)))
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert average_precision(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.1])) == pytest.approx(
        (1.0 + 2 / 3) / 2
    )


def test_youden_threshold_and_binary_metrics():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
    t = youden_threshold(y, p)
    assert 0.3 < t <= 0.6
    m = binary_metrics(y, p, t)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0 and m["accuracy"] == 1.0
    assert youden_threshold(np.ones(3), np.array([0.1, 0.2, 0.3])) == 0.5


# --- model, training & inference -------------------------------------------


@pytest.mark.parametrize("arch", ["small", "large"])
def test_mobilenet_binary_head_non_square(arch):
    model = build_mobilenet_v3(arch, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 96, 128))
    assert out.shape == (2, 1)
    with pytest.raises(ValueError):
        build_mobilenet_v3("tiny")  # type: ignore[arg-type]


def test_train_then_predict_user_images(converted, tmp_path, waveform):
    cfg = TrainingConfig(
        image_dir=str(converted),
        output_dir=str(tmp_path / "model"),
        arch="small",
        pretrained=False,
        image_size=(96, 64),
        epochs=2,
        batch_size=4,
        num_workers=0,
        device="cpu",
    )
    conversion = load_conversion_config(converted)
    summary = train_classifier(cfg, conversion.canonicalization)
    ckpt_path = Path(summary["checkpoint"])
    assert ckpt_path.is_file()
    assert len(summary["history"]) == 2
    assert 0.0 <= summary["test"]["accuracy"] <= 1.0
    assert (tmp_path / "model" / "training_summary.json").is_file()

    ckpt = torch.load(ckpt_path, weights_only=False)
    assert ckpt["image_size"] == [96, 64]
    assert tuple(ckpt["canonicalization"]["canonical_size"]) == SMALL

    predictor = SHDPredictor.from_checkpoint(ckpt_path, device="cpu")
    # A "user photo": a rendered ECG simulated as a smartphone capture, saved to disk.
    photo = ECGImageConverter(_fast_config(canonical_size=(600, 407))).convert(
        waveform, RenderStyle("standard_pink", 1.3, 6.0, "none", 0)
    )
    photo_path = tmp_path / "user_ecg.jpg"
    photo.save(photo_path)

    preds = predictor.predict([photo_path, photo, np.asarray(photo)])
    assert len(preds) == 3
    for pred in preds:
        assert 0.0 <= pred.probability <= 1.0
        assert pred.predicted_label == int(pred.probability >= pred.threshold)
    assert preds[0].source == str(photo_path)
    assert predictor.preprocess(photo).shape == (3, 64, 96)

    strict = SHDPredictor.from_checkpoint(ckpt_path, device="cpu", threshold=1.01)
    assert strict.predict(photo)[0].predicted_label == 0
