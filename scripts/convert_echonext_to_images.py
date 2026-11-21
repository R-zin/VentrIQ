#!/usr/bin/env python3
"""scripts/convert_echonext_to_images.py - Convert EchoNext waveforms into a CNN image dataset.

Example (full dataset, 8 processes, 2 randomized variants per training record):

    uv run python scripts/convert_echonext_to_images.py \
        --data-dir data/raw/echonext --output-dir data/images/shd \
        --num-workers 8 --variants-per-record 2

Writes ``{output-dir}/{split}/{record_id}_v{k}.jpg``, ``image_manifest.csv`` and
``conversion_config.json``. Re-running resumes: existing images are reused.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from src.data.dataset import DatasetManifest
from src.utils.paths import DATA_DIR, get_raw_echonext_dir
from src.vision.conversion import (
    PHOTO_PRESETS,
    ImageConversionConfig,
    convert_echonext_to_images,
)


def _parse_weights(spec: str) -> dict:
    """Parse 'none=0.5,mild=0.3,moderate=0.2' into a dict."""
    out = {}
    for part in spec.split(","):
        name, _, value = part.partition("=")
        out[name.strip()] = float(value) if value else 1.0
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", type=Path, default=get_raw_echonext_dir())
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional Phase 3 dataset_manifest.json (e.g. with QC exclusions applied).",
    )
    p.add_argument("--output-dir", type=Path, default=DATA_DIR / "images" / "shd")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test", "no_split"],
    )
    p.add_argument(
        "--label",
        default="shd",
        choices=["shd", "lvef_lte_45"],
        help="Skip records missing this label.",
    )
    p.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Random subsample per split (quick experiments).",
    )
    p.add_argument(
        "--variants-per-record",
        type=int,
        default=1,
        help="Independently randomized images per record (offline augmentation).",
    )
    p.add_argument(
        "--photo-mix",
        default="none=0.5,mild=0.3,moderate=0.2",
        help=f"Smartphone-photo simulation mix. Presets: {', '.join(PHOTO_PRESETS)}.",
    )
    p.add_argument("--dpi", type=int, default=100, help="Paper render resolution.")
    p.add_argument(
        "--canonical-size",
        type=int,
        nargs=2,
        default=(800, 543),
        metavar=("W", "H"),
        help="Stored image size (keep the 280:190 aspect ratio).",
    )
    p.add_argument("--format", default="jpg", choices=["jpg", "png"])
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable illumination normalization + CLAHE in the canonical view.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = ImageConversionConfig(
        dpi=args.dpi,
        canonical_size=tuple(args.canonical_size),
        image_format=args.format,
        jpeg_quality=args.jpeg_quality,
        variants_per_record=args.variants_per_record,
        photo_preset_weights=_parse_weights(args.photo_mix),
        enhance=not args.no_enhance,
        seed=args.seed,
    )
    records = DatasetManifest.load(args.manifest_path).records if args.manifest_path else None

    manifest = convert_echonext_to_images(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=cfg,
        records=records,
        splits=args.splits,
        label=args.label,
        max_per_split=args.max_per_split,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )

    print("\nImages per split / label:")
    if len(manifest):
        label_col = "shd_flag" if args.label == "shd" else "lvef_lte_45_flag"
        print(manifest.groupby(["split", label_col]).size().unstack(fill_value=0).to_string())
    print(f"\nManifest: {args.output_dir / 'image_manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
