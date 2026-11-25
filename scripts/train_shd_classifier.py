#!/usr/bin/env python3
"""scripts/train_shd_classifier.py - Train MobileNetV3 to predict SHD from ECG images.

Run ``scripts/convert_echonext_to_images.py`` first, then e.g.:

    uv run python scripts/train_shd_classifier.py \
        --image-dir data/images/shd --output-dir models/shd_mobilenetv3 \
        --arch large --pretrained --epochs 20 --batch-size 32 --num-workers 8

Writes ``best.pt`` (weights + preprocessing settings + decision threshold) and
``training_summary.json`` (per-epoch validation metrics and final test metrics).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from src.utils.paths import DATA_DIR, PROJECT_ROOT
from src.vision.conversion import load_conversion_config
from src.vision.dataset import DEFAULT_IMAGE_SIZE
from src.vision.training import TrainingConfig, train_classifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image-dir", type=Path, default=DATA_DIR / "images" / "shd")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "shd_mobilenetv3")
    p.add_argument("--label", default="shd", choices=["shd", "lvef_lte_45"])
    p.add_argument("--arch", default="large", choices=["small", "large"])
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="Start from ImageNet weights (downloaded by torchvision).",
    )
    p.add_argument(
        "--image-size", type=int, nargs=2, default=DEFAULT_IMAGE_SIZE, metavar=("W", "H")
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--patience", type=int, default=5, help="Early-stopping patience (epochs).")
    p.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help="Cap batches per epoch (smoke tests).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    conversion = load_conversion_config(args.image_dir)
    cfg = TrainingConfig(
        image_dir=str(args.image_dir),
        output_dir=str(args.output_dir),
        label=args.label,
        arch=args.arch,
        pretrained=args.pretrained,
        image_size=(args.image_size[0], args.image_size[1]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        max_batches_per_epoch=args.max_batches_per_epoch,
        early_stopping_patience=args.patience,
    )
    summary = train_classifier(cfg, conversion.canonicalization)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
