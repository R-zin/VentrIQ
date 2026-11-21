#!/usr/bin/env python3
"""scripts/predict_shd.py - Score user-supplied ECG images with a trained SHD model.

    uv run python scripts/predict_shd.py photo1.jpg scan2.png \
        --checkpoint models/shd_mobilenetv3/best.pt

Accepts smartphone photos (the paper is detected and rectified automatically) or
flat scans / screenshots of a standard 12-lead (3x4 + rhythm strip) ECG printout.

Research use only - not a diagnostic device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from src.utils.paths import PROJECT_ROOT
from src.vision.inference import SHDPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument(
        "--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "shd_mobilenetv3" / "best.pt"
    )
    p.add_argument(
        "--rectify",
        choices=["auto", "always", "never"],
        default=None,
        help="Override paper detection/rectification (default: from checkpoint).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the decision threshold stored in the checkpoint.",
    )
    p.add_argument(
        "--save-canonical",
        type=Path,
        default=None,
        help="Directory to save the canonicalized view the model actually sees.",
    )
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    predictor = SHDPredictor.from_checkpoint(
        args.checkpoint, device=args.device, threshold=args.threshold
    )

    missing = [str(p) for p in args.images if not p.is_file()]
    if missing:
        print(f"Image(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    if args.save_canonical:
        from PIL import Image

        args.save_canonical.mkdir(parents=True, exist_ok=True)
        for path in args.images:
            with Image.open(path) as img:
                predictor.canonicalizer(img.convert("RGB"), rectify=args.rectify).save(
                    args.save_canonical / f"{path.stem}_canonical.png"
                )

    for pred in predictor.predict(list(args.images), rectify=args.rectify):
        print(json.dumps(pred.to_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
