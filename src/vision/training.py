"""src/vision/training.py - Training / evaluation loop and metrics for the SHD image CNN."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ventriq_worker_init_fn
from src.vision.canonical import CanonicalizationConfig
from src.vision.dataset import DEFAULT_IMAGE_SIZE, ECGImageDataset, build_image_transform
from src.vision.model import MobileNetArch, build_mobilenet_v3

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


# ---------------------------------------------------------------------------
# Metrics (NumPy only, no scikit-learn dependency)
# ---------------------------------------------------------------------------


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties averaged)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve, step-wise)."""
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    hits = labels[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / n_pos)


def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Probability threshold maximizing sensitivity + specificity - 1."""
    labels = np.asarray(labels).astype(bool)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.all() or (~labels).all():
        return 0.5
    best_t, best_j = 0.5, -1.0
    for t in np.unique(probs):
        pred = probs >= t
        sens = (pred & labels).sum() / labels.sum()
        spec = (~pred & ~labels).sum() / (~labels).sum()
        if sens + spec - 1.0 > best_j:
            best_j, best_t = sens + spec - 1.0, float(t)
    return best_t


def binary_metrics(
    labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    labels = np.asarray(labels).astype(bool)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs >= threshold
    tp = int((pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    return {
        "n": int(len(labels)),
        "prevalence": float(labels.mean()) if len(labels) else float("nan"),
        "auroc": roc_auc(labels, probs),
        "auprc": average_precision(labels, probs),
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(1, len(labels)),
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "ppv": tp / max(1, tp + fp),
        "npv": tn / max(1, tn + fn),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    image_dir: str
    output_dir: str
    label: str = "shd"
    arch: MobileNetArch = "large"
    pretrained: bool = True
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "auto"
    seed: int = 42
    max_batches_per_epoch: Optional[int] = None  # for smoke tests
    early_stopping_patience: int = 5
    extra: Dict[str, Any] = field(default_factory=dict)


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(labels, probabilities, mean_loss)`` over ``loader``."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    labels, probs, total_loss, n = [], [], 0.0, 0
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x).reshape(-1)
        total_loss += float(criterion(logits, y))
        n += len(y)
        labels.append(y.cpu().numpy())
        probs.append(torch.sigmoid(logits).cpu().numpy())
    if not labels:
        return np.zeros(0), np.zeros(0), float("nan")
    return np.concatenate(labels), np.concatenate(probs), total_loss / max(1, n)


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    config: TrainingConfig,
    canonicalization: CanonicalizationConfig,
    threshold: float,
    metrics: Dict[str, Any],
    epoch: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model_state": model.state_dict(),
            "arch": config.arch,
            "label": config.label,
            "image_size": list(config.image_size),
            "canonicalization": canonicalization.to_dict(),
            "threshold": float(threshold),
            "epoch": int(epoch),
            "metrics": metrics,
            "training_config": asdict(config),
        },
        path,
    )
    return path


def _make_loader(ds: ECGImageDataset, cfg: TrainingConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        worker_init_fn=ventriq_worker_init_fn if cfg.num_workers > 0 else None,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )


def train_classifier(
    cfg: TrainingConfig,
    canonicalization: CanonicalizationConfig,
) -> Dict[str, Any]:
    """Train MobileNetV3 on converted ECG images; returns a summary dict.

    The best epoch (by validation AUROC, falling back to validation loss) is saved to
    ``{output_dir}/best.pt`` together with the preprocessing settings needed by
    ``SHDPredictor``. The decision threshold stored in the checkpoint is the Youden
    threshold on the validation set. If a test split exists it is evaluated once,
    with the best checkpoint, at the end.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = build_image_transform(cfg.image_size, train=True)
    eval_tf = build_image_transform(cfg.image_size, train=False)
    train_ds = ECGImageDataset(cfg.image_dir, split="train", label=cfg.label, transform=train_tf)
    val_ds = ECGImageDataset(cfg.image_dir, split="val", label=cfg.label, transform=eval_tf)
    test_ds = ECGImageDataset(cfg.image_dir, split="test", label=cfg.label, transform=eval_tf)
    if len(train_ds) == 0:
        raise ValueError(f"No labelled training images found in {cfg.image_dir}")
    logger.info(
        "Images: train=%d (pos %.1f%%) val=%d test=%d | device=%s",
        len(train_ds),
        100 * train_ds.positive_rate(),
        len(val_ds),
        len(test_ds),
        device,
    )

    model = build_mobilenet_v3(cfg.arch, pretrained=cfg.pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([train_ds.pos_weight()], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)
    if cfg.max_batches_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_batches_per_epoch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=max(1, cfg.epochs * steps_per_epoch),
        pct_start=0.1,
    )

    train_loader = _make_loader(train_ds, cfg, shuffle=True)
    val_loader = _make_loader(val_ds, cfg, shuffle=False) if len(val_ds) else None

    history = []
    best_score, best_epoch, epochs_without_improvement = -math.inf, -1, 0
    best_path = out_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for b, batch in enumerate(train_loader):
            if b >= steps_per_epoch:
                break
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x).reshape(-1), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)

        record: Dict[str, Any] = {"epoch": epoch, "train_loss": running / max(1, seen)}
        threshold = 0.5
        if val_loader is not None:
            y_val, p_val, val_loss = predict_dataset(
                model, val_loader, device, cfg.max_batches_per_epoch
            )
            threshold = youden_threshold(y_val, p_val)
            record.update({"val_loss": val_loss, "val": binary_metrics(y_val, p_val, threshold)})
            auroc = record["val"]["auroc"]
            score = auroc if not math.isnan(auroc) else -val_loss
        else:
            score = -record["train_loss"]
        record["seconds"] = round(time.time() - t0, 1)
        history.append(record)
        logger.info(
            "epoch %d | train_loss %.4f | val_loss %s | val_auroc %s | %.0fs",
            epoch,
            record["train_loss"],
            f"{record.get('val_loss', float('nan')):.4f}",
            f"{record.get('val', {}).get('auroc', float('nan')):.4f}",
            record["seconds"],
        )

        if score > best_score:
            best_score, best_epoch, epochs_without_improvement = score, epoch, 0
            save_checkpoint(best_path, model, cfg, canonicalization, threshold, record, epoch)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                logger.info(
                    "Early stopping after %d epochs without improvement.",
                    epochs_without_improvement,
                )
                break

    summary: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "checkpoint": str(best_path),
        "history": history,
    }

    if len(test_ds):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        y_test, p_test, test_loss = predict_dataset(
            model, _make_loader(test_ds, cfg, shuffle=False), device
        )
        summary["test"] = binary_metrics(y_test, p_test, ckpt["threshold"])
        summary["test"]["loss"] = test_loss
        logger.info("Test metrics (best epoch %d): %s", best_epoch, summary["test"])

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
