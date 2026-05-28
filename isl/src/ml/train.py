"""
Training script for ISL sign language recognition models.

The training path is designed for honest model improvement work:
  - CPU/GPU auto selection with optional CUDA AMP
  - Stratified train/validation split
  - Weighted sampling for class balance
  - Configurable landmark augmentation
  - Multiple model families through --model-type
  - Top-1/top-3/top-5, macro F1, per-class metrics, and confusion matrix
  - Checkpoint metadata that the backend recognizer can load
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.ml.dataset import AugmentationConfig, create_data_loaders, FEATURE_DIM
from src.ml.model import (
    ISLModel,
    ISLModelHybrid,
    ISLModelLite,
    ISLModelTCN,
    ISLModelTransformer,
)

MODEL_TYPES = ("hybrid", "transformer", "tcn", "full", "lite")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for repeatable splits and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve the requested training device."""
    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but this PyTorch build does not support it")
    return resolved


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Apply mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = max(self.warmup_epochs + 1, total_epochs)
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * factor)


def build_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Build a boolean mask where True marks padded positions."""
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps >= lengths.unsqueeze(1)


def forward_model(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward helper that applies sequence masks when lengths are available."""
    mask = None
    if batch_lengths is not None:
        mask = build_padding_mask(batch_lengths, batch_x.size(1))
    return model(batch_x, mask=mask, lengths=batch_lengths)


def build_model(
    model_type: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_classes: int,
) -> nn.Module:
    """Construct a model by name."""
    if model_type == "lite":
        return ISLModelLite(
            input_dim=FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    if model_type == "hybrid":
        return ISLModelHybrid(
            input_dim=FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
    if model_type == "transformer":
        return ISLModelTransformer(
            input_dim=FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
    if model_type == "tcn":
        return ISLModelTCN(
            input_dim=FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
    if model_type == "full":
        return ISLModel(
            input_dim=FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def _empty_confusion(num_classes: int) -> np.ndarray:
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def compute_metrics(
    targets: Sequence[int],
    preds: Sequence[int],
    probs: np.ndarray,
    num_classes: int,
    class_names: Sequence[str],
) -> Dict:
    """Compute top-k and per-class classification metrics without sklearn."""
    targets_np = np.asarray(targets, dtype=np.int64)
    preds_np = np.asarray(preds, dtype=np.int64)
    n = int(len(targets_np))
    confusion = _empty_confusion(num_classes)

    for true_idx, pred_idx in zip(targets_np, preds_np):
        confusion[int(true_idx), int(pred_idx)] += 1

    if n == 0:
        return {
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": [],
            "confusion_matrix": confusion.tolist(),
        }

    top_metrics = {}
    if probs.size:
        order = np.argsort(-probs, axis=1)
        for k in (1, 3, 5):
            kk = min(k, probs.shape[1])
            correct = [targets_np[i] in order[i, :kk] for i in range(n)]
            top_metrics[f"top{k}"] = float(np.mean(correct))
    else:
        top_metrics = {"top1": float(np.mean(targets_np == preds_np)), "top3": 0.0, "top5": 0.0}

    per_class = []
    precisions = []
    recalls = []
    f1s = []
    weighted_f1_total = 0.0

    for idx, name in enumerate(class_names):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - confusion[idx, idx])
        fn = float(confusion[idx, :].sum() - confusion[idx, idx])
        support = int(confusion[idx, :].sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_f1_total += f1 * support
        per_class.append(
            {
                "class_index": idx,
                "class_name": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    return {
        **top_metrics,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(weighted_f1_total / max(n, 1)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_model(
    model: nn.Module,
    data_loader,
    criterion,
    device: torch.device,
    num_classes: int,
    class_names: Sequence[str],
) -> Dict:
    """Evaluate a model on a data loader."""
    model.eval()
    total_loss = 0.0
    total = 0
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                batch_x, batch_y, batch_lengths = batch
            else:
                batch_x, batch_y = batch
                batch_lengths = None

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if batch_lengths is not None:
                batch_lengths = batch_lengths.to(device)

            logits = forward_model(model, batch_x, batch_lengths)
            loss = criterion(logits, batch_y)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total += bs
            all_targets.extend(batch_y.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.append(probs.detach().cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, num_classes))
    metrics = compute_metrics(all_targets, all_preds, probs_np, num_classes, class_names)
    metrics["loss"] = float(total_loss / max(total, 1))
    metrics["samples"] = int(total)
    return metrics


def save_confusion_csv(path: Path, class_names: Sequence[str], matrix: Sequence[Sequence[int]]) -> None:
    """Save a readable confusion matrix CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row])


def train_model(
    data_dir: str = "extracted_data",
    output_dir: str = "models",
    model_type: str = "hybrid",
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.35,
    epochs: int = 150,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    min_lr: float = 1e-6,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    use_mixup: bool = True,
    mixup_alpha: float = 0.15,
    seq_length: int = 30,
    patience: int = 25,
    warmup_epochs: int = 10,
    val_ratio: float = 0.2,
    seed: int = 42,
    device: str = "auto",
    use_amp: bool = True,
    num_workers: int = 0,
    augment: bool = True,
    augment_repeats: int = 2,
    mirror_augment: bool = False,
    noise_std: float = 0.012,
    frame_drop_prob: float = 0.08,
    jitter_shift: float = 0.025,
    time_stretch_min: float = 0.85,
    time_stretch_max: float = 1.18,
    temporal_crop_prob: float = 0.25,
) -> Dict:
    """Train one ISL recognition model and return summary metrics."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type}")

    set_seed(seed)
    device_obj = resolve_device(device)
    amp_enabled = use_amp and device_obj.type == "cuda"
    pin_memory = device_obj.type == "cuda"

    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"Device: {device_obj}")
    if device_obj.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP: {'enabled' if amp_enabled else 'disabled'}")

    augmentation_config = AugmentationConfig(
        enabled=augment,
        noise_std=noise_std,
        time_stretch_min=time_stretch_min,
        time_stretch_max=time_stretch_max,
        frame_drop_prob=frame_drop_prob,
        mirror_prob=0.5 if mirror_augment else 0.0,
        jitter_shift=jitter_shift,
        temporal_crop_prob=temporal_crop_prob,
    )

    train_loader, val_loader, class_names, meta = create_data_loaders(
        data_dir,
        seq_length=seq_length,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=num_workers,
        augmentation_config=augmentation_config,
        augment_repeats=augment_repeats,
        pin_memory=pin_memory,
    )

    num_classes = len(class_names)
    print(f"\nModel: {model_type}, Hidden: {hidden_dim}, Layers: {num_layers}")
    print(f"Features: {FEATURE_DIM}, Seq length: {seq_length}, Classes: {num_classes}")
    print(f"Augment: {augment}, Repeats/epoch: {augment_repeats}, Mirror: {mirror_augment}")

    model = build_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device_obj)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {train_params:,} trainable")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        min_lr=min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "min_lr": min_lr,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "use_mixup": use_mixup,
        "mixup_alpha": mixup_alpha,
        "seq_length": seq_length,
        "patience": patience,
        "warmup_epochs": warmup_epochs,
        "val_ratio": val_ratio,
        "seed": seed,
        "device": str(device_obj),
        "use_amp": amp_enabled,
        "num_workers": num_workers,
        "augment": augment,
        "augment_repeats": augment_repeats,
        "mirror_augment": mirror_augment,
        "noise_std": noise_std,
        "frame_drop_prob": frame_drop_prob,
        "jitter_shift": jitter_shift,
        "time_stretch_min": time_stretch_min,
        "time_stretch_max": time_stretch_max,
        "temporal_crop_prob": temporal_crop_prob,
        "feature_dim": FEATURE_DIM,
        "num_classes": num_classes,
        "total_params": total_params,
        "trainable_params": train_params,
    }
    with open(out_path / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    recommended_min_frames = max(10, seq_length // 2)
    recommended_confidence = 0.58 if model_type in ("hybrid", "transformer", "tcn") else 0.60
    recommended_margin = 0.08
    recommended_max_entropy = 0.55

    best_val_acc = 0.0
    best_epoch = 0
    best_metrics: Optional[Dict] = None
    patience_counter = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top3": [],
        "val_top5": [],
        "val_macro_f1": [],
        "lr": [],
    }

    print(
        f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>10} | {'Val Acc':>9} | {'Top-3':>7} | {'Macro F1':>8} | {'LR':>10}"
    )
    print("-" * 92)
    t0 = time.time()

    for epoch in range(epochs):
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]

        model.train()
        train_loss = 0.0
        train_correct = 0.0
        train_total = 0

        for batch in train_loader:
            if len(batch) == 3:
                batch_x, batch_y, batch_lengths = batch
            else:
                batch_x, batch_y = batch
                batch_lengths = None

            batch_x = batch_x.to(device_obj, non_blocking=pin_memory)
            batch_y = batch_y.to(device_obj, non_blocking=pin_memory)
            if batch_lengths is not None:
                batch_lengths = batch_lengths.to(device_obj, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if use_mixup and random.random() < 0.5:
                    batch_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, mixup_alpha)
                    logits = forward_model(model, batch_x, batch_lengths)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                    predicted = logits.argmax(dim=1)
                    train_correct += (
                        lam * (predicted == y_a).float()
                        + (1 - lam) * (predicted == y_b).float()
                    ).sum().item()
                else:
                    logits = forward_model(model, batch_x, batch_lengths)
                    loss = criterion(logits, batch_y)
                    predicted = logits.argmax(dim=1)
                    train_correct += (predicted == batch_y).sum().item()

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_total += batch_x.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device_obj,
            num_classes,
            class_names,
        )
        val_acc = float(val_metrics["top1"])

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(val_acc)
        history["val_top3"].append(float(val_metrics["top3"]))
        history["val_top5"].append(float(val_metrics["top5"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
        history["lr"].append(float(lr))

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            marker = " *"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": float(val_metrics["loss"]),
                    "val_top3": float(val_metrics["top3"]),
                    "val_top5": float(val_metrics["top5"]),
                    "val_macro_f1": float(val_metrics["macro_f1"]),
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "model_type": model_type,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "input_dim": FEATURE_DIM,
                    "seq_length": seq_length,
                    "recommended_min_frames": recommended_min_frames,
                    "recommended_confidence_threshold": recommended_confidence,
                    "recommended_margin_threshold": recommended_margin,
                    "recommended_max_entropy": recommended_max_entropy,
                    "uses_padding_mask": True,
                    "training_config": config,
                },
                str(out_path / "best_model.pt"),
            )
        else:
            patience_counter += 1

        print(
            f"{epoch + 1:5d} | {train_loss:10.4f} | {train_acc:8.1%} | "
            f"{val_metrics['loss']:10.4f} | {val_acc:8.1%} | "
            f"{val_metrics['top3']:6.1%} | {val_metrics['macro_f1']:7.1%} | {lr:10.6f}{marker}"
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (best: epoch {best_epoch + 1})")
            break

    final_metrics = evaluate_model(
        model,
        val_loader,
        criterion,
        device_obj,
        num_classes,
        class_names,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes,
            "model_type": model_type,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "input_dim": FEATURE_DIM,
            "seq_length": seq_length,
            "recommended_min_frames": recommended_min_frames,
            "recommended_confidence_threshold": recommended_confidence,
            "recommended_margin_threshold": recommended_margin,
            "recommended_max_entropy": recommended_max_entropy,
            "uses_padding_mask": True,
            "training_config": config,
            "val_acc": float(final_metrics["top1"]),
            "val_loss": float(final_metrics["loss"]),
            "val_top3": float(final_metrics["top3"]),
            "val_top5": float(final_metrics["top5"]),
            "val_macro_f1": float(final_metrics["macro_f1"]),
        },
        str(out_path / "last_model.pt"),
    )

    with open(out_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_path / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    report = {
        "best_epoch": best_epoch + 1,
        "best_val_acc": float(best_val_acc),
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "config": config,
        "dataset_metadata": meta,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if best_metrics is not None:
        save_confusion_csv(
            out_path / "confusion_matrix.csv",
            class_names,
            best_metrics["confusion_matrix"],
        )

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2%} (epoch {best_epoch + 1})")
    if best_metrics is not None:
        print(
            f"Best top-3: {best_metrics['top3']:.2%}, "
            f"top-5: {best_metrics['top5']:.2%}, "
            f"macro F1: {best_metrics['macro_f1']:.2%}"
        )
    print(f"Artifacts saved to: {out_path}")

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "history": history,
        "class_names": class_names,
        "output_dir": str(out_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ISL recognition model")
    parser.add_argument("--data-dir", default="extracted_data", help="Path to extracted landmarks")
    parser.add_argument("--output-dir", default="models", help="Output directory for models")
    parser.add_argument("--model-type", default="hybrid", choices=MODEL_TYPES)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-mixup", action="store_true", help="Disable mixup")
    parser.add_argument("--mixup-alpha", type=float, default=0.15)
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true", help="Disable landmark augmentation")
    parser.add_argument("--augment-repeats", type=int, default=2)
    parser.add_argument(
        "--mirror-augment",
        action="store_true",
        help="Enable left/right mirroring. Off by default because it can change sign meaning.",
    )
    parser.add_argument("--noise-std", type=float, default=0.012)
    parser.add_argument("--frame-drop-prob", type=float, default=0.08)
    parser.add_argument("--jitter-shift", type=float, default=0.025)
    parser.add_argument("--time-stretch-min", type=float, default=0.85)
    parser.add_argument("--time-stretch-max", type=float, default=1.18)
    parser.add_argument("--temporal-crop-prob", type=float, default=0.25)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        use_mixup=not args.no_mixup,
        mixup_alpha=args.mixup_alpha,
        seq_length=args.seq_length,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        augment_repeats=args.augment_repeats,
        mirror_augment=args.mirror_augment,
        noise_std=args.noise_std,
        frame_drop_prob=args.frame_drop_prob,
        jitter_shift=args.jitter_shift,
        time_stretch_min=args.time_stretch_min,
        time_stretch_max=args.time_stretch_max,
        temporal_crop_prob=args.temporal_crop_prob,
    )


if __name__ == "__main__":
    main()
