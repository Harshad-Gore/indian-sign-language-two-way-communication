"""Benchmark multiple ISL landmark recognition models.

This runner trains each requested architecture with the same split/seed and saves
real metrics for comparison. It does not invent an accuracy number; the best
checkpoint is selected from validation top-1 accuracy and copied to the benchmark
root for easy deployment after you review the metrics.
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List

from src.ml.train import MODEL_TYPES, train_model


def model_defaults(model_type: str, hidden_dim: int, num_layers: int, dropout: float) -> Dict:
    """Use sensible per-model defaults while still allowing CLI overrides."""
    if model_type == "lite":
        return {
            "hidden_dim": min(hidden_dim, 128),
            "num_layers": 1,
            "dropout": min(dropout, 0.30),
        }
    if model_type == "transformer":
        return {
            "hidden_dim": hidden_dim,
            "num_layers": max(num_layers, 3),
            "dropout": min(dropout, 0.30),
        }
    if model_type == "tcn":
        return {
            "hidden_dim": hidden_dim,
            "num_layers": max(num_layers, 4),
            "dropout": min(dropout, 0.32),
        }
    if model_type == "full":
        return {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }
    return {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
    }


def write_results_csv(path: Path, rows: List[Dict]) -> None:
    """Write a compact benchmark table."""
    fields = [
        "rank",
        "model_type",
        "status",
        "best_val_acc",
        "best_top3",
        "best_top5",
        "best_macro_f1",
        "best_epoch",
        "output_dir",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def benchmark(args) -> Dict:
    started = time.time()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    for model_type in args.models:
        run_dir = root / f"{model_type}_seed{args.seed}"
        defaults = model_defaults(model_type, args.hidden_dim, args.num_layers, args.dropout)
        print("\n" + "=" * 80)
        print(f"Benchmarking model: {model_type}")
        print(f"Run dir: {run_dir}")
        print("=" * 80)

        try:
            result = train_model(
                data_dir=args.data_dir,
                output_dir=str(run_dir),
                model_type=model_type,
                hidden_dim=defaults["hidden_dim"],
                num_layers=defaults["num_layers"],
                dropout=defaults["dropout"],
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
            metrics = result.get("best_metrics") or {}
            results.append(
                {
                    "model_type": model_type,
                    "status": "ok",
                    "best_val_acc": float(result.get("best_val_acc", 0.0)),
                    "best_top3": float(metrics.get("top3", 0.0)),
                    "best_top5": float(metrics.get("top5", 0.0)),
                    "best_macro_f1": float(metrics.get("macro_f1", 0.0)),
                    "best_epoch": int(result.get("best_epoch", 0)) + 1,
                    "output_dir": str(run_dir),
                    "error": "",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model_type": model_type,
                    "status": "failed",
                    "best_val_acc": 0.0,
                    "best_top3": 0.0,
                    "best_top5": 0.0,
                    "best_macro_f1": 0.0,
                    "best_epoch": 0,
                    "output_dir": str(run_dir),
                    "error": str(exc),
                }
            )
            print(f"Model {model_type} failed: {exc}")

    ranked = sorted(
        results,
        key=lambda row: (row["status"] == "ok", row["best_val_acc"], row["best_macro_f1"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    best = ranked[0] if ranked else None
    if best and best["status"] == "ok":
        best_ckpt = Path(best["output_dir"]) / "best_model.pt"
        if best_ckpt.exists():
            shutil.copy2(best_ckpt, root / "best_model.pt")
            class_names_path = Path(best["output_dir"]) / "class_names.json"
            if class_names_path.exists():
                shutil.copy2(class_names_path, root / "class_names.json")
            print(f"\nBest checkpoint copied to: {root / 'best_model.pt'}")

    summary = {
        "best_model": best,
        "results": ranked,
        "elapsed_seconds": round(time.time() - started, 2),
        "command_settings": vars(args),
    }
    with open(root / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_results_csv(root / "benchmark_results.csv", ranked)

    print("\nBenchmark complete")
    for row in ranked:
        print(
            f"#{row['rank']} {row['model_type']:12s} "
            f"acc={row['best_val_acc']:.2%} top3={row['best_top3']:.2%} "
            f"f1={row['best_macro_f1']:.2%} status={row['status']}"
        )

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark ISL recognition models")
    parser.add_argument("--data-dir", default="extracted_data")
    parser.add_argument("--output-dir", default="models/benchmark")
    parser.add_argument("--models", nargs="+", default=["hybrid", "transformer", "tcn", "full", "lite"], choices=MODEL_TYPES)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-mixup", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=0.15)
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--augment-repeats", type=int, default=3)
    parser.add_argument("--mirror-augment", action="store_true")
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
    benchmark(args)


if __name__ == "__main__":
    main()
