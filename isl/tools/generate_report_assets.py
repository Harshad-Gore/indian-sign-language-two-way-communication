"""Generate lightweight SVG report assets from saved benchmark results.

This keeps the README publish-friendly: the visual outputs are small text-based
SVGs that document the real benchmark metrics without committing large model
checkpoints or notebook-only artifacts.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


SVG_NS = "http://www.w3.org/2000/svg"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def fnum(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class SvgBuilder:
    def __init__(self, width: int, height: int, background: str = "#09111f") -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-labelledby="title desc">'
        ]
        self.add('<title id="title">ISL benchmark report asset</title>')
        self.add(
            '<desc id="desc">SVG report asset generated from the saved benchmark metrics and training histories.</desc>'
        )
        self.add(
            "<defs>"
            '<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">'
            '<stop offset="0%" stop-color="#07101d"/>'
            '<stop offset="60%" stop-color="#0b1324"/>'
            '<stop offset="100%" stop-color="#081726"/>'
            "</linearGradient>"
            '<linearGradient id="accent" x1="0%" y1="0%" x2="100%" y2="0%">'
            '<stop offset="0%" stop-color="#22d3ee"/>'
            '<stop offset="55%" stop-color="#6366f1"/>'
            '<stop offset="100%" stop-color="#a855f7"/>'
            "</linearGradient>"
            '<linearGradient id="good" x1="0%" y1="0%" x2="100%" y2="0%">'
            '<stop offset="0%" stop-color="#4ade80"/>'
            '<stop offset="100%" stop-color="#22c55e"/>'
            "</linearGradient>"
            '<linearGradient id="warn" x1="0%" y1="0%" x2="100%" y2="0%">'
            '<stop offset="0%" stop-color="#fbbf24"/>'
            '<stop offset="100%" stop-color="#f97316"/>'
            "</linearGradient>"
            "</defs>"
        )
        self.rect(0, 0, width, height, fill="url(#bg)")
        # Decorative glow blobs for a less sterile report look.
        self.circle(width * 0.08, height * 0.16, 150, fill="#22d3ee", opacity=0.10)
        self.circle(width * 0.92, height * 0.14, 180, fill="#6366f1", opacity=0.12)
        self.circle(width * 0.78, height * 0.92, 240, fill="#a855f7", opacity=0.08)

    def add(self, snippet: str) -> None:
        self.parts.append(snippet)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1,
        rx: float = 0,
        opacity: float = 1.0,
    ) -> None:
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}" '
            f'opacity="{opacity:.3f}" />'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = "#94a3b8",
        stroke_width: float = 1.0,
        opacity: float = 1.0,
        dasharray: str | None = None,
    ) -> None:
        dash = f' stroke-dasharray="{dasharray}"' if dasharray else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.1f}" opacity="{opacity:.3f}"{dash} />'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1,
        opacity: float = 1.0,
    ) -> None:
        self.add(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.1f}" opacity="{opacity:.3f}" />'
        )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        *,
        stroke: str = "#22d3ee",
        stroke_width: float = 3.0,
        fill: str = "none",
        opacity: float = 1.0,
        dasharray: str | None = None,
    ) -> None:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash = f' stroke-dasharray="{dasharray}"' if dasharray else ""
        self.add(
            f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity:.3f}"{dash} />'
        )

    def text(
        self,
        x: float,
        y: float,
        value: object,
        *,
        size: float = 16,
        fill: str = "#e5eefb",
        weight: int = 400,
        anchor: str = "start",
        opacity: float = 1.0,
        family: str = "Inter, Segoe UI, system-ui, sans-serif",
        extra: str = "",
    ) -> None:
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="{family}" '
            f'font-size="{size:.1f}" font-weight="{weight}" text-anchor="{anchor}" '
            f'opacity="{opacity:.3f}" {extra}>{esc(value)}</text>'
        )

    def path(
        self,
        d: str,
        *,
        stroke: str = "#22d3ee",
        stroke_width: float = 3.0,
        fill: str = "none",
        opacity: float = 1.0,
    ) -> None:
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity:.3f}" />'
        )

    def finalize(self) -> str:
        self.add("</svg>")
        return "\n".join(self.parts)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_text_block(svg: SvgBuilder, x: float, y: float, title: str, subtitle: str) -> None:
    svg.text(x, y, title, size=40, weight=900, fill="#f8fbff")
    svg.text(x, y + 28, subtitle, size=15, fill="#9fb2d0", opacity=0.95)


def make_card(svg: SvgBuilder, x: float, y: float, w: float, h: float, label: str, value: str, accent: str) -> None:
    svg.rect(x, y, w, h, fill="#0b1630", stroke="#1e3358", stroke_width=1.2, rx=22, opacity=0.92)
    svg.rect(x, y, 8, h, fill=accent, stroke="none", rx=4, opacity=0.98)
    svg.text(x + 20, y + 28, label, size=11, fill="#a9bdd9", weight=700)
    svg.text(x + 20, y + 62, value, size=24, fill="#f8fbff", weight=900)


def draw_benchmark_comparison(output_path: Path, rows: list[dict[str, str]]) -> None:
    models = [row["model_type"] for row in rows]
    series = [
        ("best_val_acc", "Best val acc", "#22d3ee"),
        ("best_top3", "Top-3", "#6366f1"),
        ("best_macro_f1", "Macro F1", "#f59e0b"),
    ]

    svg = SvgBuilder(1440, 840)
    make_text_block(
        svg,
        72,
        68,
        "ISL Recognition Benchmark",
        "Five-model sweep on 1,120 landmark videos across 71 classes, with augmentation and CPU/GPU auto selection.",
    )

    best = rows[0]
    make_card(svg, 980, 38, 168, 92, "Winner", best["model_type"].title(), "url(#accent)")
    make_card(svg, 1160, 38, 128, 92, "Best top-1", pct(float(best["best_val_acc"])), "url(#good)")
    make_card(svg, 1300, 38, 118, 92, "Macro-F1", pct(float(best["best_macro_f1"])), "url(#warn)")

    chart_x = 96
    chart_y = 180
    chart_w = 1260
    chart_h = 500
    svg.rect(chart_x, chart_y, chart_w, chart_h, fill="#07101e", stroke="#1b2f50", stroke_width=1.2, rx=24, opacity=0.95)

    left, right, top, bottom = 90, 30, 24, 78
    plot_x = chart_x + left
    plot_y = chart_y + top
    plot_w = chart_w - left - right
    plot_h = chart_h - top - bottom

    # Grid and y-axis.
    for tick in [0.80, 0.85, 0.90, 0.95, 1.00]:
        y = plot_y + plot_h - (tick * plot_h)
        svg.line(plot_x, y, plot_x + plot_w, y, stroke="#1f3355", stroke_width=1, opacity=0.95)
        svg.text(plot_x - 14, y + 4, pct(tick), size=12, fill="#8ea3bf", anchor="end")

    svg.line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke="#385074", stroke_width=1.2, opacity=0.75)
    svg.line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#385074", stroke_width=1.2, opacity=0.75)

    group_w = plot_w / len(models)
    inner_w = group_w * 0.72
    bar_w = inner_w / len(series) - 6
    max_value = 1.0
    for i, row in enumerate(rows):
        group_x = plot_x + i * group_w
        start_x = group_x + (group_w - inner_w) / 2
        model_name = row["model_type"]
        svg.text(group_x + group_w / 2, plot_y + plot_h + 30, model_name.upper(), size=14, fill="#d5def0", weight=800, anchor="middle")
        svg.text(group_x + group_w / 2, plot_y + plot_h + 48, f"rank #{row['rank']}", size=11, fill="#7e92ad", anchor="middle")
        for s_idx, (field, _, color) in enumerate(series):
            value = float(row[field])
            bar_h = value / max_value * plot_h
            x = start_x + s_idx * (bar_w + 6)
            y = plot_y + plot_h - bar_h
            svg.rect(x, y, bar_w, bar_h, fill=color, stroke="none", rx=8, opacity=0.92)
            svg.text(x + bar_w / 2, y - 6, pct(value), size=11, fill="#bcd0ea", anchor="middle")

    # Legend
    legend_x = 110
    legend_y = 704
    for idx, (_, label, color) in enumerate(series):
        lx = legend_x + idx * 180
        svg.rect(lx, legend_y, 16, 16, fill=color, rx=5)
        svg.text(lx + 24, legend_y + 13, label, size=12, fill="#dbe6f4", weight=700)

    svg.text(
        96,
        780,
        f"Best model: {best['model_type']}  |  Validation accuracy: {pct(float(best['best_val_acc']))}  |  "
        f"Top-3: {pct(float(best['best_top3']))}  |  Macro-F1: {pct(float(best['best_macro_f1']))}",
        size=13,
        fill="#8fa6c7",
    )
    output_path.write_text(svg.finalize(), encoding="utf-8")


def plot_series(
    svg: SvgBuilder,
    x: float,
    y: float,
    w: float,
    h: float,
    epochs: Sequence[int],
    series: Sequence[tuple[str, Sequence[float], str, str, float]],
    *,
    y_min: float,
    y_max: float,
    y_label: str,
    best_epoch: int | None = None,
) -> None:
    svg.rect(x, y, w, h, fill="#07101e", stroke="#1b2f50", stroke_width=1.2, rx=22, opacity=0.95)
    svg.text(x + 24, y + 30, y_label, size=18, fill="#f4f8ff", weight=900)

    pad_left = 64
    pad_right = 20
    pad_top = 24
    pad_bottom = 42
    px = x + pad_left
    py = y + pad_top
    pw = w - pad_left - pad_right
    ph = h - pad_top - pad_bottom

    # Grid.
    grid_ticks = 5
    for idx in range(grid_ticks + 1):
        frac = idx / grid_ticks
        gy = py + ph - frac * ph
        svg.line(px, gy, px + pw, gy, stroke="#1f3355", stroke_width=1, opacity=0.95)
        value = y_min + frac * (y_max - y_min)
        svg.text(px - 10, gy + 4, fnum(value, 2), size=11, fill="#8ea3bf", anchor="end")

    for idx in range(0, len(epochs), max(1, len(epochs) // 9)):
        ex = px + (epochs[idx] - epochs[0]) / max(1, epochs[-1] - epochs[0]) * pw
        svg.line(ex, py, ex, py + ph, stroke="#17304e", stroke_width=1, opacity=0.8)
        svg.text(ex, py + ph + 22, str(epochs[idx]), size=11, fill="#8ea3bf", anchor="middle")

    svg.line(px, py, px, py + ph, stroke="#385074", stroke_width=1.2, opacity=0.75)
    svg.line(px, py + ph, px + pw, py + ph, stroke="#385074", stroke_width=1.2, opacity=0.75)

    if best_epoch is not None and epochs[0] <= best_epoch <= epochs[-1]:
        ex = px + (best_epoch - epochs[0]) / max(1, epochs[-1] - epochs[0]) * pw
        svg.line(ex, py, ex, py + ph, stroke="#f59e0b", stroke_width=1.5, opacity=0.85, dasharray="7,6")
        svg.text(ex + 8, py + 16, f"best epoch {best_epoch}", size=11, fill="#fbbf24", weight=700)

    for label, values, color, dash, opacity in series:
        points: list[tuple[float, float]] = []
        for epoch, value in zip(epochs, values):
            ex = px + (epoch - epochs[0]) / max(1, epochs[-1] - epochs[0]) * pw
            normalized = (value - y_min) / max(1e-9, y_max - y_min)
            ey = py + ph - normalized * ph
            points.append((ex, ey))
        svg.polyline(points, stroke=color, stroke_width=3.0, opacity=opacity, dasharray=dash or None)

    legend_x = x + 24
    legend_y = y + h - 18
    for idx, (label, _, color, dash, _) in enumerate(series):
        lx = legend_x + idx * 180
        svg.rect(lx, legend_y - 13, 16, 16, fill=color, rx=4)
        if dash:
            svg.line(lx + 2, legend_y - 5, lx + 14, legend_y - 5, stroke="#09111f", stroke_width=2.0, dasharray=dash)
        svg.text(lx + 24, legend_y + 1, label, size=11, fill="#dbe6f4", weight=700)


def draw_training_curves(output_path: Path, history: dict, metrics: dict) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    best_epoch = int(metrics["best_epoch"])
    best_val_acc = float(metrics["best_val_acc"])

    svg = SvgBuilder(1440, 1080)
    make_text_block(
        svg,
        72,
        68,
        "Hybrid model training curves",
        "The selected checkpoint was trained for 93 recorded epochs and reached its best validation score at epoch 58.",
    )

    # Top-right callout
    make_card(svg, 1040, 38, 132, 92, "Best acc", pct(best_val_acc), "url(#good)")
    make_card(svg, 1182, 38, 116, 92, "Final acc", pct(float(history["val_acc"][-1])), "url(#accent)")
    make_card(svg, 1308, 38, 92, 92, "Epochs", str(len(epochs)), "url(#warn)")

    loss_values = list(history["train_loss"]) + list(history["val_loss"])
    loss_min = max(0.0, min(loss_values) * 0.90)
    loss_max = max(loss_values) * 1.08
    acc_values = list(history["train_acc"]) + list(history["val_acc"]) + list(history["val_top3"]) + list(history["val_top5"]) + list(history["val_macro_f1"])
    acc_min = 0.0
    acc_max = min(1.0, max(acc_values) * 1.05)
    if acc_max < 0.98:
        acc_max = 1.0

    plot_series(
        svg,
        88,
        150,
        1264,
        380,
        epochs,
        [
            ("Train loss", history["train_loss"], "#22d3ee", "", 0.98),
            ("Val loss", history["val_loss"], "#a855f7", "8,5", 0.96),
        ],
        y_min=loss_min,
        y_max=loss_max,
        y_label="Loss trajectory",
        best_epoch=best_epoch,
    )
    plot_series(
        svg,
        88,
        570,
        1264,
        380,
        epochs,
        [
            ("Train acc", history["train_acc"], "#4ade80", "", 0.98),
            ("Val acc", history["val_acc"], "#f59e0b", "8,5", 0.98),
        ],
        y_min=acc_min,
        y_max=acc_max,
        y_label="Accuracy trajectory",
        best_epoch=best_epoch,
    )

    svg.text(
        88,
        1014,
        f"Best validation accuracy: {pct(best_val_acc)} at epoch {best_epoch} | "
        f"Final validation accuracy: {pct(float(history['val_acc'][-1]))} | "
        f"Final macro-F1: {pct(float(history['val_macro_f1'][-1]))}",
        size=13,
        fill="#8fa6c7",
    )
    output_path.write_text(svg.finalize(), encoding="utf-8")


def draw_per_class_f1(output_path: Path, per_class: Sequence[dict], class_counts: dict[str, int]) -> None:
    rows = sorted(per_class, key=lambda row: float(row["f1"]))
    row_h = 22
    top_margin = 126
    left_label_x = 48
    support_x = 360
    bar_x = 410
    bar_w = 860
    bottom_margin = 56
    height = top_margin + len(rows) * row_h + bottom_margin
    svg = SvgBuilder(1440, height)

    make_text_block(
        svg,
        72,
        68,
        "Per-class F1 distribution",
        "Sorted ascending so the weakest classes are easiest to spot. Support counts come from the validation subset.",
    )

    make_card(svg, 1140, 38, 96, 92, "Classes", str(len(rows)), "url(#accent)")
    make_card(
        svg,
        1246,
        38,
        132,
        92,
        "Avg support",
        f"{sum(class_counts.values()) / max(1, len(class_counts)):.2f}",
        "url(#good)",
    )

    svg.rect(88, 148, 1264, len(rows) * row_h + 24, fill="#07101e", stroke="#1b2f50", stroke_width=1.2, rx=22, opacity=0.95)
    svg.text(left_label_x, 132, "Class", size=12, fill="#8ea3bf", weight=700)
    svg.text(support_x, 132, "Support", size=12, fill="#8ea3bf", weight=700)
    svg.text(bar_x, 132, "F1 score", size=12, fill="#8ea3bf", weight=700)

    # Average guide line.
    for frac, label in [(0.25, "0.25"), (0.50, "0.50"), (0.75, "0.75"), (1.00, "1.00")]:
        gx = bar_x + frac * bar_w
        svg.line(gx, 156, gx, 148 + len(rows) * row_h + 12, stroke="#1f3355", stroke_width=1, opacity=0.85)
        svg.text(gx, 152, label, size=11, fill="#8ea3bf", anchor="middle")

    for idx, row in enumerate(rows):
        y = 168 + idx * row_h
        f1 = float(row["f1"])
        class_name = row["class_name"]
        support = int(row["support"])
        bar_len = clamp(f1, 0.0, 1.0) * bar_w
        # Color moves from orange to green.
        if f1 >= 0.95:
            fill = "#22c55e"
        elif f1 >= 0.80:
            fill = "#84cc16"
        elif f1 >= 0.60:
            fill = "#f59e0b"
        else:
            fill = "#ef4444"

        svg.text(left_label_x, y + 11, class_name, size=12, fill="#e3ecfa")
        svg.text(support_x, y + 11, f"n={support}", size=12, fill="#9fb2d0")
        svg.rect(bar_x, y - 1, bar_w, 14, fill="#11213a", stroke="#22395f", stroke_width=0.8, rx=7, opacity=0.95)
        svg.rect(bar_x, y - 1, bar_len, 14, fill=fill, stroke="none", rx=7, opacity=0.95)
        svg.text(bar_x + bar_len + 10, y + 11, pct(f1), size=11, fill="#cfe0f2")

    lowest = rows[0]
    highest = rows[-1]
    svg.text(
        88,
        height - 18,
        f"Weakest class: {lowest['class_name']} ({pct(float(lowest['f1']))}) | "
        f"Strongest class: {highest['class_name']} ({pct(float(highest['f1']))})",
        size=13,
        fill="#8fa6c7",
    )
    output_path.write_text(svg.finalize(), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SVG report assets from benchmark metrics.")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Path to isl/models/benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write docs/figures SVGs into",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_root = Path(__file__).resolve().parents[2]
    benchmark_dir = args.benchmark_dir or (script_root / "isl" / "models" / "benchmark")
    output_dir = args.output_dir or (script_root / "docs" / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_results = load_csv(benchmark_dir / "benchmark_results.csv")
    benchmark_results = sorted(benchmark_results, key=lambda row: int(row["rank"]))
    best_run = benchmark_results[0]

    best_run_dir = benchmark_dir / Path(best_run["output_dir"]).name
    best_metrics = load_json(best_run_dir / "metrics.json")
    training_history = load_json(best_run_dir / "training_history.json")
    per_class = best_metrics["best_metrics"]["per_class"]
    class_counts = best_metrics["dataset_metadata"]["class_counts"]

    draw_benchmark_comparison(output_dir / "benchmark-results.svg", benchmark_results)
    draw_training_curves(output_dir / "hybrid-training-curves.svg", training_history, best_metrics)
    draw_per_class_f1(output_dir / "hybrid-per-class-f1.svg", per_class, class_counts)

    # Small machine-readable summary for quick reference and future report scripts.
    summary = {
        "best_model": best_run["model_type"],
        "best_val_acc": float(best_run["best_val_acc"]),
        "best_top3": float(best_run["best_top3"]),
        "best_top5": float(best_run["best_top5"]),
        "best_macro_f1": float(best_run["best_macro_f1"]),
        "best_epoch": int(best_run["best_epoch"]),
        "num_models": len(benchmark_results),
        "num_classes": best_metrics["dataset_metadata"]["num_classes"],
        "total_videos": best_metrics["dataset_metadata"]["total_videos"],
    }
    (output_dir / "benchmark-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote SVG report assets to {output_dir}")


if __name__ == "__main__":
    main()
