#!/usr/bin/env python3
"""Compare multiple depth-evaluation summaries and render a compact bar chart.

Accepted input formats:
- JSON produced by `evaluate_sparse_depth_predictions.py`
- JSON produced by `eval_kitti_scape_spike.py`
- JSON summaries with a `metrics` object
- plain-text logs containing lines like `_abs_rel_diff : 0.123`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_ALIASES = {
    "abs_rel": "abs_rel",
    "abs_rel_diff": "abs_rel",
    "squ_rel": "squ_rel",
    "squ_rel_diff": "squ_rel",
    "rms_linear": "rms_linear",
    "RMS_linear": "rms_linear",
    "rms_log": "rms_log",
    "RMS_log": "rms_log",
    "silog": "silog",
    "SILog": "silog",
    "mean_err": "mean_err",
    "mean_depth_error": "mean_err",
    "median_diff": "median_diff",
    "delta_1.25": "delta_1.25",
    "threshold_delta_1.25": "delta_1.25",
    "delta_1.25^2": "delta_1.25^2",
    "threshold_delta_1.25^2": "delta_1.25^2",
    "delta_1.25^3": "delta_1.25^3",
    "threshold_delta_1.25^3": "delta_1.25^3",
}

DEFAULT_METRICS = ("abs_rel", "rms_linear", "rms_log", "mean_err", "delta_1.25")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        action="append",
        required=True,
        help="Input specification in the form `Label=/path/to/summary`.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated canonical metric names to plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Depth Evaluation Comparison",
        help="Plot title.",
    )
    return parser.parse_args()


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected `Label=/path`, got `{spec}`")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label:
        raise ValueError(f"Missing label in `{spec}`")
    if not path.exists():
        raise FileNotFoundError(path)
    return label, path


def canonicalize_metrics(raw_metrics: dict[str, object]) -> OrderedDict[str, float]:
    result: OrderedDict[str, float] = OrderedDict()
    for key, value in raw_metrics.items():
        canonical_key = METRIC_ALIASES.get(key)
        if canonical_key is None:
            continue
        result[canonical_key] = float(value)
    return result


def parse_plaintext_metrics(path: Path) -> OrderedDict[str, float]:
    pattern = re.compile(r"^(_?[A-Za-z0-9_.^]+)\s*:\s*([-+0-9.eE]+)\s*$")
    metrics: OrderedDict[str, float] = OrderedDict()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        raw_key = match.group(1).lstrip("_")
        if re.match(r"^\d+_", raw_key):
            continue
        canonical_key = METRIC_ALIASES.get(raw_key)
        if canonical_key is None:
            continue
        metrics[canonical_key] = float(match.group(2))
    if not metrics:
        raise RuntimeError(f"No parseable metrics found in {path}")
    return metrics


def load_summary_metrics(path: Path) -> OrderedDict[str, float]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "metrics_frame_mean" in payload:
                return canonicalize_metrics(payload["metrics_frame_mean"])
            if "metrics" in payload:
                return canonicalize_metrics(payload["metrics"])
        raise RuntimeError(f"Unsupported JSON summary layout: {path}")
    return parse_plaintext_metrics(path)


def save_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(rows: list[dict[str, object]], metrics: list[str], title: str, out_path: Path) -> None:
    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(labels))
    colors = ["#0f766e", "#2563eb", "#dc2626", "#7c3aed", "#f59e0b", "#059669"]

    for idx, metric_name in enumerate(metrics):
        ax = axes[idx // ncols, idx % ncols]
        values = [float(row.get(metric_name, np.nan)) for row in rows]
        bars = ax.bar(x, values, color=[colors[i % len(colors)] for i in range(len(labels))])
        ax.set_title(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
        if metric_name.startswith("delta_"):
            ax.set_ylim(0.0, 1.05)

    total_axes = nrows * ncols
    for idx in range(len(metrics), total_axes):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]

    rows: list[dict[str, object]] = []
    for spec in args.input:
        label, path = parse_input_spec(spec)
        metric_map = load_summary_metrics(path)
        row: OrderedDict[str, object] = OrderedDict([("label", label), ("source", str(path))])
        for metric_name in metrics:
            row[metric_name] = metric_map.get(metric_name, math.nan)
        rows.append(row)

    save_csv(rows, args.output_dir / "comparison_metrics.csv")
    plot_comparison(rows, metrics, args.title, args.output_dir / "comparison_overview.png")
    print(f"[done] compared={len(rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
