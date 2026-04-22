#!/usr/bin/env python3
"""Plot comparison of depth evaluation results across multiple runs.

Collects metrics from:
- eval_metrics*.log files (plain-text format: `_abs_rel_diff : 0.123`)
- metrics_summary.json files (from eval_kitti_scape_spike.py)
- eval_metrics_recomputed*.json files (from evaluate_saved_npy.py)
- analysis/summary.json files (from export_eval_results_csv.py)

Usage:
    python plot_eval_comparison.py --eval-root /path/to/runs/eval --output-dir /path/to/out
    python plot_eval_comparison.py --eval-root /path/to/runs/eval --output-dir /path/to/out \
        --include kitti_0001 test_mambassm_bidivim
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Metric aliases: normalize all known key variants to canonical names
# ---------------------------------------------------------------------------
METRIC_ALIASES: dict[str, str] = {
    "abs_rel": "abs_rel",
    "abs_rel_diff": "abs_rel",
    "_abs_rel_diff": "abs_rel",
    "squ_rel": "squ_rel",
    "squ_rel_diff": "squ_rel",
    "_squ_rel_diff": "squ_rel",
    "rms_linear": "rms_linear",
    "RMS_linear": "rms_linear",
    "_RMS_linear": "rms_linear",
    "rms_log": "rms_log",
    "RMS_log": "rms_log",
    "_RMS_log": "rms_log",
    "silog": "silog",
    "SILog": "silog",
    "_SILog": "silog",
    "mean_err": "mean_err",
    "mean_depth_error": "mean_err",
    "_mean_depth_error": "mean_err",
    "median_diff": "median_diff",
    "_median_diff": "median_diff",
    "delta_1.25": "delta_1.25",
    "delta1": "delta_1.25",
    "threshold_delta_1.25": "delta_1.25",
    "_threshold_delta_1.25": "delta_1.25",
    "delta_1.25^2": "delta_1.25^2",
    "delta2": "delta_1.25^2",
    "threshold_delta_1.25^2": "delta_1.25^2",
    "_threshold_delta_1.25^2": "delta_1.25^2",
    "delta_1.25^3": "delta_1.25^3",
    "delta3": "delta_1.25^3",
    "threshold_delta_1.25^3": "delta_1.25^3",
    "_threshold_delta_1.25^3": "delta_1.25^3",
}

CANONICAL_METRICS = (
    "abs_rel", "squ_rel", "rms_linear", "rms_log", "silog",
    "mean_err", "median_diff", "delta_1.25", "delta_1.25^2", "delta_1.25^3",
)

# Metrics where lower is better (for coloring)
LOWER_IS_BETTER = {"abs_rel", "squ_rel", "rms_linear", "rms_log", "silog", "mean_err", "median_diff"}

PLOT_COLORS = [
    "#0f766e", "#2563eb", "#dc2626", "#7c3aed",
    "#f59e0b", "#059669", "#db2777", "#ea580c",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _canonicalize(raw: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in raw.items():
        canon = METRIC_ALIASES.get(k) or METRIC_ALIASES.get(k.lstrip("_"))
        if canon and isinstance(v, (int, float)):
            out[canon] = float(v)
    return out


def load_log(path: Path) -> dict[str, float]:
    pattern = re.compile(r"^(_?[A-Za-z0-9_.^]+)\s*:\s*([-+0-9.eE]+)\s*$")
    raw: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        key = m.group(1)
        # skip depth-binned lines like _10_abs_rel_diff
        if re.match(r"^_?\d+_", key):
            continue
        canon = METRIC_ALIASES.get(key) or METRIC_ALIASES.get(key.lstrip("_"))
        if canon:
            raw[canon] = float(m.group(2))
    return raw


def load_json(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    # metrics_summary.json from eval_kitti_scape_spike.py
    if "metrics_frame_mean" in payload:
        return _canonicalize(payload["metrics_frame_mean"])
    # eval_metrics_recomputed*.json
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return _canonicalize(payload["metrics"])
    # analysis/summary.json
    if "overall" in payload:
        return _canonicalize(payload["overall"])
    # plain flat dict
    return _canonicalize(payload)


def discover_runs(eval_root: Path, include: list[str] | None) -> list[tuple[str, dict[str, float]]]:
    """Walk eval_root and collect (label, metrics) for each run."""
    results: list[tuple[str, dict[str, float]]] = []
    seen_labels: set[str] = set()

    for run_dir in sorted(eval_root.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if include and not any(pat in name for pat in include):
            continue

        metrics: dict[str, float] = {}

        # Priority 1: metrics_summary.json
        p = run_dir / "metrics_summary.json"
        if p.exists():
            metrics = load_json(p)

        # Priority 2: eval_metrics_recomputed*.json
        if not metrics:
            for p in sorted(run_dir.glob("eval_metrics_recomputed*.json")):
                metrics = load_json(p)
                if metrics:
                    break

        # Priority 3: analysis/summary.json
        if not metrics:
            p = run_dir / "analysis" / "summary.json"
            if p.exists():
                metrics = load_json(p)

        # Priority 4: eval_metrics*.log
        if not metrics:
            for p in sorted(run_dir.glob("eval_metrics*.log")):
                metrics = load_log(p)
                if metrics:
                    break

        if not metrics:
            continue

        label = name
        if label in seen_labels:
            label = f"{name}_dup"
        seen_labels.add(label)
        results.append((label, metrics))

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _shorten_label(label: str, max_len: int = 28) -> str:
    """Shorten long run names for axis labels."""
    # strip common prefixes
    for prefix in ("test_mambassm_", "test_", "val_", "kitti_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    if len(label) > max_len:
        label = label[:max_len - 2] + ".."
    return label


def plot_metric_grid(
    runs: list[tuple[str, dict[str, float]]],
    metrics: list[str],
    title: str,
    out_path: Path,
) -> None:
    labels = [_shorten_label(r[0]) for r in runs]
    n_metrics = len(metrics)
    ncols = min(3, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).flatten()

    x = np.arange(len(labels))

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        values = [r[1].get(metric, np.nan) for r in runs]
        colors = [PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(len(labels))]
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if metric.startswith("delta_"):
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel("accuracy (higher=better)", fontsize=8)
        else:
            ax.set_ylabel("error (lower=better)", fontsize=8)

        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + ax.get_ylim()[1] * 0.01,
                    f"{value:.4f}",
                    ha="center", va="bottom", fontsize=6.5,
                )

    # hide unused axes
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_radar(
    runs: list[tuple[str, dict[str, float]]],
    out_path: Path,
    title: str = "Metric Radar",
) -> None:
    """Radar chart for a normalized view of key metrics."""
    radar_metrics = ["abs_rel", "rms_linear", "rms_log", "mean_err", "delta_1.25"]
    labels_radar = radar_metrics

    # collect finite values per metric for normalization
    all_vals: dict[str, list[float]] = {m: [] for m in radar_metrics}
    for _, mdict in runs:
        for m in radar_metrics:
            v = mdict.get(m, np.nan)
            if np.isfinite(v):
                all_vals[m].append(v)

    def normalize(metric: str, value: float) -> float:
        vals = all_vals[metric]
        if not vals:
            return 0.5
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return 0.5
        norm = (value - lo) / (hi - lo)
        # for error metrics: invert so that better = larger on radar
        if metric in LOWER_IS_BETTER:
            norm = 1.0 - norm
        return float(np.clip(norm, 0.0, 1.0))

    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.grid(alpha=0.3)

    for i, (label, mdict) in enumerate(runs):
        values_norm = [normalize(m, mdict.get(m, np.nan)) for m in radar_metrics]
        values_norm += values_norm[:1]
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        ax.plot(angles, values_norm, color=color, linewidth=1.8, label=_shorten_label(label))
        ax.fill(angles, values_norm, color=color, alpha=0.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_delta_thresholds(
    runs: list[tuple[str, dict[str, float]]],
    out_path: Path,
    title: str = "Threshold Accuracy (δ)",
) -> None:
    """Grouped bar chart for δ1.25 / δ1.25² / δ1.25³."""
    threshold_metrics = ["delta_1.25", "delta_1.25^2", "delta_1.25^3"]
    labels = [_shorten_label(r[0]) for r in runs]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5), constrained_layout=True)
    offsets = [-width, 0, width]
    bar_colors = ["#2563eb", "#0f766e", "#f59e0b"]

    for i, (metric, offset, color) in enumerate(zip(threshold_metrics, offsets, bar_colors)):
        values = [r[1].get(metric, np.nan) for r in runs]
        bars = ax.bar(x + offset, values, width, label=metric, color=color, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    v + 0.005,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy (higher = better)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def save_comparison_csv(runs: list[tuple[str, dict[str, float]]], out_path: Path) -> None:
    fieldnames = ["run"] + list(CANONICAL_METRICS)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, mdict in runs:
            row: dict[str, object] = {"run": label}
            for m in CANONICAL_METRICS:
                row[m] = f"{mdict[m]:.6f}" if m in mdict and np.isfinite(mdict[m]) else ""
            writer.writerow(row)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/root/shared-nvme/spikemamba/runs/eval"),
        help="Root directory containing per-run eval subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/shared-nvme/spikemamba/runs/eval/comparison_plots"),
        help="Output directory for plots and CSV.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Only include runs whose directory name contains one of these substrings.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="SpikeMamba Depth Evaluation Comparison",
        help="Title for the plots.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="abs_rel,rms_linear,rms_log,silog,mean_err,median_diff,delta_1.25,delta_1.25^2,delta_1.25^3",
        help="Comma-separated metrics to include in the grid plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {args.eval_root}")
    runs = discover_runs(args.eval_root, args.include)
    if not runs:
        raise SystemExit("No runs with parseable metrics found.")

    print(f"Found {len(runs)} runs:")
    for label, mdict in runs:
        abs_rel = mdict.get("abs_rel", float("nan"))
        delta = mdict.get("delta_1.25", float("nan"))
        print(f"  {label:<60}  abs_rel={abs_rel:.4f}  delta_1.25={delta:.4f}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print("\nGenerating plots...")
    plot_metric_grid(runs, metrics, args.title, args.output_dir / "comparison_grid.png")
    plot_radar(runs, args.output_dir / "comparison_radar.png", title=args.title + " (Radar)")
    plot_delta_thresholds(runs, args.output_dir / "comparison_delta_thresholds.png", title=args.title + " — δ Thresholds")
    save_comparison_csv(runs, args.output_dir / "comparison_metrics.csv")

    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
