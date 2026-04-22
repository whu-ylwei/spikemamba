#!/usr/bin/env python3
"""Evaluate depth predictions against KITTI sparse ground-truth depth maps.

Designed for the KITTI-style evaluation where GT depth is sparse (LiDAR projection).
Supports:
  - Sparse GT carried by zero/NaN values (auto-detected)
  - Depth-range binned metrics (0-10m, 10-20m, 20-30m, 30-80m, 80-250m)
  - Per-frame CSV + summary JSON + trend plots + sample comparison grid
  - Optional scale alignment (median scaling) for scale-ambiguous predictions

Typical usage:
    python eval_kitti_sparse.py \
        --pred-dir runs/eval/kitti_0001_image02_scape_spike_ep80_best_20260412/pred_metric \
        --gt-dir   runs/eval/kitti_0001_image02_scape_spike_ep80_best_20260412/gt_metric \
        --output-dir runs/eval/kitti_0001_sparse_eval \
        --dataset-name kitti_0001_scape_spike_ep80

    # With scale alignment:
    python eval_kitti_sparse.py ... --scale-align median

    # Limit depth range (KITTI standard: 1-80m):
    python eval_kitti_sparse.py ... --min-depth 1.0 --max-depth 80.0
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_METRICS = (
    "abs_rel", "squ_rel", "rms_linear", "rms_log", "silog",
    "mean_err", "median_diff", "delta_1.25", "delta_1.25^2", "delta_1.25^3",
)

# Depth bins for range-specific analysis (upper bound in metres, label)
DEPTH_BINS: list[tuple[float, float, str]] = [
    (0.0,   10.0,  "0-10m"),
    (10.0,  20.0,  "10-20m"),
    (20.0,  30.0,  "20-30m"),
    (30.0,  80.0,  "30-80m"),
    (80.0,  250.0, "80-250m"),
]

PLOT_COLORS = ["#0f766e", "#2563eb", "#dc2626", "#7c3aed", "#f59e0b"]


# ---------------------------------------------------------------------------
# Depth utilities
# ---------------------------------------------------------------------------

def squeeze_depth(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {arr.shape}")
    return arr


def infer_space(arr: np.ndarray) -> str:
    """Guess whether array is metric depth or normalized-log depth."""
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        return "metric"
    p99 = float(np.percentile(finite, 99.0))
    p01 = float(np.percentile(finite, 1.0))
    if p99 <= 1.5 and p01 >= -0.25:
        return "normalized_log"
    return "metric"


def normalized_log_to_metric(arr: np.ndarray, clip_distance: float = 1000.0, reg_factor: float = 5.7) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    if np.any(finite):
        converted = np.exp(reg_factor * (arr[finite] - 1.0)) * clip_distance
        converted = np.clip(converted, np.exp(-reg_factor) * clip_distance, clip_distance)
        out[finite] = converted
    return out


def to_metric(arr: np.ndarray, clip_distance: float, reg_factor: float) -> np.ndarray:
    space = infer_space(arr)
    if space == "normalized_log":
        return normalized_log_to_metric(arr, clip_distance, reg_factor)
    return np.asarray(arr, dtype=np.float32)


def build_sparse_mask(gt: np.ndarray, min_depth: float, max_depth: float | None) -> np.ndarray:
    """Valid mask: finite, positive, within depth range."""
    mask = np.isfinite(gt) & (gt > min_depth)
    if max_depth is not None:
        mask &= gt <= max_depth
    return mask


def center_crop_to(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return arr[top:top + target_h, left:left + target_w]


def align_shapes(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop the larger array to match the smaller one."""
    if pred.shape == gt.shape:
        return pred, gt
    target_h = min(pred.shape[0], gt.shape[0])
    target_w = min(pred.shape[1], gt.shape[1])
    if pred.shape != (target_h, target_w):
        pred = center_crop_to(pred, target_h, target_w)
    if gt.shape != (target_h, target_w):
        gt = center_crop_to(gt, target_h, target_w)
    return pred, gt


def scale_align(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """Optionally rescale prediction to match GT scale."""
    if mode == "none":
        return pred
    t = gt[mask]
    p = pred[mask]
    if p.size == 0 or np.median(p) == 0:
        return pred
    if mode == "median":
        scale = np.median(t) / np.median(p)
    elif mode == "mean":
        scale = np.mean(t) / np.mean(p)
    else:
        raise ValueError(f"Unknown scale-align mode: {mode}")
    return pred * float(scale)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> OrderedDict[str, float]:
    eps = 1e-5
    t = gt[mask].astype(np.float64)
    p = pred[mask].astype(np.float64)
    if t.size == 0:
        return OrderedDict((k, math.nan) for k in CANONICAL_METRICS)

    abs_diff = np.abs(t - p)
    ratio = np.maximum(t / (p + eps), p / (t + eps))
    log_diff = np.log(t + eps) - np.log(p + eps)

    return OrderedDict([
        ("abs_rel",    float(np.mean(abs_diff / (t + eps)))),
        ("squ_rel",    float(np.mean(abs_diff ** 2 / (t ** 2 + eps)))),
        ("rms_linear", float(np.sqrt(np.mean(abs_diff ** 2)))),
        ("rms_log",    float(np.sqrt(np.mean(log_diff ** 2)))),
        ("silog",      float(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2)),
        ("mean_err",   float(np.mean(abs_diff))),
        ("median_diff",float(np.median(abs_diff))),
        ("delta_1.25", float(np.mean(ratio <= 1.25))),
        ("delta_1.25^2", float(np.mean(ratio <= 1.25 ** 2))),
        ("delta_1.25^3", float(np.mean(ratio <= 1.25 ** 3))),
    ])


def average_metrics(rows: list[dict]) -> OrderedDict[str, float]:
    out = OrderedDict()
    for k in CANONICAL_METRICS:
        vals = [float(r[k]) for r in rows if k in r and math.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else math.nan
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def extract_token(path: Path) -> str:
    m = re.search(r"(\d+)$", path.stem)
    if m is None:
        raise ValueError(f"Cannot extract frame token from {path.name}")
    return m.group(1)


def sorted_npy(folder: Path, glob: str = "*.npy") -> list[Path]:
    return sorted(folder.glob(glob))


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_trends(rows: list[dict], out_path: Path) -> None:
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    def _plot(ax, key, color, ylabel, title, ylim=None):
        vals = [float(r.get(key, np.nan)) for r in rows]
        ax.plot(x, vals, color=color, linewidth=1.5)
        ax.fill_between(x, vals, alpha=0.15, color=color)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Frame index")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)

    _plot(axes[0, 0], "valid_pixels", "#0f766e", "pixels", "Valid Sparse Pixels")
    _plot(axes[0, 1], "abs_rel",      "#dc2626", "error",  "Abs Rel Error")
    _plot(axes[0, 2], "rms_linear",   "#7c3aed", "metres", "RMS Linear (m)")
    _plot(axes[1, 0], "rms_log",      "#f59e0b", "log",    "RMS Log")
    _plot(axes[1, 1], "silog",        "#059669", "silog",  "SILog")
    _plot(axes[1, 2], "delta_1.25",   "#2563eb", "ratio",  "δ1.25 Accuracy", ylim=(0, 1))

    fig.suptitle("Per-Frame Sparse Depth Metrics", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_depth_bin_comparison(bin_metrics: dict[str, OrderedDict[str, float]], out_path: Path) -> None:
    """Bar chart comparing abs_rel and delta_1.25 across depth bins."""
    bin_names = list(bin_metrics.keys())
    abs_rels = [bin_metrics[b].get("abs_rel", np.nan) for b in bin_names]
    deltas   = [bin_metrics[b].get("delta_1.25", np.nan) for b in bin_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(bin_names))

    bars1 = ax1.bar(x, abs_rels, color=PLOT_COLORS[:len(bin_names)], edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_names, rotation=20, ha="right")
    ax1.set_title("Abs Rel by Depth Bin", fontweight="bold")
    ax1.set_ylabel("abs_rel (lower=better)")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, v in zip(bars1, abs_rels):
        if np.isfinite(v):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(x, deltas, color=PLOT_COLORS[:len(bin_names)], edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_names, rotation=20, ha="right")
    ax2.set_title("δ1.25 Accuracy by Depth Bin", fontweight="bold")
    ax2.set_ylabel("δ1.25 (higher=better)")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar, v in zip(bars2, deltas):
        if np.isfinite(v):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Sparse Depth Evaluation by Depth Range", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(
    sample_rows: list[dict],
    out_path: Path,
    pred_dir: Path,
    gt_dir: Path,
    clip_distance: float,
    reg_factor: float,
    min_depth: float,
    max_depth: float | None,
    scale_align_mode: str,
) -> None:
    if not sample_rows:
        return
    n = len(sample_rows)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.8 * n), constrained_layout=True)
    if n == 1:
        axes = np.asarray([axes])

    for row_axes, row in zip(axes, sample_rows):
        token = str(row["frame_token"])
        pred_path = pred_dir / f"depth_{token}.npy"
        gt_path   = gt_dir   / f"depth_{token}.npy"
        if not pred_path.exists() or not gt_path.exists():
            # try glob fallback
            preds = list(pred_dir.glob(f"*{token}*.npy"))
            gts   = list(gt_dir.glob(f"*{token}*.npy"))
            if not preds or not gts:
                continue
            pred_path, gt_path = preds[0], gts[0]

        pred_raw = squeeze_depth(np.load(pred_path))
        gt_raw   = squeeze_depth(np.load(gt_path))
        pred_m   = to_metric(pred_raw, clip_distance, reg_factor)
        gt_m     = to_metric(gt_raw,   clip_distance, reg_factor)
        pred_m, gt_m = align_shapes(pred_m, gt_m)
        mask = build_sparse_mask(gt_m, min_depth, max_depth)
        pred_m = scale_align(pred_m, gt_m, mask, scale_align_mode)

        valid_gt = gt_m[mask]
        vmin = float(np.min(valid_gt)) if valid_gt.size else 0.0
        vmax = float(np.max(valid_gt)) if valid_gt.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

        masked_gt   = np.where(mask, gt_m,   np.nan)
        masked_pred = np.where(mask, pred_m, np.nan)
        abs_err     = np.where(mask, np.abs(gt_m - pred_m), np.nan)

        im0 = row_axes[0].imshow(masked_gt,   cmap="plasma", vmin=vmin, vmax=vmax)
        row_axes[0].set_title(f"GT sparse  [{token}]", fontsize=9)
        fig.colorbar(im0, ax=row_axes[0], fraction=0.046, pad=0.04)

        im1 = row_axes[1].imshow(masked_pred, cmap="plasma", vmin=vmin, vmax=vmax)
        row_axes[1].set_title(f"Pred (at GT pixels)", fontsize=9)
        fig.colorbar(im1, ax=row_axes[1], fraction=0.046, pad=0.04)

        err_vmax = float(np.nanpercentile(abs_err, 95)) if np.any(np.isfinite(abs_err)) else 1.0
        if not np.isfinite(err_vmax) or err_vmax <= 0:
            err_vmax = 1.0
        im2 = row_axes[2].imshow(abs_err, cmap="magma", vmin=0, vmax=err_vmax)
        row_axes[2].set_title(f"|Error|  abs_rel={float(row['abs_rel']):.4f}", fontsize=9)
        fig.colorbar(im2, ax=row_axes[2], fraction=0.046, pad=0.04)

        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Sparse Depth Sample Comparison", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir",    type=Path, required=True, help="Directory with prediction .npy files.")
    p.add_argument("--gt-dir",      type=Path, required=True, help="Directory with GT .npy files (sparse, zeros or NaN for invalid).")
    p.add_argument("--output-dir",  type=Path, required=True, help="Output directory.")
    p.add_argument("--dataset-name", type=str, default="kitti_sparse_eval")
    p.add_argument("--min-depth",   type=float, default=0.0,    help="Ignore GT pixels <= this depth (m).")
    p.add_argument("--max-depth",   type=float, default=None,   help="Ignore GT pixels > this depth (m). Default: no limit.")
    p.add_argument("--clip-distance", type=float, default=1000.0, help="Clip distance for normalized-log conversion.")
    p.add_argument("--reg-factor",  type=float, default=5.7,    help="Reg factor for normalized-log conversion.")
    p.add_argument("--scale-align", choices=("none", "median", "mean"), default="none",
                   help="Per-frame scale alignment before computing metrics.")
    p.add_argument("--min-valid-pixels", type=int, default=10,  help="Skip frames with fewer valid GT pixels.")
    p.add_argument("--max-frames",  type=int, default=None,     help="Evaluate only the first N matched frames.")
    p.add_argument("--sample-count", type=int, default=6,       help="Number of sample frames to render.")
    p.add_argument("--depth-bins",  action="store_true", default=True,
                   help="Compute per-depth-bin metrics (default: on).")
    p.add_argument("--no-depth-bins", dest="depth_bins", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    pred_files = sorted_npy(args.pred_dir)
    gt_files   = sorted_npy(args.gt_dir)
    if not pred_files:
        raise SystemExit(f"No .npy files in {args.pred_dir}")
    if not gt_files:
        raise SystemExit(f"No .npy files in {args.gt_dir}")

    gt_by_token = {extract_token(p): p for p in gt_files}

    matched: list[tuple[str, Path, Path]] = []
    for pred_path in pred_files:
        token = extract_token(pred_path)
        gt_path = gt_by_token.get(token)
        if gt_path is not None:
            matched.append((token, pred_path, gt_path))

    if not matched:
        raise SystemExit("No matching pred/GT frame tokens found.")
    if args.max_frames:
        matched = matched[:args.max_frames]

    print(f"Matched {len(matched)} frames. Evaluating...")

    frame_rows: list[dict] = []
    skipped: list[dict] = []

    # Accumulators for depth-bin metrics
    bin_pixel_lists: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {b[2]: [] for b in DEPTH_BINS}

    for token, pred_path, gt_path in matched:
        pred_raw = squeeze_depth(np.load(pred_path))
        gt_raw   = squeeze_depth(np.load(gt_path))

        pred_m = to_metric(pred_raw, args.clip_distance, args.reg_factor)
        gt_m   = to_metric(gt_raw,   args.clip_distance, args.reg_factor)
        pred_m, gt_m = align_shapes(pred_m, gt_m)

        mask = build_sparse_mask(gt_m, args.min_depth, args.max_depth)
        valid_pixels = int(np.count_nonzero(mask))

        if valid_pixels < args.min_valid_pixels:
            skipped.append({"frame_token": token, "valid_pixels": valid_pixels, "reason": "too_few_valid_pixels"})
            continue

        pred_aligned = scale_align(pred_m, gt_m, mask, args.scale_align)
        metrics = compute_metrics(gt_m, pred_aligned, mask)

        row = OrderedDict([
            ("frame_token",    token),
            ("valid_pixels",   valid_pixels),
            ("valid_fraction", float(valid_pixels / mask.size)),
            ("pred_path",      str(pred_path)),
            ("gt_path",        str(gt_path)),
        ])
        row.update(metrics)
        frame_rows.append(row)

        # Accumulate for depth bins
        if args.depth_bins:
            for lo, hi, bin_name in DEPTH_BINS:
                bin_mask = mask & (gt_m > lo) & (gt_m <= hi)
                if np.any(bin_mask):
                    bin_pixel_lists[bin_name].append((gt_m[bin_mask], pred_aligned[bin_mask]))

    if not frame_rows:
        raise SystemExit("All frames skipped. Check --min-depth / --max-depth / --min-valid-pixels.")

    # Summary
    summary_metrics = average_metrics(frame_rows)

    # Depth-bin metrics
    bin_metrics: dict[str, OrderedDict[str, float]] = {}
    if args.depth_bins:
        for lo, hi, bin_name in DEPTH_BINS:
            pairs = bin_pixel_lists[bin_name]
            if not pairs:
                bin_metrics[bin_name] = OrderedDict((k, math.nan) for k in CANONICAL_METRICS)
                continue
            gt_all   = np.concatenate([p[0] for p in pairs])
            pred_all = np.concatenate([p[1] for p in pairs])
            full_mask = np.ones(gt_all.shape, dtype=bool)
            bin_metrics[bin_name] = compute_metrics(gt_all, pred_all, full_mask)

    summary = OrderedDict([
        ("dataset_name",       args.dataset_name),
        ("pred_dir",           str(args.pred_dir)),
        ("gt_dir",             str(args.gt_dir)),
        ("scale_align",        args.scale_align),
        ("min_depth",          args.min_depth),
        ("max_depth",          args.max_depth),
        ("clip_distance",      args.clip_distance),
        ("reg_factor",         args.reg_factor),
        ("matched_frames",     len(matched)),
        ("evaluated_frames",   len(frame_rows)),
        ("skipped_frames",     len(skipped)),
        ("mean_valid_pixels",  float(np.mean([r["valid_pixels"] for r in frame_rows]))),
        ("mean_valid_fraction",float(np.mean([r["valid_fraction"] for r in frame_rows]))),
        ("metrics_frame_mean", summary_metrics),
    ])
    if bin_metrics:
        summary["depth_bin_metrics"] = {k: dict(v) for k, v in bin_metrics.items()}

    # Save outputs
    save_csv(frame_rows, args.output_dir / "frame_metrics.csv")
    with (args.output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "skipped_frames.json").open("w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2)

    # Plots
    plot_trends(frame_rows, plots_dir / "frame_metric_trends.png")

    if bin_metrics:
        plot_depth_bin_comparison(bin_metrics, plots_dir / "depth_bin_comparison.png")

    # Sample grid: pick evenly spaced from best to worst by abs_rel
    ranked = sorted(frame_rows, key=lambda r: float(r["abs_rel"]))
    if len(ranked) <= args.sample_count:
        sample_rows = ranked
    else:
        indices = np.linspace(0, len(ranked) - 1, args.sample_count, dtype=int)
        sample_rows = [ranked[i] for i in indices]

    plot_sample_grid(
        sample_rows,
        plots_dir / "sample_grid.png",
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        clip_distance=args.clip_distance,
        reg_factor=args.reg_factor,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        scale_align_mode=args.scale_align,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset : {args.dataset_name}")
    print(f"Frames  : evaluated={len(frame_rows)}  skipped={len(skipped)}")
    print(f"Scale   : {args.scale_align}")
    print(f"{'='*60}")
    print(f"{'Metric':<18} {'Value':>12}")
    print(f"{'-'*32}")
    for k, v in summary_metrics.items():
        print(f"  {k:<16} {v:>12.6f}")
    if bin_metrics:
        print(f"\n{'Depth Bin':<12} {'abs_rel':>10} {'rms_lin':>10} {'delta_1.25':>12}")
        print(f"{'-'*46}")
        for bin_name, bm in bin_metrics.items():
            print(f"  {bin_name:<10} {bm.get('abs_rel', float('nan')):>10.4f} "
                  f"{bm.get('rms_linear', float('nan')):>10.4f} "
                  f"{bm.get('delta_1.25', float('nan')):>12.4f}")
    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
