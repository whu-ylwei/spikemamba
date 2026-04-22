#!/usr/bin/env python3
"""Evaluate sparse depth predictions stored as `.npy` files.

This script is intended for KITTI-style sparse supervision where only a subset of
pixels are valid in each depth map. It supports:

- sparse GT carried by `NaN` values inside the target depth maps
- optional external valid-mask `.npy` files
- metric-depth arrays as well as the repository's normalized log-depth arrays
- center-crop alignment when predictions are smaller than the sparse GT map
- per-frame CSV export, summary JSON export, trend plots, and sample comparison plots
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


CANONICAL_METRICS = (
    "abs_rel",
    "squ_rel",
    "rms_linear",
    "rms_log",
    "silog",
    "mean_err",
    "median_diff",
    "delta_1.25",
    "delta_1.25^2",
    "delta_1.25^3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory with prediction `.npy` files.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory with GT `.npy` files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV/JSON/PNG outputs.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Optional directory with valid-mask `.npy` files.",
    )
    parser.add_argument("--pred-glob", type=str, default="*.npy")
    parser.add_argument("--gt-glob", type=str, default="*.npy")
    parser.add_argument("--mask-glob", type=str, default="*.npy")
    parser.add_argument(
        "--prediction-space",
        choices=("auto", "metric", "normalized_log"),
        default="auto",
        help="Depth representation used by prediction files.",
    )
    parser.add_argument(
        "--target-space",
        choices=("auto", "metric", "normalized_log"),
        default="auto",
        help="Depth representation used by GT files.",
    )
    parser.add_argument(
        "--align-mode",
        choices=("center_to_prediction", "center_to_smaller", "error"),
        default="center_to_prediction",
        help="How to align prediction and GT arrays when their shapes differ.",
    )
    parser.add_argument(
        "--clip-distance",
        type=float,
        default=1000.0,
        help="Metric clip distance used by normalized-log depth conversion.",
    )
    parser.add_argument(
        "--reg-factor",
        type=float,
        default=5.7,
        help="Regularization factor used by normalized-log depth conversion.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.0,
        help="Ignore GT values smaller than or equal to this threshold.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Optional GT upper bound. Pixels deeper than this are ignored.",
    )
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=50,
        help="Frames with fewer valid pixels after masking are skipped.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only evaluate the first N matched frames.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sparse_depth_eval",
        help="Stored in the summary JSON for easier tracking.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=6,
        help="How many evenly-spaced frames to render from best to worst.",
    )
    parser.add_argument(
        "--sample-sort-metric",
        type=str,
        default="abs_rel",
        choices=CANONICAL_METRICS,
        help="Metric used to rank sample frames.",
    )
    return parser.parse_args()


def sorted_files(folder: Path, pattern: str) -> list[Path]:
    return sorted(folder.glob(pattern))


def extract_frame_token(path: Path) -> str:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Unable to extract frame token from {path}")
    return match.group(1)


def squeeze_depth(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D depth map or a singleton 3D depth map, got shape {arr.shape}")
    return arr


def infer_space(arr: np.ndarray, requested: str) -> str:
    if requested != "auto":
        return requested

    finite = np.asarray(arr[np.isfinite(arr)], dtype=np.float32)
    if finite.size == 0:
        return "metric"

    p99 = float(np.percentile(finite, 99.0))
    p01 = float(np.percentile(finite, 1.0))
    if p99 <= 1.5 and p01 >= -0.25:
        return "normalized_log"
    return "metric"


def normalized_log_to_metric(arr: np.ndarray, clip_distance: float, reg_factor: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out

    converted = np.exp(reg_factor * (arr[finite] - 1.0))
    converted *= clip_distance
    converted = np.clip(converted, np.exp(-reg_factor) * clip_distance, clip_distance)
    out[finite] = converted
    return out


def to_metric_depth(arr: np.ndarray, space: str, clip_distance: float, reg_factor: float) -> np.ndarray:
    if space == "metric":
        return np.asarray(arr, dtype=np.float32)
    if space == "normalized_log":
        return normalized_log_to_metric(arr, clip_distance, reg_factor)
    raise ValueError(f"Unsupported depth space: {space}")


def center_crop(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = arr.shape
    target_h, target_w = target_shape
    if target_h > h or target_w > w:
        raise ValueError(f"Cannot center crop {arr.shape} to larger shape {target_shape}")
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return arr[top : top + target_h, left : left + target_w]


def align_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    align_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if prediction.shape == target.shape:
        return prediction, target, mask

    if align_mode == "error":
        raise ValueError(f"Shape mismatch: pred={prediction.shape}, target={target.shape}")

    if align_mode == "center_to_prediction":
        if target.shape[0] < prediction.shape[0] or target.shape[1] < prediction.shape[1]:
            raise ValueError(
                "center_to_prediction requires GT to be at least as large as the prediction. "
                f"Got pred={prediction.shape}, target={target.shape}"
            )
        target = center_crop(target, prediction.shape)
        mask = center_crop(mask.astype(np.uint8), prediction.shape).astype(bool)
        return prediction, target, mask

    if align_mode == "center_to_smaller":
        target_shape = (min(prediction.shape[0], target.shape[0]), min(prediction.shape[1], target.shape[1]))
        prediction = center_crop(prediction, target_shape)
        target = center_crop(target, target_shape)
        mask = center_crop(mask.astype(np.uint8), target_shape).astype(bool)
        return prediction, target, mask

    raise ValueError(f"Unsupported align mode: {align_mode}")


def build_valid_mask(
    target_metric: np.ndarray,
    external_mask: np.ndarray | None,
    min_depth: float,
    max_depth: float | None,
) -> np.ndarray:
    mask = np.isfinite(target_metric) & (target_metric > min_depth)
    if max_depth is not None:
        mask &= target_metric <= max_depth
    if external_mask is not None:
        mask &= external_mask.astype(bool, copy=False)
    return mask


def metric_block(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> OrderedDict[str, float]:
    eps = 1e-5
    t = target[mask]
    p = prediction[mask]
    if t.size == 0:
        return OrderedDict((key, math.nan) for key in CANONICAL_METRICS)

    ratio = np.maximum(t / (p + eps), p / (t + eps))
    log_diff = np.log(t + eps) - np.log(p + eps)
    abs_diff = np.abs(t - p)

    return OrderedDict(
        [
            ("abs_rel", float(np.mean(abs_diff / (t + eps)))),
            ("squ_rel", float(np.mean((abs_diff ** 2) / (t ** 2 + eps)))),
            ("rms_linear", float(np.sqrt(np.mean(abs_diff ** 2)))),
            ("rms_log", float(np.sqrt(np.mean(log_diff ** 2)))),
            ("silog", float(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2)),
            ("mean_err", float(np.mean(abs_diff))),
            ("median_diff", float(np.median(abs_diff))),
            ("delta_1.25", float(np.mean(ratio <= 1.25))),
            ("delta_1.25^2", float(np.mean(ratio <= 1.25 ** 2))),
            ("delta_1.25^3", float(np.mean(ratio <= 1.25 ** 3))),
        ]
    )


def average_metrics(rows: list[dict[str, object]]) -> OrderedDict[str, float]:
    summary = OrderedDict()
    for key in CANONICAL_METRICS:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[key] = float(np.nanmean(values))
    return summary


def save_frame_metrics_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_trends(rows: list[dict[str, object]], out_path: Path) -> None:
    x = np.arange(len(rows), dtype=np.int32)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(x, [int(row["valid_pixels"]) for row in rows], color="#0f766e", linewidth=2)
    axes[0, 0].set_title("Valid Pixels")
    axes[0, 0].set_xlabel("Evaluated Frame")
    axes[0, 0].set_ylabel("pixels")

    axes[0, 1].plot(x, [float(row["abs_rel"]) for row in rows], color="#dc2626", linewidth=2)
    axes[0, 1].set_title("Abs Rel")
    axes[0, 1].set_xlabel("Evaluated Frame")
    axes[0, 1].set_ylabel("error")

    axes[1, 0].plot(x, [float(row["rms_linear"]) for row in rows], color="#7c3aed", linewidth=2)
    axes[1, 0].set_title("RMS Linear")
    axes[1, 0].set_xlabel("Evaluated Frame")
    axes[1, 0].set_ylabel("meters")

    axes[1, 1].plot(x, [float(row["delta_1.25"]) for row in rows], color="#2563eb", linewidth=2)
    axes[1, 1].set_title("Delta 1.25")
    axes[1, 1].set_xlabel("Evaluated Frame")
    axes[1, 1].set_ylabel("ratio")
    axes[1, 1].set_ylim(0.0, 1.0)

    fig.suptitle("Sparse Depth Evaluation Trends", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_summary_metrics(summary: OrderedDict[str, float], out_path: Path) -> None:
    subset = OrderedDict(
        [
            ("abs_rel", summary["abs_rel"]),
            ("rms_linear", summary["rms_linear"]),
            ("rms_log", summary["rms_log"]),
            ("mean_err", summary["mean_err"]),
            ("delta_1.25", summary["delta_1.25"]),
        ]
    )
    labels = list(subset.keys())
    values = list(subset.values())

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#dc2626", "#7c3aed", "#f59e0b", "#0f766e", "#2563eb"])
    ax.set_title("Sparse Eval Summary")
    ax.set_ylabel("value")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_frame_triplet(
    row: dict[str, object],
    clip_distance: float,
    reg_factor: float,
    align_mode: str,
    min_depth: float,
    max_depth: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = squeeze_depth(np.load(Path(str(row["pred_path"]))))
    target = squeeze_depth(np.load(Path(str(row["gt_path"]))))

    prediction_space = str(row["prediction_space"])
    target_space = str(row["target_space"])
    pred_metric = to_metric_depth(pred, prediction_space, clip_distance, reg_factor)
    target_metric = to_metric_depth(target, target_space, clip_distance, reg_factor)

    mask_path_str = str(row["mask_path"])
    external_mask = None if not mask_path_str else squeeze_depth(np.load(Path(mask_path_str))).astype(bool)
    mask = build_valid_mask(target_metric, external_mask, min_depth, max_depth)
    pred_metric, target_metric, mask = align_arrays(pred_metric, target_metric, mask, align_mode)
    return pred_metric, target_metric, mask


def render_sample_grid(
    rows: list[dict[str, object]],
    out_path: Path,
    clip_distance: float,
    reg_factor: float,
    align_mode: str,
    min_depth: float,
    max_depth: float | None,
    sort_metric: str,
) -> None:
    if not rows:
        return

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3.6 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.asarray([axes])

    for row_axes, row in zip(axes, rows):
        prediction, target, mask = load_frame_triplet(
            row,
            clip_distance=clip_distance,
            reg_factor=reg_factor,
            align_mode=align_mode,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        valid_target = target[mask]
        vmin = float(np.min(valid_target))
        vmax = float(np.max(valid_target))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        masked_target = np.where(mask, target, np.nan)
        masked_prediction = np.where(mask, prediction, np.nan)
        abs_error = np.where(mask, np.abs(target - prediction), np.nan)

        im0 = row_axes[0].imshow(masked_target, cmap="viridis", vmin=vmin, vmax=vmax)
        row_axes[0].set_title(f"GT {row['frame_token']}")
        fig.colorbar(im0, ax=row_axes[0], fraction=0.046, pad=0.04)

        im1 = row_axes[1].imshow(masked_prediction, cmap="viridis", vmin=vmin, vmax=vmax)
        row_axes[1].set_title(f"Pred {row['frame_token']}")
        fig.colorbar(im1, ax=row_axes[1], fraction=0.046, pad=0.04)

        err_vmax = float(np.nanpercentile(abs_error, 95.0))
        if not np.isfinite(err_vmax) or err_vmax <= 0.0:
            err_vmax = 1e-6
        im2 = row_axes[2].imshow(abs_error, cmap="magma", vmin=0.0, vmax=err_vmax)
        row_axes[2].set_title(f"|Err| {sort_metric}={float(row[sort_metric]):.4f}")
        fig.colorbar(im2, ax=row_axes[2], fraction=0.046, pad=0.04)

        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Sparse Depth Samples", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def select_sample_rows(rows: list[dict[str, object]], count: int, metric_name: str) -> list[dict[str, object]]:
    if not rows:
        return []
    ranked = sorted(rows, key=lambda row: float(row[metric_name]))
    if len(ranked) <= count:
        return ranked

    indices = np.linspace(0, len(ranked) - 1, count, dtype=int)
    unique_indices = []
    seen = set()
    for idx in indices.tolist():
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)
    return [ranked[idx] for idx in unique_indices]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted_files(args.pred_dir, args.pred_glob)
    gt_files = sorted_files(args.gt_dir, args.gt_glob)
    if not pred_files:
        raise SystemExit(f"No prediction files found in {args.pred_dir}")
    if not gt_files:
        raise SystemExit(f"No GT files found in {args.gt_dir}")

    gt_by_token = {extract_frame_token(path): path for path in gt_files}
    mask_by_token: dict[str, Path] = {}
    if args.mask_dir is not None:
        mask_by_token = {extract_frame_token(path): path for path in sorted_files(args.mask_dir, args.mask_glob)}

    matched: list[tuple[str, Path, Path, Path | None]] = []
    for pred_path in pred_files:
        token = extract_frame_token(pred_path)
        gt_path = gt_by_token.get(token)
        if gt_path is None:
            continue
        matched.append((token, pred_path, gt_path, mask_by_token.get(token)))

    if not matched:
        raise SystemExit("No matching prediction/GT frame tokens were found.")
    if args.max_frames is not None:
        matched = matched[: args.max_frames]

    frame_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for eval_index, (token, pred_path, gt_path, mask_path) in enumerate(matched):
        pred_raw = squeeze_depth(np.load(pred_path))
        gt_raw = squeeze_depth(np.load(gt_path))

        prediction_space = infer_space(pred_raw, args.prediction_space)
        target_space = infer_space(gt_raw, args.target_space)
        pred_metric = to_metric_depth(pred_raw, prediction_space, args.clip_distance, args.reg_factor)
        gt_metric = to_metric_depth(gt_raw, target_space, args.clip_distance, args.reg_factor)

        external_mask = None
        if mask_path is not None:
            external_mask = squeeze_depth(np.load(mask_path)).astype(bool)
        valid_mask = build_valid_mask(gt_metric, external_mask, args.min_depth, args.max_depth)
        pred_metric, gt_metric, valid_mask = align_arrays(pred_metric, gt_metric, valid_mask, args.align_mode)

        valid_pixels = int(np.count_nonzero(valid_mask))
        valid_fraction = float(valid_pixels / valid_mask.size) if valid_mask.size else 0.0
        if valid_pixels < args.min_valid_pixels:
            skipped_rows.append(
                {
                    "frame_token": token,
                    "pred_path": str(pred_path),
                    "gt_path": str(gt_path),
                    "reason": "too_few_valid_pixels",
                    "valid_pixels": valid_pixels,
                }
            )
            continue

        metrics = metric_block(gt_metric, pred_metric, valid_mask)
        row = OrderedDict(
            [
                ("eval_index", eval_index),
                ("frame_token", token),
                ("pred_path", str(pred_path)),
                ("gt_path", str(gt_path)),
                ("mask_path", "" if mask_path is None else str(mask_path)),
                ("prediction_space", prediction_space),
                ("target_space", target_space),
                ("height", int(pred_metric.shape[0])),
                ("width", int(pred_metric.shape[1])),
                ("valid_pixels", valid_pixels),
                ("valid_fraction", valid_fraction),
            ]
        )
        for key, value in metrics.items():
            row[key] = value
        frame_rows.append(row)

    if not frame_rows:
        raise SystemExit("All matched frames were skipped. Check mask coverage and depth thresholds.")

    summary_metrics = average_metrics(frame_rows)
    summary = OrderedDict(
        [
            ("dataset_name", args.dataset_name),
            ("prediction_dir", str(args.pred_dir)),
            ("gt_dir", str(args.gt_dir)),
            ("mask_dir", None if args.mask_dir is None else str(args.mask_dir)),
            ("prediction_space_requested", args.prediction_space),
            ("target_space_requested", args.target_space),
            ("align_mode", args.align_mode),
            ("clip_distance", args.clip_distance),
            ("reg_factor", args.reg_factor),
            ("min_depth", args.min_depth),
            ("max_depth", args.max_depth),
            ("matched_frames", len(matched)),
            ("evaluated_frames", len(frame_rows)),
            ("skipped_frames", len(skipped_rows)),
            ("mean_valid_pixels", float(np.mean([int(row["valid_pixels"]) for row in frame_rows])),
            ),
            ("mean_valid_fraction", float(np.mean([float(row["valid_fraction"]) for row in frame_rows])),
            ),
            ("metrics_frame_mean", summary_metrics),
            (
                "best_frame_by_abs_rel",
                {
                    "frame_token": min(frame_rows, key=lambda row: float(row["abs_rel"]))["frame_token"],
                    "abs_rel": float(min(frame_rows, key=lambda row: float(row["abs_rel"]))["abs_rel"]),
                },
            ),
            (
                "worst_frame_by_abs_rel",
                {
                    "frame_token": max(frame_rows, key=lambda row: float(row["abs_rel"]))["frame_token"],
                    "abs_rel": float(max(frame_rows, key=lambda row: float(row["abs_rel"]))["abs_rel"]),
                },
            ),
        ]
    )

    save_frame_metrics_csv(frame_rows, args.output_dir / "frame_metrics.csv")
    with (args.output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "skipped_frames.json").open("w", encoding="utf-8") as f:
        json.dump(skipped_rows, f, indent=2)

    plot_metric_trends(frame_rows, plots_dir / "frame_metric_trends.png")
    plot_summary_metrics(summary_metrics, plots_dir / "summary_metrics.png")
    sample_rows = select_sample_rows(frame_rows, args.sample_count, args.sample_sort_metric)
    render_sample_grid(
        sample_rows,
        plots_dir / "sample_grid.png",
        clip_distance=args.clip_distance,
        reg_factor=args.reg_factor,
        align_mode=args.align_mode,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        sort_metric=args.sample_sort_metric,
    )

    print(f"[done] evaluated={len(frame_rows)} skipped={len(skipped_rows)} output={args.output_dir}")
    print(json.dumps(summary_metrics, indent=2))


if __name__ == "__main__":
    main()
