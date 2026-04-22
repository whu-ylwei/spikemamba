#!/usr/bin/env python3
"""Render saved depth predictions and GT arrays into eval_vis comparison images.

This mirrors the visual style used by evaluation_DENSE.py:
- two vertically stacked panels
- top: Target
- bottom: Prediction
- shared depth colormap range derived from the target frame
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True, help="Directory with GT depth npy files.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory with predicted depth npy files.")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Optional directory with valid-mask npy files. If omitted, all pixels are treated as valid.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write comparison PNGs.")
    parser.add_argument("--target-glob", type=str, default="depth_*.npy")
    parser.add_argument("--pred-glob", type=str, default="depth_*.npy")
    parser.add_argument("--mask-prefix", type=str, default="mask_", help="Mask filename prefix.")
    parser.add_argument("--mask-suffix", type=str, default=".npy", help="Mask filename suffix.")
    parser.add_argument("--colormap", type=str, default="tab20c")
    parser.add_argument("--max-frames", type=int, default=None, help="Only render the first N matching frames.")
    return parser.parse_args()


def sorted_files(folder: Path, pattern: str) -> list[Path]:
    return sorted(folder.glob(pattern))


def extract_frame_token(path: Path) -> str:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Unable to extract frame index from {path}")
    return match.group(1)


def load_mask(mask_dir: Path | None, frame_token: str, shape: tuple[int, int], prefix: str, suffix: str) -> np.ndarray:
    if mask_dir is None:
        return np.ones(shape, dtype=bool)
    mask_path = mask_dir / f"{prefix}{frame_token}{suffix}"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask for frame {frame_token}: {mask_path}")
    mask = np.load(mask_path)
    return mask.astype(bool, copy=False)


def second_largest_or_max(data: np.ndarray) -> float:
    unique = np.unique(data)
    if unique.size == 0:
        return 1.0
    if unique.size == 1:
        return float(unique[0])
    return float(unique[-2])


def render_frame(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray, out_path: Path, colormap: str) -> None:
    valid_target = target[mask]
    if valid_target.size == 0:
        raise ValueError(f"No valid pixels for {out_path.name}")

    fill_value = float(np.max(valid_target))
    display_target = np.where(mask, target, fill_value)
    display_prediction = np.where(mask, prediction, fill_value)

    vmax = second_largest_or_max(display_target)
    vmin = float(np.min(valid_target))
    if vmax <= vmin:
        vmax = float(np.max(valid_target))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    fig, ax = plt.subplots(ncols=1, nrows=2)

    target_plot = np.flip(np.fliplr(display_target))
    pcm = ax[0].pcolormesh(target_plot, cmap=colormap, vmin=vmin, vmax=vmax)
    ax[0].set_xticklabels([])
    ax[0].set_title("Target")
    fig.colorbar(pcm, ax=ax[0], extend="both", orientation="vertical")

    prediction_plot = np.flip(np.fliplr(display_prediction))
    pcm = ax[1].pcolormesh(prediction_plot, cmap=colormap, vmin=vmin, vmax=vmax)
    ax[1].set_title("Prediction")
    fig.colorbar(pcm, ax=ax[1], extend="both", orientation="vertical")

    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted_files(args.pred_dir, args.pred_glob)
    target_files = sorted_files(args.target_dir, args.target_glob)
    if not pred_files:
        raise SystemExit(f"No prediction files found in {args.pred_dir}")
    if not target_files:
        raise SystemExit(f"No target files found in {args.target_dir}")

    target_by_token = {extract_frame_token(path): path for path in target_files}
    matched: list[tuple[str, Path, Path]] = []
    for pred_path in pred_files:
        token = extract_frame_token(pred_path)
        target_path = target_by_token.get(token)
        if target_path is None:
            continue
        matched.append((token, pred_path, target_path))

    if not matched:
        raise SystemExit("No matching prediction/target frames found.")
    if args.max_frames is not None:
        matched = matched[: args.max_frames]

    for index, (token, pred_path, target_path) in enumerate(matched, start=1):
        prediction = np.load(pred_path)
        target = np.load(target_path)
        if prediction.ndim == 3:
            prediction = prediction[0]
        if target.ndim == 3:
            target = target[0]
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch for frame {token}: {prediction.shape} vs {target.shape}")

        mask = load_mask(args.mask_dir, token, target.shape, args.mask_prefix, args.mask_suffix)
        if mask.shape != target.shape:
            raise ValueError(f"Mask shape mismatch for frame {token}: {mask.shape} vs {target.shape}")

        out_path = args.output_dir / f"frame_{token}.png"
        render_frame(target, prediction, mask, out_path, args.colormap)

        if index == 1 or index % 20 == 0 or index == len(matched):
            print(f"[render] {index}/{len(matched)} -> {out_path}")


if __name__ == "__main__":
    main()
