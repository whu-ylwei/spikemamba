#!/usr/bin/env python3
"""Simulate SCAPE-style spike tensors on KITTI and evaluate with the 80-epoch best model.

Notes:
- KITTI depth supervision in `proj_depth/groundtruth/image_02` is aligned with KITTI's left RGB camera
  `image_02`, not with the grayscale `image_00` camera.
- The SCAPE spike simulator is reused directly from `/root/shared-nvme/scape/simulate_dense_spike_camera.py`.
- To match the existing SpikeMamba evaluation pipeline, inference uses a center crop of 224 x 224.
  The simulator is pixel-wise independent, so simulating on the cropped RGB frames is equivalent to
  simulating at full resolution and then cropping the spike tensors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.S2DepthNet import S2DepthTransformerUNetConv
from model.metric import abs_rel_diff, mean_error, rms_linear, scale_invariant_error, squ_rel_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-seq",
        type=Path,
        default=Path("/root/shared-nvme/kitti/2011_09_26/2011_09_26_drive_0001_sync"),
        help="KITTI raw drive root.",
    )
    parser.add_argument(
        "--raw-camera",
        type=str,
        default="image_02",
        help="KITTI camera folder under the raw sequence. Use image_02 for left RGB.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("/root/shared-nvme/kitti/data_depth_annotated/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02"),
        help="KITTI annotated depth PNG directory for the aligned camera.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/root/shared-nvme/spikemamba/configs/train_s2d_MambaSSM_bidivim_ep40_continue40_halflr_20260320_bs2.json"),
        help="Model config used by the 80-epoch best checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/shared-nvme/spikemamba/runs/train/train_s2d_MambaSSM_bidivim_ep40_continue40_halflr_20260320_bs2/checkpoints/model_best.pth.tar"),
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--sim-script",
        type=Path,
        default=Path("/root/shared-nvme/scape/simulate_dense_spike_camera.py"),
        help="Path to the SCAPE spike simulator script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/shared-nvme/spikemamba/runs/eval/kitti_0001_image02_scape_spike_ep80_best_20260412"),
        help="Directory for simulated spikes, predictions, and metrics.",
    )
    parser.add_argument("--crop-size", type=int, default=224, help="Center crop size used for inference.")
    parser.add_argument("--num-bins", type=int, default=128, help="Number of spike bins.")
    parser.add_argument("--threshold", type=float, default=4.0, help="Integrate-and-fire threshold.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the initial accumulator state.")
    parser.add_argument(
        "--init-state",
        choices=("random", "zero", "half"),
        default="random",
        help="Initial accumulator state, matching the SCAPE simulator.",
    )
    parser.add_argument(
        "--save-spikes",
        action="store_true",
        help="Save cropped spike tensors used for evaluation.",
    )
    parser.add_argument(
        "--overwrite-spikes",
        action="store_true",
        help="Recompute and overwrite saved spike tensors.",
    )
    parser.add_argument(
        "--max-gt-frames",
        type=int,
        default=None,
        help="Only evaluate the first N GT frames. Useful for smoke tests.",
    )
    return parser.parse_args()


def load_simulator_module(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("scape_spike_sim", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load simulator module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"))


def center_crop_slices(height: int, width: int, crop_size: int) -> tuple[slice, slice]:
    if crop_size > height or crop_size > width:
        raise ValueError(f"Crop size {crop_size} exceeds input size {(height, width)}")
    top = int(round((height - crop_size) / 2.0))
    left = int(round((width - crop_size) / 2.0))
    return slice(top, top + crop_size), slice(left, left + crop_size)


def metric_depth_from_prediction(pred_log: np.ndarray, clip_distance: float, reg_factor: float) -> np.ndarray:
    pred = np.exp(reg_factor * (pred_log - np.ones_like(pred_log, dtype=np.float32)))
    pred *= clip_distance
    return np.clip(pred, np.exp(-reg_factor) * clip_distance, clip_distance)


def metric_block(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> OrderedDict[str, float]:
    eps = 1e-5
    t = target[mask]
    p = prediction[mask]
    if t.size == 0:
        return OrderedDict(
            [
                ("abs_rel", math.nan),
                ("squ_rel", math.nan),
                ("rms_linear", math.nan),
                ("rms_log", math.nan),
                ("silog", math.nan),
                ("mean_err", math.nan),
                ("median_diff", math.nan),
                ("delta_1.25", math.nan),
                ("delta_1.25^2", math.nan),
                ("delta_1.25^3", math.nan),
            ]
        )

    ratio = np.maximum(t / (p + eps), p / (t + eps))
    log_diff = np.log(t + eps) - np.log(p + eps)
    return OrderedDict(
        [
            ("abs_rel", float(abs_rel_diff(p, t))),
            ("squ_rel", float(squ_rel_diff(p, t))),
            ("rms_linear", float(rms_linear(p, t))),
            ("rms_log", float(np.sqrt((log_diff ** 2).mean()))),
            ("silog", float(scale_invariant_error(np.log(p + eps), np.log(t + eps)))),
            ("mean_err", float(mean_error(p, t))),
            ("median_diff", float(np.abs(np.median(t) - np.median(p)))),
            ("delta_1.25", float(np.mean(ratio <= 1.25))),
            ("delta_1.25^2", float(np.mean(ratio <= 1.25 ** 2))),
            ("delta_1.25^3", float(np.mean(ratio <= 1.25 ** 3))),
        ]
    )


def average_metric_blocks(blocks: list[OrderedDict[str, float]]) -> OrderedDict[str, float]:
    if not blocks:
        raise RuntimeError("No frames were evaluated.")
    result: OrderedDict[str, float] = OrderedDict()
    keys = list(blocks[0].keys())
    for key in keys:
        values = np.asarray([b[key] for b in blocks], dtype=np.float64)
        result[key] = float(np.nanmean(values))
    return result


def load_model(config_path: Path, checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any], float, float]:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_cfg = config["model"]
    model_cfg["gpu"] = config["gpu"]
    model_cfg["every_x_rgb_frame"] = config["data_loader"]["train"]["every_x_rgb_frame"]
    model_cfg["baseline"] = config["data_loader"]["train"]["baseline"]
    model_cfg["loss_composition"] = config["trainer"]["loss_composition"]

    model = S2DepthTransformerUNetConv(model_cfg)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    clip_distance = float(config["data_loader"]["validation"].get("clip_distance", 1000.0))
    reg_factor_raw = config["data_loader"]["train"].get("reg_factor", 5.7)
    reg_factor = 5.7 if reg_factor_raw is None else float(reg_factor_raw)
    return model, config, clip_distance, reg_factor


def main() -> None:
    args = parse_args()
    if args.raw_camera != "image_02":
        print(
            f"[warn] raw camera is {args.raw_camera}. "
            "KITTI annotated left-depth GT in proj_depth/groundtruth/image_02 is aligned with image_02."
        )

    raw_rgb_dir = args.raw_seq / args.raw_camera / "data"
    raw_frames = list_pngs(raw_rgb_dir)
    gt_files = list_pngs(args.gt_dir)

    if len(raw_frames) < 2:
        raise RuntimeError(f"Need at least two RGB frames in {raw_rgb_dir}")
    if not gt_files:
        raise RuntimeError(f"No GT depth PNG files found in {args.gt_dir}")

    if args.max_gt_frames is not None:
        gt_files = gt_files[: args.max_gt_frames]

    gt_indices = [int(p.stem) for p in gt_files]
    if min(gt_indices) < 1:
        raise RuntimeError("Ground-truth frame index 0 cannot be evaluated because spike_0000000000 does not exist.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spikes_dir = args.output_dir / "spikes_crop224"
    pred_dir = args.output_dir / "pred_metric"
    gt_metric_dir = args.output_dir / "gt_metric"
    valid_mask_dir = args.output_dir / "valid_mask"
    for folder in (pred_dir, gt_metric_dir, valid_mask_dir):
        folder.mkdir(parents=True, exist_ok=True)
    if args.save_spikes:
        spikes_dir.mkdir(parents=True, exist_ok=True)

    simulator = load_simulator_module(args.sim_script)
    model, config, clip_distance, reg_factor = load_model(args.config, args.checkpoint)

    first_gray = simulator.load_gray_frame(raw_frames[0])
    crop_y, crop_x = center_crop_slices(first_gray.shape[0], first_gray.shape[1], args.crop_size)
    prev_gray = first_gray[crop_y, crop_x]
    rng = np.random.default_rng(args.seed)
    accumulator = simulator.initialize_state(prev_gray.shape, args.threshold, args.init_state, rng)
    bin_centers = (np.arange(args.num_bins, dtype=np.float32) + 0.5) / float(args.num_bins)

    gt_index_to_path = {int(p.stem): p for p in gt_files}
    metrics_per_frame: list[OrderedDict[str, float]] = []

    expected_last_gt = max(gt_indices)
    last_required_frame = min(expected_last_gt, len(raw_frames) - 1)

    print(f"[info] simulator: {args.sim_script}")
    print(f"[info] input camera: {raw_rgb_dir}")
    print(f"[info] gt camera: {args.gt_dir}")
    print(f"[info] crop: {args.crop_size}x{args.crop_size}")
    print(f"[info] evaluating {len(gt_files)} GT frames from {gt_indices[0]} to {gt_indices[-1]}")
    print(f"[info] clip_distance={clip_distance}, reg_factor={reg_factor}")

    with torch.no_grad():
        for frame_idx in range(1, last_required_frame + 1):
            spike_path = spikes_dir / f"spike_{frame_idx:010d}.npy"
            curr_gray = simulator.load_gray_frame(raw_frames[frame_idx])[crop_y, crop_x]
            if args.save_spikes and spike_path.exists() and not args.overwrite_spikes:
                spike_tensor = np.load(spike_path)
                # Even when reusing a saved tensor we must keep the accumulator in sync for later frames.
                _, accumulator = simulator.simulate_interval(
                    prev_gray=prev_gray,
                    curr_gray=curr_gray,
                    accumulator=accumulator,
                    threshold=args.threshold,
                    bin_centers=bin_centers,
                )
            else:
                spike_tensor, accumulator = simulator.simulate_interval(
                    prev_gray=prev_gray,
                    curr_gray=curr_gray,
                    accumulator=accumulator,
                    threshold=args.threshold,
                    bin_centers=bin_centers,
                )
            prev_gray = curr_gray

            if frame_idx not in gt_index_to_path:
                continue

            if args.save_spikes and (args.overwrite_spikes or not spike_path.exists()):
                np.save(spike_path, spike_tensor)

            input_tensor = torch.from_numpy(spike_tensor.astype(np.float32, copy=False)).unsqueeze(0).cuda()
            pred_dict, _, _ = model({"image": input_tensor}, None, {})
            pred_log = pred_dict["image"][0, 0].detach().cpu().numpy()
            pred_metric = metric_depth_from_prediction(pred_log, clip_distance, reg_factor)

            gt_png = np.asarray(Image.open(gt_index_to_path[frame_idx]), dtype=np.uint16)
            gt_metric = np.clip(gt_png.astype(np.float32) / 256.0, 0.0, clip_distance)
            gt_metric = gt_metric[crop_y, crop_x]
            valid_mask = gt_metric > 0.0

            metrics = metric_block(gt_metric, pred_metric, valid_mask)
            metrics_per_frame.append(metrics)

            np.save(pred_dir / f"depth_{frame_idx:010d}.npy", pred_metric.astype(np.float32))
            np.save(gt_metric_dir / f"depth_{frame_idx:010d}.npy", gt_metric.astype(np.float32))
            np.save(valid_mask_dir / f"mask_{frame_idx:010d}.npy", valid_mask.astype(np.uint8))

            if len(metrics_per_frame) == 1 or len(metrics_per_frame) % 20 == 0:
                print(
                    f"[eval] frame={frame_idx:04d} "
                    f"valid_pixels={int(valid_mask.sum())} "
                    f"abs_rel={metrics['abs_rel']:.4f} "
                    f"rms={metrics['rms_linear']:.4f} "
                    f"delta1={metrics['delta_1.25']:.4f}"
                )

    summary = {
        "raw_sequence": str(args.raw_seq),
        "raw_camera": args.raw_camera,
        "gt_dir": str(args.gt_dir),
        "simulator_script": str(args.sim_script),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "crop_size": args.crop_size,
        "num_bins": args.num_bins,
        "threshold": args.threshold,
        "init_state": args.init_state,
        "seed": args.seed,
        "clip_distance": clip_distance,
        "reg_factor": reg_factor,
        "num_raw_frames": len(raw_frames),
        "num_gt_frames": len(gt_files),
        "evaluated_frames": len(metrics_per_frame),
        "first_gt_frame": gt_indices[0],
        "last_gt_frame": gt_indices[-1],
        "metrics_frame_mean": average_metric_blocks(metrics_per_frame),
    }

    summary_path = args.output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[done] metrics saved to", summary_path)
    for key, value in summary["metrics_frame_mean"].items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
