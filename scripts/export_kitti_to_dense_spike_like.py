#!/usr/bin/env python3
"""Export a KITTI raw sequence to a DENSE-spike-like folder layout.

Matched to DENSE-spike:
- directory layout: rgb/frames, depth/data, depth/frames, spike/r128
- filename conventions: frame_XXXXXXXXXX.png, depth_XXXXXXXXXX.npy, spike_XXXXXXXXXX.npy
- spike simulation rule: same SCAPE/DENSE-style integrate-and-fire simulator
- timestamps.txt format: two columns [index, timestamp_seconds]

Supported export modes:
- native KITTI timestamps / native FPS
- uniform target FPS by time-resampling RGB with linear interpolation before spike simulation

Depth handling stays conservative:
- original sparse KITTI depth is never temporally interpolated per pixel
- in target-FPS mode, each original GT frame is assigned to the nearest resampled timestamp
- all other frames keep NaN depth so no pseudo-ground-truth is fabricated
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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
        help="KITTI camera folder. image_02 is the left RGB camera aligned with proj_depth/groundtruth/image_02.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("/root/shared-nvme/kitti/data_depth_annotated/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02"),
        help="KITTI left-depth ground-truth PNG folder.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/shared-nvme/kitti/DENSE_spike_like"),
        help="Root directory for exported DENSE-like data.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name to create under output-root.",
    )
    parser.add_argument(
        "--sequence-name",
        type=str,
        default="test_sequence_00_kitti_0001_image02",
        help="Exported sequence directory name.",
    )
    parser.add_argument(
        "--sim-script",
        type=Path,
        default=Path("/root/shared-nvme/scape/simulate_dense_spike_camera.py"),
        help="Path to the DENSE-style spike simulator script.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="If set, resample RGB to a uniform FPS grid before simulating spikes.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Optional output height. Use 260 to match DENSE-spike spatial size.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optional output width. Use 346 to match DENSE-spike spatial size.",
    )
    parser.add_argument("--num-bins", type=int, default=128, help="Number of temporal bins.")
    parser.add_argument("--threshold", type=float, default=4.0, help="Integrate-and-fire threshold.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the initial accumulator.")
    parser.add_argument(
        "--init-state",
        choices=("random", "zero", "half"),
        default="random",
        help="Accumulator initialization mode.",
    )
    parser.add_argument(
        "--clip-distance",
        type=float,
        default=1000.0,
        help="Depth clipping distance used for visualization only.",
    )
    parser.add_argument(
        "--rgb-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Only used in native-FPS mode without resize. Whether to copy or symlink RGB frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing exported sequence directory.",
    )
    return parser.parse_args()


def load_simulator_module(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("scape_spike_sim", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load simulator module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_kitti_timestamp_line(line: str) -> float:
    line = line.strip()
    main, frac = line.split(".")
    frac6 = (frac + "000000")[:6]
    dt = datetime.strptime(main + "." + frac6, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def read_kitti_timestamps(path: Path) -> np.ndarray:
    values = [parse_kitti_timestamp_line(line) for line in path.read_text().splitlines() if line.strip()]
    values = np.asarray(values, dtype=np.float64)
    values -= values[0]
    return values


def write_indexed_timestamps(path: Path, stamps: np.ndarray) -> None:
    rows = np.column_stack([np.arange(len(stamps), dtype=np.int64), stamps.astype(np.float64)])
    np.savetxt(path, rows, fmt=["%.0f", "%.8f"])


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"{path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def depth_to_vis(depth_m: np.ndarray, clip_distance: float) -> np.ndarray:
    valid = np.isfinite(depth_m)
    vis = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        scaled = np.clip(depth_m[valid] / clip_distance, 0.0, 1.0)
        vis_vals = np.maximum(1, np.round(scaled * 255.0)).astype(np.uint8)
        vis[valid] = vis_vals
    return vis


def maybe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def load_rgb_uint8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb_uint8(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)


def maybe_resize_rgb_uint8(rgb: np.ndarray, output_shape: tuple[int, int] | None) -> np.ndarray:
    if output_shape is None:
        return rgb
    output_h, output_w = output_shape
    if rgb.shape[0] == output_h and rgb.shape[1] == output_w:
        return rgb
    return np.asarray(Image.fromarray(rgb, mode="RGB").resize((output_w, output_h), Image.BILINEAR), dtype=np.uint8)


def load_metric_depth_from_png(path: Path, output_shape: tuple[int, int] | None = None) -> np.ndarray:
    depth_png_img = Image.open(path)
    if output_shape is not None:
        output_h, output_w = output_shape
        if depth_png_img.size != (output_w, output_h):
            depth_png_img = depth_png_img.resize((output_w, output_h), Image.NEAREST)
    depth_png = np.asarray(depth_png_img, dtype=np.uint16)
    depth_m = depth_png.astype(np.float32) / 256.0
    depth_m[depth_png == 0] = np.nan
    return depth_m


def build_target_timestamps(source_timestamps: np.ndarray, target_fps: float | None) -> np.ndarray:
    if target_fps is None:
        return source_timestamps.copy()
    if target_fps <= 0:
        raise ValueError("--target-fps must be > 0")
    dt = 1.0 / target_fps
    count = int(np.floor(source_timestamps[-1] / dt)) + 1
    return np.arange(count, dtype=np.float64) * dt


def export_directory_skeleton(output_seq_dir: Path, num_bins: int) -> dict[str, Path]:
    paths = {
        "rgb_frames": output_seq_dir / "rgb" / "frames",
        "depth_data": output_seq_dir / "depth" / "data",
        "depth_frames": output_seq_dir / "depth" / "frames",
        "spike": output_seq_dir / "spike" / f"r{num_bins}",
    }
    extras = [
        output_seq_dir / "events" / "data",
        output_seq_dir / "events" / "frames",
        output_seq_dir / "events" / "frames_white",
        output_seq_dir / "events" / "voxels",
        output_seq_dir / "semantic" / "data",
        output_seq_dir / "semantic" / "frames",
    ]
    for folder in [*paths.values(), *extras]:
        folder.mkdir(parents=True, exist_ok=True)
    return paths


def export_depth_sequence(
    target_timestamps: np.ndarray,
    gt_by_source_index: dict[int, Path],
    source_timestamps: np.ndarray,
    output_shape: tuple[int, int],
    depth_data_dir: Path,
    depth_frames_dir: Path,
    clip_distance: float,
) -> dict[str, Any]:
    height, width = output_shape
    target_depths: list[np.ndarray] = [
        np.full((height, width), np.nan, dtype=np.float32) for _ in range(len(target_timestamps))
    ]

    mapping_records: list[dict[str, Any]] = []
    used_targets: dict[int, dict[str, Any]] = {}

    for source_idx, gt_path in sorted(gt_by_source_index.items()):
        target_idx = int(np.argmin(np.abs(target_timestamps - source_timestamps[source_idx])))
        time_error = float(abs(target_timestamps[target_idx] - source_timestamps[source_idx]))
        record = {
            "source_frame_index": int(source_idx),
            "target_frame_index": int(target_idx),
            "source_timestamp_sec": float(source_timestamps[source_idx]),
            "target_timestamp_sec": float(target_timestamps[target_idx]),
            "time_error_sec": time_error,
            "gt_path": str(gt_path),
        }
        previous = used_targets.get(target_idx)
        if previous is None or time_error < previous["time_error_sec"]:
            used_targets[target_idx] = record
        mapping_records.append(record)

    for target_idx, record in sorted(used_targets.items()):
        target_depths[target_idx] = load_metric_depth_from_png(Path(record["gt_path"]), output_shape=output_shape)

    for idx, depth_m in enumerate(target_depths):
        np.save(depth_data_dir / f"depth_{idx:010d}.npy", depth_m.astype(np.float32))
        Image.fromarray(depth_to_vis(depth_m, clip_distance)).save(depth_frames_dir / f"frame_{idx:010d}.png")

    return {
        "num_depth_frames_with_gt": len(used_targets),
        "num_depth_frames_total": len(target_timestamps),
        "gt_mapping": mapping_records,
        "gt_target_indices_used": sorted(int(k) for k in used_targets.keys()),
        "max_gt_time_error_sec": float(max((r["time_error_sec"] for r in used_targets.values()), default=0.0)),
    }


def export_native_rgb_frames(raw_frames: list[Path], rgb_frames_dir: Path, rgb_mode: str) -> None:
    for idx, src in enumerate(raw_frames):
        dst = rgb_frames_dir / f"frame_{idx:010d}.png"
        maybe_link_or_copy(src, dst, rgb_mode)


def export_native_rgb_frames_resized(
    raw_frames: list[Path], rgb_frames_dir: Path, output_shape: tuple[int, int]
) -> None:
    for idx, src in enumerate(raw_frames):
        rgb = maybe_resize_rgb_uint8(load_rgb_uint8(src), output_shape)
        save_rgb_uint8(rgb_frames_dir / f"frame_{idx:010d}.png", rgb)


def export_resampled_rgb_and_spikes(
    raw_frames: list[Path],
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    rgb_frames_dir: Path,
    spike_dir: Path,
    simulator: Any,
    num_bins: int,
    threshold: float,
    init_state: str,
    seed: int,
    output_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    raw_ptr = 0
    left_rgb = maybe_resize_rgb_uint8(load_rgb_uint8(raw_frames[0]), output_shape)
    right_rgb = (
        maybe_resize_rgb_uint8(load_rgb_uint8(raw_frames[1]), output_shape)
        if len(raw_frames) > 1
        else left_rgb.copy()
    )

    def get_interpolated_rgb(target_t: float) -> np.ndarray:
        nonlocal raw_ptr, left_rgb, right_rgb
        while raw_ptr + 1 < len(source_timestamps) - 1 and source_timestamps[raw_ptr + 1] < target_t:
            raw_ptr += 1
            left_rgb = right_rgb
            right_rgb = maybe_resize_rgb_uint8(load_rgb_uint8(raw_frames[raw_ptr + 1]), output_shape)

        t0 = source_timestamps[raw_ptr]
        t1 = source_timestamps[min(raw_ptr + 1, len(source_timestamps) - 1)]
        if t1 <= t0:
            alpha = 0.0
        else:
            alpha = float((target_t - t0) / (t1 - t0))
        alpha = min(max(alpha, 0.0), 1.0)
        interp = (1.0 - alpha) * left_rgb.astype(np.float32) + alpha * right_rgb.astype(np.float32)
        return np.clip(np.round(interp), 0, 255).astype(np.uint8)

    bin_centers = (np.arange(num_bins, dtype=np.float32) + 0.5) / float(num_bins)
    prev_rgb = get_interpolated_rgb(float(target_timestamps[0]))
    save_rgb_uint8(rgb_frames_dir / "frame_0000000000.png", prev_rgb)
    prev_gray = ((prev_rgb.astype(np.float32) / 255.0) @ simulator.GRAY_WEIGHTS).astype(np.float32, copy=False)
    accumulator = simulator.initialize_state(prev_gray.shape, threshold, init_state, np.random.default_rng(seed))

    total_spikes = 0
    for idx in range(1, len(target_timestamps)):
        curr_rgb = get_interpolated_rgb(float(target_timestamps[idx]))
        save_rgb_uint8(rgb_frames_dir / f"frame_{idx:010d}.png", curr_rgb)
        curr_gray = ((curr_rgb.astype(np.float32) / 255.0) @ simulator.GRAY_WEIGHTS).astype(np.float32, copy=False)
        spike_tensor, accumulator = simulator.simulate_interval(
            prev_gray=prev_gray,
            curr_gray=curr_gray,
            accumulator=accumulator,
            threshold=threshold,
            bin_centers=bin_centers,
        )
        np.save(spike_dir / f"spike_{idx:010d}.npy", spike_tensor)
        total_spikes += int(spike_tensor.sum())
        prev_gray = curr_gray
        if idx % 50 == 0:
            print(f"[progress] resampled frames={idx}/{len(target_timestamps)-1}")

    dt = float(target_timestamps[1] - target_timestamps[0]) if len(target_timestamps) > 1 else float("nan")
    return {
        "num_rgb_frames": len(target_timestamps),
        "num_spike_tensors": max(len(target_timestamps) - 1, 0),
        "image_resolution": [int(prev_gray.shape[0]), int(prev_gray.shape[1])],
        "mean_spikes_per_pixel_per_interval": float(
            total_spikes / max(len(target_timestamps) - 1, 1) / prev_gray.size
        ),
        "target_dt_sec": dt,
    }


def export_native_spikes(
    raw_frames: list[Path],
    spike_dir: Path,
    simulator: Any,
    num_bins: int,
    threshold: float,
    init_state: str,
    seed: int,
    output_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    bin_centers = (np.arange(num_bins, dtype=np.float32) + 0.5) / float(num_bins)
    if output_shape is None:
        prev_gray = simulator.load_gray_frame(raw_frames[0])
    else:
        prev_rgb = maybe_resize_rgb_uint8(load_rgb_uint8(raw_frames[0]), output_shape)
        prev_gray = ((prev_rgb.astype(np.float32) / 255.0) @ simulator.GRAY_WEIGHTS).astype(np.float32, copy=False)
    accumulator = simulator.initialize_state(prev_gray.shape, threshold, init_state, np.random.default_rng(seed))

    total_spikes = 0
    for idx in range(1, len(raw_frames)):
        if output_shape is None:
            curr_gray = simulator.load_gray_frame(raw_frames[idx])
        else:
            curr_rgb = maybe_resize_rgb_uint8(load_rgb_uint8(raw_frames[idx]), output_shape)
            curr_gray = ((curr_rgb.astype(np.float32) / 255.0) @ simulator.GRAY_WEIGHTS).astype(np.float32, copy=False)
        spike_tensor, accumulator = simulator.simulate_interval(
            prev_gray=prev_gray,
            curr_gray=curr_gray,
            accumulator=accumulator,
            threshold=threshold,
            bin_centers=bin_centers,
        )
        np.save(spike_dir / f"spike_{idx:010d}.npy", spike_tensor)
        total_spikes += int(spike_tensor.sum())
        prev_gray = curr_gray

    return {
        "num_rgb_frames": len(raw_frames),
        "num_spike_tensors": max(len(raw_frames) - 1, 0),
        "image_resolution": [int(prev_gray.shape[0]), int(prev_gray.shape[1])],
        "mean_spikes_per_pixel_per_interval": float(total_spikes / max(len(raw_frames) - 1, 1) / prev_gray.size),
    }


def main() -> None:
    args = parse_args()
    if args.raw_camera != "image_02":
        print(
            f"[warn] exporting {args.raw_camera}. "
            "The provided depth GT folder is aligned with image_02; use image_02 unless you reproject depth."
        )

    raw_rgb_dir = args.raw_seq / args.raw_camera / "data"
    raw_ts_path = args.raw_seq / args.raw_camera / "timestamps.txt"
    raw_frames = sorted(raw_rgb_dir.glob("*.png"))
    if not raw_frames:
        raise RuntimeError(f"No raw RGB frames found in {raw_rgb_dir}")
    if not raw_ts_path.exists():
        raise RuntimeError(f"Missing KITTI timestamps file {raw_ts_path}")

    gt_paths = sorted(args.gt_dir.glob("*.png"))
    if not gt_paths:
        raise RuntimeError(f"No GT depth PNG files found in {args.gt_dir}")
    gt_by_source_index = {int(p.stem): p for p in gt_paths}

    source_timestamps = read_kitti_timestamps(raw_ts_path)
    if len(source_timestamps) != len(raw_frames):
        raise RuntimeError(
            f"Timestamp count mismatch: {len(source_timestamps)} timestamps vs {len(raw_frames)} RGB frames."
        )

    target_timestamps = build_target_timestamps(source_timestamps, args.target_fps)
    output_seq_dir = args.output_root / args.split / args.sequence_name
    ensure_clean_dir(output_seq_dir, args.overwrite)
    paths = export_directory_skeleton(output_seq_dir, args.num_bins)

    sample_gt = np.asarray(Image.open(gt_paths[0]), dtype=np.uint16)
    sample_shape = tuple(int(x) for x in sample_gt.shape)
    if args.resize_height is None and args.resize_width is None:
        output_shape = sample_shape
    elif args.resize_height is not None and args.resize_width is not None:
        if args.resize_height <= 0 or args.resize_width <= 0:
            raise ValueError("--resize-height and --resize-width must be positive")
        output_shape = (int(args.resize_height), int(args.resize_width))
    else:
        raise ValueError("Provide both --resize-height and --resize-width, or neither.")

    simulator = load_simulator_module(args.sim_script)

    if args.target_fps is None:
        if output_shape == sample_shape:
            export_native_rgb_frames(raw_frames, paths["rgb_frames"], args.rgb_mode)
        else:
            export_native_rgb_frames_resized(raw_frames, paths["rgb_frames"], output_shape)
        rgb_spike_info = export_native_spikes(
            raw_frames=raw_frames,
            spike_dir=paths["spike"],
            simulator=simulator,
            num_bins=args.num_bins,
            threshold=args.threshold,
            init_state=args.init_state,
            seed=args.seed,
            output_shape=None if output_shape == sample_shape else output_shape,
        )
        export_mode = "native_fps"
    else:
        rgb_spike_info = export_resampled_rgb_and_spikes(
            raw_frames=raw_frames,
            source_timestamps=source_timestamps,
            target_timestamps=target_timestamps,
            rgb_frames_dir=paths["rgb_frames"],
            spike_dir=paths["spike"],
            simulator=simulator,
            num_bins=args.num_bins,
            threshold=args.threshold,
            init_state=args.init_state,
            seed=args.seed,
            output_shape=output_shape,
        )
        export_mode = "resampled_rgb_then_simulated_spike"

    write_indexed_timestamps(paths["rgb_frames"] / "timestamps.txt", target_timestamps)
    write_indexed_timestamps(paths["spike"] / "timestamps.txt", target_timestamps)

    depth_info = export_depth_sequence(
        target_timestamps=target_timestamps,
        gt_by_source_index=gt_by_source_index,
        source_timestamps=source_timestamps,
        output_shape=output_shape,
        depth_data_dir=paths["depth_data"],
        depth_frames_dir=paths["depth_frames"],
        clip_distance=args.clip_distance,
    )
    write_indexed_timestamps(paths["depth_data"] / "timestamps.txt", target_timestamps)
    write_indexed_timestamps(paths["depth_frames"] / "timestamps.txt", target_timestamps)

    mean_dt = float(np.diff(target_timestamps).mean()) if len(target_timestamps) > 1 else float("nan")
    metadata = {
        "source_dataset": "KITTI raw + annotated depth",
        "source_sequence": str(args.raw_seq),
        "source_camera": args.raw_camera,
        "depth_gt_dir": str(args.gt_dir),
        "output_sequence_dir": str(output_seq_dir),
        "simulator_script": str(args.sim_script),
        "export_mode": export_mode,
        "target_fps_requested": None if args.target_fps is None else float(args.target_fps),
        "num_source_rgb_frames": len(raw_frames),
        "num_source_gt_frames": len(gt_by_source_index),
        "num_export_rgb_frames": rgb_spike_info["num_rgb_frames"],
        "num_export_depth_frames": depth_info["num_depth_frames_total"],
        "num_export_depth_frames_with_gt": depth_info["num_depth_frames_with_gt"],
        "num_export_spike_tensors": rgb_spike_info["num_spike_tensors"],
        "image_resolution": rgb_spike_info["image_resolution"],
        "source_image_resolution": [int(sample_shape[0]), int(sample_shape[1])],
        "timestamps_start_sec": float(target_timestamps[0]),
        "timestamps_end_sec": float(target_timestamps[-1]),
        "mean_frame_dt_sec": mean_dt,
        "approx_fps": float(1.0 / mean_dt) if mean_dt > 0 else float("nan"),
        "dense_reference_fps": 30.0,
        "num_bins": args.num_bins,
        "threshold": args.threshold,
        "init_state": args.init_state,
        "seed": args.seed,
        "mean_spikes_per_pixel_per_interval": rgb_spike_info["mean_spikes_per_pixel_per_interval"],
        "gt_target_indices_used": depth_info["gt_target_indices_used"],
        "max_gt_time_error_sec": depth_info["max_gt_time_error_sec"],
        "notes": [
            "Directory layout and file naming follow DENSE-spike conventions for rgb/depth/spike.",
            "If target_fps_requested is not null, RGB frames are linearly interpolated in time on a uniform grid before spike simulation.",
            "Sparse KITTI depth is never temporally interpolated per pixel; GT is only attached to the nearest exported frame.",
            "events/ and semantic/ directories are created as empty placeholders because KITTI does not provide matching content here.",
        ],
    }
    with (output_seq_dir / "export_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with (output_seq_dir / "gt_frame_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(depth_info["gt_mapping"], f, indent=2, ensure_ascii=False)

    print("[done] exported sequence to", output_seq_dir)
    print("[done] mode =", export_mode)
    print("[done] approx fps =", metadata["approx_fps"])
    print(
        "[done] rgb/depth/spikes =",
        metadata["num_export_rgb_frames"],
        metadata["num_export_depth_frames"],
        metadata["num_export_spike_tensors"],
    )
    print("[done] mean_spikes_per_pixel =", metadata["mean_spikes_per_pixel_per_interval"])


if __name__ == "__main__":
    main()
