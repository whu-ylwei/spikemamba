#!/usr/bin/env python3
"""Approximate DENSE-style spike tensors from raw event windows.

This script reads DENSE-format raw event files:

    <sequence>/events/data/events_XXXXXXXXXX.npy
    <sequence>/events/data/timestamps.txt

and writes proxy spike tensors:

    <sequence>/<output_subdir>/spike_XXXXXXXXXX.npy
    <sequence>/<output_subdir>/timestamps.txt

Important limitation:
The published DENSE-spike data cannot be exactly recovered from the raw event
windows alone. The released spike tensors are much denser than the raw event
counts, so this script implements a configurable proxy simulator rather than an
exact inversion of the original unpublished simulator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np


DEFAULT_HEIGHT = 0
DEFAULT_WIDTH = 0


@dataclass(frozen=True)
class EventWindow:
    index: int
    path: Path
    t_start_units: float
    t_end_units: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approximate spike tensors from DENSE raw events")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Sequence directory or split directory containing sequence subfolders.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="spike/sim_r128",
        help="Relative output subdirectory inside each sequence.",
    )
    parser.add_argument(
        "--events-subdir",
        type=str,
        default="events/data",
        help="Relative input event subdirectory inside each sequence.",
    )
    parser.add_argument(
        "--reference-spike-subdir",
        type=str,
        default="",
        help="Optional reference spike folder, e.g. spike/r128, used only for burst search/reporting.",
    )
    parser.add_argument("--num-bins", type=int, default=128, help="Number of output temporal bins.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Output height.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Output width.")
    parser.add_argument(
        "--polarity-mode",
        choices=("abs", "positive", "negative"),
        default="abs",
        help="How raw event polarity contributes to spikes.",
    )
    parser.add_argument(
        "--burst-length",
        type=int,
        default=1,
        help="Number of consecutive bins activated per raw event.",
    )
    parser.add_argument(
        "--output-index-offset",
        type=int,
        default=1,
        help="Output filename offset. DENSE-spike uses events_000... -> spike_000...001.",
    )
    parser.add_argument(
        "--search-burst",
        action="store_true",
        help="Search burst_length against reference spike density before generation.",
    )
    parser.add_argument(
        "--search-min",
        type=int,
        default=1,
        help="Minimum burst_length included in search.",
    )
    parser.add_argument(
        "--search-max",
        type=int,
        default=128,
        help="Maximum burst_length included in search.",
    )
    parser.add_argument(
        "--search-windows",
        type=int,
        default=8,
        help="Maximum paired windows used for burst search/reporting.",
    )
    parser.add_argument(
        "--limit-sequences",
        type=int,
        default=0,
        help="If > 0, only process the first N discovered sequences.",
    )
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=0,
        help="If > 0, only process the first N windows per sequence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing spike files.",
    )
    return parser.parse_args()


def discover_sequences(root: Path, events_subdir: str) -> list[Path]:
    if (root / events_subdir).is_dir():
        return [root]

    sequences = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / events_subdir).is_dir():
            sequences.append(child)
    return sequences


def load_timestamps(path: Path) -> np.ndarray:
    stamps = np.loadtxt(path)
    if stamps.ndim == 1:
        stamps = stamps.reshape(1, -1)
    return stamps


def build_windows(
    sequence_dir: Path,
    events_subdir: str,
    limit_windows: int = 0,
    time_scale: float = 1.0,
) -> list[EventWindow]:
    events_dir = sequence_dir / events_subdir
    timestamps = load_timestamps(events_dir / "timestamps.txt")
    times_sec = timestamps[:, 1].astype(np.float64)
    dt_sec = float(np.median(np.diff(times_sec))) if len(times_sec) > 1 else 1.0 / 30.0

    windows: list[EventWindow] = []
    for row_idx, t_end_sec in enumerate(times_sec):
        event_path = events_dir / f"events_{row_idx:010d}.npy"
        if not event_path.exists():
            continue
        t_start_sec = times_sec[row_idx - 1] if row_idx > 0 else t_end_sec - dt_sec
        windows.append(
            EventWindow(
                index=row_idx,
                path=event_path,
                t_start_units=t_start_sec * time_scale,
                t_end_units=t_end_sec * time_scale,
            )
        )
        if limit_windows > 0 and len(windows) >= limit_windows:
            break
    return windows


def filter_events(
    events: np.ndarray,
    height: int,
    width: int,
    polarity_mode: str,
    t_start_units: float,
    t_end_units: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if events.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty

    t_units = events[:, 0].astype(np.float64)
    x = events[:, 1].astype(np.int64)
    y = events[:, 2].astype(np.int64)
    polarity = events[:, 3]

    mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    mask &= (t_units >= t_start_units) & (t_units <= t_end_units)
    if polarity_mode == "positive":
        mask &= polarity > 0
    elif polarity_mode == "negative":
        mask &= polarity < 0

    return t_units[mask], x[mask], y[mask]


def simulate_window(
    events: np.ndarray,
    *,
    t_start_units: float,
    t_end_units: float,
    height: int,
    width: int,
    num_bins: int,
    burst_length: int,
    polarity_mode: str,
) -> np.ndarray:
    burst_length = max(1, min(int(burst_length), int(num_bins)))
    window_units = max(t_end_units - t_start_units, 1.0)
    t_units, x, y = filter_events(events, height, width, polarity_mode, t_start_units, t_end_units)

    if t_units.size == 0:
        return np.zeros((num_bins, height, width), dtype=np.uint8)

    bins = np.floor((t_units - t_start_units) / window_units * num_bins).astype(np.int64)
    bins = np.clip(bins, 0, num_bins - 1)

    flat_idx = y * width + x
    end_bins = np.minimum(bins + burst_length, num_bins)

    diff = np.zeros((num_bins + 1, height * width), dtype=np.int32)
    np.add.at(diff, (bins, flat_idx), 1)
    np.add.at(diff, (end_bins, flat_idx), -1)
    occupied = np.cumsum(diff[:-1], axis=0) > 0
    return occupied.reshape(num_bins, height, width).astype(np.uint8)


def reference_path_for_window(
    sequence_dir: Path,
    reference_spike_subdir: str,
    window_index: int,
    output_index_offset: int,
) -> Path:
    return sequence_dir / reference_spike_subdir / f"spike_{window_index + output_index_offset:010d}.npy"


def detect_time_scale(sequence_dir: Path, events_subdir: str) -> float:
    windows = build_windows(sequence_dir, events_subdir, limit_windows=1)
    if not windows:
        return 1e9

    events = np.load(windows[0].path)
    if events.size == 0:
        return 1e9

    event_end = float(np.max(events[:, 0]))
    timestamps = load_timestamps(sequence_dir / events_subdir / "timestamps.txt")
    ts_end_sec = float(timestamps[min(windows[0].index, len(timestamps) - 1), 1])

    candidates = [1.0, 1e3, 1e6, 1e9]
    best_scale = 1e9
    best_error = float("inf")
    for scale in candidates:
        target = ts_end_sec * scale
        if target <= 0 or event_end <= 0:
            continue
        error = abs(np.log10(event_end) - np.log10(target))
        if error < best_error:
            best_error = error
            best_scale = scale
    return best_scale


def infer_resolution(sequence_dir: Path, args: argparse.Namespace) -> tuple[int, int]:
    if args.reference_spike_subdir:
        ref_dir = sequence_dir / args.reference_spike_subdir
        ref_files = sorted(ref_dir.glob("spike_*.npy"))
        if ref_files:
            ref = np.load(ref_files[0], mmap_mode="r")
            _, height, width = ref.shape
            return int(height), int(width)

    windows = build_windows(sequence_dir, args.events_subdir, limit_windows=1)
    if not windows:
        raise FileNotFoundError(f"No event windows found under {sequence_dir / args.events_subdir}")

    events = np.load(windows[0].path)
    if events.size == 0:
        raise ValueError(f"First event file is empty: {windows[0].path}")

    width = int(np.max(events[:, 1])) + 1
    height = int(np.max(events[:, 2])) + 1
    return height, width


def search_burst_length(
    sequences: Iterable[Path],
    args: argparse.Namespace,
) -> int:
    if not args.reference_spike_subdir:
        raise ValueError("--search-burst requires --reference-spike-subdir")

    samples: list[tuple[EventWindow, Path]] = []
    for sequence_dir in sequences:
        for window in build_windows(
            sequence_dir,
            args.events_subdir,
            limit_windows=args.search_windows,
            time_scale=args.event_time_scale,
        ):
            ref_path = reference_path_for_window(
                sequence_dir,
                args.reference_spike_subdir,
                window.index,
                args.output_index_offset,
            )
            if ref_path.exists():
                samples.append((window, ref_path))
            if len(samples) >= args.search_windows:
                break
        if len(samples) >= args.search_windows:
            break

    if not samples:
        raise FileNotFoundError("No paired event/spike samples found for burst search")

    cached_events = [(window, np.load(window.path), np.load(ref_path)) for window, ref_path in samples]

    best_burst = args.burst_length
    best_error = float("inf")
    for burst in range(args.search_min, args.search_max + 1):
        errors = []
        for window, events, reference in cached_events:
            simulated = simulate_window(
                events,
                t_start_units=window.t_start_units,
                t_end_units=window.t_end_units,
                height=args.height,
                width=args.width,
                num_bins=args.num_bins,
                burst_length=burst,
                polarity_mode=args.polarity_mode,
            )
            errors.append(abs(float(simulated.mean()) - float(reference.mean())))
        mean_error = float(np.mean(errors))
        if mean_error < best_error:
            best_error = mean_error
            best_burst = burst

    print(
        f"[search] best burst_length={best_burst} "
        f"(mean occupancy error={best_error:.6f}, samples={len(samples)})"
    )
    return best_burst


def write_metadata(output_dir: Path, args: argparse.Namespace, burst_length: int) -> None:
    metadata = {
        "simulator": "approximate_event_to_spike_burst",
        "note": (
            "This is a proxy simulator from raw events to dense spike tensors. "
            "It is not an exact reconstruction of the original DENSE-spike generator."
        ),
        "num_bins": args.num_bins,
        "height": args.height,
        "width": args.width,
        "polarity_mode": args.polarity_mode,
        "burst_length": burst_length,
        "output_index_offset": args.output_index_offset,
    }
    with (output_dir / "simulation_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def process_sequence(sequence_dir: Path, args: argparse.Namespace, burst_length: int) -> None:
    windows = build_windows(
        sequence_dir,
        args.events_subdir,
        limit_windows=args.limit_windows,
        time_scale=args.event_time_scale,
    )
    output_dir = sequence_dir / args.output_subdir
    print(f"[sequence] {sequence_dir}")
    print(f"  windows={len(windows)} output={output_dir}")

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps_src = sequence_dir / args.events_subdir / "timestamps.txt"
    shutil.copy2(timestamps_src, output_dir / "timestamps.txt")
    write_metadata(output_dir, args, burst_length)

    for window in windows:
        events = np.load(window.path)
        simulated = simulate_window(
            events,
            t_start_units=window.t_start_units,
            t_end_units=window.t_end_units,
            height=args.height,
            width=args.width,
            num_bins=args.num_bins,
            burst_length=burst_length,
            polarity_mode=args.polarity_mode,
        )
        output_path = output_dir / f"spike_{window.index + args.output_index_offset:010d}.npy"
        np.save(output_path, simulated.astype(np.uint8))


def main() -> None:
    args = parse_args()
    sequences = discover_sequences(args.input_root, args.events_subdir)
    if args.limit_sequences > 0:
        sequences = sequences[: args.limit_sequences]
    if not sequences:
        raise FileNotFoundError(f"No sequences with {args.events_subdir!r} found under {args.input_root}")

    if args.height <= 0 or args.width <= 0:
        inferred_height, inferred_width = infer_resolution(sequences[0], args)
        args.height = inferred_height
        args.width = inferred_width
        print(f"[config] inferred resolution={args.height}x{args.width}")

    time_scale = detect_time_scale(sequences[0], args.events_subdir)
    print(f"[config] inferred event time scale={time_scale:g} ticks/sec")
    args.event_time_scale = time_scale

    burst_length = args.burst_length
    if args.search_burst:
        burst_length = search_burst_length(sequences, args)
    else:
        print(f"[config] burst_length={burst_length}")

    for sequence_dir in sequences:
        process_sequence(sequence_dir, args, burst_length)


if __name__ == "__main__":
    main()
