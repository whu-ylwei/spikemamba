import argparse
import glob
import os
from collections import OrderedDict

import numpy as np

from model.metric import (
    abs_rel_diff,
    mean_error,
    median_error,
    mse,
    rms_linear,
    scale_invariant_error,
    squ_rel_diff,
)


DEPTH_VALUES = [10, 20, 30, 80, 250, 500]


def parse_args():
    parser = argparse.ArgumentParser("Evaluate saved prediction npy files against GT npy files.")
    parser.add_argument("--target_dataset", required=True, type=str)
    parser.add_argument("--predictions_dataset", required=True, type=str)
    parser.add_argument("--output_log", required=True, type=str)
    parser.add_argument("--crop_ymax", default=260, type=int)
    parser.add_argument("--clip_distance", default=80.0, type=float)
    parser.add_argument("--reg_factor", default=5.7, type=float)
    parser.add_argument("--target_glob", default="*.npy", type=str)
    parser.add_argument("--pred_glob", default="*.npy", type=str)
    return parser.parse_args()


def build_metric_keys():
    keys = [
        "_abs_rel_diff",
        "_squ_rel_diff",
        "_RMS_linear",
        "_RMS_log",
        "_SILog",
        "_mean_depth_error",
        "_median_diff",
        "_threshold_delta_1.25",
        "_threshold_delta_1.25^2",
        "_threshold_delta_1.25^3",
    ]
    for k in DEPTH_VALUES:
        keys.extend(
            [
                f"_{k}_abs_rel_diff",
                f"_{k}_squ_rel_diff",
                f"_{k}_RMS_linear",
                f"_{k}_RMS_log",
                f"_{k}_SILog",
                f"_{k}_mean_depth_error",
                f"_{k}_median_diff",
                f"_{k}_threshold_delta_1.25",
                f"_{k}_threshold_delta_1.25^2",
                f"_{k}_threshold_delta_1.25^3",
            ]
        )
    return keys


def prepare_depth_data(target, prediction, clip_distance, reg_factor):
    # Mirror evaluation_DENSE.py behavior.
    prediction = np.exp(reg_factor * (prediction - np.ones_like(prediction, dtype=np.float32)))
    target = np.exp(reg_factor * (target - np.ones_like(target, dtype=np.float32)))
    target *= clip_distance
    prediction *= clip_distance
    prediction = np.clip(prediction, np.exp(-1 * reg_factor) * clip_distance, clip_distance)
    return target, prediction


def metric_block(target, prediction, mask):
    eps = 1e-5
    t = target[mask]
    p = prediction[mask]
    if t.size == 0:
        return {
            "abs_rel": np.nan,
            "squ_rel": np.nan,
            "rms_linear": np.nan,
            "rms_log": np.nan,
            "silog": np.nan,
            "mean_err": np.nan,
            "median_diff": np.nan,
            "d1": np.nan,
            "d2": np.nan,
            "d3": np.nan,
        }

    ratio = np.maximum(t / (p + eps), p / (t + eps))
    log_diff = np.log(t + eps) - np.log(p + eps)
    return {
        "abs_rel": abs_rel_diff(p, t),
        "squ_rel": squ_rel_diff(p, t),
        "rms_linear": rms_linear(p, t),
        "rms_log": np.sqrt((log_diff**2).mean()),
        "silog": scale_invariant_error(np.log(p + eps), np.log(t + eps)),
        "mean_err": mean_error(p, t),
        "median_diff": np.abs(np.median(t) - np.median(p)),
        "d1": np.mean(ratio <= 1.25),
        "d2": np.mean(ratio <= 1.25**2),
        "d3": np.mean(ratio <= 1.25**3),
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(args.predictions_dataset, args.pred_glob)))
    tgt_files = sorted(glob.glob(os.path.join(args.target_dataset, args.target_glob)))
    if len(pred_files) == 0 or len(tgt_files) == 0:
        raise RuntimeError("No npy files found in prediction or target dataset.")
    if len(pred_files) != len(tgt_files):
        raise RuntimeError(f"Count mismatch: pred={len(pred_files)}, target={len(tgt_files)}")

    keys = build_metric_keys()
    metrics = OrderedDict((k, 0.0) for k in keys)
    metrics2 = []

    n = len(pred_files)
    for i in range(n):
        pred = np.load(pred_files[i])[: args.crop_ymax]
        tgt = np.load(tgt_files[i])[: args.crop_ymax]

        pred_2d = pred[0] if pred.ndim == 3 else pred
        tgt_2d = tgt[0] if tgt.ndim == 3 else tgt
        tgt_abs, pred_abs = prepare_depth_data(tgt_2d, pred_2d, args.clip_distance, args.reg_factor)
        if pred_abs.shape != tgt_abs.shape:
            raise RuntimeError(f"Shape mismatch at index {i}: {pred_abs.shape} vs {tgt_abs.shape}")

        full_mask = np.ones_like(tgt_abs, dtype=bool)
        b = metric_block(tgt_abs, pred_abs, full_mask)
        metrics["_abs_rel_diff"] += b["abs_rel"]
        metrics["_squ_rel_diff"] += b["squ_rel"]
        metrics["_RMS_linear"] += b["rms_linear"]
        metrics["_RMS_log"] += b["rms_log"]
        metrics["_SILog"] += b["silog"]
        metrics["_mean_depth_error"] += b["mean_err"]
        metrics["_median_diff"] += b["median_diff"]
        metrics["_threshold_delta_1.25"] += b["d1"]
        metrics["_threshold_delta_1.25^2"] += b["d2"]
        metrics["_threshold_delta_1.25^3"] += b["d3"]

        # Preserve "total metrics" from original script.
        metrics2.append(
            np.array(
                [
                    mse(pred_abs[None, None], tgt_abs[None, None]),
                    abs_rel_diff(pred_abs[None, None], tgt_abs[None, None]),
                    scale_invariant_error(pred_abs[None, None], tgt_abs[None, None]),
                    median_error(pred_abs[None, None], tgt_abs[None, None]),
                    mean_error(pred_abs[None, None], tgt_abs[None, None]),
                    rms_linear(pred_abs[None, None], tgt_abs[None, None]),
                ]
            )
        )

        for d in DEPTH_VALUES:
            m = full_mask & (np.nan_to_num(tgt_abs) < d)
            b = metric_block(tgt_abs, pred_abs, m)
            metrics[f"_{d}_abs_rel_diff"] += b["abs_rel"]
            metrics[f"_{d}_squ_rel_diff"] += b["squ_rel"]
            metrics[f"_{d}_RMS_linear"] += b["rms_linear"]
            metrics[f"_{d}_RMS_log"] += b["rms_log"]
            metrics[f"_{d}_SILog"] += b["silog"]
            metrics[f"_{d}_mean_depth_error"] += b["mean_err"]
            metrics[f"_{d}_median_diff"] += b["median_diff"]
            metrics[f"_{d}_threshold_delta_1.25"] += b["d1"]
            metrics[f"_{d}_threshold_delta_1.25^2"] += b["d2"]
            metrics[f"_{d}_threshold_delta_1.25^3"] += b["d3"]

    with open(args.output_log, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k} : {v / n:.6f}\n")
        f.write("----------------------------------------------\n")
        for _, v in metrics.items():
            f.write(f"{v / n:.6f}\n")
        total = np.sum(np.array(metrics2), 0) / len(metrics2)
        f.write(f"total metrics:  {total}\n")


if __name__ == "__main__":
    main()
