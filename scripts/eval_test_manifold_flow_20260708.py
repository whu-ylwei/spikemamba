"""Evaluate a trained ablation checkpoint on the TEST split.

Loads model_best.pth.tar for one arm, runs the test sequences, prints depth metrics.
Reuses the training runner's dataset/resize/val logic for consistency.
"""
import argparse
import glob
import json
import os
import sys
from os.path import join

import numpy as np
import torch

sys.path.insert(0, ".")

from data_loader.SpikesDENSE_dataset import SequenceSynchronizedFramesSpikesDENSEDataset
from model.S2DepthNet import S2DepthTransformerUNetConv
from model.metric import (
    abs_rel_diff, squ_rel_diff, rms_linear, scale_invariant_error,
    mean_error, median_error, mse,
)
from scripts.ablation_manifold_flow_20260708 import (
    CONFIG, DATAROOT, resize_sequence, val_loss_only,
)


def build_test(cfg):
    part = cfg["data_loader"]["validation"]  # reuse folder layout (spike/depth/frame folders)
    base = join(DATAROOT, "DENSE/test")
    subs = sorted(s for s in glob.glob(join(base, "*")) if "ipynb" not in s and os.path.isdir(s))
    dss = []
    for s in subs:
        dss.append(SequenceSynchronizedFramesSpikesDENSEDataset(
            base_folder=s, spike_folder=part["spike_folder"], depth_folder=part["depth_folder"],
            frame_folder=part["frame_folder"], start_time=0.0, stop_time=0.0,
            sequence_length=cfg["trainer"]["sequence_length"], step_size=part["step_size"],
            clip_distance=part["clip_distance"], normalize=cfg["data_loader"].get("normalize", True),
            scale_factor=part["scale_factor"], every_x_rgb_frame=part["every_x_rgb_frame"],
            baseline=part["baseline"], loss_composition=cfg["trainer"]["loss_composition"],
        ))
    from torch.utils.data import ConcatDataset
    return ConcatDataset(dss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", choices=["on", "off"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=CONFIG,
                    help="Config to build the model from; must match the arch the ckpt was trained with.")
    ap.add_argument("--max-seq", type=int, default=-1)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    m = cfg["model"]
    m["gpu"] = 0
    m["every_x_rgb_frame"] = cfg["data_loader"]["train"]["every_x_rgb_frame"]
    m["baseline"] = cfg["data_loader"]["train"]["baseline"]
    m["loss_composition"] = cfg["trainer"]["loss_composition"]
    m["manifold_flow"]["enabled"] = (args.flow == "on")

    dev = torch.device("cuda:0")
    model = S2DepthTransformerUNetConv(m).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    ds = build_test(cfg)
    n = len(ds) if args.max_seq < 0 else min(len(ds), args.max_seq)

    acc = {k: [] for k in [
        "test_loss", "abs_rel_diff", "squ_rel_diff", "rms_linear", "rms_log",
        "scale_invariant_error", "mean_error", "median_error", "mse",
        "delta_1.25", "delta_1.25^2", "delta_1.25^3",
    ]}
    eps = 1e-6
    with torch.no_grad():
        for i in range(n):
            seq = resize_sequence(ds[i])
            l, pred, tgt = val_loss_only(model, seq, dev)
            p = pred.detach().cpu().numpy(); t = tgt.detach().cpu().numpy()
            valid = ~np.isnan(t)
            pv, tv = p[valid], t[valid]
            ratio = np.maximum(pv / (tv + eps), tv / (pv + eps))
            log_diff = np.log(tv + eps) - np.log(pv + eps)
            acc["test_loss"].append(l)
            acc["abs_rel_diff"].append(abs_rel_diff(p, t))
            acc["squ_rel_diff"].append(squ_rel_diff(p, t))
            acc["rms_linear"].append(rms_linear(p, t))
            acc["rms_log"].append(float(np.sqrt((log_diff ** 2).mean())))
            acc["scale_invariant_error"].append(scale_invariant_error(p, t))
            acc["mean_error"].append(mean_error(p, t))
            acc["median_error"].append(median_error(p, t))
            acc["mse"].append(mse(p, t))
            acc["delta_1.25"].append(float(np.mean(ratio <= 1.25)))
            acc["delta_1.25^2"].append(float(np.mean(ratio <= 1.25 ** 2)))
            acc["delta_1.25^3"].append(float(np.mean(ratio <= 1.25 ** 3)))

    res = {"flow": args.flow, "ckpt_epoch": ck.get("epoch"), "n_test_seq": n}
    res.update({k: float(np.mean(v)) for k, v in acc.items()})
    print("TEST_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
