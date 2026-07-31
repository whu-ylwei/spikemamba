"""End-to-end export of residual-bridge (manifold flow) predictions on the TEST split,
using the SAME preprocessing/forward as training (resize to 112x112, recurrent state).

Writes normalized-depth NPYs compatible with evaluation_DENSE.py:
  <out>/npy/image/depth_XXXXXXXXXX.npy            (prediction, [1,H,W])
  <out>/ground_truth/npy/depth_image/frame_XXXXXXXXXX.npy   (GT, [1,H,W])

This fixes the resolution mismatch bug: test_DENSE.py fed 224x224 CenterCrop to a
model trained on 112x112 resized inputs, collapsing predictions. Here we resize
exactly like scripts.ablation_manifold_flow_20260708.resize_sequence.
"""
import argparse, json, glob, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
import scripts.ablation_manifold_flow_20260708 as _abl
from scripts.ablation_manifold_flow_20260708 import (
    DATAROOT, resize_sequence, batchify, SPATIAL,
)
from data_loader.SpikesDENSE_dataset import SequenceSynchronizedFramesSpikesDENSEDataset
from model.S2DepthNet import S2DepthTransformerUNetConv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_s2d_spiketransformer_mambaflow_residual_bridge_40ep_20260710.json")
    ap.add_argument("--ckpt", default="runs/flow_reB_40plus40_20260710/model_best.pth.tar")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=2, help="skip first N steps per sequence (settle recurrent state)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    m = cfg["model"]; m["gpu"] = 0
    m["every_x_rgb_frame"] = cfg["data_loader"]["train"]["every_x_rgb_frame"]
    m["baseline"] = cfg["data_loader"]["train"]["baseline"]
    m["loss_composition"] = cfg["trainer"]["loss_composition"]

    # Mirror training preprocessing (resize vs centercrop, target size) so export口径
    # matches exactly. resize_sequence reads these module globals at call time.
    preproc = cfg.get("preproc", {})
    _abl.PREPROC_MODE = str(preproc.get("mode", "resize")).lower()
    if "size" in preproc:
        _abl.SPATIAL = tuple(int(x) for x in preproc["size"])
    elif "spatial_resolution" in m:
        _abl.SPATIAL = tuple(int(x) for x in m["spatial_resolution"])
    print(f"[export] PREPROC_MODE={_abl.PREPROC_MODE} SPATIAL={_abl.SPATIAL}", flush=True)

    dev = torch.device("cuda:0")
    model = S2DepthTransformerUNetConv(m).to(dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    print("loaded ckpt epoch", ck.get("epoch"), flush=True)

    part = cfg["data_loader"]["validation"]
    base = os.path.join(DATAROOT, "DENSE-spike/DENSE/test")
    subs = sorted(s for s in glob.glob(os.path.join(base, "*")) if "ipynb" not in s and os.path.isdir(s))

    pred_dir = os.path.join(args.out, "npy/image")
    gt_dir = os.path.join(args.out, "ground_truth/npy/depth_image")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    idx = 0
    with torch.no_grad():
        for sub in subs:
            ds = SequenceSynchronizedFramesSpikesDENSEDataset(
                base_folder=sub, spike_folder=part["spike_folder"], depth_folder=part["depth_folder"],
                frame_folder=part["frame_folder"], start_time=0.0, stop_time=0.0,
                sequence_length=cfg["trainer"]["sequence_length"], step_size=part["step_size"],
                clip_distance=part["clip_distance"], normalize=cfg["data_loader"].get("normalize", True),
                scale_factor=part["scale_factor"], every_x_rgb_frame=part["every_x_rgb_frame"],
                baseline=part["baseline"], loss_composition=cfg["trainer"]["loss_composition"],
            )
            print(f"sub {os.path.basename(sub)}: {len(ds)} seqs", flush=True)
            for si in range(len(ds)):
                seq = resize_sequence(ds[si])
                prev_lstm = {"image": None}
                prev_super = {"image": None}
                pred = tgt = None
                for step, item in enumerate(seq):
                    b = batchify(item)
                    pred_dict, super_states, prev_lstm = model(b, prev_super["image"], prev_lstm)
                    prev_super = {k: v for k, v in super_states.items()
                                  if k not in ("aux_losses", "aux_outputs")} if isinstance(super_states, dict) else {"image": None}
                    if "image" not in prev_super:
                        prev_super["image"] = None
                    pred = pred_dict["image"]
                    tgt = b["depth_image"].to(dev)
                # take last step of each (length-2) sequence; sequences already slide by step_size
                p = pred[0].detach().cpu().numpy().astype(np.float32)   # [1,H,W]
                t = tgt[0].detach().cpu().numpy().astype(np.float32)
                np.save(os.path.join(pred_dir, f"depth_{idx:010d}.npy"), p)
                np.save(os.path.join(gt_dir, f"frame_{idx:010d}.npy"), t)
                idx += 1
    print("TOTAL exported:", idx, flush=True)


if __name__ == "__main__":
    main()
