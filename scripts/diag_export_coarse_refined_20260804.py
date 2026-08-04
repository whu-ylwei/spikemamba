"""Diagnostic export: dump BOTH coarse (frozen backbone) and refined (coarse+flow delta)
predictions on the TEST split, so we can evaluate each separately and see whether the
flow head actually improves over coarse (residual-bridge health check).

Writes evaluation_DENSE-compatible NPY layout under <out>:
  refined/npy/image/depth_*.npy      + refined/ground_truth/npy/depth_image/frame_*.npy
  coarse/npy/image/depth_*.npy       + coarse/ground_truth/npy/depth_image/frame_*.npy
(GT duplicated in both so each can be evaluated standalone with the same tool.)

Mirrors export_reB_predictions_20260730 preprocessing (config-driven centercrop224).
"""
import argparse, json, glob, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
import scripts.ablation_manifold_flow_20260708 as _abl
from scripts.ablation_manifold_flow_20260708 import DATAROOT, resize_sequence, batchify
from data_loader.SpikesDENSE_dataset import SequenceSynchronizedFramesSpikesDENSEDataset
from model.S2DepthNet import S2DepthTransformerUNetConv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    m = cfg["model"]; m["gpu"] = 0
    m["every_x_rgb_frame"] = cfg["data_loader"]["train"]["every_x_rgb_frame"]
    m["baseline"] = cfg["data_loader"]["train"]["baseline"]
    m["loss_composition"] = cfg["trainer"]["loss_composition"]
    preproc = cfg.get("preproc", {})
    _abl.PREPROC_MODE = str(preproc.get("mode", "resize")).lower()
    if "size" in preproc:
        _abl.SPATIAL = tuple(int(x) for x in preproc["size"])
    elif "spatial_resolution" in m:
        _abl.SPATIAL = tuple(int(x) for x in m["spatial_resolution"])
    print(f"[diag] PREPROC_MODE={_abl.PREPROC_MODE} SPATIAL={_abl.SPATIAL}", flush=True)

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

    dirs = {}
    for tag in ("refined", "coarse"):
        pd = os.path.join(args.out, tag, "npy/image")
        gd = os.path.join(args.out, tag, "ground_truth/npy/depth_image")
        os.makedirs(pd, exist_ok=True); os.makedirs(gd, exist_ok=True)
        dirs[tag] = (pd, gd)

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
                prev_lstm = {"image": None}; prev_super = {"image": None}
                refined = coarse = tgt = None
                for item in seq:
                    b = batchify(item)
                    pred_dict, super_states, prev_lstm = model(b, prev_super["image"], prev_lstm)
                    aux_out = super_states.get("aux_outputs", {}) if isinstance(super_states, dict) else {}
                    prev_super = {k: v for k, v in super_states.items()
                                  if k not in ("aux_losses", "aux_outputs")} if isinstance(super_states, dict) else {"image": None}
                    if "image" not in prev_super:
                        prev_super["image"] = None
                    refined = pred_dict["image"]
                    coarse = aux_out.get("coarse_prediction")
                    tgt = b["depth_image"].to(dev)
                t = tgt[0].detach().cpu().numpy().astype(np.float32)
                r = refined[0].detach().cpu().numpy().astype(np.float32)
                np.save(os.path.join(dirs["refined"][0], f"depth_{idx:010d}.npy"), r)
                np.save(os.path.join(dirs["refined"][1], f"frame_{idx:010d}.npy"), t)
                if torch.is_tensor(coarse):
                    c = coarse[0].detach().cpu().numpy().astype(np.float32)
                    np.save(os.path.join(dirs["coarse"][0], f"depth_{idx:010d}.npy"), c)
                    np.save(os.path.join(dirs["coarse"][1], f"frame_{idx:010d}.npy"), t)
                idx += 1
    print("TOTAL exported:", idx, flush=True)


if __name__ == "__main__":
    main()

