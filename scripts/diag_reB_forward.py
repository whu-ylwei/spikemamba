"""Diagnostic: compare coarse vs refined (manifold flow) outputs on test data,
using the SAME dataset/resize/forward as training (val_loss_only path).
Goal: determine whether the residual-bridge prediction collapse happens in
the coarse decoder or in the manifold flow refinement, and whether the
training forward gives sane depth on the TEST split.
"""
import json, glob, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from scripts.ablation_manifold_flow_20260708 import (
    CONFIG, DATAROOT, resize_sequence, batchify, SPATIAL,
)
from data_loader.SpikesDENSE_dataset import SequenceSynchronizedFramesSpikesDENSEDataset
from model.S2DepthNet import S2DepthTransformerUNetConv

CFG = "configs/train_s2d_spiketransformer_mambaflow_residual_bridge_40ep_20260710.json"
CK = "runs/flow_reB_40plus40_20260710/model_best.pth.tar"

cfg = json.load(open(CFG))
m = cfg["model"]; m["gpu"] = 0
m["every_x_rgb_frame"] = cfg["data_loader"]["train"]["every_x_rgb_frame"]
m["baseline"] = cfg["data_loader"]["train"]["baseline"]
m["loss_composition"] = cfg["trainer"]["loss_composition"]
dev = torch.device("cuda:0")
model = S2DepthTransformerUNetConv(m).to(dev)
ck = torch.load(CK, map_location=dev, weights_only=False)
sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
model.load_state_dict(sd)
model.eval()
print("loaded ckpt epoch", ck.get("epoch"))


def stat(name, t):
    v = t[~torch.isnan(t)]
    print(f"  {name}: min{v.min():.4f} max{v.max():.4f} mean{v.mean():.4f} median{v.median():.4f}")


# build test dataset the same way eval_test does
part = cfg["data_loader"]["validation"]
base = os.path.join(DATAROOT, "DENSE-spike/DENSE/test")
subs = sorted(s for s in glob.glob(os.path.join(base, "*")) if "ipynb" not in s and os.path.isdir(s))
ds = SequenceSynchronizedFramesSpikesDENSEDataset(
    base_folder=subs[0], spike_folder=part["spike_folder"], depth_folder=part["depth_folder"],
    frame_folder=part["frame_folder"], start_time=0.0, stop_time=0.0,
    sequence_length=cfg["trainer"]["sequence_length"], step_size=part["step_size"],
    clip_distance=part["clip_distance"], normalize=cfg["data_loader"].get("normalize", True),
    scale_factor=part["scale_factor"], every_x_rgb_frame=part["every_x_rgb_frame"],
    baseline=part["baseline"], loss_composition=cfg["trainer"]["loss_composition"],
)
print("test seqs:", len(ds), "seq_len:", cfg["trainer"]["sequence_length"])

# run the training-style forward over a few sequences, maintaining recurrent state
with torch.no_grad():
    for si in [0, 5, 10]:
        seq = resize_sequence(ds[si])
        prev_lstm = {"image": None}
        prev_super = {"image": None}
        for item in seq:
            b = batchify(item)
            pred_dict, super_states, prev_lstm = model(b, prev_super["image"], prev_lstm)
            prev_super = {k: v for k, v in super_states.items()
                          if k not in ("aux_losses", "aux_outputs")} if isinstance(super_states, dict) else {"image": None}
            if "image" not in prev_super:
                prev_super["image"] = None
        pred = pred_dict["image"]
        tgt = b["depth_image"].to(dev)
        aux = super_states.get("aux_outputs", {}) if isinstance(super_states, dict) else {}
        print(f"=== seq {si} (last step) ===")
        stat("refined_pred", pred)
        if "coarse_prediction" in aux and torch.is_tensor(aux["coarse_prediction"]):
            stat("coarse_pred ", aux["coarse_prediction"])
        stat("target     ", tgt)
