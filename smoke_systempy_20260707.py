"""Standalone smoke test for system-python: dataset -> model -> loss -> backward.

Bypasses train_parallel.py's mp.spawn / DistributedSampler path (env-specific
UCX segfault + unconditional DistributedSampler bug) to validate the real
dataflow on the system interpreter.
"""
import json
import sys
from os.path import join

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from data_loader.SpikesDENSE_dataset import SequenceSynchronizedFramesSpikesDENSEDataset
from model.S2DepthNet import S2DepthTransformerUNetConv
from model.loss import scale_invariant_loss

CONFIG = "configs/train_s2d_spiketransformer_mambaflow_systempy_smoke_1ep_20260424.json"
DATAROOT = "/root/shared-nvme/spikemamba"


def build_one_sequence_dataset(cfg):
    tr = cfg["data_loader"]["train"]
    base = join(DATAROOT, tr["base_folder"])
    # Pick the first sequence subfolder.
    import glob
    seq = sorted(glob.glob(join(base, "*")))
    seq = [s for s in seq if "ipynb" not in s]
    print("sequence folder:", seq[0])
    ds = SequenceSynchronizedFramesSpikesDENSEDataset(
        base_folder=seq[0],
        spike_folder=tr["spike_folder"],
        depth_folder=tr["depth_folder"],
        frame_folder=tr["frame_folder"],
        start_time=0.0, stop_time=0.0,
        sequence_length=cfg["trainer"]["sequence_length"],
        step_size=tr["step_size"],
        clip_distance=tr["clip_distance"],
        normalize=cfg["data_loader"].get("normalize", True),
        scale_factor=tr["scale_factor"],
        every_x_rgb_frame=tr["every_x_rgb_frame"],
        baseline=tr["baseline"],
        loss_composition=cfg["trainer"]["loss_composition"],
    )
    print("dataset length:", len(ds))
    return ds


def main():
    cfg = json.load(open(CONFIG))
    cfg["model"]["gpu"] = 0
    cfg["model"]["every_x_rgb_frame"] = cfg["data_loader"]["train"]["every_x_rgb_frame"]
    cfg["model"]["baseline"] = cfg["data_loader"]["train"]["baseline"]
    cfg["model"]["loss_composition"] = cfg["trainer"]["loss_composition"]

    torch.manual_seed(111)
    device = torch.device("cuda:0")

    model = eval(cfg["arch"])(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", n_params)

    ds = build_one_sequence_dataset(cfg)
    if len(ds) == 0:
        raise RuntimeError("Empty dataset for smoke test.")
    sequence = ds[0]  # list of length L of dicts
    # Model declares spatial_resolution; the real pipeline resizes to it.
    # Raw DENSE frames are 260x346 -> after /2 patch stride give odd W, which
    # breaks 2x2 spatial folding. Resize spike + depth to the configured size.
    import torch.nn.functional as F
    sr = cfg["model"].get("spatial_resolution", [112, 112])
    Ht, Wt = int(sr[0]), int(sr[1])
    for item in sequence:
        for k in list(item.keys()):
            v = item[k]
            if torch.is_tensor(v) and v.ndim == 3:
                mode = "nearest" if "depth" in k else "bilinear"
                kw = {} if mode == "nearest" else {"align_corners": False}
                item[k] = F.interpolate(v.unsqueeze(0).float(), size=(Ht, Wt),
                                        mode=mode, **kw).squeeze(0)
    print("resized to:", (Ht, Wt))
    print("sequence L:", len(sequence), "keys:", list(sequence[0].keys()))
    for k, v in sequence[0].items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    prev_states_lstm = {"image": None}
    prev_super = {"image": None}
    total = None
    for l, item in enumerate(sequence):
        # add batch dim
        batched = {}
        for k, v in item.items():
            batched[k] = v.unsqueeze(0) if torch.is_tensor(v) else v
        pred_dict, super_states, prev_states_lstm = model(
            batched, prev_super["image"], prev_states_lstm
        )
        target = batched["depth_image"].to(device)
        pred = pred_dict["image"]
        loss = scale_invariant_loss(pred, target, weight=1.0, n_lambda=1.0)
        aux = super_states.get("aux_losses", {}) if isinstance(super_states, dict) else {}
        for name, val in aux.items():
            if torch.is_tensor(val):
                loss = loss + val
        print(f"step {l}: pred {tuple(pred.shape)} loss={loss.item():.4f} "
              f"aux={[ (k, round(float(v),4)) for k,v in aux.items() if torch.is_tensor(v)]}")
        total = loss if total is None else total + loss
        prev_super = {k: v for k, v in super_states.items()
                      if k not in ("aux_losses", "aux_outputs")} if isinstance(super_states, dict) else {"image": None}
        if "image" not in prev_super:
            prev_super["image"] = None

    total = total / len(sequence)
    opt.zero_grad()
    total.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
    opt.step()
    print(f"backward OK. total_loss={total.item():.4f} grad_norm={float(gnorm):.4f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
