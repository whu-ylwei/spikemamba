"""Ablation runner: Manifold Flow ON vs OFF.

Standalone driver (bypasses train_parallel.py's DistributedSampler/UCX blockers,
following the validated pattern in smoke_systempy_20260707.py).

Both arms are identical except config["model"]["manifold_flow"]["enabled"].
Same seed, same data order, same optimizer/clip/seq_len. The FLOW arm adds its
intrinsic aux losses (L_coarse/L_rec/L_flow/L_geo) exactly as SpikeTTrainer does.

Usage:
    python3 scripts/ablation_manifold_flow_20260708.py --flow on  --epochs 40 --out runs/ablation/flow
    python3 scripts/ablation_manifold_flow_20260708.py --flow off --epochs 40 --out runs/ablation/base
"""
import argparse
import glob
import json
import os
import sys
import time
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, ".")

from data_loader.SpikesDENSE_dataset import SequenceSynchronizedFramesSpikesDENSEDataset
from model.S2DepthNet import S2DepthTransformerUNetConv
from model.loss import scale_invariant_loss, multi_scale_grad_loss
from model.metric import abs_rel_diff, rms_linear, scale_invariant_error, median_error

CONFIG = "configs/train_s2d_spiketransformer_mambaflow_bidivim_seq2_40ep_20260423.json"
DATAROOT = "/root/shared-nvme/spikemamba"
SEED = 111
GRAD_CLIP = 1.0
GRAD_LOSS_W = 0.25
SPATIAL = (112, 112)
# 3a: scale-invariance strength for the SI-log loss. 1.0 = fully scale-invariant (legacy);
# <1.0 keeps part of the global-scale penalty to improve absolute accuracy (abs_rel).
# Set from config manifold_flow.si_n_lambda in main().
SI_N_LAMBDA = 1.0


def build_concat(cfg, split_key):
    """ConcatDataset over every sequence subfolder under the split, like train_parallel.py."""
    part = cfg["data_loader"][split_key]
    base = join(DATAROOT, part["base_folder"])
    subs = sorted(s for s in glob.glob(join(base, "*")) if "ipynb" not in s and os.path.isdir(s))
    datasets = []
    for s in subs:
        datasets.append(SequenceSynchronizedFramesSpikesDENSEDataset(
            base_folder=s,
            spike_folder=part["spike_folder"],
            depth_folder=part["depth_folder"],
            frame_folder=part["frame_folder"],
            start_time=0.0, stop_time=0.0,
            sequence_length=cfg["trainer"]["sequence_length"],
            step_size=part["step_size"],
            clip_distance=part["clip_distance"],
            normalize=cfg["data_loader"].get("normalize", True),
            scale_factor=part["scale_factor"],
            every_x_rgb_frame=part["every_x_rgb_frame"],
            baseline=part["baseline"],
            loss_composition=cfg["trainer"]["loss_composition"],
        ))
    return ConcatDataset(datasets), subs


def resize_sequence(sequence):
    """Resize spike + depth tensors to SPATIAL (raw 260x346 breaks 2x2 folding)."""
    Ht, Wt = SPATIAL
    for item in sequence:
        for k in list(item.keys()):
            v = item[k]
            if torch.is_tensor(v) and v.ndim == 3:
                mode = "nearest" if "depth" in k else "bilinear"
                kw = {} if mode == "nearest" else {"align_corners": False}
                item[k] = F.interpolate(v.unsqueeze(0).float(), size=(Ht, Wt),
                                        mode=mode, **kw).squeeze(0)
    return sequence


def batchify(item):
    return {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in item.items()}


def _seq_collate(batch):
    """batch_size=1 passthrough; resize happens here so DataLoader workers do it in parallel."""
    return resize_sequence(batch[0])


def run_sequence(model, sequence, dev, train, opt=None):
    """One temporal sequence: mirrors SpikeTTrainer.forward_pass_sequence loss composition."""
    prev_lstm = {"image": None}
    prev_super = {"image": None}
    total = None
    last_pred = None
    last_tgt = None
    for item in sequence:
        b = batchify(item)
        pred_dict, super_states, prev_lstm = model(b, prev_super["image"], prev_lstm)
        pred = pred_dict["image"]
        tgt = b["depth_image"].to(dev)

        loss = scale_invariant_loss(pred, tgt, weight=1.0, n_lambda=SI_N_LAMBDA)
        loss = loss + GRAD_LOSS_W * multi_scale_grad_loss(pred, tgt)

        aux = super_states.get("aux_losses", {}) if isinstance(super_states, dict) else {}
        aux_out = super_states.get("aux_outputs", {}) if isinstance(super_states, dict) else {}
        # L_coarse: coarse prediction supervised the same way as trainer.
        coarse = aux_out.get("coarse_prediction")
        cw = float(aux_out.get("coarse_weight", 0.0))
        if torch.is_tensor(coarse) and cw > 0.0:
            loss = loss + cw * scale_invariant_loss(coarse, tgt, weight=1.0, n_lambda=SI_N_LAMBDA)
        for _, val in aux.items():
            if torch.is_tensor(val):
                loss = loss + val

        total = loss if total is None else total + loss
        prev_super = {k: v for k, v in super_states.items()
                      if k not in ("aux_losses", "aux_outputs")} if isinstance(super_states, dict) else {"image": None}
        if "image" not in prev_super:
            prev_super["image"] = None
        last_pred, last_tgt = pred, tgt

    total = total / len(sequence)
    if train:
        opt.zero_grad()
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        return float(total.item()), float(gnorm)
    return float(total.item()), last_pred, last_tgt


def val_loss_only(model, sequence, dev):
    """Validation: monitored val_loss = si + 0.25*grad (mean over steps), plus depth metrics on last step."""
    prev_lstm = {"image": None}
    prev_super = {"image": None}
    losses = []
    last_pred = None
    last_tgt = None
    for item in sequence:
        b = batchify(item)
        pred_dict, super_states, prev_lstm = model(b, prev_super["image"], prev_lstm)
        pred = pred_dict["image"]
        tgt = b["depth_image"].to(dev)
        l = scale_invariant_loss(pred, tgt, 1.0, SI_N_LAMBDA) + GRAD_LOSS_W * multi_scale_grad_loss(pred, tgt)
        losses.append(float(l.item()))
        prev_super = {k: v for k, v in super_states.items()
                      if k not in ("aux_losses", "aux_outputs")} if isinstance(super_states, dict) else {"image": None}
        if "image" not in prev_super:
            prev_super["image"] = None
        last_pred, last_tgt = pred, tgt
    return float(np.mean(losses)), last_pred, last_tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", choices=["on", "off"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-train-seq", type=int, default=-1)
    ap.add_argument("--max-val-seq", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=CONFIG,
                    help="Training config JSON (defaults to the seq2 40ep baseline).")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint (model_last.pth.tar) to warm-start weights from (stage-2 continue).")
    ap.add_argument("--lr", type=float, default=None,
                    help="Override optimizer lr (e.g. halved lr for the continue stage).")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader worker processes for parallel spike-file prefetch (I/O bound).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # Append when resuming so both stages accumulate in one epochs.jsonl.
    jsonl = open(join(args.out, "epochs.jsonl"), "a" if args.resume else "w")

    cfg = json.load(open(args.config))
    m = cfg["model"]
    m["gpu"] = 0
    m["every_x_rgb_frame"] = cfg["data_loader"]["train"]["every_x_rgb_frame"]
    m["baseline"] = cfg["data_loader"]["train"]["baseline"]
    m["loss_composition"] = cfg["trainer"]["loss_composition"]
    m["manifold_flow"]["enabled"] = (args.flow == "on")

    # 3a: pick up scale-invariance strength from config (default 1.0 = legacy).
    global SI_N_LAMBDA
    SI_N_LAMBDA = float(m.get("manifold_flow", {}).get("si_n_lambda", 1.0))
    print(f"[cfg] SI_N_LAMBDA={SI_N_LAMBDA}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda:0")

    model = S2DepthTransformerUNetConv(m).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.flow}] trainable params: {n_params}", flush=True)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[{args.flow}] resumed from {args.resume} @ep{start_epoch}", flush=True)

    train_ds, train_subs = build_concat(cfg, "train")
    val_ds, val_subs = build_concat(cfg, "validation")
    n_train = len(train_ds) if args.max_train_seq < 0 else min(len(train_ds), args.max_train_seq)
    n_val = len(val_ds) if args.max_val_seq < 0 else min(len(val_ds), args.max_val_seq)
    print(f"[{args.flow}] train seqs: {n_train}/{len(train_ds)} from {len(train_subs)} subfolders; "
          f"val seqs: {n_val}/{len(val_ds)}", flush=True)

    lr = args.lr if args.lr is not None else cfg["optimizer"]["lr"]
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=cfg["optimizer"]["weight_decay"])
    print(f"[{args.flow}] lr={lr}", flush=True)

    # Fixed data order across arms (seeded permutation of the same indices).
    # DataLoader workers only prefetch/resize; the fixed order is baked into a Subset
    # so ordering is identical to the pre-DataLoader single-process version.
    g = torch.Generator().manual_seed(SEED)
    train_order = torch.randperm(len(train_ds), generator=g).tolist()[:n_train]
    val_order = list(range(n_val))

    train_loader = DataLoader(
        Subset(train_ds, train_order), batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=_seq_collate,
        pin_memory=False, persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        Subset(val_ds, val_order), batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=_seq_collate,
        pin_memory=False, persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    best = {"val_loss": float("inf"), "epoch": -1}
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses = []
        nan_hit = False
        for seq in train_loader:
            tl, gnorm = run_sequence(model, seq, dev, train=True, opt=opt)
            if not np.isfinite(tl):
                nan_hit = True
                print(f"[{args.flow}] NON-FINITE train loss at epoch {epoch}", flush=True)
                break
            tr_losses.append(tl)
        if nan_hit:
            jsonl.write(json.dumps({"epoch": epoch, "status": "nan"}) + "\n")
            jsonl.flush()
            break

        model.eval()
        vl_list = []
        abs_rel, rmse, sie, med = [], [], [], []
        with torch.no_grad():
            for seq in val_loader:
                vl, pred, tgt = val_loss_only(model, seq, dev)
                vl_list.append(vl)
                p = pred.detach().cpu().numpy()
                t = tgt.detach().cpu().numpy()
                abs_rel.append(abs_rel_diff(p, t))
                rmse.append(rms_linear(p, t))
                sie.append(scale_invariant_error(p, t))
                med.append(median_error(p, t))

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(tr_losses)),
            "val_loss": float(np.mean(vl_list)),
            "abs_rel_diff": float(np.mean(abs_rel)),
            "rms_linear": float(np.mean(rmse)),
            "scale_invariant_error": float(np.mean(sie)),
            "median_error": float(np.mean(med)),
            "sec": round(time.time() - t0, 1),
        }
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        print(f"[{args.flow}] ep{epoch:02d} train={rec['train_loss']:.5f} "
              f"val={rec['val_loss']:.5f} abs_rel={rec['abs_rel_diff']:.4f} "
              f"rmse={rec['rms_linear']:.4f} ({rec['sec']}s)", flush=True)

        if rec["val_loss"] < best["val_loss"]:
            best = {**rec}
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "flow": args.flow},
                       join(args.out, "model_best.pth.tar"))
        # Always keep the latest weights so a continue stage can warm-start from here.
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "flow": args.flow},
                   join(args.out, "model_last.pth.tar"))

    summary = {"flow": args.flow, "params": n_params, "epochs_run": epoch,
               "n_train": n_train, "n_val": n_val, "best": best}
    json.dump(summary, open(join(args.out, "summary.json"), "w"), indent=2)
    jsonl.close()
    print(f"[{args.flow}] DONE best_val={best['val_loss']:.5f} @ep{best['epoch']}", flush=True)


if __name__ == "__main__":
    main()
