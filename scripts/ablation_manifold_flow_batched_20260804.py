"""Batched stage-2 flow trainer (2026-08-04).

Purpose: saturate the GPU by batching N temporal sequences per step, instead of
the bs=1 per-sequence loop in ablation_manifold_flow_20260708.py. The original
script is left untouched so historical runs stay reproducible.

What is batched: ONLY the train loop. The val loop stays bs=1 and reuses
abl.val_loss_only + the same metrics, so val_loss/abs_rel/rmse keep the exact
historical口径 (directly comparable to prior residual-bridge runs and 0320).

Loss-semantics guard: scale_invariant_loss (model/loss.py) flattens all pixels
and takes a GLOBAL mean, so its (mean)^2 scale-invariance term is per-image only
at bs=1. To preserve identical semantics at bs=N we compute the SI term
per-sample and average (si_per_image below). All other aux losses are plain
means with no cross-sample term, so they are batch-safe as-is.

Reuses model/backbone/AE setup, build_concat, resize_sequence, val_loss_only,
and metrics from the 0708 module. lr / freeze / ae-ckpt semantics unchanged.
"""
import argparse
import gc
import json
import os
import sys
import time
from os.path import join

import numpy as np
import torch

sys.path.insert(0, ".")

import scripts.ablation_manifold_flow_20260708 as abl
from model.S2DepthNet import S2DepthTransformerUNetConv
from model.loss import scale_invariant_loss, multi_scale_grad_loss

def batched_collate(batch):
    """batch = list of N sequences; each sequence is a list of L frame-dicts.
    Resize each sequence (workers do this in parallel), then stack the N frames
    at each time index into tensors of shape [N, C, H, W]. Returns a list of L
    'batched frames' (dicts of [N,...] tensors). Assumes all sequences share the
    same length L (true: sequence_length is fixed in config)."""
    seqs = [abl.resize_sequence(s) for s in batch]
    L = len(seqs[0])
    out = []
    for l in range(L):
        frames = [s[l] for s in seqs]
        merged = {}
        for k, v in frames[0].items():
            if torch.is_tensor(v):
                merged[k] = torch.stack([f[k] for f in frames], dim=0)  # [N,C,H,W]
            else:
                merged[k] = v
        out.append(merged)
    return out


def si_per_image(pred, tgt, n_lambda):
    """Per-sample scale-invariant loss averaged over the batch, preserving the
    exact bs=1 semantics (each image gets its own scale-normalization)."""
    n = pred.shape[0]
    return sum(scale_invariant_loss(pred[i:i + 1], tgt[i:i + 1], 1.0, n_lambda)
               for i in range(n)) / n


def run_batched(model, batched_frames, dev, opt):
    """One batched training step over a temporal sequence with batch dim N.
    Mirrors abl.run_sequence loss composition, but SI is computed per-image."""
    prev_lstm = {"image": None}
    prev_super = {"image": None}
    total = None
    for b in batched_frames:
        b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        pred_dict, super_states, prev_lstm = model(b, prev_super["image"], prev_lstm)
        pred = pred_dict["image"]
        tgt = b["depth_image"].to(dev)

        loss = si_per_image(pred, tgt, abl.SI_N_LAMBDA)
        loss = loss + abl.GRAD_LOSS_W * multi_scale_grad_loss(pred, tgt)

        aux = super_states.get("aux_losses", {}) if isinstance(super_states, dict) else {}
        aux_out = super_states.get("aux_outputs", {}) if isinstance(super_states, dict) else {}
        coarse = aux_out.get("coarse_prediction")
        cw = float(aux_out.get("coarse_weight", 0.0))
        if torch.is_tensor(coarse) and cw > 0.0:
            loss = loss + cw * si_per_image(coarse, tgt, abl.SI_N_LAMBDA)
        for _, val in aux.items():
            if torch.is_tensor(val):
                loss = loss + val

        total = loss if total is None else total + loss
        prev_super = {k: v for k, v in super_states.items()
                      if k not in ("aux_losses", "aux_outputs")} if isinstance(super_states, dict) else {"image": None}
        if "image" not in prev_super:
            prev_super["image"] = None

    total = total / len(batched_frames)
    opt.zero_grad()
    total.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), abl.GRAD_CLIP)
    opt.step()
    return float(total.item()), float(gnorm)


def build_model_and_data(args):
    """Replicates the 0708 main() setup (config, model, backbone/AE load, freeze,
    datasets) via the abl module globals so behavior is identical."""
    cfg = json.load(open(args.config))
    m = cfg["model"]
    m["gpu"] = 0
    m["every_x_rgb_frame"] = cfg["data_loader"]["train"]["every_x_rgb_frame"]
    m["baseline"] = cfg["data_loader"]["train"]["baseline"]
    m["loss_composition"] = cfg["trainer"]["loss_composition"]
    m["manifold_flow"]["enabled"] = (args.flow == "on")

    abl.SI_N_LAMBDA = float(m.get("manifold_flow", {}).get("si_n_lambda", 1.0))
    print(f"[cfg] SI_N_LAMBDA={abl.SI_N_LAMBDA}", flush=True)
    preproc = cfg.get("preproc", {})
    abl.PREPROC_MODE = str(preproc.get("mode", "resize")).lower()
    if "size" in preproc:
        abl.SPATIAL = tuple(int(x) for x in preproc["size"])
    elif "spatial_resolution" in m:
        abl.SPATIAL = tuple(int(x) for x in m["spatial_resolution"])
    print(f"[cfg] PREPROC_MODE={abl.PREPROC_MODE} SPATIAL={abl.SPATIAL}", flush=True)

    torch.manual_seed(abl.SEED)
    np.random.seed(abl.SEED)
    dev = torch.device("cuda:0")
    model = S2DepthTransformerUNetConv(m).to(dev)

    if args.backbone_ckpt:
        bck = torch.load(args.backbone_ckpt, map_location=dev, weights_only=False)
        bsd = bck.get("state_dict", bck)
        bsd = {(k[7:] if k.startswith("module.") else k): v for k, v in bsd.items()}
        missing, unexpected = model.load_state_dict(bsd, strict=False)
        other_missing = [k for k in missing if not k.startswith("manifold_flow.")]
        assert not other_missing and not unexpected, \
            f"backbone load mismatch: missing={other_missing[:3]} unexpected={unexpected[:3]}"
        print(f"[{args.flow}] backbone warm-start from {args.backbone_ckpt} "
              f"({len(bsd)} keys; {len(missing)} manifold keys at init)", flush=True)
    if args.ae_ckpt:
        ae = torch.load(args.ae_ckpt, map_location=dev, weights_only=False)
        mf = model.manifold_flow
        mf.condition_encoder.load_state_dict(ae["condition_encoder"])
        mf.recon_decoder.load_state_dict(ae["recon_decoder"])
        if not mf.share_encoder and "target_encoder" in ae:
            mf.target_encoder.load_state_dict(ae["target_encoder"])
        if mf.variational and "quant_conv" in ae:
            mf.quant_conv.load_state_dict(ae["quant_conv"])
        print(f"[{args.flow}] AE loaded from {args.ae_ckpt} @ep{ae.get('epoch')}", flush=True)
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("manifold_flow."):
                p.requires_grad_(False)
        print(f"[{args.flow}] backbone frozen (only manifold_flow trainable)", flush=True)
    print(f"[{args.flow}] trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)

    train_ds, train_subs = abl.build_concat(cfg, "train")
    val_ds, val_subs = abl.build_concat(cfg, "validation")
    return cfg, m, model, dev, train_ds, val_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", choices=["on", "off"], default="on")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Number of temporal sequences batched per train step.")
    ap.add_argument("--backbone-ckpt", default=None)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--ae-ckpt", default=None)
    ap.add_argument("--max-train-seq", type=int, default=-1)
    ap.add_argument("--max-val-seq", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true",
                    help="Run 1 epoch on a few sequences, report peak GPU mem, then exit.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg, m, model, dev, train_ds, val_ds = build_model_and_data(args)

    from torch.utils.data import DataLoader, Subset
    n_train = len(train_ds) if args.max_train_seq < 0 else min(len(train_ds), args.max_train_seq)
    n_val = len(val_ds) if args.max_val_seq < 0 else min(len(val_ds), args.max_val_seq)
    g = torch.Generator().manual_seed(abl.SEED)
    train_order = torch.randperm(len(train_ds), generator=g).tolist()[:n_train]
    val_order = list(range(n_val))
    print(f"[{args.flow}] train seqs: {n_train}/{len(train_ds)}; val seqs: {n_val}/{len(val_ds)}; "
          f"batch_size={args.batch_size}", flush=True)

    # Train: batched. Drop last partial batch so every step is full-size (stable timing/mem).
    # Host-RAM guard: spike tensors are 46MB/frame → a bs=N batch is huge, and the container
    # cgroup memory.max is 64GB. Two OOMs (nw=8/pf=2, then nw=3 persistent) traced to
    # persistent_workers=True: workers stay alive across epochs and their CephFS page-cache
    # references accumulate until OOM at ep03. Fix: persistent_workers=False (workers +
    # their cache are torn down every epoch), prefetch_factor=1, and gc + cache drop between
    # epochs (see main loop). Val also non-persistent, prefetch=1.
    train_loader = DataLoader(
        Subset(train_ds, train_order), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=batched_collate, drop_last=True,
        pin_memory=False, persistent_workers=False,
        prefetch_factor=(1 if args.num_workers > 0 else None))
    # Val: bs=1, identical to historical口径 (reuses abl.val_loss_only + metrics).
    val_loader = DataLoader(
        Subset(val_ds, val_order), batch_size=1, shuffle=False,
        num_workers=min(2, args.num_workers), collate_fn=abl._seq_collate,
        pin_memory=False, persistent_workers=False,
        prefetch_factor=(1 if args.num_workers > 0 else None))

    lr = args.lr if args.lr is not None else cfg["optimizer"]["lr"]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr,
                           weight_decay=cfg["optimizer"]["weight_decay"])
    print(f"[{args.flow}] lr={lr}", flush=True)

    if args.smoke:
        model.train(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i, seq in enumerate(train_loader):
            tl, gn = run_batched(model, seq, dev, opt)
            print(f"[smoke] step{i} loss={tl:.5f} gnorm={gn:.3f} "
                  f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
            if i >= 3:
                break
        print(f"[smoke] bs={args.batch_size} peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
              f"{(time.time()-t0):.1f}s for {i+1} steps", flush=True)
        return

    jsonl = open(join(args.out, "epochs.jsonl"), "w")
    best = {"val_loss": float("inf"), "epoch": -1}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses = []
        for seq in train_loader:
            tl, gnorm = run_batched(model, seq, dev, opt)
            if not np.isfinite(tl):
                print(f"[{args.flow}] NON-FINITE train loss at ep{epoch}", flush=True)
                break
            tr_losses.append(tl)

        model.eval()
        vl_list, abs_rel, rmse, sie, med = [], [], [], [], []
        with torch.no_grad():
            for seq in val_loader:
                vl, pred, tgt = abl.val_loss_only(model, seq, dev)
                vl_list.append(vl)
                p, t = pred.detach().cpu().numpy(), tgt.detach().cpu().numpy()
                abs_rel.append(abl.abs_rel_diff(p, t)); rmse.append(abl.rms_linear(p, t))
                sie.append(abl.scale_invariant_error(p, t)); med.append(abl.median_error(p, t))
        rec = {"epoch": epoch, "train_loss": float(np.mean(tr_losses)),
               "val_loss": float(np.mean(vl_list)), "abs_rel_diff": float(np.mean(abs_rel)),
               "rms_linear": float(np.mean(rmse)), "scale_invariant_error": float(np.mean(sie)),
               "median_error": float(np.mean(med)), "sec": round(time.time() - t0, 1)}
        jsonl.write(json.dumps(rec) + "\n"); jsonl.flush()
        print(f"[{args.flow}] ep{epoch:02d} train={rec['train_loss']:.5f} val={rec['val_loss']:.5f} "
              f"abs_rel={rec['abs_rel_diff']:.4f} rmse={rec['rms_linear']:.4f} ({rec['sec']}s)", flush=True)

        if rec["val_loss"] < best["val_loss"]:
            best = {**rec}
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "flow": args.flow},
                       join(args.out, "model_best.pth.tar"))
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "flow": args.flow},
                   join(args.out, "model_last.pth.tar"))
        # Host-RAM hygiene: reclaim Python garbage + CUDA cache each epoch so page-cache
        # / worker leftovers don't accumulate toward the 64GB cgroup limit.
        gc.collect(); torch.cuda.empty_cache()
        try:
            cur = int(open("/sys/fs/cgroup/memory.current").read()) / 1e9
            print(f"[mem] after ep{epoch}: cgroup.current={cur:.1f}GB / 64.4GB", flush=True)
        except Exception:
            pass
    json.dump({"best": best, "n_train": n_train, "n_val": n_val},
              open(join(args.out, "summary.json"), "w"), indent=2)
    jsonl.close()
    print(f"[{args.flow}] DONE best_val={best['val_loss']:.5f} @ep{best['epoch']}", flush=True)


if __name__ == "__main__":
    main()



