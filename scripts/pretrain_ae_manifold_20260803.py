"""Stage 1 (latent-manifold fix, 2026-08-03): pretrain the depth autoencoder.

See LATENT_MANIFOLD_FIX_20260803.md. We fix the latent manifold BEFORE training the
flow: treat (manifold_flow.condition_encoder + manifold_flow.recon_decoder) as a plain
autoencoder and train it with a large-weight reconstruction L1 until it converges.

Trained on BOTH gt depth AND the frozen-backbone coarse depth, so the shared encoder
encodes both distributions well (z_coarse and z_gt land on the same good manifold).

Outputs manifold_flow.condition_encoder + recon_decoder weights to <out>/ae_pretrained.pth.tar
(stage 2 loads these and freezes them via freeze_autoencoder=True).

Usage:
    python3 scripts/pretrain_ae_manifold_20260803.py \
        --config configs/train_..._latentfix_20260803.json \
        --backbone-ckpt bivmamba_best/checkpoint/model_best_ep70.pth.tar \
        --epochs 30 --lr 2e-4 --out runs/ae_pretrain_latentfix_20260803
"""
import argparse
import json
import os
import sys
import time
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, ".")

import scripts.ablation_manifold_flow_20260708 as abl
from model.S2DepthNet import S2DepthTransformerUNetConv
from model.manifold_flow import _masked_l1, _sanitize_depth

SEED = abl.SEED


def load_backbone(model, ckpt_path, dev):
    """Load 0320 MambaSSM weights into the S2DepthNet backbone (strict=False:
    the 68 manifold_flow keys are missing_keys and stay at init). Verified: 306
    backbone keys match exactly with zero shape conflict."""
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    manifold_missing = [k for k in missing if k.startswith("manifold_flow.")]
    other_missing = [k for k in missing if not k.startswith("manifold_flow.")]
    assert not other_missing, f"backbone keys unexpectedly missing: {other_missing[:5]}"
    assert not unexpected, f"unexpected keys in ckpt: {unexpected[:5]}"
    print(f"[ae] backbone loaded: {len(sd)} keys, {len(manifold_missing)} manifold keys left at init", flush=True)


def freeze_backbone(model):
    """Freeze everything except the manifold_flow module (encoder+recon_decoder are
    trained here; velocity_field/residual_decoder are untouched but harmless)."""
    for name, p in model.named_parameters():
        if not name.startswith("manifold_flow."):
            p.requires_grad_(False)


def ae_forward_loss(mf, depth, dev):
    """Autoencoder reconstruction loss: ||recon_decoder(encoder(depth)) - depth||_1
    over valid pixels, plus (if variational) kl_weight * KL(q(z|x)||N(0,I)).
    Uses the same split-decoder recon + VAE encode path as training."""
    clean, mask = _sanitize_depth(depth.to(dev))
    latent, kl = mf._encode_latent(mf.condition_encoder, clean, sample=True)  # shared encoder (E)
    recon = mf._decode_recon(latent)              # recon_decoder (D_rec)
    rec_l1 = _masked_l1(recon, clean, mask)
    if mf.variational and mf.kl_weight > 0.0 and kl is not None:
        return rec_l1 + mf.kl_weight * kl, rec_l1.detach(), kl.detach()
    return rec_l1, rec_l1.detach(), (kl.detach() if kl is not None else None)


@torch.no_grad()
def coarse_from_backbone(model, seq, dev):
    """Run the frozen backbone over a temporal sequence, returning the coarse depth
    predictions (one per frame). Mirrors S2DepthNet.forward up to coarse_prediction."""
    coarses = []
    prev_lstm = {"image": None}
    for item in seq:
        b = abl.batchify(item)
        # Reuse the model forward but grab coarse via aux_outputs (manifold on).
        pred_dict, super_states, prev_lstm = model(b, None, prev_lstm)
        aux_out = super_states.get("aux_outputs", {}) if isinstance(super_states, dict) else {}
        c = aux_out.get("coarse_prediction")
        if torch.is_tensor(c):
            coarses.append(c.detach())
    return coarses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone-ckpt", default="bivmamba_best/checkpoint/model_best_ep70.pth.tar")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--train-on-coarse", choices=["on", "off"], default="on",
                    help="Also reconstruct frozen-backbone coarse depth (default on) so the "
                         "encoder encodes the coarse distribution seen at inference.")
    ap.add_argument("--max-train-seq", type=int, default=-1)
    ap.add_argument("--max-val-seq", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jsonl = open(join(args.out, "epochs.jsonl"), "w")

    cfg = json.load(open(args.config))
    m = cfg["model"]
    m["gpu"] = 0
    m["every_x_rgb_frame"] = cfg["data_loader"]["train"]["every_x_rgb_frame"]
    m["baseline"] = cfg["data_loader"]["train"]["baseline"]
    m["loss_composition"] = cfg["trainer"]["loss_composition"]
    m["manifold_flow"]["enabled"] = True
    # AE pretrain requires the split recon_decoder to exist.
    m["manifold_flow"]["split_decoders"] = True
    m["manifold_flow"]["freeze_autoencoder"] = False  # we train the AE here

    # preproc口径 (centercrop224) — set the same module globals ablation uses.
    preproc = cfg.get("preproc", {"mode": "resize", "size": m["spatial_resolution"]})
    abl.PREPROC_MODE = str(preproc.get("mode", "resize")).lower()
    abl.SPATIAL = tuple(int(x) for x in preproc.get("size", m["spatial_resolution"]))
    print(f"[ae] PREPROC_MODE={abl.PREPROC_MODE} SPATIAL={abl.SPATIAL}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda:0")

    model = S2DepthTransformerUNetConv(m).to(dev)
    load_backbone(model, args.backbone_ckpt, dev)
    freeze_backbone(model)
    mf = model.manifold_flow

    # Only the AE (encoder + recon_decoder [+ quant_conv if variational]) is optimized.
    # residual_decoder/velocity_field exist but get no gradient (never used in the AE loss).
    ae_params = list(mf.condition_encoder.parameters()) + list(mf.recon_decoder.parameters())
    if not mf.share_encoder:
        ae_params += list(mf.target_encoder.parameters())
    if mf.variational:
        ae_params += list(mf.quant_conv.parameters())
    n_ae = sum(p.numel() for p in ae_params)
    opt = torch.optim.Adam(ae_params, lr=args.lr, weight_decay=0.0)
    print(f"[ae] trainable AE params: {n_ae}  lr={args.lr}  "
          f"variational={mf.variational} kl_weight={mf.kl_weight}", flush=True)

    train_ds, _ = abl.build_concat(cfg, "train")
    val_ds, _ = abl.build_concat(cfg, "validation")
    n_train = len(train_ds) if args.max_train_seq < 0 else min(len(train_ds), args.max_train_seq)
    n_val = len(val_ds) if args.max_val_seq < 0 else min(len(val_ds), args.max_val_seq)
    g = torch.Generator().manual_seed(SEED)
    train_order = torch.randperm(len(train_ds), generator=g).tolist()[:n_train]
    val_order = list(range(n_val))
    mk = lambda ds, order: DataLoader(
        Subset(ds, order), batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=abl._seq_collate, pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None))
    train_loader = mk(train_ds, train_order)
    val_loader = mk(val_ds, val_order)
    print(f"[ae] train seqs {n_train}  val seqs {n_val}", flush=True)

    def run_seq(seq, train):
        # GT depth targets for every frame; optionally add frozen-backbone coarse.
        depths = [abl.batchify(item)["depth_image"] for item in seq]
        coarses = coarse_from_backbone(model, seq, dev) if args.train_on_coarse == "on" else []
        targets = depths + coarses
        recs, kls = [], []
        for d in targets:
            loss, rec_l1, kl = ae_forward_loss(mf, d, dev)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            recs.append(float(rec_l1))
            if kl is not None:
                kls.append(float(kl))
        return (float(np.mean(recs)) if recs else 0.0,
                float(np.mean(kls)) if kls else 0.0)

    best = {"val_rec": float("inf"), "epoch": -1}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.eval()          # backbone frozen; AE modules have no BN/dropout that matter
        mf.condition_encoder.train(); mf.recon_decoder.train()
        tr = [run_seq(seq, train=True) for seq in train_loader]
        mf.condition_encoder.eval(); mf.recon_decoder.eval()
        with torch.no_grad():
            vl = [run_seq(seq, train=False) for seq in val_loader]
        rec = {"epoch": epoch,
               "train_rec": float(np.mean([r for r, _ in tr])),
               "train_kl": float(np.mean([k for _, k in tr])),
               "val_rec": float(np.mean([r for r, _ in vl])),
               "val_kl": float(np.mean([k for _, k in vl])),
               "sec": round(time.time() - t0, 1)}
        jsonl.write(json.dumps(rec) + "\n"); jsonl.flush()
        print(f"[ae] ep{epoch:02d} train_rec={rec['train_rec']:.5f} val_rec={rec['val_rec']:.5f} "
              f"train_kl={rec['train_kl']:.3f} val_kl={rec['val_kl']:.3f} ({rec['sec']}s)", flush=True)

        save = {"epoch": epoch,
                "condition_encoder": mf.condition_encoder.state_dict(),
                "recon_decoder": mf.recon_decoder.state_dict(),
                "share_encoder": mf.share_encoder,
                "variational": mf.variational}
        if not mf.share_encoder:
            save["target_encoder"] = mf.target_encoder.state_dict()
        if mf.variational:
            save["quant_conv"] = mf.quant_conv.state_dict()
        torch.save(save, join(args.out, "ae_last.pth.tar"))
        # Monitor pure reconstruction (manifold quality), not rec+KL.
        if rec["val_rec"] < best["val_rec"]:
            best = {**rec}
            torch.save(save, join(args.out, "ae_pretrained.pth.tar"))

    json.dump({"best": best, "n_train": n_train, "n_val": n_val},
              open(join(args.out, "summary.json"), "w"), indent=2)
    jsonl.close()
    print(f"[ae] DONE best_val_rec={best['val_rec']:.5f} @ep{best['epoch']}", flush=True)


if __name__ == "__main__":
    main()
