"""Probe A (FAST version) — go/no-go gate for the spike-conditioned flow direction.

Question it answers: does raw spike contain far-field depth information that the frozen
0320 backbone's downsampled latent has LOST? If a tiny CNN reading ONLY raw spike
(bypassing the backbone entirely) can match/beat coarse's far-field abs_rel (0.707),
then spike has usable far info → flow refinement has headroom. If it can't, far-field
failure is a sensor/physics limit → the whole flow-enhancement direction should be dropped.

FAST design (to finish in ~1h): center-crop 224² then downsample to 112², shallow CNN,
subset of train seqs, a few epochs. Goal is the TREND, not a precise number.

Reuses the dataset + preprocessing + evaluation from the existing residual-bridge pipeline.
Standalone (no backbone, no mamba, no manifold_flow).
"""
import argparse, json, glob, os, sys, time
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
import scripts.ablation_manifold_flow_20260708 as abl
from data_loader.SpikesDENSE_dataset import SequenceSynchronizedFramesSpikesDENSEDataset

DATAROOT = "/root/shared-nvme/spikemamba"
SEED = 111

class SpikeDepthProbe(nn.Module):
    """Shallow encoder-decoder: raw spike (C_bins, H, W) -> depth (1, H, W).
    Deliberately small; only tests whether the info is THERE, not SOTA quality."""
    def __init__(self, in_bins, ch=48):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_bins, ch, 3, stride=2, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),      # /2
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.GroupNorm(8, ch * 2), nn.SiLU(),   # /4
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.GroupNorm(8, ch * 2), nn.SiLU(),
        )
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch * 2, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),                 # x2
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),                                          # x4
            nn.Conv2d(ch, 1, 3, padding=1), nn.Sigmoid(),   # depth in normalized [0,1]
        )

    def forward(self, spike):
        return self.dec(self.enc(spike))


def _center_crop(v, s):
    _, H, W = v.shape
    i, j = (H - s) // 2, (W - s) // 2
    return v[:, i:i + s, j:j + s]


def prep(item, size112):
    """center-crop 224 then resize to size (fast). spike->(bins,size,size), depth->(1,size,size)."""
    sp = _center_crop(item["image"].float(), 224)
    dp = _center_crop(item["depth_image"].float(), 224)
    sp = F.interpolate(sp.unsqueeze(0), size=(size112, size112), mode="bilinear", align_corners=False)[0]
    dp = F.interpolate(dp.unsqueeze(0), size=(size112, size112), mode="nearest")[0]
    return sp, dp


def far_weighted_l1(pred, tgt, far_thresh):
    """L1 but only on far-field valid pixels (tgt in normalized (far_thresh,1], >0)."""
    mask = (tgt > far_thresh) & (tgt > 0)
    if mask.sum() < 1:
        return (pred - tgt).abs().mean() * 0.0
    return (pred - tgt).abs()[mask].mean()


def build_seqs(cfg, split, n_max):
    part = cfg["data_loader"][split]
    base = join(DATAROOT, "DENSE-spike/DENSE", "train" if split == "train" else "test")
    subs = sorted(s for s in glob.glob(join(base, "*")) if "ipynb" not in s and os.path.isdir(s))
    items = []
    for sub in subs:
        ds = SequenceSynchronizedFramesSpikesDENSEDataset(
            base_folder=sub, spike_folder=part["spike_folder"], depth_folder=part["depth_folder"],
            frame_folder=part["frame_folder"], start_time=0.0, stop_time=0.0,
            sequence_length=cfg["trainer"]["sequence_length"], step_size=part["step_size"],
            clip_distance=part["clip_distance"], normalize=cfg["data_loader"].get("normalize", True),
            scale_factor=part["scale_factor"], every_x_rgb_frame=part["every_x_rgb_frame"],
            baseline=part["baseline"], loss_composition=cfg["trainer"]["loss_composition"])
        for si in range(len(ds)):
            items.append((sub, si))
            if len(items) >= n_max:
                return items, subs, part
    return items, subs, part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_s2d_spiketransformer_mambaflow_residual_bridge_latentfix_20260803.json")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--far-thresh", type=float, default=0.7, help="normalized-depth far-field cutoff")
    ap.add_argument("--out", default="runs/probeA_spike_farfield_20260805")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda:0")
    cfg = json.load(open(args.config))
    in_bins = cfg["model"].get("num_bins_rgb", cfg["model"].get("num_bins_events", 128))

    # Fixed subsets (seeded) — cache tensors in host RAM (small at 112²) to avoid re-reading CephFS.
    tr_items, _, tr_part = build_seqs(cfg, "train", args.n_train)
    va_items, _, _ = build_seqs(cfg, "validation", args.n_val)
    print(f"[probeA] train {len(tr_items)} / val {len(va_items)} seqs, size={args.size}, far>{args.far_thresh}", flush=True)

    def load(items, part, split):
        base = join(DATAROOT, "DENSE-spike/DENSE", "train" if split == "train" else "test")
        cache = {}
        out = []
        for sub, si in items:
            if sub not in cache:
                cache[sub] = SequenceSynchronizedFramesSpikesDENSEDataset(
                    base_folder=sub, spike_folder=part["spike_folder"], depth_folder=part["depth_folder"],
                    frame_folder=part["frame_folder"], start_time=0.0, stop_time=0.0,
                    sequence_length=cfg["trainer"]["sequence_length"], step_size=part["step_size"],
                    clip_distance=part["clip_distance"], normalize=cfg["data_loader"].get("normalize", True),
                    scale_factor=part["scale_factor"], every_x_rgb_frame=part["every_x_rgb_frame"],
                    baseline=part["baseline"], loss_composition=cfg["trainer"]["loss_composition"])
            seq = cache[sub][si]
            sp, dp = prep(seq[-1], args.size)   # last frame of the length-2 seq
            out.append((sp, dp))
        return out

    t0 = time.time()
    tr = load(tr_items, tr_part, "train")
    va = load(va_items, tr_part, "validation")
    print(f"[probeA] data loaded in {time.time()-t0:.0f}s", flush=True)

    model = SpikeDepthProbe(in_bins).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[probeA] params={sum(p.numel() for p in model.parameters())}", flush=True)

    def evaluate():
        model.eval()
        far_l1, all_l1, far_delta_n, far_delta_d = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for sp, dp in va:
                sp, dp = sp.unsqueeze(0).to(dev), dp.unsqueeze(0).to(dev)
                pr = model(sp)
                m_all = dp > 0
                m_far = (dp > args.far_thresh) & (dp > 0)
                all_l1 += float((pr - dp).abs()[m_all].mean())
                if m_far.sum() > 0:
                    far_l1 += float((pr - dp).abs()[m_far].mean())
                    # abs_rel proxy in normalized space + delta<1.25^0.1 proxy (normalized ratio)
                    p, t = pr[m_far], dp[m_far]
                    ratio = torch.maximum(p / (t + 1e-6), t / (p + 1e-6))
                    far_delta_n += float((ratio < 1.25).float().sum())
                    far_delta_d += int(m_far.sum())
        n = len(va)
        return all_l1 / n, far_l1 / n, (far_delta_n / max(1, far_delta_d))

    for ep in range(1, args.epochs + 1):
        model.train(); te = time.time(); losses = []
        idx = torch.randperm(len(tr))
        for i in idx:
            sp, dp = tr[i]
            sp, dp = sp.unsqueeze(0).to(dev), dp.unsqueeze(0).to(dev)
            pr = model(sp)
            # far-weighted + light all-pixel term so it still learns global structure
            loss = far_weighted_l1(pr, dp, args.far_thresh) + 0.2 * (pr - dp).abs()[dp > 0].mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        all_l1, far_l1, far_delta = evaluate()
        print(f"[probeA] ep{ep:02d} train_loss={np.mean(losses):.4f} val_far_L1={far_l1:.4f} "
              f"val_all_L1={all_l1:.4f} far_delta<1.25(proxy)={far_delta:.3f} ({time.time()-te:.0f}s)", flush=True)

    print("[probeA] DONE. Interpretation: compare val_far_L1 trend + far_delta proxy. "
          "If far metrics keep improving and far_delta approaches coarse's _500 δ=0.636, "
          "spike HAS usable far info (GO). If far metrics plateau far worse, likely sensor limit (NO-GO).", flush=True)


if __name__ == "__main__":
    main()


