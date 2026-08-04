"""CPU-only diagnostic: why do 0320 baseline and residual-bridge have similar
delta<1.25 but very different abs_rel? Decompose abs_rel by depth bin and by
error tail. Replicates evaluation_DENSE.prepare_depth_data (reg=5.7, clip=1000),
NO median rescale (matches historical runs)."""
import glob, os, re, sys
import numpy as np

REG = 5.7
CLIP = 1000.0

def decode(x):
    # exp decode normalized[0,1] -> absolute meters, then clip (evaluation_DENSE)
    x = np.exp(REG * (x - 1.0)) * CLIP
    return np.clip(x, np.exp(-REG) * CLIP, CLIP)

def frame_id(path):
    m = re.search(r"(\d{7,})", os.path.basename(path))
    return int(m.group(1)) if m else None

def load_map(d, sub):
    out = {}
    for f in glob.glob(os.path.join(d, sub, "*.npy")):
        i = frame_id(f)
        if i is not None:
            out[i] = f
    return out

def run(name, pred_dir, gt_dir):
    preds = load_map(pred_dir, "")
    gts = load_map(gt_dir, "")
    ids = sorted(set(preds) & set(gts))
    print(f"\n===== {name} =====")
    print(f"  matched frames: {len(ids)} (pred {len(preds)}, gt {len(gts)})")
    if not ids:
        return None

    # per-pixel accumulators
    absrel_sum = 0.0; npix = 0
    d1_hit = 0
    # depth-bin decomposition of abs_rel contribution
    edges = np.array([0, 5, 10, 20, 30, 50, 80, 150, 300, 1001], dtype=np.float32)
    bin_relsum = np.zeros(len(edges) - 1)
    bin_cnt = np.zeros(len(edges) - 1)
    bin_d1hit = np.zeros(len(edges) - 1)
    per_img_absrel = []

    for i in ids:
        t = decode(np.load(gts[i]).astype(np.float32).squeeze())
        p = decode(np.load(preds[i]).astype(np.float32).squeeze())
        if t.shape != p.shape:
            h = min(t.shape[0], p.shape[0]); w = min(t.shape[1], p.shape[1])
            t = t[:h, :w]; p = p[:h, :w]
        m = np.isfinite(t) & np.isfinite(p) & (t > 1e-3)
        t = t[m]; p = p[m]
        rel = np.abs(t - p) / t
        ratio = np.maximum(p / t, t / p)
        absrel_sum += rel.sum(); npix += rel.size
        d1_hit += (ratio < 1.25).sum()
        per_img_absrel.append(rel.mean())
        bi = np.digitize(t, edges) - 1
        bi = np.clip(bi, 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            sel = bi == b
            if sel.any():
                bin_relsum[b] += rel[sel].sum()
                bin_cnt[b] += sel.sum()
                bin_d1hit[b] += (ratio[sel] < 1.25).sum()

    absrel = absrel_sum / npix
    d1 = d1_hit / npix
    print(f"  abs_rel(pixel-mean) = {absrel:.4f}    delta<1.25 = {d1:.4f}")
    pim = np.array(per_img_absrel)
    print(f"  per-image abs_rel: median={np.median(pim):.4f} p90={np.percentile(pim,90):.4f} "
          f"p99={np.percentile(pim,99):.4f} max={pim.max():.4f}")
    print(f"  {'bin(m)':>12} {'pix%':>7} {'abs_rel':>9} {'d1':>6} {'rel_contrib%':>12}")
    total_contrib = bin_relsum.sum()
    for b in range(len(edges) - 1):
        if bin_cnt[b] == 0: continue
        lbl = f"{edges[b]:.0f}-{edges[b+1]:.0f}"
        print(f"  {lbl:>12} {100*bin_cnt[b]/npix:7.2f} "
              f"{bin_relsum[b]/bin_cnt[b]:9.3f} {bin_d1hit[b]/bin_cnt[b]:6.3f} "
              f"{100*bin_relsum[b]/total_contrib:12.2f}")
    return dict(absrel=absrel, d1=d1)

if __name__ == "__main__":
    base = "/root/shared-nvme/spikemamba"
    jobs = [
        ("0320_baseline",
         f"{base}/runs/eval/val_mambassm_bidivim_continue40_best_20260320_sys/npy/image",
         f"{base}/runs/eval/val_mambassm_bidivim_continue40_best_20260320_sys/ground_truth/npy/depth_image"),
        ("reB_centercrop224_ep3",
         f"{base}/eval_reB_centercrop224_20260731/npy/image",
         f"{base}/eval_reB_centercrop224_20260731/ground_truth/npy/depth_image"),
    ]
    res = {}
    for name, pd, gd in jobs:
        res[name] = run(name, pd, gd)
    print("\n===== SUMMARY =====")
    for k, v in res.items():
        if v: print(f"  {k}: abs_rel={v['absrel']:.4f} delta1={v['d1']:.4f}")
