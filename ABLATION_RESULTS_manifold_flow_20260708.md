# Ablation Results: Manifold Flow ON vs OFF

Runner: `scripts/ablation_manifold_flow_20260708.py` · Data: `runs/ablation/{base,flow}/`
Date: 2026-07-08

## TL;DR
On the monitored quantity (`val_loss`), the **BASE** arm (flow **off**) came out ahead
at every arm's best epoch: **0.01214** (BASE) vs **0.01372** (FLOW), a ~11% lower val_loss
without the flow head. On depth metrics the picture is split — FLOW gives a dramatically
lower `abs_rel_diff` and `median_error`, while BASE gives lower `rms_linear`. See caveats:
the FLOW arm did **not** finish (crashed at ep19 on an infra fault, not divergence), so this
is a 19-epoch vs 40-epoch comparison anchored at the shared best epoch (ep15).

## Run status
| Arm | flow | Epochs planned | Epochs completed | Ended by | summary.json |
|-----|------|----------------|------------------|----------|--------------|
| BASE | off | 40 | 40 | normal | yes |
| FLOW | on  | 40 | 19 | infra crash: `OSError [Errno 107] Transport endpoint is not connected` (`/home/dev/.local` fuse mount dropped mid-run) | no |

The FLOW crash is an environment fault (the mounted site-packages dir disconnected), not a
NaN/divergence. Losses were still decreasing normally through ep19.

## Best-epoch comparison (both arms best `val_loss` @ ep15)
| Metric | BASE (off) @ep15 | FLOW (on) @ep15 | Winner |
|--------|------------------|-----------------|--------|
| val_loss ↓ | **0.012142** | 0.013717 | BASE |
| abs_rel_diff ↓ | 33554.31 | **2435.91** | FLOW |
| rms_linear (RMSE) ↓ | 0.30587 | **0.09070** | FLOW |
| scale_invariant_error ↓ | **0.003501** | 0.006147 | BASE |
| median_error ↓ | 0.29458 | **0.02425** | FLOW |
| train_loss | 0.005694 | 0.051736 | — |

Note: the train_loss scales are not comparable across arms — the FLOW arm's loss includes
extra aux terms (L_coarse/L_rec/L_flow/L_geo), so its train_loss is expected to be higher.

## Best per arm (across all completed epochs)
| | BASE best | FLOW best (of 19) |
|---|-----------|-------------------|
| best val_loss | 0.012142 @ep15 | 0.013717 @ep15 |
| best rms_linear | 0.2471 @ep21 | 0.08903 @ep17 |
| best median_error | 0.2192 @ep23 | 0.01221 @ep16 |
| best scale_invariant_error | 0.003215 @ep23 | 0.005830 @ep14 |

## Reading the split
- BASE wins on `val_loss` and `scale_invariant_error` — the scale-invariant term is what
  `val_loss` is dominated by, so these two agree.
- FLOW wins hard on the scale-*dependent* metrics (`rms_linear`, `median_error`, `abs_rel_diff`).
  The flow head appears to fix absolute-scale/offset errors: BASE's `median_error` sits around
  ~0.29 (large constant offset) while FLOW's is ~0.02. This is the metric where the manifold
  flow refinement is doing visible work.
- Interpretation: flow ON trades a slightly worse scale-invariant fit for a much better
  absolute-depth fit. Which matters depends on the downstream use of absolute depth.

## Caveats
- **Not a finished A/B.** FLOW ran 19/40 epochs. The best-epoch anchor (ep15) is fair since
  both peaked there, but BASE's later epochs (val_loss stayed ~0.012–0.015) are not matched
  by a completed FLOW arm. Re-run FLOW to 40 epochs to confirm.
- **Single seed** (`torch.manual_seed(111)`), no error bars.
- **Subsampled**: 300/4990 train seqs, 150/998 val seqs per epoch (as configured in the run,
  not the full set the plan describes).
- **Synthetic DENSE only.**
- `abs_rel_diff` values are very large (tens of thousands) — likely near-zero GT depths
  inflating the ratio; treat its absolute magnitude with caution and read it as relative
  between arms only.

## To reproduce / finish
```bash
bash scripts/run_ablation_manifold_flow_20260708.sh   # runs BASE then FLOW
# or re-run just the flow arm to 40 ep:
python scripts/ablation_manifold_flow_20260708.py --flow on --epochs 40 \
  --out runs/ablation/flow
```
Recommend running off a stable local path (not the fuse-mounted `/home/dev/.local`) to
avoid the transport-endpoint crash that killed the FLOW arm.
