# Ablation Plan: Manifold Flow ON vs OFF

## Goal
Measure whether the `ConditionedDepthManifoldFlow` refinement head actually improves
depth estimation, by training two otherwise-identical models and comparing on the
validation set.

## Why a standalone runner (not train_parallel.py)
`train_parallel.py` has two known blockers on this box (documented in `20260707.md`):
- unconditional `DistributedSampler` without `init_process_group` on the single-GPU path
- `mp.spawn` UCX segfault in the NGC container

The working, validated path is the standalone driver pattern in
`smoke_systempy_20260707.py`. I will build the ablation runner on that same pattern.

## Environment (verified)
- GPU: RTX 4090 (sm_89), 24 GB
- `mamba_ssm` + `selective_scan_cuda` import AND run a real forward on this GPU (verified)
- One seq2 train step (bs=1): ~0.58 s, ~3.8 GB peak

## Design (fair A/B)
Both arms are byte-identical except `model.manifold_flow.enabled`:
- **Arm FLOW**: config as-is (`enabled: true`)
- **Arm BASE**: same config, `enabled: false` (pure encoder→decoder, no flow head/aux losses)

Controlled to be identical across arms:
- same base config `train_s2d_spiketransformer_mambaflow_bidivim_seq2_40ep_20260423.json`
- same seed (`torch.manual_seed(111)`), same data order (no shuffle, fixed index walk)
- same optimizer (Adam lr=1e-4), same grad clip (norm 1.0), same seq_len=2
- same 112×112 resize of spike+depth (established working approach from smoke driver)
- same primary loss (scale_invariant, w=1.0) + 0.25·multi_scale_grad_loss
- FLOW arm additionally adds its aux losses (L_coarse/L_rec/L_flow/L_geo) exactly as the
  trainer does — this is intrinsic to "flow ON", not an unfair extra.

## Scope (user-approved: full-ish, faithful)
- **40 epochs** per arm (matches real stage-1 schedule)
- All 5 train subfolders, all val sequences per epoch
- Val every epoch; track best `val_loss` (scale_invariant + 0.25·grad, mean over val)

## Metrics (user-approved)
Report per epoch and best-epoch summary for BOTH arms:
- `val_loss` (monitored quantity)
- depth metrics on val: `abs_rel_diff`, `rms_linear` (RMSE), `scale_invariant_error`, `median_error`
Reuse `model/metric.py` and `model/loss.py` — no reimplementation.

## Deliverables
1. `scripts/ablation_manifold_flow_20260708.py` — standalone runner:
   - `--flow {on,off}`, `--epochs`, `--max-train-seq`, `--max-val-seq`, `--out`
   - builds ConcatDataset over all train subfolders (like train_parallel.py)
   - trains, validates each epoch, writes per-epoch JSONL + final summary JSON
   - saves best checkpoint per arm
2. `scripts/run_ablation_manifold_flow_20260708.sh` — runs BASE then FLOW sequentially,
   logs to `runs/ablation/<arm>/`
3. `ABLATION_RESULTS_manifold_flow_20260708.md` — written after the run: side-by-side
   table (best val_loss + depth metrics per arm), verdict, and caveats.

## Execution
- Runs in background (~4+ hrs total for both arms). I will launch, monitor progress,
  and report the comparison table + verdict when done.
- No changes to any existing model/config/training file. Ablation is additive only.

## Risk / caveats to note in results
- Single seed (no error bars) — will state this.
- Synthetic DENSE only.
- If flow arm diverges/NaNs, runner logs it rather than silently continuing.
