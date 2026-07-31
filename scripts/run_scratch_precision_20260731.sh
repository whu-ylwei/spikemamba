#!/usr/bin/env bash
# Residual-bridge (route B) manifold flow, PRECISION config, trained FROM SCRATCH.
# No --resume: random init, fresh Adam, all 0730 precision switches active from ep1.
# Stage: 20 epochs @ lr=1e-4 (config default). Probe to see if new loss combo learns.
# If val curve looks healthy, continue to 40ep in a second stage (decided by monitor).
set -euo pipefail

WORKDIR="/root/shared-nvme/spikemamba"
cd "$WORKDIR"

CFG="configs/train_s2d_spiketransformer_mambaflow_residual_bridge_precision_20260730.json"
OUT="runs/flow_reB_scratch_precision_20260731"
LOG="run_logs/flow_reB_scratch_precision_20260731.log"
mkdir -p "$OUT" run_logs

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SCRATCH start: 20ep lr=1e-4 (precision cfg)" | tee -a "$LOG"
env -u PYTHONNOUSERSITE python3 scripts/ablation_manifold_flow_20260708.py \
    --flow on --epochs 20 --config "$CFG" --out "$OUT" --num-workers 16 \
    >> "$LOG" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SCRATCH stage done (20ep)" | tee -a "$LOG"
