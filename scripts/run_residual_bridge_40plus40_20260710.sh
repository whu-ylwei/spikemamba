#!/usr/bin/env bash
# Residual-bridge (route B) manifold flow: 40+40 epoch schedule.
# Stage 1: 40 epochs @ lr=1e-4 (fresh).
# Stage 2: continue 40 epochs @ lr=5e-5 (halved), warm-started from stage-1 model_last.
set -euo pipefail

WORKDIR="/root/shared-nvme/spikemamba"
cd "$WORKDIR"

CFG="configs/train_s2d_spiketransformer_mambaflow_residual_bridge_40ep_20260710.json"
OUT="runs/flow_reB_40plus40_20260710"
LOG="run_logs/flow_reB_40plus40_20260710.log"
mkdir -p "$OUT" run_logs

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] STAGE1 start: 40ep lr=1e-4" | tee -a "$LOG"
python3 scripts/ablation_manifold_flow_20260708.py \
    --flow on --epochs 40 --config "$CFG" --out "$OUT" --num-workers 12 \
    >> "$LOG" 2>&1

if [[ ! -f "$OUT/model_last.pth.tar" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] STAGE1 FAILED: no model_last.pth.tar" | tee -a "$LOG"
    exit 1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] STAGE2 start: continue 40ep lr=5e-5" | tee -a "$LOG"
python3 scripts/ablation_manifold_flow_20260708.py \
    --flow on --epochs 40 --config "$CFG" --out "$OUT" --num-workers 12 \
    --resume "$OUT/model_last.pth.tar" --lr 5e-5 \
    >> "$LOG" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE 80 epochs total" | tee -a "$LOG"
