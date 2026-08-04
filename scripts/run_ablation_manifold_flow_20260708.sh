#!/usr/bin/env bash
# Manifold Flow ablation: train BASE (off) then FLOW (on), then eval both best
# checkpoints on the TEST split. Capped seqs/epoch, 40 epochs, same seed both arms.
set -euo pipefail
cd /root/shared-nvme/spikemamba

EPOCHS=40
MAX_TRAIN=300
MAX_VAL=150
PY=python3
RUN=scripts/ablation_manifold_flow_20260708.py
EVAL=scripts/eval_test_manifold_flow_20260708.py

echo "=== TRAIN BASE (flow off) ==="
$PY $RUN --flow off --epochs $EPOCHS --max-train-seq $MAX_TRAIN --max-val-seq $MAX_VAL --out runs/ablation/base

echo "=== TRAIN FLOW (flow on) ==="
$PY $RUN --flow on  --epochs $EPOCHS --max-train-seq $MAX_TRAIN --max-val-seq $MAX_VAL --out runs/ablation/flow

echo "=== EVAL ON TEST ==="
$PY $EVAL --flow off --ckpt runs/ablation/base/model_best.pth.tar | tee runs/ablation/test_base.txt
$PY $EVAL --flow on  --ckpt runs/ablation/flow/model_best.pth.tar | tee runs/ablation/test_flow.txt

echo "=== ALL DONE ==="
