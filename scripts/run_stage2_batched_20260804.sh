#!/usr/bin/env bash
# Stage-2 batched flow training (2026-08-04). Reuses stage-1 ep29 AE (frozen),
# freezes backbone, trains only flow head. batch_size=40 to saturate GPU (~19.6GB),
# lr 5e-5 (unchanged), 20 epochs. Val stays bs=1 for identical historical口径.
set -euo pipefail
cd /root/shared-nvme/spikemamba
env -u PYTHONNOUSERSITE python3 scripts/ablation_manifold_flow_batched_20260804.py \
  --flow on --epochs 20 --lr 5e-5 --batch-size 32 --num-workers 6 \
  --config configs/train_s2d_spiketransformer_mambaflow_residual_bridge_latentfix_20260803.json \
  --backbone-ckpt bivmamba_best/checkpoint/model_best_ep70.pth.tar \
  --freeze-backbone \
  --ae-ckpt runs/ae_pretrain_latentfix_20260803/ae_pretrained.pth.tar \
  --out runs/flow_reB_latentfix_batched_20260804
echo "=== batched stage-2 done -> runs/flow_reB_latentfix_batched_20260804/model_best.pth.tar ==="
