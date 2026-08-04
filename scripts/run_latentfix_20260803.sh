#!/usr/bin/env bash
# Latent-manifold-fix (2026-08-03): two-stage DepthFM-style training.
# See LATENT_MANIFOLD_FIX_20260803.md.
#   Stage 1: pretrain the depth autoencoder (encoder + recon_decoder) to fix the latent manifold.
#   Stage 2: warm-start + freeze the 0320 backbone, freeze the pretrained AE, train ONLY the
#            flow (velocity_field + residual_decoder) at small lr for 20 epochs.
#
# Env trap (see memory residual-bridge-training-0710): must run with env -u PYTHONNOUSERSITE.
set -euo pipefail
cd /root/shared-nvme/spikemamba

CFG=configs/train_s2d_spiketransformer_mambaflow_residual_bridge_latentfix_20260803.json
BACKBONE=bivmamba_best/checkpoint/model_best_ep70.pth.tar
AE_OUT=runs/ae_pretrain_latentfix_20260803
FLOW_OUT=runs/flow_reB_latentfix_20260803
RUN="env -u PYTHONNOUSERSITE python3"

echo "=== STAGE 1: AE pretrain (fix latent manifold) ==="
$RUN scripts/pretrain_ae_manifold_20260803.py \
  --config "$CFG" \
  --backbone-ckpt "$BACKBONE" \
  --epochs 30 --lr 2e-4 --num-workers 8 \
  --out "$AE_OUT"
echo "=== STAGE 1 done. AE -> $AE_OUT/ae_pretrained.pth.tar ==="

echo "=== STAGE 2: freeze backbone + AE, train flow @ small lr 20ep ==="
$RUN scripts/ablation_manifold_flow_20260708.py \
  --flow on --epochs 20 --lr 5e-5 --num-workers 8 \
  --config "$CFG" \
  --backbone-ckpt "$BACKBONE" \
  --freeze-backbone \
  --ae-ckpt "$AE_OUT/ae_pretrained.pth.tar" \
  --out "$FLOW_OUT"
echo "=== STAGE 2 done. flow -> $FLOW_OUT/model_best.pth.tar ==="
