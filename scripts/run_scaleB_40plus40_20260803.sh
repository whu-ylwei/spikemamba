#!/usr/bin/env bash
# 病灶切分实验 scaleB (0803) @ 224² CenterCrop 同口径 (可比 0320)
# 相对 centercrop224 基线只改三个损失权重 (纯净切分两病灶):
#   reconstruction_weight 0 -> 0.3   (病灶A: 给 latent 流形加重建约束)
#   refined_metric_weight 0.1 -> 0.4 (病灶B: 提高绝对尺度直接监督)
#   si_n_lambda 0.5 -> 0.25          (病灶B: 松开完全尺度不变)
# 另: L_flow 已在 manifold_flow.py 全局改为 smooth_l1 (diff 1-A)。
# stage1: 随机初始化 40ep@lr1e-4; stage2: resume ep40, 40ep@lr5e-5 -> ep41-80。
set -euo pipefail
cd /root/shared-nvme/spikemamba

OUT=runs/flow_reB_scaleB_20260803
CFG=configs/train_s2d_spiketransformer_mambaflow_residual_bridge_scaleB_20260803.json
LOG=run_logs/flow_reB_scaleB_20260803.log
PY="env -u PYTHONNOUSERSITE python3"
SCRIPT=scripts/ablation_manifold_flow_20260708.py

mkdir -p "$OUT"

{
  echo "===== STAGE1 start $(date) : 40ep @ lr1e-4 (scratch) ====="
  $PY $SCRIPT --flow on --epochs 40 --lr 1e-4 --num-workers 16 \
      --config "$CFG" --out "$OUT"
  echo "===== STAGE1 done $(date) ====="

  echo "===== STAGE2 start $(date) : 40ep @ lr5e-5 (resume ep40) ====="
  $PY $SCRIPT --flow on --epochs 40 --lr 5e-5 --num-workers 16 \
      --config "$CFG" --out "$OUT" \
      --resume "$OUT/model_last.pth.tar"
  echo "===== STAGE2 done $(date) ====="
} >> "$LOG" 2>&1
