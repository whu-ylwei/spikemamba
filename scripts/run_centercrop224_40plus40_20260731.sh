#!/usr/bin/env bash
# 全新正规 40+40 @ 224² CenterCrop 同口径 (0731)
# stage1: 随机初始化, 连续 40ep@lr1e-4
# stage2: 从 stage1 model_last(ep40) resume, 40ep@lr5e-5 -> ep41-80
# best 只在 val 改善时更新; stage2 从 summary.json 继承 stage1 best 避免覆盖。
set -euo pipefail
cd /root/shared-nvme/spikemamba

OUT=runs/flow_reB_centercrop224_40plus40_20260731
CFG=configs/train_s2d_spiketransformer_mambaflow_residual_bridge_centercrop224_20260731.json
LOG=run_logs/flow_reB_centercrop224_40plus40_20260731.log
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
