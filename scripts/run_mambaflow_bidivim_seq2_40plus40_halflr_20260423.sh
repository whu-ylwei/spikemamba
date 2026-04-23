#!/usr/bin/env bash

set -euo pipefail

WORKDIR="/root/shared-nvme/spikemamba"
DATA_ROOT="$WORKDIR/DENSE-spike"

CFG_STAGE1="$WORKDIR/configs/train_s2d_spiketransformer_mambaflow_bidivim_seq2_40ep_20260423.json"
CFG_STAGE2="$WORKDIR/configs/train_s2d_spiketransformer_mambaflow_bidivim_seq2_ep40_continue40_halflr_20260423.json"

RUN_DIR_STAGE1="$WORKDIR/runs/train/train_s2d_MambaFlow_bidivim_seq2_40ep_20260423_bs2"
RUN_DIR_STAGE2="$WORKDIR/runs/train/train_s2d_MambaFlow_bidivim_seq2_ep40_continue40_halflr_20260423_bs2"
RESUME_DIR="$WORKDIR/runs/resume/mambaflow_20260423"
RESUME_CKPT="$RESUME_DIR/train_s2d_MambaFlow_bidivim_seq2_ep40_continue40_halflr_20260423_resume.pth.tar"

MASTER_LOG="$WORKDIR/run_logs/train_s2d_MambaFlow_bidivim_seq2_40plus40_halflr_20260423.master.log"
TRAIN_LOG_STAGE1="$WORKDIR/run_logs/train_s2d_MambaFlow_bidivim_seq2_40ep_20260423.log"
TRAIN_LOG_STAGE2="$WORKDIR/run_logs/train_s2d_MambaFlow_bidivim_seq2_ep40_continue40_halflr_20260423.log"

mkdir -p "$RESUME_DIR" "$WORKDIR/run_logs"

if [[ -e "$RUN_DIR_STAGE1" ]]; then
    echo "Stage-1 run directory already exists: $RUN_DIR_STAGE1" >&2
    exit 1
fi

if [[ -e "$RUN_DIR_STAGE2" ]]; then
    echo "Stage-2 run directory already exists: $RUN_DIR_STAGE2" >&2
    exit 1
fi

if [[ -e "$RESUME_CKPT" ]]; then
    echo "Stage-2 resume checkpoint already exists: $RESUME_CKPT" >&2
    exit 1
fi

echo "[$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ)] stage1 start: 40 epochs, lr=1e-4" | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES=0 /usr/bin/python "$WORKDIR/train_parallel.py" \
    -c "$CFG_STAGE1" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1280 \
    > "$TRAIN_LOG_STAGE1" 2>&1

echo "[$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ)] stage1 finished, preparing stage2 resume from epoch040" | tee -a "$MASTER_LOG"
/usr/bin/python - <<'PY'
import glob
import json
import os
import torch

workdir = "/root/shared-nvme/spikemamba"
stage1_dir = os.path.join(workdir, "runs/train/train_s2d_MambaFlow_bidivim_seq2_40ep_20260423_bs2/checkpoints")
cfg_path = os.path.join(workdir, "configs/train_s2d_spiketransformer_mambaflow_bidivim_seq2_ep40_continue40_halflr_20260423.json")
out_path = os.path.join(workdir, "runs/resume/mambaflow_20260423/train_s2d_MambaFlow_bidivim_seq2_ep40_continue40_halflr_20260423_resume.pth.tar")

matches = sorted(glob.glob(os.path.join(stage1_dir, "checkpoint-epoch040-loss-*.pth.tar")))
if not matches:
    raise FileNotFoundError(f"Could not find epoch-040 checkpoint in {stage1_dir}")
base_ckpt = matches[-1]

checkpoint = torch.load(base_ckpt, map_location="cpu")
with open(cfg_path, "r", encoding="utf-8") as fp:
    config = json.load(fp)
checkpoint["config"] = config
lr = config["optimizer"]["lr"]
for group in checkpoint["optimizer"]["param_groups"]:
    group["lr"] = lr
    if "initial_lr" in group:
        group["initial_lr"] = lr
torch.save(checkpoint, out_path)
print(out_path)
PY

echo "[$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ)] stage2 start: continue to 80 epochs, lr=5e-5" | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES=0 /usr/bin/python "$WORKDIR/train_parallel.py" \
    -r "$RESUME_CKPT" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1281 \
    > "$TRAIN_LOG_STAGE2" 2>&1

echo "[$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ)] stage2 finished" | tee -a "$MASTER_LOG"
