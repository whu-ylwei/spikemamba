#!/usr/bin/env bash

set -euo pipefail

WORKDIR="/root/shared-nvme/spikemamba"
EXPDIR="$WORKDIR/0323"
DATA_ROOT="$WORKDIR/DENSE-spike"

CFG_STAGE1="$EXPDIR/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_60ep_20260323.json"
CFG_STAGE2="$EXPDIR/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_ep60_to120_halflr_20260323.json"

RUN_ROOT="$WORKDIR/runs/train/0323"
RUN_DIR_STAGE1="$RUN_ROOT/train_s2d_MambaSSM_bidivim_nograd_2enc_60ep_20260323_bs2"
RUN_DIR_STAGE2="$RUN_ROOT/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to120_halflr_20260323_bs2"

RESUME_DIR="$WORKDIR/runs/resume/0323"
RESUME_CKPT="$RESUME_DIR/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to120_halflr_20260323_resume.pth.tar"

MASTER_LOG="$WORKDIR/run_logs/0323/run_0323_60plus60_halflr.log"
TRAIN_LOG_STAGE1="$WORKDIR/run_logs/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_60ep_20260323.log"
TRAIN_LOG_STAGE2="$WORKDIR/run_logs/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to120_halflr_20260323.log"

mkdir -p "$RUN_ROOT" "$RESUME_DIR" "$(dirname "$MASTER_LOG")"

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

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage1 start: 60 epochs, lr=1e-4" | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES=0 /usr/bin/python "$WORKDIR/train_parallel.py" \
    -c "$CFG_STAGE1" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1272 \
    > "$TRAIN_LOG_STAGE1" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage1 finished, preparing stage2 resume from epoch060" | tee -a "$MASTER_LOG"
/usr/bin/python - <<'PY'
import glob
import json
import os
import torch

workdir = "/root/shared-nvme/spikemamba"
stage1_dir = os.path.join(workdir, "runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_60ep_20260323_bs2/checkpoints")
cfg_path = os.path.join(workdir, "0323/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_ep60_to120_halflr_20260323.json")
out_path = os.path.join(workdir, "runs/resume/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to120_halflr_20260323_resume.pth.tar")

matches = sorted(glob.glob(os.path.join(stage1_dir, "checkpoint-epoch060-loss-*.pth.tar")))
if not matches:
    raise FileNotFoundError(f"Could not find epoch-060 checkpoint in {stage1_dir}")
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

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage2 start: continue to 120 epochs, lr=5e-5" | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES=0 /usr/bin/python "$WORKDIR/train_parallel.py" \
    -r "$RESUME_CKPT" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1273 \
    > "$TRAIN_LOG_STAGE2" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage2 finished" | tee -a "$MASTER_LOG"
