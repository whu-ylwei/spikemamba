#!/usr/bin/env bash

set -euo pipefail

WORKDIR="/root/shared-nvme/spikemamba"
DATA_ROOT="$WORKDIR/DENSE-spike"
BASE_CKPT="$WORKDIR/runs/train/train_s2d_MambaSSM_spatialseq_40ep_20260318_bs2/checkpoints/checkpoint-epoch040-loss-0.0031.pth.tar"
CFG_PATH="$WORKDIR/configs/train_s2d_spiketransformer_mambassm_spatialseq_ep40_to80_halflr_20260318.json"
RESUME_DIR="$WORKDIR/runs/resume"
RUN_DIR="$WORKDIR/runs/train/train_s2d_MambaSSM_spatialseq_ep40_to80_halflr_20260318_bs2"
RESUME_CKPT="$RESUME_DIR/resume_ep40_to80_halflr_20260318.pth.tar"
TRAIN_LOG="$WORKDIR/run_logs/train_s2d_mambassm_spatialseq_ep40_to80_halflr_20260318.log"

mkdir -p "$RESUME_DIR"

if [[ -d "$RUN_DIR" ]]; then
    echo "Target run directory already exists: $RUN_DIR" >&2
    exit 1
fi

/usr/bin/python3 - <<'PY'
import json
import torch

base_ckpt = "/root/shared-nvme/spikemamba/runs/train/train_s2d_MambaSSM_spatialseq_40ep_20260318_bs2/checkpoints/checkpoint-epoch040-loss-0.0031.pth.tar"
cfg_path = "/root/shared-nvme/spikemamba/configs/train_s2d_spiketransformer_mambassm_spatialseq_ep40_to80_halflr_20260318.json"
out_path = "/root/shared-nvme/spikemamba/runs/resume/resume_ep40_to80_halflr_20260318.pth.tar"

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

CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 "$WORKDIR/train_parallel.py" \
    -r "$RESUME_CKPT" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1263 \
    > "$TRAIN_LOG" 2>&1
