#!/usr/bin/env bash

set -euo pipefail

WORKDIR="/root/shared-nvme/spikemamba"
DATA_ROOT="$WORKDIR/DENSE-spike"
BASE_CKPT="$WORKDIR/runs/train/train_s2d_MambaSSM_spatialseq_30ep_20260317_bs2/checkpoints/model_best.pth.tar"
STAGE1_CFG="$WORKDIR/configs/train_s2d_spiketransformer_mambassm_spatialseq_ep28_to30_keep_lr_20260317.json"
STAGE2_CFG="$WORKDIR/configs/train_s2d_spiketransformer_mambassm_spatialseq_ep30_to60_halflr_20260317.json"
RESUME_DIR="$WORKDIR/runs/resume"
STAGE1_RUN_DIR="$WORKDIR/runs/train/train_s2d_MambaSSM_spatialseq_ep28_to30_keep_lr_20260317_bs2"
STAGE2_RUN_DIR="$WORKDIR/runs/train/train_s2d_MambaSSM_spatialseq_ep30_to60_halflr_20260317_bs2"
STAGE1_RESUME="$RESUME_DIR/resume_ep28_to30_keep_lr_20260317.pth.tar"
STAGE2_RESUME="$RESUME_DIR/resume_ep30_to60_halflr_20260317.pth.tar"
STAGE1_LOG="$WORKDIR/run_logs/train_s2d_mambassm_spatialseq_ep28_to30_keep_lr_20260317.log"
STAGE2_LOG="$WORKDIR/run_logs/train_s2d_mambassm_spatialseq_ep30_to60_halflr_20260317.log"
MASTER_LOG="$WORKDIR/run_logs/train_s2d_mambassm_spatialseq_ep28_plus32_schedule_20260317.log"

mkdir -p "$RESUME_DIR"

if [[ -d "$STAGE1_RUN_DIR" || -d "$STAGE2_RUN_DIR" ]]; then
    echo "Target run directory already exists. Refusing to overwrite." >&2
    exit 1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] preparing stage1 resume checkpoint" | tee -a "$MASTER_LOG"
python - <<'PY'
import json
import torch

base_ckpt = "/root/shared-nvme/spikemamba/runs/train/train_s2d_MambaSSM_spatialseq_30ep_20260317_bs2/checkpoints/model_best.pth.tar"
cfg_path = "/root/shared-nvme/spikemamba/configs/train_s2d_spiketransformer_mambassm_spatialseq_ep28_to30_keep_lr_20260317.json"
out_path = "/root/shared-nvme/spikemamba/runs/resume/resume_ep28_to30_keep_lr_20260317.pth.tar"

checkpoint = torch.load(base_ckpt, map_location="cpu")
config = json.load(open(cfg_path))
checkpoint["config"] = config
lr = config["optimizer"]["lr"]
for group in checkpoint["optimizer"]["param_groups"]:
    group["lr"] = lr
    if "initial_lr" in group:
        group["initial_lr"] = lr
torch.save(checkpoint, out_path)
print(out_path)
PY

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage1 training: epoch 29-30 at lr=1e-4" | tee -a "$MASTER_LOG"
python "$WORKDIR/train_parallel.py" \
    -r "$STAGE1_RESUME" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1261 \
    > "$STAGE1_LOG" 2>&1

shopt -s nullglob
stage1_ckpts=("$STAGE1_RUN_DIR"/checkpoints/checkpoint-epoch030-loss-*.pth.tar)
shopt -u nullglob

if [[ ${#stage1_ckpts[@]} -ne 1 ]]; then
    echo "Expected exactly one epoch030 checkpoint after stage1, got ${#stage1_ckpts[@]}" >&2
    exit 1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] preparing stage2 resume checkpoint from ${stage1_ckpts[0]}" | tee -a "$MASTER_LOG"
python - <<'PY'
import json
import torch

base_ckpt = None
import glob
matches = glob.glob("/root/shared-nvme/spikemamba/runs/train/train_s2d_MambaSSM_spatialseq_ep28_to30_keep_lr_20260317_bs2/checkpoints/checkpoint-epoch030-loss-*.pth.tar")
if len(matches) != 1:
    raise SystemExit(f"Expected one stage1 epoch030 checkpoint, got {len(matches)}")
base_ckpt = matches[0]
cfg_path = "/root/shared-nvme/spikemamba/configs/train_s2d_spiketransformer_mambassm_spatialseq_ep30_to60_halflr_20260317.json"
out_path = "/root/shared-nvme/spikemamba/runs/resume/resume_ep30_to60_halflr_20260317.pth.tar"

checkpoint = torch.load(base_ckpt, map_location="cpu")
config = json.load(open(cfg_path))
checkpoint["config"] = config
lr = config["optimizer"]["lr"]
for group in checkpoint["optimizer"]["param_groups"]:
    group["lr"] = lr
    if "initial_lr" in group:
        group["initial_lr"] = lr
torch.save(checkpoint, out_path)
print(out_path)
PY

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage2 training: epoch 31-60 at lr=5e-5" | tee -a "$MASTER_LOG"
python "$WORKDIR/train_parallel.py" \
    -r "$STAGE2_RESUME" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1262 \
    > "$STAGE2_LOG" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] schedule finished" | tee -a "$MASTER_LOG"
