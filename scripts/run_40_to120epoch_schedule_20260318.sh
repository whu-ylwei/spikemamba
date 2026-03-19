#!/usr/bin/env bash

set -euo pipefail

WORKDIR="/root/shared-nvme/spikemamba"
DATA_ROOT="$WORKDIR/DENSE-spike"
BASE_CKPT="$WORKDIR/runs/train/train_s2d_MambaSSM_spatialseq_40ep_20260318_bs2/checkpoints/checkpoint-epoch040-loss-0.0031.pth.tar"
STAGE1_CFG="$WORKDIR/configs/train_s2d_spiketransformer_mambassm_spatialseq_120epoch_stage1_ep40_to60_keep_lr_20260318.json"
STAGE2_CFG="$WORKDIR/configs/train_s2d_spiketransformer_mambassm_spatialseq_120epoch_ep60_to120_halflr_20260318.json"
RESUME_DIR="$WORKDIR/runs/resume"
STAGE1_RUN_DIR="$WORKDIR/runs/train/train_s2d_MambaSSM_spatialseq_120epoch_stage1_ep40_to60_keep_lr_20260318_bs2"
STAGE2_RUN_DIR="$WORKDIR/runs/train/train_s2d_MambaSSM_spatialseq_120epoch_20260318_bs2"
STAGE1_RESUME="$RESUME_DIR/resume_120epoch_stage1_ep40_to60_keep_lr_20260318.pth.tar"
STAGE2_RESUME="$RESUME_DIR/resume_120epoch_stage2_ep60_to120_halflr_20260318.pth.tar"
STAGE1_LOG="$WORKDIR/run_logs/train_s2d_mambassm_spatialseq_120epoch_stage1_ep40_to60_keep_lr_20260318.log"
STAGE2_LOG="$WORKDIR/run_logs/train_s2d_mambassm_spatialseq_120epoch_20260318.log"
MASTER_LOG="$WORKDIR/run_logs/train_s2d_mambassm_spatialseq_40_to120epoch_schedule_20260318.log"

mkdir -p "$RESUME_DIR"

if [[ -d "$STAGE1_RUN_DIR" || -d "$STAGE2_RUN_DIR" ]]; then
    echo "Target run directory already exists. Refusing to overwrite." >&2
    exit 1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] preparing stage1 resume checkpoint from epoch40" | tee -a "$MASTER_LOG"
/usr/bin/python3 - <<'PY'
import json
import torch

base_ckpt = "/root/shared-nvme/spikemamba/runs/train/train_s2d_MambaSSM_spatialseq_40ep_20260318_bs2/checkpoints/checkpoint-epoch040-loss-0.0031.pth.tar"
cfg_path = "/root/shared-nvme/spikemamba/configs/train_s2d_spiketransformer_mambassm_spatialseq_120epoch_stage1_ep40_to60_keep_lr_20260318.json"
out_path = "/root/shared-nvme/spikemamba/runs/resume/resume_120epoch_stage1_ep40_to60_keep_lr_20260318.pth.tar"

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

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage1 training: epoch 41-60 at lr=1e-4" | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 "$WORKDIR/train_parallel.py" \
    -r "$STAGE1_RESUME" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1264 \
    > "$STAGE1_LOG" 2>&1

shopt -s nullglob
stage1_ckpts=("$STAGE1_RUN_DIR"/checkpoints/checkpoint-epoch060-loss-*.pth.tar)
shopt -u nullglob

if [[ ${#stage1_ckpts[@]} -ne 1 ]]; then
    echo "Expected exactly one epoch060 checkpoint after stage1, got ${#stage1_ckpts[@]}" >&2
    exit 1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] preparing stage2 resume checkpoint from ${stage1_ckpts[0]}" | tee -a "$MASTER_LOG"
/usr/bin/python3 - <<'PY'
import glob
import json
import torch

matches = glob.glob("/root/shared-nvme/spikemamba/runs/train/train_s2d_MambaSSM_spatialseq_120epoch_stage1_ep40_to60_keep_lr_20260318_bs2/checkpoints/checkpoint-epoch060-loss-*.pth.tar")
if len(matches) != 1:
    raise SystemExit(f"Expected one stage1 epoch060 checkpoint, got {len(matches)}")
base_ckpt = matches[0]
cfg_path = "/root/shared-nvme/spikemamba/configs/train_s2d_spiketransformer_mambassm_spatialseq_120epoch_ep60_to120_halflr_20260318.json"
out_path = "/root/shared-nvme/spikemamba/runs/resume/resume_120epoch_stage2_ep60_to120_halflr_20260318.pth.tar"

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

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage2 training: epoch 61-120 at lr=5e-5" | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 "$WORKDIR/train_parallel.py" \
    -r "$STAGE2_RESUME" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1265 \
    > "$STAGE2_LOG" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] schedule finished" | tee -a "$MASTER_LOG"
