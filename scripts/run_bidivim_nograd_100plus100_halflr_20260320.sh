#!/usr/bin/env bash

set -euo pipefail

WORKDIR="/root/shared-nvme/spikemamba"
DATA_ROOT="$WORKDIR/DENSE-spike"

CFG_STAGE1="$WORKDIR/configs/train_s2d_spiketransformer_mambassm_bidivim_nograd_100ep_20260320.json"
CFG_STAGE2="$WORKDIR/configs/train_s2d_spiketransformer_mambassm_bidivim_nograd_ep100_to200_halflr_20260320.json"

RUN_DIR_STAGE1="$WORKDIR/runs/train/train_s2d_MambaSSM_bidivim_nograd_100ep_20260320_bs2"
RUN_DIR_STAGE2="$WORKDIR/runs/train/train_s2d_MambaSSM_bidivim_nograd_ep100_continue100_halflr_20260320_bs2"
RESUME_DIR="$WORKDIR/runs/resume_staging"
RESUME_CKPT="$RESUME_DIR/train_s2d_MambaSSM_bidivim_nograd_ep100_continue100_halflr_20260320_resume.pth.tar"

TRAIN_LOG_STAGE1="$WORKDIR/run_logs/train_s2d_MambaSSM_bidivim_nograd_100ep_20260320.log"
TRAIN_LOG_STAGE2="$WORKDIR/run_logs/train_s2d_MambaSSM_bidivim_nograd_ep100_continue100_halflr_20260320.log"

mkdir -p "$RESUME_DIR" "$WORKDIR/run_logs"

if [[ -d "$RUN_DIR_STAGE1" ]]; then
    echo "Stage-1 run directory already exists: $RUN_DIR_STAGE1" >&2
    exit 1
fi

if [[ -d "$RUN_DIR_STAGE2" ]]; then
    echo "Stage-2 run directory already exists: $RUN_DIR_STAGE2" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 "$WORKDIR/train_parallel.py" \
    -c "$CFG_STAGE1" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1266 \
    > "$TRAIN_LOG_STAGE1" 2>&1

/usr/bin/python3 - <<'PY'
import glob
import json
import os
import torch

workdir = "/root/shared-nvme/spikemamba"
stage1_dir = os.path.join(workdir, "runs/train/train_s2d_MambaSSM_bidivim_nograd_100ep_20260320_bs2/checkpoints")
cfg_path = os.path.join(workdir, "configs/train_s2d_spiketransformer_mambassm_bidivim_nograd_ep100_to200_halflr_20260320.json")
out_path = os.path.join(workdir, "runs/resume_staging/train_s2d_MambaSSM_bidivim_nograd_ep100_continue100_halflr_20260320_resume.pth.tar")

matches = sorted(glob.glob(os.path.join(stage1_dir, "checkpoint-epoch100-loss-*.pth.tar")))
if not matches:
    raise FileNotFoundError("Could not find epoch-100 checkpoint in {}".format(stage1_dir))
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

CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 "$WORKDIR/train_parallel.py" \
    -r "$RESUME_CKPT" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1267 \
    > "$TRAIN_LOG_STAGE2" 2>&1
