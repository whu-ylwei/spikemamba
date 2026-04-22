#!/usr/bin/env bash

set -euo pipefail

WORKDIR="/root/shared-nvme/spikemamba"
DATA_ROOT="$WORKDIR/DENSE-spike"
CFG="$WORKDIR/configs/train_s2d_MambaSSM_mhc_residual_100ep_20260416_bs2.json"
LOG="$WORKDIR/run_logs/train_mhc_residual_100ep_20260416.log"

mkdir -p "$WORKDIR/run_logs"

CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 "$WORKDIR/train_parallel.py" \
    -c "$CFG" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1270 \
    > "$LOG" 2>&1
