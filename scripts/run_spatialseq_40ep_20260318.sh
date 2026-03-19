#!/usr/bin/env bash

set -euo pipefail

WORKDIR="/root/shared-nvme/spikemamba"
DATA_ROOT="$WORKDIR/DENSE-spike"
CONFIG="$WORKDIR/configs/train_s2d_spiketransformer_mambassm_spatialseq_40ep_20260318.json"
RUN_DIR="$WORKDIR/runs/train/train_s2d_MambaSSM_spatialseq_40ep_20260318_bs2"
LOG="$WORKDIR/run_logs/train_s2d_mambassm_spatialseq_40ep_20260318.log"

mkdir -p "$WORKDIR/run_logs"

if [[ -d "$RUN_DIR" ]]; then
    echo "Target run directory already exists. Refusing to overwrite." >&2
    exit 1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] training: epochs 1-40 at lr=1e-4" | tee -a "$LOG"
/usr/bin/python3 "$WORKDIR/train_parallel.py" \
    -c "$CONFIG" \
    -f "$DATA_ROOT" \
    --gpu 0 \
    --multiprocessing_distributed \
    --dist_url tcp://127.0.0.1:1264 \
    >> "$LOG" 2>&1
