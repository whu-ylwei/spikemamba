# 0410 Branch Snapshot

This branch packages the current core SpikeMamba code state together with the exact 80-epoch configuration and evaluation snapshot used for the latest `scape-spike` test.

## Base code snapshot

- Base commit: `947c104ac51dcf3c3e29aa3438f5727eb1086906`
- Source branch at snapshot time: `0320`
- New delivery branch: `0410`

Core entry points and modules already tracked in this branch:

- Training entry: `train_parallel.py`
- Inference entry: `test_DENSE.py`
- Evaluation entry: `evaluation_DENSE.py`
- Main model: `model/S2DepthNet.py`
- Encoder and temporal backbone: `model/encoder_transformer.py`, `model/mamba_3d.py`
- Data loaders: `data_loader/SpikesDENSE_dataset.py`, `data_loader/spike_dataset.py`
- Trainer: `trainer/spiket_trainer.py`

## 80-Epoch Best Configuration

Tracked config copy:

- `configs/train_s2d_MambaSSM_bidivim_ep40_continue40_halflr_20260320_bs2.json`

This config corresponds to the checkpoint used for the 2026-04-10 `scape-spike` evaluation.

## 80-Epoch Best Checkpoint Metadata

The raw checkpoint file is not committed in this branch because the local file is 1.4G and exceeds practical plain-git branch packaging.

- Local checkpoint path:
  `runs/train/train_s2d_MambaSSM_bidivim_ep40_continue40_halflr_20260320_bs2/checkpoints/model_best.pth.tar`
- File size:
  `1.4G`
- SHA256:
  `767e2b7ec2f99947d4ff9b07deecb18169ea71a9076eb20970b29bad3b8f0022`

## Latest `scape-spike` Evaluation Snapshot

Tracked evaluation summary:

- `results/reports/analysis/scape_spike_80epoch_eval_20260410.json`

Main metrics:

- Abs Rel: `0.5289149568936038`
- Sq Rel: `5.53868782534452`
- RMS linear: `146.75209547391697`
- RMS log: `0.6071548301930099`
- SILog: `0.2295740446722553`
- delta < 1.25: `0.3857915150546563`
- delta < 1.25^2: `0.7773988529497322`
- delta < 1.25^3: `0.8844009230028118`

## Evaluation Command

```bash
cd /root/shared-nvme/spikemamba

python3 evaluation_DENSE.py \
  --target_dataset /root/shared-nvme/spikemamba/runs/eval/test_mambassm_bidivim_continue40_best_20260320_scape_spike_20260410/ground_truth/npy/depth_image \
  --predictions_dataset /root/shared-nvme/spikemamba/runs/eval/test_mambassm_bidivim_continue40_best_20260320_scape_spike_20260410/npy/image \
  --clip_distance 1000 \
  --reg_factor 5.7 \
  --output_folder /root/shared-nvme/spikemamba/runs/eval/test_mambassm_bidivim_continue40_best_20260320_scape_spike_20260410/eval_vis
```
