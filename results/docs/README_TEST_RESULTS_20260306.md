# GPU Test and Evaluation Report (2026-03-06)
## 1. Objective
Run `test_DENSE.py` and `evaluation_DENSE.py` on GPU in `shared-nvme/spikemamba`, then summarize reproducible metrics in paper-style format.

## 2. Environment
- Date: 2026-03-06 (UTC)
- GPU: NVIDIA GeForce RTX 5090
- Driver / CUDA runtime: 570.144 / 12.8
- Python: 3.12.3 (`./.venv/bin/python`)
- PyTorch: 2.10.0+cu128 (`torch.cuda.is_available() = True`)

## 3. Model and Data
- Checkpoint: `runs/train/train_verify_20260306/checkpoints/model_best.pth.tar`
- Config: `runs/train/train_verify_20260306/config.json`
- Test dataset root: `DENSE-spike/DENSE/test`
- Test output: `runs/eval/test_gpu_20260306`

## 4. Commands Executed
```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python test_DENSE.py \
  --path_to_model runs/train/train_verify_20260306/checkpoints/model_best.pth.tar \
  --config runs/train/train_verify_20260306/config.json \
  --output_path /root/shared-nvme/spikemamba/runs/eval/test_gpu_20260306 \
  --data_folder /root/shared-nvme/spikemamba/DENSE-spike/DENSE/test
```

```bash
./.venv/bin/python evaluation_DENSE.py \
  --target_dataset /root/shared-nvme/spikemamba/runs/eval/test_gpu_20260306/ground_truth/npy/depth_image \
  --predictions_dataset /root/shared-nvme/spikemamba/runs/eval/test_gpu_20260306/npy/image \
  --clip_distance 1000 \
  --reg_factor 5.7 \
  --output_folder /root/shared-nvme/spikemamba/runs/eval/test_gpu_20260306/eval_vis
```

## 5. Run Status
- `test_DENSE.py`: completed (exit code `0`)
- `evaluation_DENSE.py`: completed (exit code `0`)
- Inference output files (excluding eval log/vis): `8973`
- Evaluation pairs: `997`
- Evaluation visualization images: `997`

## 6. Main Metrics (Research Format)
Dataset size for evaluation: 997 frames/pairs.

| Method | Abs Rel ↓ | Sq Rel ↓ | RMS log ↓ | SI log ↓ | delta<1.25 ↑ | delta<1.25^2 ↑ | delta<1.25^3 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `S2DepthTransformerUNetConv` (`train_verify_20260306/model_best.pth.tar`) | 1.306131 | 5.317150 | 2.424063 | 2.167130 | 0.096819 | 0.155237 | 0.219581 |

Additional vector reported by `evaluation_DENSE.py`:
- `total metrics [mse, abs_rel_diff, scale_invariant_error, median_error, mean_error, rms_linear]`
- `[2.04911904e+05, 1.30613113e+00, 1.41769802e+05, 2.81084020e+01, 2.32978471e+02, 4.31365349e+02]`

## 7. Important Observation
Current runtime did not import `mamba_ssm`; code used fallback `Mamba` implementation from `model/mamba_3d.py`.

Environment check:
- `Mamba module: model.mamba_3d`
- `mamba_ssm: missing`

This may significantly affect final accuracy compared with a full `mamba_ssm` setup.

## 8. Artifacts
- Full inference + evaluation folder: `/root/shared-nvme/spikemamba/runs/eval/test_gpu_20260306`
- Evaluation log: `/root/shared-nvme/spikemamba/runs/eval/test_gpu_20260306/eval_metrics_20260306.log`
- Evaluation visualizations: `/root/shared-nvme/spikemamba/runs/eval/test_gpu_20260306/eval_vis`
