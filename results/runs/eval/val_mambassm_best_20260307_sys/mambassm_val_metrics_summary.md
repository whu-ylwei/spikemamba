# mamba_ssm Validation Metrics Summary

Model checkpoint:
`/root/shared-nvme/spikemamba/train_runs/checkpoints/train_s2d_MambaSSM_STRICT_20260306_124138_bs2/model_best.pth.tar`

Dataset:
`/root/shared-nvme/spikemamba/DENSE-spike/DENSE/validation`

Metrics source:
`/root/shared-nvme/spikemamba/logs/val_mambassm_best_20260307_sys/eval_metrics.log`

| Metric | Value |
|---|---:|
| Abs Rel (down) | 0.455506 |
| Sq Rel (down) | 8.500888 |
| RMS log (down) | 0.517973 |
| SI log (down) | 0.204664 |
| delta<1.25 (up) | 0.692932 |
| delta<1.25^2 (up) | 0.814251 |
| delta<1.25^3 (up) | 0.880025 |
