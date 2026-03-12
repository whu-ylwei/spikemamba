# Unified Naming

命名规则：`dayXX_runYY`（按日期和时间顺序）。以下为映射：

| unified | date | time | type | original |
|---|---|---|---|---|
| day01_run01 | 20260305 | 000000 | train_log | /root/shared-nvme/spikemamba/logs/train/train_rtx5090_20260305.log |
| day01_run02 | 20260305 | 104342 | train_log | /root/shared-nvme/spikemamba/logs/train/train_20260305_104342.log |
| day01_run03 | 20260305 | 113710 | train_log | /root/shared-nvme/spikemamba/logs/train/train_opt_20260305_113710.log |
| day01_run04 | 20260305 | 114542 | train_log | /root/shared-nvme/spikemamba/logs/train/train_opt_20260305_114542.log |
| day01_run05 | 20260305 | 125250 | train_log | /root/shared-nvme/spikemamba/logs/train/train_resume_20260305_125250.log |
| day01_run06 | 20260305 | 135122 | train_log | /root/shared-nvme/spikemamba/logs/train/train_fromscratch_check_20260305_135122.log |
| day01_run07 | 20260305 | 135133 | train_log | /root/shared-nvme/spikemamba/logs/train/train_fromscratch_check_20260305_135133.log |
| day02_run01 | 20260306 | 044351 | train_log | /root/shared-nvme/spikemamba/logs/train/train_run_20260306_044351.log |
| day02_run02 | 20260306 | 120000 | eval_dir | /root/shared-nvme/spikemamba/runs/eval/test_gpu_20260306 |
| day02_run03 | 20260306 | 120000 | eval_dir | /root/shared-nvme/spikemamba/runs/eval/test_gpu_best_epoch19_20260306 |
| day02_run04 | 20260306 | 124138 | checkpoint_dir | /root/shared-nvme/spikemamba/runs/train/train_s2d_MambaSSM_STRICT_20260306_124138 |
| day02_run05 | 20260306 | 124138 | checkpoint_dir | /root/shared-nvme/spikemamba/runs/train/train_s2d_MambaSSM_STRICT_20260306_124138_bs2 |
| day02_run06 | 20260306 | 124138 | checkpoint_dir | /root/shared-nvme/spikemamba/runs/train/train_s2d_MambaSSM_STRICT_20260306_124138_bs2_20260308_055419 |
| day03_run01 | 20260307 | 120000 | eval_dir | /root/shared-nvme/spikemamba/runs/eval/test_mambassm_best_20260307 |
| day03_run02 | 20260307 | 120000 | eval_dir | /root/shared-nvme/spikemamba/runs/eval/test_mambassm_best_20260307_sys |
| day03_run03 | 20260307 | 120000 | eval_dir | /root/shared-nvme/spikemamba/runs/eval/val_mambassm_best_20260307_sys |
| day04_run01 | 20260308 | 000000 | train_dir | /root/shared-nvme/spikemamba/runs/train/train_s2d_SpikeTransformer_mkl_once_20260308 |
| day04_run02 | 20260308 | 120000 | eval_dir | /root/shared-nvme/spikemamba/runs/eval/mamba_mkl_20260308 |
