This folder is a curated snapshot that contains only:

1. The current project code snapshot under `code/`
2. The third result's parameters under `run3/params/`
3. The third result's best checkpoint under `run3/weights/`
4. The third result's core evaluation outputs under `run3/results/`

Included run:
- `test_mambassm_best_20260307_sys`
- Checkpoint source: `train_s2d_MambaSSM_STRICT_20260306_124138_bs2/model_best.pth.tar`
- Best checkpoint epoch: `28`

Not included:
- Other training runs
- Other test runs
- Other checkpoints
- TensorBoard directories
- Large raw visualization/video/npy outputs for unrelated runs
- Datasets, virtual environments, wheels, git metadata

Note:
- This is the minimal clean package for the third run only.
- For the third run, core result files are kept as text logs/summary files in `run3/results/`.
- The runtime code snapshot is aligned to the pre-MKL run3 timeline (core code files dated 2026-03-05/06).
- Later offline-evaluation helper code and `mkl-*` package pins were removed from this package.
- In this Git branch, `run3/weights/model_best.pth.tar` is stored as split parts because GitHub rejects single files larger than 100 MB.
