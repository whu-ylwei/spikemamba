# Results Package

This directory collects the experiment artifacts that were not included in the earlier code-only push.

Included content:
- `docs/`: training and evaluation summary notes.
- `logs/`: launcher logs and training logs.
- `reports/`: analysis reports, CSV summaries, and generated plots.
- `runs/eval/`: evaluation reports, metrics logs, and visualization images that are GitHub-safe.
- `runs/train/`: lightweight training artifacts such as config snapshots and loss curves.
- `placeholders/`: README markers for directories whose large binary contents were not copied here.
- `misc/`: leftover lightweight files that were not part of the earlier push.

Excluded from this package:
- Large checkpoints such as `*.pth.tar`.
- TensorBoard event files such as `events.out.tfevents*`.
- Dense numeric dumps such as `*.npy` and `*.npz`.
- Local datasets, virtual environments, and wheel/package caches.

Reason for exclusion:
- Several omitted files exceed GitHub's single-file size limit or would make the repository impractically large.
