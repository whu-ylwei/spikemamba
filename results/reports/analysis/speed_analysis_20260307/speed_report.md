# Speed Analysis (2026-03-07)

## Method
- Training speed: extracted from TensorBoard scalar wall-time (`loss` per epoch).
- Runtime speed: computed from file mtime span over 997 output frames (`npy/image` for test, `eval_vis` for eval).

## Training
- fallback: total 4619.4s (77.0 min), avg 159.3s/epoch
- mamba_ssm: total 21861.2s (364.4 min), avg 753.8s/epoch
- mamba_ssm vs fallback: epoch time x4.73 (slower)

## Runtime (997 frames)
- fallback test: 207.6s, 4.80 fps
- mamba_ssm test: 243.9s, 4.09 fps
- test speed ratio (mamba_ssm/fallback): 0.85x
- fallback eval: 270.9s, 3.68 fps
- mamba_ssm eval: 276.8s, 3.60 fps
- eval speed ratio (mamba_ssm/fallback): 0.98x

## Figures
- training_epoch_time_curve.png
- training_cumulative_time.png
- runtime_fps_comparison.png
- runtime_duration_comparison.png
