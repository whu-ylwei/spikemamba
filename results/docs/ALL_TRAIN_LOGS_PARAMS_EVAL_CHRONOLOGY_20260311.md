# SpikeMamba 训练日志/参数/测评结果时间线（按时间顺序）

说明：
- 时间优先取文件名中的时间戳（`YYYYMMDD_HHMMSS`），无时分秒时按日期归档。
- 仅整理“已记录”的训练日志、配置参数与测评结果。
- 大体量逐 batch 明细（loss 列表）不在此全文展开，保留原日志路径。

## 1) 训练日志时间线

| 时间 | 类型 | 记录文件 | 关联参数文件 | 结果摘要 |
|---|---|---|---|---|
| 2026-03-05 10:43:42 | 训练 | `logs/train/train_20260305_104342.log` | `configs/train_s2d_spiketransformer.json` | 训练到 epoch 1 末尾后报错：`KeyError: 'image'` |
| 2026-03-05 (时间未知) | 训练 | `logs/train/train_rtx5090_20260305.log` | `configs/train_s2d_spiketransformer_rtx5090_20260305.json` | 启动阶段报错：`np.alltrue` 被 NumPy 2.0 移除 |
| 2026-03-05 11:37:10 | 训练优化尝试 | `logs/train/train_opt_20260305_113710.log` | 同期 spiketransformer 配置 | 报错：`UnboundLocalError: step` |
| 2026-03-05 11:45:42 | 训练优化尝试 | `logs/train/train_opt_20260305_114542.log` | `configs/train_s2d_spiketransformer_rtx5090_20260305.json` | 成功保存 checkpoint：epoch 8/16/24；路径指向 `runs/train/train_s2d_SpikeTransformer_rtx5090_20260305_20260305_114546/` |
| 2026-03-05 12:52:50 | 断点续训 | `logs/train/train_resume_20260305_125250.log` | 同期 spiketransformer 配置 | 日志包含大量训练/验证 loss 记录（原始日志保留） |
| 2026-03-05 13:51:22 | from-scratch检查 | `logs/train/train_fromscratch_check_20260305_135122.log` | `configs/train_s2d_spiketransformer_rtx5090_20260305_fromscratch_check.json` | 启动命令问题：`/usr/bin/time` 缺失 |
| 2026-03-05 13:51:33 | from-scratch检查 | `logs/train/train_fromscratch_check_20260305_135133.log` | `configs/train_s2d_spiketransformer_rtx5090_20260305_fromscratch_check.json` | 成功保存 `model_best.pth.tar`；保存 epoch1/epoch2 checkpoint（loss 约 0.2253/0.2258） |
| 2026-03-06 04:43:51 | 训练运行 | `logs/train/train_run_20260306_044351.log` | 同期 spiketransformer/mamba 实验 | 含训练过程与验证 loss 明细（原始日志保留） |
| 2026-03-06 12:41:38 | MambaSSM 启动 #1 | `logs/train/launcher/train_s2d_MambaSSM_STRICT_20260306_124138.log` | `configs/train_s2d_spiketransformer_mambassm_strict_20260306_124138.json` | 报错：缺少 `kornia` |
| 2026-03-06 12:41:38 | MambaSSM 启动 #2 | `logs/train/launcher/train_s2d_MambaSSM_STRICT_20260306_124138.retry1.log` | 同上 | 报错：缺少 `skimage` |
| 2026-03-06 12:41:38 | MambaSSM 启动 #3 | `logs/train/launcher/train_s2d_MambaSSM_STRICT_20260306_124138.retry2.log` | 同上 | 报错：NumPy 2 与 OpenCV 编译不兼容 |
| 2026-03-06 12:41:38 | MambaSSM 启动 #4 | `logs/train/launcher/train_s2d_MambaSSM_STRICT_20260306_124138.retry3.log` | 同上 | 训练中报错：`CUDA out of memory` |
| 2026-03-06 (bs2) | MambaSSM 训练 | `logs/train/launcher/train_s2d_MambaSSM_STRICT_20260306_124138_bs2.retry4.log` | `runs/train/train_s2d_MambaSSM_STRICT_20260306_124138_bs2/config.json` | 成功跑通并在 `runs/train/..._bs2/` 产出 `model_best` 与 epoch 8/16/24 checkpoint |
| 2026-03-08 05:54:19 | MambaSSM 续跑 | （由 checkpoint 文件夹命名） | `runs/train/train_s2d_MambaSSM_STRICT_20260306_124138_bs2_20260308_055419/config.json` | 成功产出更优 checkpoint：epoch24 loss 0.0039 |
| 2026-03-08 (一次性实验) | MKL命名实验 | （无单独训练log） | `runs/train/train_s2d_SpikeTransformer_mkl_once_20260308/config.json` | 记录到 `model_best.pth.tar`（epochs=1） |

## 2) 参数配置快照（按时间）

| 时间 | 配置文件 | name | arch | batch_size | lr | epochs | optimizer/scheduler | Mamba相关 |
|---|---|---|---|---:|---:|---:|---|---|
| 2026-03-05 | `configs/train_s2d_spiketransformer.json` | train_s2d_SpikeTransformer | S2DepthTransformerUNetConv | 1 | 3e-4 | 30 | Adam / ExponentialLR | 无 |
| 2026-03-05 | `configs/train_s2d_spiketransformer_rtx5090_20260305.json` | train_s2d_SpikeTransformer_rtx5090_20260305 | S2DepthTransformerUNetConv | 8 | 1e-4 | 30 | Adam / ExponentialLR | 无 |
| 2026-03-05 | `configs/train_s2d_spiketransformer_rtx5090_20260305_fromscratch_check.json` | train_s2d_SpikeTransformer_fromscratch_check_20260305 | S2DepthTransformerUNetConv | 8 | 1e-4 | 2 | Adam / ExponentialLR | 无 |
| 2026-03-06 12:41:38 | `configs/train_s2d_spiketransformer_mambassm_strict_20260306_124138.json` | train_s2d_MambaSSM_STRICT_20260306_124138_bs2 | S2DepthTransformerUNetConv | 2 | 1e-4 | 30 | Adam / ExponentialLR | `mamba_backend=mamba_ssm`, `mamba_require_ssm=true` |
| 2026-03-06 | `runs/train/train_s2d_MambaSSM_STRICT_20260306_124138/config.json` | train_s2d_MambaSSM_STRICT_20260306_124138 | S2DepthTransformerUNetConv | 8 | 1e-4 | 30 | Adam / ExponentialLR | 同上 |
| 2026-03-06 | `runs/train/train_s2d_MambaSSM_STRICT_20260306_124138_bs2/config.json` | train_s2d_MambaSSM_STRICT_20260306_124138_bs2 | S2DepthTransformerUNetConv | 2 | 1e-4 | 30 | Adam / ExponentialLR | 同上 |
| 2026-03-08 05:54:19 | `runs/train/train_s2d_MambaSSM_STRICT_20260306_124138_bs2_20260308_055419/config.json` | train_s2d_MambaSSM_STRICT_20260306_124138_bs2_20260308_055419 | S2DepthTransformerUNetConv | 2 | 1e-4 | 30 | Adam / ExponentialLR | 同上 |
| 2026-03-08 | `runs/train/train_s2d_SpikeTransformer_mkl_once_20260308/config.json` | train_s2d_SpikeTransformer_mkl_once_20260308 | S2DepthTransformerUNetConv | 1 | 3e-4 | 1 | Adam / ExponentialLR | 文件名含 `mkl_once` |

## 3) 测评结果时间线（按时间）

| 时间 | 测评日志 | 场景 | AbsRel | RMS_linear | RMS_log | SILog | δ1.25 |
|---|---|---|---:|---:|---:|---:|---:|
| 2026-03-06 | `runs/eval/test_gpu_20260306/eval_metrics_20260306.log` | GPU测试 | 1.306131 | 431.365349 | 2.424063 | 2.167130 | 0.096819 |fallback的epoch1
| 2026-03-06 | `runs/eval/test_gpu_best_epoch19_20260306/eval_metrics.log` | GPU best epoch19 测试 | 0.779262 | 388.739310 | 1.891718 | 1.160968 | 0.154195 |fallback的epoch19
| 2026-03-07 | `runs/eval/test_mambassm_best_20260307_sys/eval_metrics.log` | MambaSSM 测试集 | 0.859811 | 150.583446 | 0.851177 | 0.480538 | 0.514184 |mambassm的test
| 2026-03-07 | `runs/eval/val_mambassm_best_20260307_sys/eval_metrics.log` | MambaSSM 验证集 | **0.455506** | **96.525860** | **0.517973** | **0.204664** | **0.692932** |mambassm的val
| 2026-03-08（结果） / 2026-03-11（离线评测） | `runs/eval/mamba_mkl_20260308/eval_metrics_20260311/eval_metrics.log` | mamba_mkl_20260308（基于已保存npy离线评估） | 0.818071 | 11.725683 | 0.802200 | 0.456358 | 0.560028 |mambassm+mkl的epoch1

## 4) 关键产物位置（按时间）

- 2026-03-05 fromscratch 检查产物：
  - `runs/train/train_s2d_SpikeTransformer_fromscratch_check_20260305/`
- 2026-03-05 RTX5090 训练产物：
  - `runs/train/train_s2d_SpikeTransformer_rtx5090_20260305_20260305_114546/`
- 2026-03-06 MambaSSM bs2 产物：
  - `runs/train/train_s2d_MambaSSM_STRICT_20260306_124138_bs2/`
- 2026-03-08 MambaSSM bs2 续跑产物（更低训练loss）：
  - `runs/train/train_s2d_MambaSSM_STRICT_20260306_124138_bs2_20260308_055419/`
- 2026-03-08 mkl_once 产物：
  - `runs/train/train_s2d_SpikeTransformer_mkl_once_20260308/`
