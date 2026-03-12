# 保留参数分析与清理结果（2026-03-06）

## 1) 已保留的训练目录

| 目录 | config epochs | save_freq | monitor | 保存的 checkpoint epoch | model_best epoch | monitor_best |
|---|---:|---:|---|---|---:|---:|
| train_s2d_SpikeTransformer_fromscratch_check_20260305 | 2 | 1 | val_loss (min) | [1, 2] | 1 | 0.22367235159873963 |
| train_s2d_SpikeTransformer_rtx5090_20260305_20260305_114546 | 30 | 8 | val_loss (min) | [8, 16, 24] | 1 | 0.22367235159873963 |
| train_s2d_SpikeTransformer_rtx5090_20260305_20260306_044356 | 30 | 8 | val_loss (min) | [8, 16, 24] | 19 | 0.04212769944965839 |
| train_verify_20260306 | 1 | 1 | val_loss (min) | [1] | 1 | 0.14055547788739203 |

## 2) best=19 测试结果（已完成）

| 指标 | epoch1(旧测试) | best19(本次) | 变化 |
|---|---:|---:|---:|
| AbsRel(↓) | 1.306131 | 0.779262 | -0.526869 |
| SqRel(↓) | 5.317150 | 1.945662 | -3.371488 |
| RMS log(↓) | 2.424063 | 1.891718 | -0.532345 |
| SIlog(↓) | 2.167130 | 1.160968 | -1.006162 |
| delta<1.25(↑) | 0.096819 | 0.154195 | +0.057376 |
| delta<1.25^2(↑) | 0.155237 | 0.228663 | +0.073426 |
| delta<1.25^3(↑) | 0.219581 | 0.290279 | +0.070698 |

## 3) 已删除的无效文件夹

无效判定规则：无 `model_best.pth.tar` 且无 `checkpoint-epoch*.pth.tar`。

- train_s2d_SpikeTransformer
- train_s2d_SpikeTransformer_rtx5090_20260305
- train_s2d_SpikeTransformer_rtx5090_20260305_20260305_102602
- train_s2d_SpikeTransformer_rtx5090_20260305_20260305_102920
- train_s2d_SpikeTransformer_rtx5090_20260305_20260305_104347
- train_s2d_SpikeTransformer_rtx5090_20260305_20260305_113716
- train_s2d_SpikeTransformer_rtx5090_20260305_smoke
- train_s2d_SpikeTransformer_rtx5090_20260305_smoke2

## 4) 安全校验

- `train_s2d_SpikeTransformer_rtx5090_20260305_20260306_044356`（best=19）已保留。
