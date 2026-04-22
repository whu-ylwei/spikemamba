# SpikeMamba 改动/训练/评估维护 README

本文件作为 `spikemamba` 当前实验线的统一维护入口，集中整理：

- 代码与配置改动日期
- 对应训练链路和最好训练结果
- 已有的 test / val / 外部数据集评估
- 关键证据路径

说明：

- `记录日期` 优先使用 git 提交日期；未提交内容标记为 `workspace`。
- `训练结果` 优先使用 TensorBoard 中的 `val_loss`；没有时退回已整理文档或 checkpoint 命名。
- `评估结果` 优先记录 `AbsRel / RMS_linear / RMS_log / SILog / delta_1.25`。
- 本文件只保留维护视角下的摘要，逐 batch 明细、图片和大体量日志不在此展开。

## 1. 维护规则

- 新增训练链路时，至少补 6 个字段：
  - 日期
  - 变更说明
  - 配置文件
  - 训练 run 目录
  - 最好训练结果
  - 评估结果或“未补评估”
- 未提交到 git 的内容单独标 `workspace`，避免和正式提交历史混淆。
- 同一 checkpoint 在不同数据集上复评时，保留多条评估记录，不覆盖旧结果。

## 2. 快速索引

| 用途 | 文件 |
|---|---|
| 本维护总表 | `README_EXPERIMENT_HISTORY.md` |
| 03-05 到 03-11 详细时间线归档 | `results/docs/ALL_TRAIN_LOGS_PARAMS_EVAL_CHRONOLOGY_20260311.md` |
| 当前数据流说明 | `README_CURRENT_DATAFLOW_20260317.md` |
| Mamba 编码器分析 | `README_MAMBA_ENCODER_20260319.md` |
| 0323 两编码器续训记录 | `results/docs/README_0323_2ENC_100PLUS100_CONTINUATION_20260324.md` |
| 0410 80 epoch 快照说明 | `results/docs/README_0410_CORE_AND_80EPOCH_BEST.md` |

## 3. 主线时间表

| 记录日期 | 来源 | 对应实验日期 | 主要改动 | 训练结果摘要 | 评估摘要 | 关键证据 |
|---|---|---|---|---|---|---|
| 2026-03-12 | git `617c34d1`, `817665fd` | 2026-03-05 到 2026-03-08 | 补齐核心 SpikeMamba 代码、Mamba 后端、RTX5090 配置、严格 `mamba_ssm` 配置、训练/评测 README 和结果归档 | `train_s2d_SpikeTransformer_rtx5090_20260305_20260306_044356`: best `val_loss=0.042128 @ epoch 19`；`train_s2d_MambaSSM_STRICT_20260306_124138_bs2`: best `val_loss=0.016148 @ epoch 28`；`train_s2d_MambaSSM_STRICT_20260306_124138_bs2_20260308_055419`: best `val_loss=0.015840 @ epoch 24` | fallback best epoch19 test: `AbsRel=0.779262`, `RMS=388.739310`, `d1=0.154195`；MambaSSM val: `AbsRel=0.455506`, `d1=0.692932`；MambaSSM test: `AbsRel=0.859811`, `d1=0.514184`；03-11 续跑复评: `AbsRel=0.818072`, `d1=0.560016` | `configs/train_s2d_spiketransformer_rtx5090_20260305.json`; `configs/train_s2d_spiketransformer_mambassm_strict_20260306_124138.json`; `results/docs/ALL_TRAIN_LOGS_PARAMS_EVAL_CHRONOLOGY_20260311.md` |
| 2026-03-17 | git `26600d15` | 2026-03-17 | 引入 spatial-sequence 序列组织，更新 `model/mamba_3d.py`，补 `ep28->30->60` 调度脚本和配置，新增当前数据流说明 | `train_s2d_MambaSSM_spatialseq_30ep_20260317_bs2`: best `val_loss=0.013964 @ epoch 28`，final `0.013994 @ epoch 30`；`train_s2d_MambaSSM_spatialseq_ep30_to60_halflr_20260317_bs2`: best `val_loss=0.013499 @ epoch 56`，final `0.013910 @ epoch 60` | epoch28 test: `AbsRel=0.933878`, `RMS=147.006386`, `d1=0.622136`；epoch56 test: `AbsRel=0.717065`, `RMS=145.130972`, `d1=0.630309` | `configs/train_s2d_spiketransformer_mambassm_spatialseq_30ep_20260317.json`; `configs/train_s2d_spiketransformer_mambassm_spatialseq_ep30_to60_halflr_20260317.json`; `README_CURRENT_DATAFLOW_20260317.md` |
| 2026-03-18 | git `ab827394`, `e0faa027` | 2026-03-18 | 增加 `40ep / 80ep / 120ep / 40->120` 的 spatialseq 配置与调度脚本，并加入评估导出脚本 | `train_s2d_MambaSSM_spatialseq_40ep_20260318_bs2`: best `val_loss=0.013970 @ epoch 25`；`train_s2d_MambaSSM_spatialseq_120ep_20260318_bs2`: best `val_loss=0.013748 @ epoch 56`，final `0.014504 @ epoch 120` | 03-18/03-19 多次 test 记录都指向同一 best checkpoint `epoch 25`：`AbsRel=0.767677`, `RMS=148.383508`, `d1=0.605921` | `configs/train_s2d_spiketransformer_mambassm_spatialseq_120ep_20260318.json`; `configs/train_s2d_spiketransformer_mambassm_spatialseq_ep40_to80_halflr_20260318.json`; `results/reports/analysis/test_mambassm_spatialseq_best_ep25_20260318_sys_metrics.csv` |
| 2026-03-20 | git `947c104a` | 2026-03-19 到 2026-03-20 | 在 `model/mamba_3d.py` 中把单向 Mamba 改成 Vim-style 双向块，开始 bidirectional Mamba 训练线 | `train_s2d_MambaSSM_bidivim_100ep_20260319_bs2`: best `val_loss=0.014163 @ epoch 43`；`train_s2d_MambaSSM_bidivim_ep40_continue40_halflr_20260320_bs2`: best `val_loss=0.014110 @ epoch 70`，final `0.014541 @ epoch 80` | 100ep best test: `AbsRel=0.596676`, `RMS=161.351207`, `d1=0.661908`；continue40 best test: `AbsRel=0.592840`, `RMS=168.946036`, `d1=0.658142`；continue40 val: `AbsRel=0.513091`, `RMS=81.905808`, `d1=0.767781` | `model/mamba_3d.py`; `configs/train_s2d_spiketransformer_mambassm_bidivim_nograd_100ep_20260320.json`; `configs/train_s2d_MambaSSM_bidivim_ep40_continue40_halflr_20260320_bs2.json`; `README_MAMBA_ENCODER_20260319.md` |
| 2026-03-23 到 2026-03-24 | workspace | 2026-03-23 到 2026-03-24 | 开出两编码器续训链路，`num_encoders=2`，`swin_depths=[2,6]`，按 `60 -> 100 -> 200` 三段训练；stage3 实际落盘到 `/root/runs/train/0323/` | stage1 `train_s2d_MambaSSM_bidivim_nograd_2enc_60ep_20260323_bs2`: best `val_loss=0.014740 @ epoch 21`；stage2 `...ep60_to100_keep_lr...`: best `0.014877 @ epoch 73`；stage3 `...ep100_to200_halflr...`: best `0.014996 @ epoch 117`，final `0.015839 @ epoch 200`；当前全局最好仍是 stage1 epoch21 | 截至当前未发现对应最终 test / val 评估归档，应后补 | `0323/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_60ep_20260323.json`; `results/docs/README_0323_2ENC_100PLUS100_CONTINUATION_20260324.md`; `/root/runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep100_to200_halflr_20260323_bs2/` |
| 2026-04-10 | git `6ab91633` | 2026-04-10 | 固化 80 epoch best checkpoint 的配置快照，并补 `scape-spike` 评估 JSON | 该节点沿用 `train_s2d_MambaSSM_bidivim_ep40_continue40_halflr_20260320_bs2` 的 best checkpoint | `scape-spike/test`: `AbsRel=0.528915`, `RMS=146.752095`, `RMS_log=0.607155`, `SILog=0.229574`, `d1=0.385792` | `results/docs/README_0410_CORE_AND_80EPOCH_BEST.md`; `results/reports/analysis/scape_spike_80epoch_eval_20260410.json` |
| 2026-04-12 到 2026-04-13 | workspace | 2026-04-12 到 2026-04-13 | 补 `DENSE-spike` 复评、`scape-spike` KITTI 稀疏评估，并记录一条缺少 `mamba_ssm` 环境的失败测试 | 训练本身无新增；`test_mambassm_bidivim_epoch080_dense_spike_20260412.log` 记录了 `mamba_ssm` 缺失导致的测试失败 | 04-12 `DENSE-spike/DENSE/test` 复评: `AbsRel=0.592840`, `RMS=168.946036`, `d1=0.658142`；04-13 `kitti_0001_scape_spike_sparse` frame-mean: `AbsRel=0.696637`, `RMS=17.649996`, `RMS_log=1.203875`, `d1=0.001655` | `run_logs/eval_mambassm_bidivim_best80_dense_spike_20260412.log`; `results/reports/analysis/kitti_scape_sparse_eval_20260413/metrics_summary.json`; `run_logs/test_mambassm_bidivim_epoch080_dense_spike_20260412.log` |
| 2026-04-16 | workspace | 2026-04-16 | 将 `ST_Mamba_Block` 中两处残差加法（`x = x + x1`、`x = x + x2`）替换为 `MHCResidual`（Multi-Head Cross-attention 残差）：query 为主流 x，key/value 为分支输出，使用 `nn.MultiheadAttention`，num_heads=8（自动向下对齐到整除 dim 的最大值）；新增 `CLAUDE.md` 记录该设计决策；新增从头训练配置 `train_s2d_MambaSSM_mhc_residual_100ep_20260416_bs2` | 已启动但首轮 forward 即失败；`runs/train/train_s2d_MambaSSM_mhc_residual_100ep_20260416_bs2/` 已创建，但当前无有效 `loss/val_loss` 标量 | 未补评估；训练日志记录 `torch.OutOfMemoryError: Tried to allocate 150.06 GiB` | `model/mamba_3d.py`; `CLAUDE.md`; `configs/train_s2d_MambaSSM_mhc_residual_100ep_20260416_bs2.json`; `run_logs/train_mhc_residual_100ep_20260416.log` |

## 4. 当前主线 checkpoint 对照

| 训练线 | 推荐查看 checkpoint / run | 最好训练结果 | 当前已知最好评估 |
|---|---|---|---|
| fallback SpikeTransformer | `runs/train/train_s2d_SpikeTransformer_rtx5090_20260305_20260306_044356/` | `val_loss=0.042128 @ epoch 19` | DENSE test `AbsRel=0.779262` |
| strict MambaSSM bs2 | `runs/train/train_s2d_MambaSSM_STRICT_20260306_124138_bs2_20260308_055419/` | `val_loss=0.015840 @ epoch 24` | DENSE test `AbsRel=0.818072` |
| spatialseq 主线 | `runs/train/train_s2d_MambaSSM_spatialseq_ep30_to60_halflr_20260317_bs2/` | `val_loss=0.013499 @ epoch 56` | DENSE test `AbsRel=0.717065` |
| spatialseq 长训练 | `runs/train/train_s2d_MambaSSM_spatialseq_120ep_20260318_bs2/` | `val_loss=0.013748 @ epoch 56` | 长训练后 best checkpoint 仍回到早期 best，test `AbsRel=0.767677` |
| bidivim 100ep | `runs/train/train_s2d_MambaSSM_bidivim_100ep_20260319_bs2/` | `val_loss=0.014163 @ epoch 43` | DENSE test `AbsRel=0.596676` |
| bidivim continue40 best80 | `runs/train/train_s2d_MambaSSM_bidivim_ep40_continue40_halflr_20260320_bs2/` | `val_loss=0.014110 @ epoch 70` | val `AbsRel=0.513091`；scape-spike test `AbsRel=0.528915` |
| 0323 两编码器 | `runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_60ep_20260323_bs2/` 和 `/root/runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep100_to200_halflr_20260323_bs2/` | 全局最好 `val_loss=0.014740 @ epoch 21` | 最终 test / val 结果未补 |
| MHC 残差 100ep | `runs/train/train_s2d_MambaSSM_mhc_residual_100ep_20260416_bs2/` | 首轮 forward OOM，当前无有效训练曲线 | 未补评估 |

## 5. 旁支与补充记录

### 5.1 0313 / 0314 MKL 分支

这条线在仓库历史里曾以大体量结果文件形式被加入又回退，但当前工作区仍保留 run 目录和日志，可作为旁支参考：

| 记录日期 | 来源 | run | 训练结果 | 备注 |
|---|---|---|---|---|
| 2026-03-16 | 曾提交后回退的历史内容 | `runs/train/0313_mkl/` | best `val_loss=0.036335 @ epoch 16`，final `0.038237 @ epoch 125` | 主要保留了 launcher 日志和可视化产物 |
| 2026-03-16 | 曾提交后回退的历史内容 | `runs/train/0314_mkl/` | best `val_loss=0.024204 @ epoch 20`，final `0.024837 @ epoch 60` | 同样属于旁支记录，不是当前主线 |

## 6. 后续补充建议

- 新增训练后先补本文件，再补专项 README，避免历史继续分散。
- `0323` 两编码器链路建议补 3 项：
  - 最终 best checkpoint 的 test 评估
  - 最终 best checkpoint 的 val 评估
  - 与 `bidivim continue40 best80` 的并排对比表
- 新的数据集评测建议统一落到 `results/reports/analysis/` 下，并输出 JSON 或 CSV 汇总，便于后续自动汇总到本表。

## 7. 本次修改整理（2026-04-16 MHC Residual）

### 7.1 修改目标

- 将 `ST_Mamba_Block` 中原本的“直接残差相加”改成“可学习的跨注意力残差融合”。
- 目标是让主干特征 `x` 不再无条件接收分支输出，而是通过 attention 选择性吸收 `x1` 和 `x2`。

### 7.2 实际改动文件

| 文件 | 改动 |
|---|---|
| `model/mamba_3d.py` | 新增 `MHCResidual`；把 `x = x + x1`、`x = x + x2` 分别替换为 `self.mhc1(x, x1)`、`self.mhc2(x, x2)` |
| `configs/train_s2d_MambaSSM_mhc_residual_100ep_20260416_bs2.json` | 新增 100 epoch 从头训练配置，`batch_size=2`，`lr=1e-4`，`save_freq=10` |
| `CLAUDE.md` | 记录这次“Residual -> MHC”的设计决策与实现约束 |
| `run_logs/train_mhc_residual_100ep_20260416.log` | 当前训练启动日志与报错记录 |

### 7.3 代码级变化

- 新模块 `MHCResidual(dim, num_heads=8)`：
  - 输入 `x`、`branch` 形状均为 `[B, C, T, H, W]`
  - 将两者展平为 `[B, T*H*W, C]`
  - `query = norm(x)`
  - `key/value = branch`
  - 经 `nn.MultiheadAttention` 后再 reshape 回 `[B, C, T, H, W]`
  - 输出为 `x + attn(x, branch)`
- `ST_Mamba_Block.forward()` 中两处残差路径已改为：
  - `self.mhc1(x, x1)`：Mamba 输出分支融合回主流
  - `self.mhc2(x, x2)`：FFN 输出分支融合回主流

### 7.4 训练配置摘要

| 项目 | 值 |
|---|---|
| run name | `train_s2d_MambaSSM_mhc_residual_100ep_20260416_bs2` |
| epochs | `100` |
| batch_size | `2` |
| lr | `1e-4` |
| scheduler | `ExponentialLR(gamma=0.5, freq=100)` |
| num_encoders | `3` |
| base_num_channels | `96` |
| backbone | `mamba_ssm` |
| 数据 | `DENSE/train` + `DENSE/validation` |

### 7.5 当前结果

- 训练目录已创建：
  - `runs/train/train_s2d_MambaSSM_mhc_residual_100ep_20260416_bs2/`
- 当前只有配置和空 TensorBoard 事件文件：
  - 尚无 `loss`、`val_loss`、`learning_rate` 标量
- 尚无 test / val 评估产物

### 7.6 当前阻塞点

- 训练在首轮 forward 进入 `self.mhc1(x, x1)` 时 OOM。
- 日志报错：
  - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 150.06 GiB`
- 直接原因：
  - 当前 `MHCResidual` 在完整的 `[T * H * W]` token 序列上做全局 attention
  - stage 0 token 数过大，attention 矩阵规模约为 `(T*H*W) x (T*H*W)`，显存开销远超 31GB GPU 上限

### 7.7 建议的后续修正方向

- 不要在完整 `T*H*W` 上直接做全局 attention。
- 更现实的替代方案：
  - 仅在 `_fold_spatial_tokens()` 后的压缩 token 上做 MHC
  - 只对时间维做 attention，空间维保持局部
  - 对 stage 0 跳过 MHC，仅在更深层使用
  - 降低分辨率 / token 长度后再试验
