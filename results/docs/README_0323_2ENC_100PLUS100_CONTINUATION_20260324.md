# SpikeMamba 0323 两编码器续训记录（2026-03-24）

> 目的：整理 `0323` 这组两编码器 MambaSSM 实验的续训链路、关键路径、监控修复和当前状态，方便后续直接接续。  
> 范围：`epoch 1-60` 基础训练，`epoch 60-100` 保持学习率续训，`epoch 100-200` 半学习率续训。  
> 最后更新：2026-03-24（UTC）

## 0. 当前上下文

- 项目路径：`/root/shared-nvme/spikemamba`
- 实际记录对象：`0323` 的 **两编码器** 实验，不是更早的三编码器配置。
- 当前计划：保留前 60 个 epoch，不重训；改为：
  - `1 -> 60`：`lr=1e-4`
  - `60 -> 100`：`lr=1e-4`
  - `100 -> 200`：`lr=5e-5`
- 当前训练状态：`stage3_halflr` 进行中，最新已到 `epoch 135`。

## 1. 有效配置快照

| 项目 | 值 |
|---|---|
| arch | `S2DepthTransformerUNetConv` |
| batch_size | `2` |
| optimizer | `Adam` |
| stage1/stage2 lr | `1e-4` |
| stage3 lr | `5e-5` |
| loss | `scale_invariant_loss` |
| grad_loss.weight | `0.25` |
| num_encoders | `2` |
| base_num_channels | `96` |
| num_residual_blocks | `2` |
| swin_depths | `[2, 6]` |
| swin_num_heads | `[3, 6]` |
| mamba_backend | `mamba_ssm` |
| mamba_require_ssm | `true` |
| trainer epochs | `200`（stage3 配置目标） |
| save_freq | `20` |

## 2. 三段训练时间线

| 阶段 | epoch 范围 | 状态 | 当前/最后 val_loss | 本段最好 val_loss | 说明 |
|---|---:|---|---|---|---|
| stage1 | `1-60` | 已完成 | `0.015115128830 @ epoch 60` | `0.014739707112 @ epoch 21` | 初始 60 epoch 基础训练 |
| stage2_keep_lr | `61-100` | 已完成 | `0.015253812075 @ epoch 100` | `0.014876967296 @ epoch 73` | 保持 `lr=1e-4` 续到 100 |
| stage3_halflr | `101-200` | 进行中 | `0.015584134497 @ epoch 135` | `0.014995988458 @ epoch 117` | 半学习率续训，当前 live |

补充：

- 全程截至当前的全局最好 `val_loss` 仍是 `0.014739707112 @ epoch 21`。
- `stage3` 当前最新 checkpoint 仍是 `epoch 120`，因为 `save_freq=20`，live event 已经走到 `epoch 135`。

## 3. 关键路径

### 3.1 配置文件

- stage1：
  - `0323/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_60ep_20260323.json`
- stage2：
  - `0323/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_ep60_to100_keep_lr_20260323.json`
- stage3：
  - `0323/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_ep100_to200_halflr_20260323.json`
- 已废弃旧分支：
  - `0323/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_ep60_to120_halflr_20260323.json`

### 3.2 训练日志

- 主控脚本日志：
  - `run_logs/0323/run_0323_60plus60_halflr.log`
- monitor 日志：
  - `run_logs/0323/monitor_0323_60plus60_halflr.log`
- stage1 日志：
  - `run_logs/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_60ep_20260323.log`
- stage2 日志：
  - `run_logs/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to100_keep_lr_20260323.log`
- stage3 日志：
  - `run_logs/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep100_to200_halflr_20260323.log`

### 3.3 训练产物目录

- stage1 run dir：
  - `/root/shared-nvme/spikemamba/runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_60ep_20260323_bs2`
- stage2 run dir：
  - `/root/shared-nvme/spikemamba/runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to100_keep_lr_20260323_bs2`
- stage3 实际 run dir：
  - `/root/runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep100_to200_halflr_20260323_bs2`

注意：

- stage3 是从 `/root` 手动启动的，因此相对 `save_dir=runs/train/0323/` 最终落到了 `/root/runs/train/0323/...`，而不是仓库内的 `.../spikemamba/runs/train/0323/...`。

### 3.4 Resume checkpoint

- stage2 resume：
  - `/root/shared-nvme/spikemamba/runs/resume/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to100_keep_lr_20260323_resume.pth.tar`
- stage3 resume：
  - `/root/shared-nvme/spikemamba/runs/resume/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep100_to200_halflr_20260323_resume.pth.tar`
- 旧 `60 -> 120 halflr` resume 路径已被阻断：
  - `/root/shared-nvme/spikemamba/runs/resume/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to120_halflr_20260323_resume.pth.tar`

## 4. 本次维护动作

### M01. 调整续训计划

- 原链路是 `60 + 60`。
- 当前已经改成：
  - 保留 `epoch 1-60`
  - `epoch 60 -> 100` 继续 `lr=1e-4`
  - `epoch 100 -> 200` 改为 `lr=5e-5`

### M02. 阻断旧的 `60 -> 120 halflr`

- 旧 resume 路径被做成“阻断目录”，避免旧 wrapper 误启动错误分支。

### M03. 修复 monitor 崩溃

- 原因：新 TensorBoard event 文件刚生成时，还没有写入 `loss/val_loss/learning_rate` 标量，monitor 直接读取会触发 `KeyError`。
- 处理：`monitor_0323_60plus60_halflr.py` 已修复，先检查 scalar tags 是否存在，再读取标量。

### M04. 修复 monitor 路径误判

- stage3 实际写到了 `/root/runs/train/0323/...`。
- monitor 已增加 run dir 解析逻辑，会在候选目录中选择“实际有最新 event/checkpoint 的目录”。

## 5. 当前 val_loss 走势

### 5.1 历史最好

| 范围 | 最好 epoch | 最好 val_loss |
|---|---:|---:|
| stage1 | `21` | `0.014739707112` |
| stage2 | `73` | `0.014876967296` |
| stage3 | `117` | `0.014995988458` |
| 全局 | `21` | `0.014739707112` |

### 5.2 stage3 最近走势

```text
epoch 123  0.015214995481
epoch 124  0.015736067668
epoch 125  0.015151639469
epoch 126  0.015343430452
epoch 127  0.015628583729
epoch 128  0.015187397599
epoch 129  0.015242834575
epoch 130  0.015071030706
epoch 131  0.015704568475
epoch 132  0.015528169461
epoch 133  0.015501422808
epoch 134  0.016253056005
epoch 135  0.015584134497
```

观察：

- stage3 已经把本段最好压到 `0.014995988458 @ epoch 117`，但尚未刷新全局最好。
- 最近 `123 -> 135` 是有波动的震荡区间，暂时没有稳定跌穿 `0.0150` 以下。

## 6. 当前 live 状态（记录时）

- 训练进程：
  - `PID=133063`
- 监视器进程：
  - `PID=137422`
- live epoch：
  - `135`
- live train loss：
  - `0.002200220944`
- live val_loss：
  - `0.015584134497`
- live lr：
  - `5e-5`
- 最新已保存 checkpoint：
  - `checkpoint-epoch120-loss-0.0022.pth.tar`

## 7. 后续接续建议

- 若继续观察训练，只看两个来源即可：
  - `run_logs/0323/monitor_0323_60plus60_halflr.log`
  - `/root/runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep100_to200_halflr_20260323_bs2/tensorboard/`
- 如果只想看“当前有效指标”，优先看 event 中的 `val_loss`，不要只看 checkpoint 名字。
- 若后续需要整理最终结果，应在 `epoch 200` 结束后补一版：
  - 最终最好 checkpoint
  - 全程最佳 epoch
  - 最终 test/val 评测

