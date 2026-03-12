# SpikeMamba 训练排障 README（2026-03-05）

本文档整理了本次对话中出现的训练问题，按“现象 / 原理 / 解决方法 / 验证方式”给出可复用结论，目标是后续可直接按文档复现与排查。

## 1. 本次排障目标

- 清理并重建失效 `venv` 路径
- 修复训练启动与中途崩溃问题
- 在不停止训练前提下提升吞吐并稳定运行
- 给出训练进度和剩余时间估算方法

## 2. 当前稳定运行基线

- 项目路径：`/root/shared-nvme/spikemamba`
- 训练脚本：`train_parallel.py`
- 配置文件：`configs/train_s2d_spiketransformer_rtx5090_20260305.json`
- 当前训练总轮数：`30`
- 当前优化配置：
  - `batch_size=8`
  - `num_workers=8`
  - `pin_memory=true`
  - `lr=0.0008`
  - `verbosity=1`
  - `save_freq=8`
  - `num_previews=1`
  - `num_val_previews=1`
  - `still_previews=false`
  - `movie=false`

## 3. 问题清单（现象 / 原理 / 解决）

### 问题 1：GPU 是否可用不确定

- 现象：需要确认训练是否真的在 GPU 上运行。
- 原理：PyTorch 可能退回 CPU；仅看进程不够，需要看 `nvidia-smi` 实时占用与显存。
- 解决方法：使用 `nvidia-smi` + 训练进程检查，确认 CUDA 可见且显存持续占用。
- 验证方式：
  - `nvidia-smi`
  - `ps -ef | grep train_parallel.py`

### 问题 2：虚拟环境路径失效（旧环境污染）

- 现象：旧环境路径失效、包冲突，运行不稳定。
- 原理：历史 `venv/conda` 残留导致解释器和 site-packages 不一致。
- 解决方法：重建两个环境并清除旧路径：
  - `.venv`
  - `venv`
  - 使用 `python3 -m venv --clear ...`
- 验证方式：
  - 查看 `pyvenv.cfg`，确认 `home=/usr/bin`、`python3.12`，不再引用旧 conda 路径。

### 问题 3：NumPy 2.0 兼容问题（`np.alltrue` 移除）

- 现象：早期日志报错：`AttributeError: np.alltrue was removed in NumPy 2.0`。
- 原理：代码使用了 NumPy 2.0 已删除的 API。
- 解决方法：将 `np.alltrue(...)` 改为 `np.all(...)`。
- 修复文件：
  - `data_loader/spike_dataset.py`
  - `data_loader/SpikesDENSE_dataset.py`
- 验证方式：训练可通过数据集初始化，不再在读取时间戳阶段崩溃。

### 问题 4：运行目录已存在导致启动失败

- 现象：重复训练时 `save_dir/name` 已存在，原逻辑会触发中断。
- 原理：目录唯一性依赖手工改名，自动化重跑不友好。
- 解决方法：在 `train_parallel.py` 中增加自动重命名逻辑（追加 UTC 时间戳）。
- 修复文件：`train_parallel.py`
- 验证方式：日志出现 `auto-switching run name`，训练继续而不是退出。

### 问题 5：3D 编码器输入维度不匹配

- 现象：模型侧存在 2D/3D 输入语义冲突，训练难以稳定启动。
- 原理：3D backbone 期望输入 `(B, C, T, H, W)`，但原数据流含 `(B, bins, H, W)` 的历史路径；时间维和通道维混用。
- 解决方法：
  - `S2DepthNet` 将输入整理为 `(B,1,T,H,W)`
  - `encoder_transformer` 改用 `Mamba3DEncoder`，按 `temporal_bins` 切分 temporal chunks
- 修复文件：
  - `model/S2DepthNet.py`
  - `model/encoder_transformer.py`
  - `model/mamba_3d.py`
- 验证方式：模型可完整前向并进入训练循环。

### 问题 6：时间维被错误下采样导致特征退化

- 现象：3D stage 下采样后时间信息被压缩，影响 temporal chunk 逻辑。
- 原理：若对 T/H/W 同时下采样，时间长度会快速下降。
- 解决方法：将 downsample 改为仅空间下采样：`kernel/stride=(1,2,2)`。
- 修复文件：`model/mamba_3d.py`
- 验证方式：encoder chunk 数与配置一致，不再出现 temporal chunk 异常。

### 问题 7：预览阶段崩溃 `KeyError: 'image'`

- 现象：训练到 preview 阶段报错：`KeyError: 'image'`。
- 原理：某些 baseline 的样本项不包含 `image`，但 preview 逻辑直接按 key 索引。
- 解决方法：在 `spiket_trainer.py` 中增加回退策略：`image` 不存在时优先用 `events0` 或 `events*`。
- 修复文件：`trainer/spiket_trainer.py`
- 验证方式：日志中不再出现该 KeyError，epoch 能连续推进。

### 问题 8：预览步长变量未定义 `UnboundLocalError: step`

- 现象：在 `still_previews=false` 且 `grid_loss=true` 组合下，报错 `step` 未定义。
- 原理：`step` 只在某分支内赋值，但在分支外被使用。
- 解决方法：统一使用提前定义的 `preview_step=self.record_every_N_sample`。
- 修复文件：`trainer/spiket_trainer.py`
- 验证方式：训练/验证 preview 阶段均可正常执行。

### 问题 9：训练速度慢、GPU 利用率波动

- 现象：GPU 利用率不是持续 95%+，出现阶段性波动。
- 原理：
  - 数据加载与验证阶段会造成 GPU 等待
  - preview/histogram/video 等日志工作会占用 CPU/IO
  - 单卡训练不存在多卡并行加速
- 解决方法：
  - 提高数据吞吐：`batch_size=8`、`num_workers=8`、`pin_memory=true`
  - 降低日志与可视化开销：`verbosity=1`、关闭 `movie/still_previews`
  - 降低 checkpoint 频率：`save_freq=8`
- 验证方式：训练进程稳定，GPU 采样常见区间约 `58%~85%`，显存约 `13.4GB`。

### 问题 10：进度不透明（日志级别较低）

- 现象：`verbosity=1` 时没有每 step 的 `Train Epoch` 行，难以直接读进度。
- 原理：该项目 epoch 级指标主要写入 TensorBoard，终端日志较简化。
- 解决方法：通过 TensorBoard 事件中的 `learning_rate` 标量步数读取当前 epoch。
- 验证方式：`learning_rate step == 当前 epoch`（每个 epoch 记录一次）。

## 4. 训练时长估算方法（稳定可复用）

1. 读取 TensorBoard 的 `learning_rate` 最近 N 个点（每点对应 1 个 epoch）
2. 计算相邻 `wall_time` 差值，得到 `sec/epoch`
3. 剩余时间 = `(total_epochs - current_epoch) * sec_per_epoch`

本次运行中，典型值约 `132 秒/epoch`，因此 ETA 估算准确度较高（通常误差在若干分钟级）。

## 5. 当前问题是否已闭环

- 环境问题：已闭环
- 启动中断问题：已闭环
- 训练中途崩溃：已闭环
- 性能与稳定性：已达到可持续训练状态（非满载属正常波动）
- 进度观测：已闭环（通过 TensorBoard step）

## 6. 变更文件索引

- `model/encoder_transformer.py`
- `model/S2DepthNet.py`
- `model/mamba_3d.py`
- `train_parallel.py`
- `trainer/spiket_trainer.py`
- `data_loader/SpikesDENSE_dataset.py`
- `data_loader/spike_dataset.py`
- `configs/train_s2d_spiketransformer_rtx5090_20260305.json`

## 7. 常用巡检命令

```bash
# 训练进程
ps -eo pid,etime,%cpu,%mem,cmd | grep train_parallel.py | grep -v grep

# GPU 状态
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv,noheader

# 最新日志
tail -n 50 logs/train_opt_20260305_114542.log

# checkpoint 与 tensorboard 写入时间
find runs/train/train_s2d_SpikeTransformer_rtx5090_20260305_20260305_114546 -type f | head
```
