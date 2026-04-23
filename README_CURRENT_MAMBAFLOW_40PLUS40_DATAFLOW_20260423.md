# SpikeMamba 当前 MambaFlow 40+40 训练数据流说明

这份文档只描述当前正在运行的这条训练链路：

- 分支：`0422`
- 训练脚本：`scripts/run_mambaflow_bidivim_seq2_40plus40_halflr_20260423.sh`
- Stage 1 配置：`configs/train_s2d_spiketransformer_mambaflow_bidivim_seq2_40ep_20260423.json`
- Stage 2 配置：`configs/train_s2d_spiketransformer_mambaflow_bidivim_seq2_ep40_continue40_halflr_20260423.json`
- 数据根目录：`/root/shared-nvme/spikemamba/DENSE-spike`

文档目标有两个：

1. 说明当前训练到底在跑哪条数据流。
2. 记录当前这次训练的阶段进度和 `val_loss` 趋势。

## 1. 当前训练快照

本次核对时间：`2026-04-23 03:49:52 UTC`。

对应当前训练日志的最新写盘时间约为 `2026-04-23 11:42:47 +0800`，说明这次核对时还没有新的 epoch 标量落盘。

- 当前仍在 `stage1`
- `stage1` 目标：`40 epoch`, `lr=1e-4`
- `stage2` 将在 `epoch 40` 后自动接续到 `epoch 80`
- `stage2` 学习率会降为 `5e-5`
- 当前已写入 TensorBoard 的最新 epoch：`4`
- 本轮核对时最新写入仍停在 `epoch 4`
- 当前 `model_best.pth.tar` 对应：`epoch 4`
- 当前 best `val_loss`：`0.039643405860136885`

当前已写入的 epoch 级标量为：

| epoch | train loss | val_loss | lr |
| --- | ---: | ---: | ---: |
| 1 | 0.089361 | 0.066120 | 1.0e-4 |
| 2 | 0.047861 | 0.069676 | 1.0e-4 |
| 3 | 0.035869 | 0.044331 | 1.0e-4 |
| 4 | 0.032436 | 0.039643 | 1.0e-4 |

可以看到当前 `val_loss` 趋势是：

- `epoch 1 -> 2` 先小幅变差：`0.0661 -> 0.0697`
- `epoch 2 -> 3` 明显改善：`0.0697 -> 0.0443`
- `epoch 3 -> 4` 继续改善：`0.0443 -> 0.0396`

因此，当前趋势不是单调下降，但在最近两个 epoch 上是明显向下的。

## 2. 当前配置摘要

当前 `stage1` 配置核心项如下：

- `dataset type = SequenceSynchronizedFramesSpikesDENSEDataset`
- `baseline = "s"`
- `sequence_length = 2`
- `batch_size = 2`
- `num_workers = 4`
- `spike_folder = spike/r128`
- `depth_folder = depth/data`
- `frame_folder = rgb/frames`
- `clip_distance = 1000.0`
- `spatial_resolution = [112, 112]`
- `num_bins_rgb = 128`
- `arch = S2DepthTransformerUNetConv`
- `mamba_backend = mamba_ssm`
- `manifold_flow.enabled = true`
- `flow_steps = 6`

损失配置如下：

- 主损失：`scale_invariant_loss(weight=1.0, n_lambda=1.0)`
- 梯度损失：`grad_loss.weight = 0.25`
- Manifold Flow 辅助项权重：`coarse_weight = 0.5`, `reconstruction_weight = 0.1`, `flow_weight = 0.1`, `geometry_weight = 0.05`

## 3. 数据从磁盘到 DataLoader

训练入口在 `train_parallel.py`。

它会做以下事情：

1. 读取配置中的训练集路径 `DENSE/train` 和验证集路径 `DENSE/validation`
2. 在数据根目录 `/root/shared-nvme/spikemamba/DENSE-spike` 下拼出实际路径
3. 对训练集和验证集下面的每个子目录，各实例化一个 `SequenceSynchronizedFramesSpikesDENSEDataset`
4. 用 `ConcatDataset` 把这些子数据集拼起来
5. 再交给 `DataLoader`

因此当前的真实训练路径是：

`/root/shared-nvme/spikemamba/DENSE-spike`
-> `DENSE/train/<subfolder>`
-> `spike/r128/spike_XXXXXXXXXX.npy`
-> `depth/data/depth_XXXXXXXXXX.npy`

## 4. 单个时间步样本如何构造

内层数据集是 `SynchronizedFramesSpikesDENSEDataset`。

对某个索引 `j`，它会：

1. 在 `spike/r128/` 下读取 `spike_XXXXXXXXXX.npy`
2. 从 `timestamps.txt` 读出 spike timestamp
3. 在深度时间戳里找同步的 `depth_XXXXXXXXXX.npy`
4. 读取并预处理深度真值

### 4.1 Spike 张量

当前 spike 文件读出后是：

- 形状：`[128, H, W]`
- 类型：`float32`

如果 `normalize=true`，会只对非零位置做标准化：

- 非零值减去非零均值
- 再除以非零标准差

### 4.2 深度真值

深度图读取后做如下处理：

1. `clip` 到 `[0, clip_distance]`
2. 除以 `clip_distance`
3. 做 `1 + log(depth) / reg_factor`
4. 再 `clip` 到 `[0, 1]`
5. 变为 `[1, H, W]`

当前 `clip_distance = 1000.0`。

## 5. 当前 baseline 下 item 里真正存的是什么

当前配置是：

- `baseline = "s"`
- `every_x_rgb_frame = 1`
- `loss_composition = "image"`

在这个分支下，数据集最终会构造：

- `item["image"] = events["events"]`
- `item["depth_image"] = frame`

也就是说：

- 键名虽然叫 `"image"`
- 但当前实际喂给模型的不是 RGB，而是 spike bins

这点非常关键。

当前每个时间步最核心的监督对是：

- 输入：`item["image"]`, 形状约为 `[128, 224, 224]`
- 真值：`item["depth_image"]`, 形状约为 `[1, 224, 224]`

## 6. sequence_length = 2 时，DataLoader 实际产出什么

外层数据集 `SequenceSynchronizedFramesSpikesDENSEDataset` 会返回一个长度为 `L=2` 的 Python 列表：

- `sequence[0]`
- `sequence[1]`

这两个元素分别对应相邻两个时间步。

再经过 `DataLoader(batch_size=2)` 后，trainer 收到的是：

- 一个长度为 2 的列表
- 列表里的每个元素都是 batched dict

因此 trainer 看到的数据可以理解成：

- `sequence[0]["image"]`: `[B, 128, 224, 224]`
- `sequence[0]["depth_image"]`: `[B, 1, 224, 224]`
- `sequence[1]["image"]`: `[B, 128, 224, 224]`
- `sequence[1]["depth_image"]`: `[B, 1, 224, 224]`

这里当前 `B = 2`。

## 7. Trainer 里的时间展开

`SpikeTTrainer.forward_pass_sequence()` 会对这个长度为 2 的 `sequence` 顺序展开：

1. 先处理 `l = 0`
2. 再处理 `l = 1`
3. 每个时间步各自前向、各自算损失
4. 最后按 `L = 2` 做时间平均

因此当前一个 batch 的总损失不是单帧损失，而是：

- 两个时间步主损失的平均
- 再加上同样按 `L=2` 平均的辅助损失

## 8. 模型入口的数据形状变化

当前模型是 `S2DepthTransformerUNetConv`。

模型前向开始时，会先从 `item` 中选输入张量。

当前会命中：

- `item["image"]`

也就是 spike tensor。

### 8.1 输入到模型前

当前输入形状：

- `[B, 128, 224, 224]`

其中：

- `128` 是 spike bin 数
- 不是外层的 `sequence_length`

### 8.2 进入 3D encoder 前

模型会把它改造成：

- `[B, 1, 128, 224, 224]`

即：

- 通道维 `C = 1`
- 时间维 `T = 128`

所以当前 3D Mamba backbone 把 spike bins 当作显式时间轴。

## 9. 编码器内部数据流

编码器是 `LongSpikeStreamEncoderConv`。

### 9.1 Patch embedding

当前 patch size：

- `(32, 2, 2)`

因此第一步 3D patch embedding 后：

- 输入：`[B, 1, 128, 224, 224]`
- 输出：`[B, 96, 4, 112, 112]`

含义：

- 时间维 `128 -> 4`
- 空间 `224 x 224 -> 112 x 112`
- 通道 `1 -> 96`

### 9.2 Mamba 编码

然后进入 3-stage 3D Mamba encoder。

编码器输出的是 3 个尺度的特征图，之后在 decoder 侧会把时间块拆开，再投影回 2D 多尺度特征。

因此 decoder 接收到的是：

- stage 0 的 2D 特征
- stage 1 的 2D 特征
- stage 2 的 2D 特征

而不是直接保留原始 3D feature 到最后。

## 10. Decoder 与粗深度图

`forward_decoder()` 会：

1. 取最深层特征
2. 过 residual blocks
3. 逐级上采样
4. 与浅层特征做 skip sum
5. 用 `1x1` 预测头输出一张单通道深度图

这一步得到的是：

- `coarse_prediction`

形状为：

- `[B, 1, 224, 224]`

## 11. Manifold Flow 精修路径

当前 `manifold_flow.enabled = true`，所以粗深度图不会直接作为最终输出。

后续流程是：

1. `coarse_prediction` 进入条件编码器
2. 条件 latent 经过 6 步 rollout
3. 得到 refined latent
4. 由目标解码器解码成 refined depth

因此当前最终监督的主输出是：

- `refined_depth`

同时模型还会返回辅助信息：

- `coarse_prediction`
- `L_rec`
- `L_flow`
- `L_geo`

其中：

- `L_coarse` 是对 `coarse_prediction` 直接加的主损失
- `L_geo` 依赖前一个时间步保留下来的 `manifold_prev`
- 因此当前跨帧时序耦合，主要来自 manifold flow 的几何一致性项

## 12. 从模型输出到损失

当前只对键 `"image"` 计算主监督。

主损失代码对应的精确形式是：

- 令 `d = y_hat - y`，其中 `y_hat = refined_depth`, `y = depth_image`
- `L_si = mean(d^2) - lambda * mean(d)^2`
- 当前 `lambda = 1.0`
- 计算时会跳过 `NaN` 监督像素

梯度损失是：

- `L_grad = multi_scale_grad_loss(y_hat, y)`
- 它先对 `y_hat - y` 做 4 个尺度的平均池化，再取每个尺度上的空间梯度绝对值均值，最后对 4 个尺度求平均
- 当前总损失里使用的是 `0.25 * L_grad`

因此每个时间步上，当前总损失可以写成：

`L_t = L_si + 0.25 * L_grad + L_coarse + L_rec + L_flow + L_geo`

其中各项来源是：

- `L_coarse = 0.5 * scale_invariant_loss(coarse_prediction, depth_image)`
- `L_rec = 0.1 * L1(target_decoder(target_encoder(depth_image)), depth_image)`
- `L_flow = 0.1 * MSE(predicted_velocity, target_velocity)`
- `L_geo = 0.05 * SmoothL1(predicted_delta, target_delta)`

补充约束：

- `L_geo` 只会在当前时间步和前一时间步都已有 latent state 时生效，因此对 `sequence_length=2` 来说通常只在后一个时间步出现
- 如果某个辅助项条件不满足，该项在该时间步不会被加入

序列级别上：

- trainer 先对两个时间步的损失按 `L=2` 求平均
- 再对整个 epoch 的 batch 做日志聚合
- 最终写成 TensorBoard 里的 `loss` 和 `val_loss`

## 13. 当前这条链路的最简摘要

当前实际数据流可以压缩成：

`DENSE-spike/DENSE/train/.../spike/r128/spike_*.npy`
-> 读出 `events`, 形状 `[128, 224, 224]`
-> 数据集写入 `item["image"]`
-> DataLoader 组成 `sequence_length=2`, `batch_size=2`
-> trainer 顺序处理两个时间步
-> 模型把 `[B,128,224,224]` 变成 `[B,1,128,224,224]`
-> 3D Mamba encoder
-> 2D decoder
-> `coarse_prediction`
-> `ConditionedDepthManifoldFlow`
-> `refined_depth`
-> 对 `depth_image` 算主损失和辅助损失
-> 形成当前 `loss / val_loss`

## 14. 当前结论

截至本文件写入时，当前训练有两个最重要的结论：

1. 当前模型的实际输入不是 RGB，而是 spike bins。
2. 当前跨帧时序建模，主要不是通过 recurrent backbone，而是通过 `sequence_length=2` 的 trainer 展开和 `manifold_flow` 的时序辅助项实现。
