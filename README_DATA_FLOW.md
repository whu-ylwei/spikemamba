# SpikeMamba 当前数据流详解（中文）

本文档面向没有机器学习背景的读者，解释 `/root/shared-nvme/spikemamba` 在当前主训练路径下，数据是如何从磁盘流到模型，再流到损失与参数更新的。

当前说明基于配置文件：
`configs/train_s2d_spiketransformer_mambassm_strict_20260306_124138.json`

---

## 1. 一句话概括

系统先读取一帧脉冲张量（`spike_*.npy`）和对应深度标签（`depth_*.npy`），做时间对齐与预处理，然后把脉冲输入送入 **Mamba 编码器 + CNN 解码器** 预测深度图，最后用损失函数比较预测与真值并反向更新参数。

---

## 2. 输入数据来自哪里

每个样本主要依赖以下文件：

- 脉冲输入：`spike/r128/spike_XXXXXXXXXX.npy`
- 脉冲时间戳：`spike/r128/timestamps.txt`
- 深度真值：`depth/data/depth_XXXXXXXXXX.npy`
- 深度时间戳：`depth/data/timestamps.txt`

### 2.1 脉冲读取

`data_loader/spike_dataset.py` 中的 `VoxelGridDENSESpikeDataset` 会读取单个 spike 文件，得到张量：

- 形状：`[C, H, W]`
- 当前配置：`C = 128`

随后对非零位置做标准化（只统计非零值的均值和方差）。

### 2.2 深度标签读取与变换

`data_loader/SpikesDENSE_dataset.py` 读取深度后执行：

1. 按 `clip_distance` 裁剪（本配置中是 `1000`）
2. 除以 `clip_distance` 做归一化
3. 做对数映射：`1 + log(depth_norm) / reg_factor`
4. 截断到 `[0, 1]`

因此模型学习的是“变换后的深度空间”，不是直接回归原始米制深度。

---

## 3. 样本字典里有哪些键

当前关键配置：

- `baseline = "s"`
- `every_x_rgb_frame = 1`
- `loss_composition = "image"`

在这个设定下，单个样本最核心的键是：

- 输入键：`image`（这里装的是 spike，不是 RGB）
- 监督键：`depth_image`

也就是说当前主监督关系是：`image -> depth_image`。

---

## 4. DataLoader 后的批数据形状

配置里 `sequence_length = 1`，所以每个 batch 是“长度为 1 的序列”：

- `sequence[0]["image"]`：`[B, 128, 224, 224]`
- `sequence[0]["depth_image"]`：`[B, 1, 224, 224]`

其中 `224` 来自训练增强里的 `RandomCrop(224)`。

---

## 5. 模型前向数据流（含张量尺寸）

主模型类是 `S2DepthTransformerUNetConv`。

### 5.1 输入键选择与重排

模型会按优先级从字典中选输入：
`image -> events -> events0 -> 其他 events* -> 其他张量`。

当前配置一般会命中 `image`。

然后把输入从 4D 变成 5D：

- 原始：`[B, 128, H, W]`
- 重排后：`[B, 1, 128, H, W]`

其中：

- `1` 是输入通道
- `128` 被当作时间维 `T`

### 5.2 Mamba 编码器阶段

编码器结构：

1. `LongSpikeStreamEncoderConv`
2. 其内部主干：`Mamba3DEncoder`
3. 核心块：`ST_Mamba_Block`

当前配置参数：

- `patch_size = (32, 2, 2)`
- `depths = [2, 2, 6]`
- `embed_dim = 96`

输入 `(B,1,128,224,224)` 后，近似特征流如下：

1. 3D Patch Embedding（`Conv3d`，步长等于 patch）  
   `(B,1,128,224,224) -> (B,96,4,112,112)`
2. Stage 0（Mamba 块）  
   `(B,96,4,112,112)`
3. 空间下采样（仅 H/W，下采样核为 `1x2x2`）  
   `(B,192,4,56,56)`
4. Stage 1  
   `(B,192,4,56,56)`
5. 再次空间下采样  
   `(B,384,4,28,28)`
6. Stage 2  
   `(B,384,4,28,28)`

`ST_Mamba_Block` 内部逻辑可理解为：

- 把每个空间位置 `(h,w)` 的时间序列单独拿出来
- 在时间维上跑 `Mamba(dim)`
- 残差相加
- 再过 FFN（MLP）并残差相加

### 5.3 从 3D 时空特征转为 2D 多尺度特征

`LongSpikeStreamEncoderConv` 会把每层特征在时间维上分块（当前 `T=4`，所以是 4 块）：

1. 时间维分成 4 个 chunk
2. 每个 chunk 过 `1x1 Conv2d`
3. 在通道维拼接

得到供解码器使用的 2D 多尺度特征：

- Level 0：`[B,96,112,112]`
- Level 1：`[B,192,56,56]`
- Level 2：`[B,384,28,28]`

### 5.4 解码器（CNN，UNet风格）

解码器使用 `UpsampleConvLayer` 逐级上采样，并与编码器跳连（当前 `skip_type = sum`）：

1. 从最深层开始：`[B,384,28,28]`
2. 上采样到：`[B,192,56,56]`
3. 与 Level 1 跳连后再上采样到：`[B,96,112,112]`
4. 与 Level 0 跳连后再上采样到：`[B,48,224,224]`
5. 最后 `1x1 Conv` 预测头：`[B,48,224,224] -> [B,1,224,224]`

输出键为：

- `predictions_dict["image"]`

---

## 6. 损失计算与反向传播

Trainer 会用输出键去匹配目标键：

- 预测键：`image`
- 目标键：`depth_image`

当前主损失：

- `scale_invariant_loss`

可选附加损失：

- 多尺度梯度损失（`grad_loss`）
- MSE 损失（`mse_loss`）

每个 batch 的训练步骤：

1. `optimizer.zero_grad()`
2. 前向计算
3. 聚合总损失
4. `loss.backward()`
5. 梯度裁剪（若开启）
6. `optimizer.step()`

---

## 7. 哪些部分是 Mamba，哪些不是

### 7.1 当前实际使用的 Mamba 部分

- `model/mamba_3d.py`
- `Mamba3DEncoder`
- `ST_Mamba_Block`
- `mamba_ssm.modules.mamba_simple.Mamba`

### 7.2 当前实际使用但非 Mamba 的部分

- 解码器：上采样 + 卷积（CNN）
- 输出头：`1x1 Conv2d`
- 训练管线：loss、优化器、调度器、日志、checkpoint

### 7.3 Transformer 在本仓库中的状态

`model/swin_transformer_3d.py` 文件仍在仓库中，但当前生效编码路径直接使用 `Mamba3DEncoder`，并未走 Swin 注意力块。

---

## 8. 完整链路总结

1. 读取 spike 与 timestamp
2. 与 depth 按时间戳对齐
3. 组织样本字典（`image`, `depth_image`）
4. DataLoader 组 batch 并裁剪
5. 输入重排到 `(B,1,T,H,W)`
6. Mamba 编码器提取多尺度特征
7. CNN 解码器恢复到全分辨率深度图
8. 与 `depth_image` 计算损失
9. 反向传播更新参数

以上过程在每个 batch、每个 epoch 重复执行。

