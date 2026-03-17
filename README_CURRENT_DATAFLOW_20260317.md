# SpikeMamba 当前数据流动说明

这份文档描述当前 `spikemamba` 项目里，从数据集输出到深度图预测的实际数据流动。重点是：

1. `spike embedding` 没有改。
2. 改动发生在 encoder 里的 `ST_Mamba_Block`，只影响送入 Mamba 的序列组织方式。
3. decoder 和深度预测头保持原有 2D U-Net 风格。

## 1. 训练时 batch 进入模型前的数据形态

当前训练配置：

- `baseline = "s"`
- `sequence_length = 1`
- `num_bins_rgb = 128`
- 训练裁剪尺寸是 `224 x 224`

对应的数据集逻辑：

- 外层数据集 `SequenceSynchronizedFramesSpikesDENSEDataset` 返回一个长度为 `L` 的 `sequence`
- 当前 `L = 1`，所以每个 batch 实际上是长度为 1 的列表
- 列表里的每个 `item` 是一个字典

对当前 `baseline = "s"` 而言，关键字段是：

- `item["image"]`: 实际存的是 spike tensor，不是 RGB 图像
- `item["depth_image"]`: 对应的深度监督

也就是说，当前模型虽然优先读取键名 `"image"`，但这个 `"image"` 在当前配置下实际内容是 spike bins。

当前单个样本在进入模型前的核心形状是：

- spike: `[B, 128, 224, 224]`
- depth target: `[B, 1, 224, 224]`

## 2. 模型入口：从 spike tensor 到 3D encoder 输入

入口在 `model/S2DepthNet.py` 的 `S2DepthTransformerUNetConv.forward()`。

### 2.1 取输入

`_select_spike_tensor()` 会按优先级尝试读取：

1. `"image"`
2. `"events"`
3. `"events0"`
4. 其他张量键

当前配置里会命中：

- `spike_tensor = item["image"]`

其形状为：

- `[B, 128, 224, 224]`

### 2.2 补出 3D encoder 所需维度

因为 encoder 期望输入是 5D：

- `[B, C, T, H, W]`

所以这里会执行：

- 把 128 个 spike bin 作为时间维 `T`
- 在前面补一个通道维 `C = 1`

得到：

- `[B, 1, 128, 224, 224]`

这里要注意：

- `T = 128` 来自 spike bins
- 训练器外层的 `sequence_length = 1` 是另一层时间概念
- 当前真正进入 encoder 的“时序长度”仍然是 spike bin 这一维

## 3. Embedding：保持不变

embedding 在 `model/encoder_transformer.py` 里的 `LongSpikeStreamEncoderConv`，实际使用的是 `model/mamba_3d.py` 里的：

- `self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)`

当前配置：

- `patch_size = (32, 2, 2)`
- `in_chans = 1`
- `embed_dim = 96`

所以：

- 输入: `[B, 1, 128, 224, 224]`
- 经过 `patch_embed`
- 输出: `[B, 96, 4, 112, 112]`

含义是：

- 时间维: `128 -> 4`
- 空间维: `224 x 224 -> 112 x 112`
- 通道维: `1 -> 96`

这一步没有被修改。

## 4. Encoder：当前真正改动的位置

改动发生在 `model/mamba_3d.py` 的 `ST_Mamba_Block.forward()`。

### 4.1 Block 输入

每个 block 的输入先是标准 3D feature：

- `[B, C, T, H, W]`

代码先做：

```python
x = x.permute(0, 3, 4, 2, 1).contiguous()
```

变成：

- `[B, H, W, T, C]`

这样做的目的，是为了更方便把局部 `2 x 2` 空间块折叠进通道维。

### 4.2 折叠到 Mamba 序列

在 `_fold_spatial_tokens()` 里，当前逻辑是：

1. 把空间按 `2 x 2` 分组
2. 每个 `2 x 2` 小块的 4 个位置合并到通道维
3. 把 `(T, H/2, W/2)` 展平成一条序列

也就是：

- 输入: `[B, H, W, T, C]`
- 输出: `[B, T * H/2 * W/2, 4C]`

对应代码里的核心步骤：

```python
x = x.reshape(B, H // 2, 2, W // 2, 2, T, C)
x = x.permute(0, 5, 1, 3, 2, 4, 6).contiguous()
x = x.reshape(B, T * (H // 2) * (W // 2), 4 * C)
```

解释如下：

- 第一步 `reshape`:
  - 把原始空间维 `H, W` 拆成 `H/2, 2, W/2, 2`
  - 等价于把整张图切成很多个 `2 x 2` 小块
- 第二步 `permute`:
  - 把维度重排成 `[B, T, H/2, W/2, 2, 2, C]`
  - 让时间维和粗网格 `(H/2, W/2)` 排在前面
- 第三步 `reshape`:
  - 将最后的 `2 x 2 x C` 合并为 `4C`
  - 同时把 `T, H/2, W/2` 拉平成一条长序列

这就实现了你要的目标：

- 序列长度从只沿 `T`
- 变成沿 `T x H/2 x W/2`

同时：

- embedding 维度从 `C`
- 变成 `4C`

### 4.3 Mamba 内部看到的 token 形状

对于当前训练配置，三个 stage 里送进 Mamba 的形状分别是：

1. Stage 0
   - feature map: `[B, 96, 4, 112, 112]`
   - Mamba input: `[B, 12544, 384]`
   - 因为 `12544 = 4 x 56 x 56`

2. Stage 1
   - feature map: `[B, 192, 4, 56, 56]`
   - Mamba input: `[B, 3136, 768]`
   - 因为 `3136 = 4 x 28 x 28`

3. Stage 2
   - feature map: `[B, 384, 4, 28, 28]`
   - Mamba input: `[B, 784, 1536]`
   - 因为 `784 = 4 x 14 x 14`

这里要强调：

- `self.mamba = Mamba(dim * 4)`
- 所以 Mamba 的输入 embedding 维度已经和 `4C` 对齐

### 4.4 从 Mamba 输出还原回 3D feature map

Mamba 输出后，在 `_unfold_spatial_tokens()` 里把 token 再还原回：

- `[B, T * H/2 * W/2, 4C]`
- 还原为 `[B, H, W, T, C]`

核心步骤：

```python
x = x.reshape(B, T, H // 2, W // 2, 2, 2, C)
x = x.permute(0, 2, 4, 3, 5, 1, 6).contiguous()
x = x.reshape(B, H, W, T, C)
```

含义是：

- 先把 `4C` 拆回 `2 x 2 x C`
- 再把粗网格 `(H/2, W/2)` 和块内坐标 `(2, 2)` 拼回完整空间

随后再 `permute` 回：

- `[B, C, T, H, W]`

所以：

- Mamba 内部使用的是新的序列组织
- block 对外输出形状仍然保持不变

这也是为什么后续 decoder 不需要跟着改。

## 5. 多 stage encoder 的完整流动

当前 encoder 主干是 `Mamba3DEncoder`。

### Stage 0

- patch embedding 输出: `[B, 96, 4, 112, 112]`
- 经过 Stage 0 Mamba blocks 后仍是: `[B, 96, 4, 112, 112]`
- 保存为第一个 skip feature

### Stage 1

- 先用 `Conv3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))`
- 只下采样空间，不改时间
- 变成: `[B, 192, 4, 56, 56]`
- 经过 Stage 1 Mamba blocks 后仍是: `[B, 192, 4, 56, 56]`
- 保存为第二个 skip feature

### Stage 2

- 再次只下采样空间
- 变成: `[B, 384, 4, 28, 28]`
- 经过 Stage 2 Mamba blocks 后仍是: `[B, 384, 4, 28, 28]`
- 保存为第三个 skip feature

## 6. 从 3D encoder 输出到 2D decoder 输入

`LongSpikeStreamEncoderConv.forward()` 不直接把 5D feature 送给 decoder，而是做了一次“按时间块分解，再转回 2D”的投影。

当前配置中：

- `temporal_bins = 128`
- `patch_size[0] = 32`
- 所以 `num_blocks = 128 / 32 = 4`

对每个 stage：

1. 沿时间维 `T` 把 feature 切成 4 份
2. 每一份形状是 `[B, C, 1, H, W]`
3. reshape 成 `[B, C, H, W]`
4. 用一个 `1x1 Conv2d` 把通道从 `C` 压到 `C/4`
5. 把 4 份结果在通道维拼接回来

这样得到送给 decoder 的 2D 多尺度特征：

1. Stage 0 -> `[B, 96, 112, 112]`
2. Stage 1 -> `[B, 192, 56, 56]`
3. Stage 2 -> `[B, 384, 28, 28]`

直观上看，这一步做的是：

- 把 3D encoder 的时间输出重新折叠回 2D 通道表示
- decoder 仍然处理标准 2D feature pyramid

## 7. Decoder 到深度图

decoder 在 `model/S2DepthNet.py`。

当前 `skip_type = "sum"`，所以用逐级相加的 skip connection。

### 7.1 bottleneck

最深层输入：

- `[B, 384, 28, 28]`

先经过 2 个 residual blocks，形状不变：

- `[B, 384, 28, 28]`

### 7.2 三次上采样

1. Decoder 0
   - `[B, 384, 28, 28]`
   - 上采样后 -> `[B, 192, 56, 56]`

2. Decoder 1
   - 与 encoder stage 1 skip 相加
   - 输入仍是 `[B, 192, 56, 56]`
   - 上采样后 -> `[B, 96, 112, 112]`

3. Decoder 2
   - 与 encoder stage 0 skip 相加
   - 输入仍是 `[B, 96, 112, 112]`
   - 上采样后 -> `[B, 48, 224, 224]`

### 7.3 prediction head

最后经过 `1x1 Conv2d`：

- `[B, 48, 224, 224] -> [B, 1, 224, 224]`

当前输出激活是：

- `identity`

所以最终深度预测是：

- `prediction["image"]`，形状 `[B, 1, 224, 224]`

训练时对应监督是：

- `item["depth_image"]`

## 8. 一句话总结当前版本

当前版本的数据流动可以概括为：

1. 数据集给出 spike bins
2. `S2DepthNet.forward()` 把 spike 整理成 `[B, 1, T, H, W]`
3. patch embedding 保持不变
4. encoder 内部每个 `ST_Mamba_Block` 把局部 `2 x 2` 空间块折叠成通道，使 Mamba 看到 `[B, T x H/2 x W/2, 4C]`
5. Mamba 输出后再还原回 `[B, C, T, H, W]`
6. encoder 末端把 3D 多尺度特征重新投影成 2D 金字塔
7. decoder 逐级上采样并输出深度图

因此，这次修改的本质是：

- 不改 spike embedding
- 不改 decoder
- 只改 Mamba 在 encoder 内部看到的序列组织方式

## 9. 当前版本最关键的形状链路

以当前训练配置为例，完整主链路是：

```text
item["image"]                         : [B, 128, 224, 224]   # 当前 baseline=s，这里实际是 spike
unsqueeze channel                     : [B, 1, 128, 224, 224]
patch_embed                           : [B, 96, 4, 112, 112]

stage0 mamba input                    : [B, 12544, 384]
stage0 output                         : [B, 96, 4, 112, 112]
downsample                            : [B, 192, 4, 56, 56]

stage1 mamba input                    : [B, 3136, 768]
stage1 output                         : [B, 192, 4, 56, 56]
downsample                            : [B, 384, 4, 28, 28]

stage2 mamba input                    : [B, 784, 1536]
stage2 output                         : [B, 384, 4, 28, 28]

2D projected encoder outputs          : [B, 96, 112, 112]
                                      : [B, 192, 56, 56]
                                      : [B, 384, 28, 28]

decoder bottleneck                    : [B, 384, 28, 28]
decoder stage0                        : [B, 192, 56, 56]
decoder stage1                        : [B, 96, 112, 112]
decoder stage2                        : [B, 48, 224, 224]
depth prediction                      : [B, 1, 224, 224]
```
