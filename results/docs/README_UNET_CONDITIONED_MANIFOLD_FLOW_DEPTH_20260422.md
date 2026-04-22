# 基于当前 U-Net 输出作为 Manifold Flow 条件的深度估计架构说明

## 1. 文档定位

本文是**架构说明文档**，不是当前代码的实现说明。

目标是回答一个明确问题：

> 在不改动当前仓库主干代码的前提下，如何把现有 U-Net 风格深度网络的输出，作为 `manifold flow` 的条件信息，用于后续深度估计与细化？

因此，本文做三件事：

1. 先说明当前仓库里真实存在的深度估计主链路。
2. 再说明 `manifold flow` 在文献里更准确的技术含义，以及它和参考架构图中的“黎曼流匹配”之间的关系。
3. 最后给出一个**最小侵入、职责清晰**的架构设计：把当前 U-Net 输出作为条件 `c`，送入独立的 manifold-flow 深度细化模块。

参考架构图来源于本地文件：

- `/root/shared-nvme/架构图.jpg`

其中红框部分对应后续要接入的流模型模块。

---

## 2. 当前仓库中的真实深度估计链路

当前主模型不是纯粹的经典 2D U-Net，而是：

- 前端：`3D spike encoder`
- 后端：`2D U-Net 风格 decoder`
- 输出：单通道深度图

核心入口与代码位置：

- `model/S2DepthNet.py`
- `model/encoder_transformer.py`
- `configs/train_s2d_spiketransformer_mambassm_bidivim_nograd_100ep_20260320.json`

### 2.1 输入和监督

当前数据集与训练器已经支持**序列样本**：

- `data_loader/SpikesDENSE_dataset.py` 中 `SequenceSynchronizedFramesSpikesDENSEDataset`
- `trainer/spiket_trainer.py` 中 `forward_pass_sequence()`

单步样本里最关键的两个量是：

- spike 输入：`item["events"]` 或被模型选中的等价 spike tensor
- 深度监督：`item["depth_image"]`

训练器按时间顺序处理一个长度为 `L` 的序列，因此后续若要引入相邻时刻的几何流形约束，数据接口本身是有基础的。

### 2.2 当前模型主干

当前配置下的关键参数是：

- `num_bins_rgb = 128`
- `base_num_channels = 96`
- `num_encoders = 3`
- `swin_depths = [2, 2, 6]`
- `skip_type = "sum"`
- `output_activation = "identity"`

主链路可以概括为：

```text
Spike Tensor
  -> S2DepthTransformerUNetConv.forward()
  -> LongSpikeStreamEncoderConv (3D encoder)
  -> 2D feature projection
  -> U-Net style decoder
  -> prediction["image"]
```

### 2.3 当前关键张量形状

在当前默认训练配置下，主链路张量大致为：

```text
input spike                         : [B, 128, 224, 224]
unsqueeze channel                   : [B,   1, 128, 224, 224]
patch embedding                     : [B,  96,   4, 112, 112]

encoder stage 0                     : [B,  96,   4, 112, 112]
encoder stage 1                     : [B, 192,   4,  56,  56]
encoder stage 2                     : [B, 384,   4,  28,  28]

projected 2D feature 0              : [B,  96, 112, 112]
projected 2D feature 1              : [B, 192,  56,  56]
projected 2D feature 2              : [B, 384,  28,  28]

decoder bottleneck                  : [B, 384,  28,  28]
decoder stage 0                     : [B, 192,  56,  56]
decoder stage 1                     : [B,  96, 112, 112]
decoder stage 2                     : [B,  48, 224, 224]
current depth output                : [B,   1, 224, 224]
```

### 2.4 本文要复用的“当前 U-Net 输出”

当前模型最终输出在：

- `model/S2DepthNet.py`
- `prediction = self.forward_decoder(encoded_xs)`
- `predictions_dict["image"] = prediction`

因此，本文定义：

- `d_u = prediction["image"]`

它的语义是：

- 由当前网络直接得到的**粗深度 / 初始深度 / 条件深度**
- 形状为 `[B, 1, 224, 224]`

后续的 manifold-flow 模块**不替代现有主干**，而是把 `d_u` 当作条件输入。

---

## 3. “Manifold Flow” 与参考图中的“黎曼流匹配”应如何理解

### 3.1 文献里的严格含义

如果严格按文献区分，当前问题实际上涉及两条技术线：

1. `manifold-learning flow (M-flow)`
   - 核心思想：先学习一个低维数据流形，再在该流形上建模概率密度。
   - 重点是“数据落在某个低维流形上”。

2. `flow matching / Riemannian Flow Matching`
   - 核心思想：直接学习一个连续时间向量场，把简单分布逐步输运到目标分布。
   - 若目标空间是流形，则对应 `Riemannian Flow Matching`。

### 3.2 参考架构图的语义

你给的参考图并不是单纯的经典 `M-flow` 图，而更接近下面这种组合结构：

```text
粗深度先验网络
  -> 条件 c
  -> 在深度隐空间上做流匹配 / 黎曼流匹配
  -> 解码成最终深度
```

下半部分的“隐式编码器 + 条件 + 高斯噪声 + 黎曼流匹配”表示：

- 先把真实深度映射到一个隐式深度流形
- 再在该流形的隐空间上学习条件输运过程
- 最后把输运后的隐变量解码回深度图

所以，从工程角度看，最合理的解释不是：

- “把 U-Net 直接替换成 manifold flow”

而是：

- “保留当前 U-Net 作为条件先验网络，再增加一个 manifold-aware latent flow refinement 模块”

这与参考图是对齐的。

---

## 4. 建议架构：用当前 U-Net 输出作为 Manifold Flow 条件的深度估计

### 4.1 总体思路

定义当前网络输出：

- `d_u = f_unet(x)`

其中：

- `x` 是当前 spike 序列输入
- `f_unet` 指当前仓库中的 `S2DepthTransformerUNetConv`
- `d_u` 是粗深度图 `[B, 1, 224, 224]`

然后新增一个独立的流模型分支：

- `c = C_phi(d_u)`：条件编码器
- `z_gt = E_psi(d_gt)`：深度流形编码器
- `z_hat = Flow_theta(noise, t, c)`：条件 manifold flow / RFM
- `d_hat = G_psi(z_hat)`：深度流形解码器

最终输出：

- `d_hat` 作为 refined depth

也就是说，**当前 U-Net 输出只充当条件，不是流模块里的被优化变量本体**。

### 4.2 为什么条件必须来自当前输出而不是原始 spike

如果直接让 flow 模块从 spike 原始输入建模，它就会重复当前主干已经做过的事情：

- 时空特征抽取
- 多尺度编码
- 初始深度成像

这会导致两个问题：

1. 模块职责重叠，训练目标混乱。
2. 新 flow 分支失去“只负责几何流形细化”的意义。

而使用 `d_u` 作为条件时，流模块看到的是：

- 当前主干已经给出的几何轮廓
- 物体边界和深度层次
- 现有模型的确定性先验

因此 flow 分支更像一个：

- **条件生成式深度细化器**
- **深度流形回投器**
- **几何一致性修正器**

---

## 5. 建议的模块划分

为和当前仓库尺度对齐，建议采用**空间 latent map**，而不是把整张深度图压成单个向量。

原因很直接：

- 深度图是密集空间场
- 单向量 latent 很容易损失局部结构
- 当前主干最深层正好已经是 `28 x 28` 的低分辨率空间表征

### 5.1 条件编码器 `C_phi`

输入：

- `d_u`，形状 `[B, 1, 224, 224]`

输出建议：

- `c`，形状 `[B, C_z, 28, 28]`

推荐语义：

- `c` 不是最终深度，而是 flow 模块的条件上下文
- 它表示“当前 U-Net 对深度的判断”

推荐结构：

```text
d_u [B,1,224,224]
  -> Conv(stride=2)
  -> Conv(stride=2)
  -> Conv(stride=2)
  -> c [B,C_z,28,28]
```

其中 `C_z` 可取：

- `64`
- `96`
- `128`

### 5.2 深度流形编码器 `E_psi`

输入：

- 训练时真实深度 `d_gt`

输出：

- `z_gt in R^{B x C_z x 28 x 28}`

作用：

- 把真实深度映射到“深度流形隐空间”
- 让 flow 模块在隐空间里学习分布和几何输运

这一步对应参考图中的：

- “隐式编码器”

### 5.3 Manifold Flow / Riemannian Flow Matching 模块 `Flow_theta`

输入：

- 噪声 `epsilon`
- 时间变量 `t`
- 条件 `c`

输出：

- 隐空间向量场 `v_theta(z_t, t, c)` 或积分后的 `z_hat`

建议形态：

```text
epsilon ~ N(0, I)
z_t = path(epsilon, z_gt, t)
v_theta(z_t, t, c) -> latent velocity
ODE solve / flow rollout -> z_hat
```

这里的核心不是“在像素空间直接跑 flow”，而是：

- 在**学习到的深度流形隐空间**上进行条件输运

这正是参考图下半部分最有价值的地方。

### 5.4 深度流形解码器 `G_psi`

输入：

- `z_hat`

输出：

- refined depth `d_hat`

建议结构：

```text
z_hat [B,C_z,28,28]
  -> upsample
  -> conv
  -> upsample
  -> conv
  -> upsample
  -> conv
  -> [B,1,224,224]
```

这一步对应参考图里红框后得到最终预测深度的过程。

---

## 6. 与参考架构图的一一对应关系

为了避免概念漂移，下面给出图中各块与当前工程的映射关系。

### 6.1 图上半部分

参考图上半部分可理解为：

```text
Spike/时窗输入
  -> 脉冲引导去噪网络
  -> 特征聚合
  -> 上采样恢复
  -> 流模型
  -> 预测结果
```

映射到当前仓库后，建议解释为：

- `Spike/时窗输入`
  - 对应当前 spike tensor 输入

- `脉冲引导去噪网络`
  - 对应当前 `S2DepthTransformerUNetConv` 整个主干
  - 即 `3D encoder + 2D decoder`

- `上采样恢复后的输出`
  - 对应当前 `d_u = prediction["image"]`

- 红框流模型
  - 对应新增的 `manifold flow / RFM refinement module`

### 6.2 图下半部分

参考图下半部分表示：

- 真实深度 `t`
- 真实深度 `t+1`
- 经过共享或冻结的隐式编码器
- 得到隐空间表示
- 在条件约束下进行流匹配

映射到当前工程后，可以拆成两个版本：

#### 版本 A：最小落地版

只使用当前时刻监督：

- `d_gt -> E_psi -> z_gt`
- `d_u -> C_phi -> c`
- `epsilon + c -> Flow_theta -> z_hat`
- `z_hat -> G_psi -> d_hat`

这个版本最容易和当前仓库兼容，因为它不强依赖额外的相邻监督设计。

#### 版本 B：时序几何约束扩展版

利用当前序列训练机制中的相邻帧：

- `d_gt^t -> E_psi -> z_t`
- `d_gt^{t+1} -> E_psi -> z_{t+1}`
- 对 `z_t` 与 `z_{t+1}` 加入流形上的几何一致性约束

这个版本更贴近参考图原意，因为它显式建模了：

- 时间维连续性
- 邻近时刻深度流形上的演化路径

---

## 7. 推荐的数据流定义

### 7.1 最小落地版

```text
输入:
  x      : spike tensor
  d_gt   : ground-truth depth

第一阶段:
  d_u = f_unet(x)

条件与流形编码:
  c    = C_phi(d_u)
  z_gt = E_psi(d_gt)

流匹配:
  epsilon ~ N(0, I)
  t ~ Uniform(0, 1)
  z_t = psi_t(epsilon, z_gt)
  v_theta(z_t, t, c)
  z_hat = FlowRollout(v_theta, epsilon, c)

解码:
  d_hat = G_psi(z_hat)
```

最终监督：

- `d_u` 对 `d_gt` 做 coarse loss
- `d_hat` 对 `d_gt` 做 refined loss

### 7.2 时序几何约束扩展版

如果使用序列样本中的相邻深度：

```text
d_gt^t     -> E_psi -> z_gt^t
d_gt^{t+1} -> E_psi -> z_gt^{t+1}
```

则可以加入：

- 邻时刻流形距离约束
- geodesic smoothness
- latent temporal consistency

这部分对应参考图里“黎曼流几何约束”的位置。

---

## 8. 损失函数建议

### 8.1 总损失

建议总目标写为：

```text
L_total
= lambda_1 * L_coarse
+ lambda_2 * L_rec
+ lambda_3 * L_fm
+ lambda_4 * L_refine
+ lambda_5 * L_geo
```

其中：

- `L_coarse`
  - 当前 U-Net 输出 `d_u` 与 `d_gt` 的监督
  - 保留现有主干的稳定深度能力

- `L_rec`
  - 深度流形自编码重建损失
  - 保证 `E_psi / G_psi` 真正学到深度流形

- `L_fm`
  - flow matching / Riemannian flow matching 主损失
  - 约束 `v_theta(z_t, t, c)` 学到正确输运方向

- `L_refine`
  - refined depth `d_hat` 与 `d_gt` 的监督
  - 可继续使用当前仓库已有的深度损失设计思想

- `L_geo`
  - 相邻时刻的流形几何一致性约束
  - 若当前阶段不启用时序几何项，可令其为 `0`

### 8.2 为什么要保留 `L_coarse`

因为本文设计是：

- 当前主干负责给出稳定、快速的粗深度
- flow 模块负责在深度流形上细化与纠偏

如果去掉 `L_coarse`，当前 U-Net 会失去独立的深度成像责任，最终变成：

- 粗分支不再“粗”
- flow 分支被迫承担全部重建工作

这不符合“当前 U-Net 输出作为条件”的设计初衷。

---

## 9. 推理流程

推理时不需要真实深度，只保留：

```text
x
 -> current U-Net
 -> d_u
 -> condition encoder C_phi
 -> flow rollout from noise
 -> z_hat
 -> depth decoder G_psi
 -> d_hat
```

因此最终在线流程是：

1. 当前模型先给出 `d_u`
2. `d_u` 作为条件引导 manifold flow
3. flow 输出 refined latent
4. 解码得到最终深度 `d_hat`

如果强调稳定性，还可以采用保守融合：

```text
d_final = alpha * d_u + (1 - alpha) * d_hat
```

但从参考图语义上看，更标准的解释是：

- `d_hat` 直接作为最终预测

---

## 10. 该设计与当前仓库的接口关系

在不改代码的前提下，可以把接口关系先定义清楚：

### 10.1 现有主干保持不变

以下部分完全不动：

- spike 数据读取
- 3D Mamba encoder
- 2D U-Net 风格 decoder
- 当前深度监督主链路

### 10.2 未来若实现，唯一最自然的条件接入点

最自然的接入点就是当前已有输出：

- `prediction["image"]`

原因：

- 这是当前主干已经完成几何推断后的结果
- 它的分辨率与监督一致
- 它不依赖额外中间状态暴露
- 作为“条件深度”最符合参考图红框前的模块语义

### 10.3 这意味着什么

从系统职责看，未来完整系统将变成：

```text
Stage A:
  Spike -> Current U-Net -> coarse depth d_u

Stage B:
  coarse depth d_u -> conditional manifold flow -> refined depth d_hat
```

这是一个**级联式两阶段深度估计框架**。

---

## 11. 方案优点与边界

### 11.1 优点

1. 不破坏当前主干
   - 现有训练和推理逻辑仍然成立

2. 职责分明
   - 当前 U-Net 负责“看懂输入并给出初始深度”
   - manifold flow 负责“把深度拉回合理流形并细化”

3. 更适合处理一对多深度不确定性
   - 相同 spike 观测下，某些区域存在深度歧义
   - flow 模块天然适合建模条件分布

4. 能自然接入时序几何约束
   - 当前仓库已经有序列样本机制
   - 参考图中的 `t / t+1` 约束有落点

### 11.2 边界

1. 严格意义上，这不是“用经典 M-flow 直接替换深度网络”
   - 而是“深度流形编码器 + 条件流匹配”的组合架构

2. 若只用 `d_u` 一个单通道图做条件，条件表达力有限
   - 所以需要 `C_phi` 做条件提升

3. 若不引入流形编码器 `E_psi / G_psi`
   - 则所谓 `manifold flow` 会退化成普通像素空间 flow
   - 这会削弱参考图里“隐式深度流形”的核心价值

---

## 12. 结论

结合当前仓库结构和参考架构图，最合理的设计不是修改现有 U-Net 主干，而是：

### 结论一句话

> 保留当前 `S2DepthTransformerUNetConv` 作为粗深度先验网络，将其输出 `prediction["image"]` 作为条件 `c`，再在学习到的深度流形隐空间上引入条件 manifold flow / Riemannian Flow Matching 模块，输出 refined depth。

### 更具体地说

当前系统负责：

- `Spike -> coarse depth`

新增 flow 系统负责：

- `coarse depth -> manifold-conditioned refinement -> final depth`

这样做的好处是：

- 与现有代码接口最自然
- 与参考图语义最一致
- 能保留当前 U-Net 的稳定性
- 能把深度估计从“单次确定性回归”提升为“受几何流形约束的条件连续输运”

---

## 13. 外部参考

1. Manifold-learning flows:
   - Johann Brehmer, Kyle Cranmer, *Flows for simultaneous manifold learning and density estimation*, NeurIPS 2020
   - https://papers.nips.cc/paper/2020/hash/051928341be67dcba03f0e04104d9047-Abstract.html

2. Flow Matching:
   - Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le, *Flow Matching for Generative Modeling*, arXiv:2210.02747
   - https://arxiv.org/abs/2210.02747

3. Riemannian Flow Matching:
   - Ricky T. Q. Chen, Yaron Lipman, *Flow Matching on General Geometries*, arXiv:2302.03660
   - https://arxiv.org/abs/2302.03660
