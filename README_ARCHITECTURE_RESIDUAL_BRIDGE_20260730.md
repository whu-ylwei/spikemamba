# SpikeMamba 架构与代码结构说明（残差桥 route B）

- 整理日期：2026-07-30
- 对应分支：`0710` / `0710-residual-bridge`（对照分支 `0710-baseline`）
- 适用代码：`model/mamba_3d.py`、`model/S2DepthNet.py`、`model/manifold_flow.py`
- 相关文档：`manifold重构方案.md`、`README_CURRENT_MAMBAFLOW_40PLUS40_DATAFLOW_20260423.md`、`CLAUDE.md`

本文档描述当前 SpikeMamba 项目的算法结构与代码结构，聚焦 0710 起进行的「残差桥」(residual-bridge, route B) manifold flow 精修工作。

## 一、项目定位

脉冲相机单目深度估计（monocular depth estimation from a spiking camera）。源自 ECCV 2022 的 Spike Transformer，编码骨干已替换为 Mamba（状态空间模型），并在解码后叠加 manifold flow 精修模块。

关键前提（历史命名遗留）：

- 模型输入不是 RGB，而是 **128-bin 脉冲张量**（spike bins），数据集里键名仍叫 `item["image"]`。
- 跨帧时序建模不依赖 recurrent backbone，而是靠 trainer 的 `sequence_length=2` 展开 + manifold flow 的跨帧几何一致性项 `L_geo`。

## 二、算法结构：两阶段流水线

整体是「粗深度预测 → manifold flow 精修」的端到端两阶段设计。

```
spike bins [B,128,H,W]
   │
   ▼  ① 3D Mamba 编码器（时间维 = 脉冲 bin 维度）
多尺度特征 features[]
   │
   ▼  ② U-Net 解码器 + skip
coarse_depth [B,1,H,W]        ← 粗深度
   │
   ▼  ③ ConditionedDepthManifoldFlow（流形流精修）
refined_depth [B,1,H,W]       ← 最终输出
```

### ① 编码器：3D Mamba（`model/mamba_3d.py`）

把脉冲相机的 128 个 spike bins 当作时间维，用 3D 结构处理 `[B, C, T, H, W]`。

- `ST_Mamba_Block` 是核心块。`_fold_spatial_tokens` 把每个 2×2 空间邻域折入通道轴，使 Mamba 看到长度 `T·H/2·W/2`、embedding `4C` 的序列，处理完再由 `_unfold_spatial_tokens` 折回。
- 双向 Mamba（`mamba` + `mamba_bwd`），共享 `in_proj`/`out_proj`，仅按方向区分 SSM 参数（对齐 Vim BiMamba v2）。
- `Mamba3DEncoder` 是 `SwinTransformer3D` 的 drop-in 替换，逐 stage 下采样（只降空间、保时间分辨率）。

注意：`CLAUDE.md` 记录的「residual → MHC」（用多头交叉注意力替换残差加法）设计意图，在当前分支代码中尚未落地——`mamba_3d.py` 仍为简单残差加法（`x = x + x1`、`x = x + x2`，约第 81、87 行）。

### ② 解码器：U-Net（`model/S2DepthNet.py`）

`S2DepthTransformerUNetConv` 继承 `BaseERGB2Depth`，标准 U-Net：`resblocks → decoders（逐级上采样 + skip）→ pred head`。

- skip 类型可配 `sum` / `concat` / `no_skip`。
- 输出头激活可配 `identity` / `sigmoid` / `softplus`，并支持 `output_scale`、`output_shift`。
- 产出 `coarse_prediction`，作为第三阶段的锚点。

### ③ Manifold Flow 精修（`model/manifold_flow.py`，当前研究核心）

核心思想：把深度图编码到低维流形隐空间，在隐空间用 flow matching（rectified flow 风格）从「粗深度隐码」流动到「真值隐码」，再解码回深度增量。

组件：

- `_DepthDownsampleEncoder` / `_DepthUpsampleDecoder`：深度图 ↔ 隐码的编解码器。
- `_ConditionalVelocityField`：条件速度场 `v(z_t, z_condition, t)`，输入拼接 `[latent_state, condition_latent, time_map]`。
- `_rollout_latent`：欧拉法积分 `flow_steps` 步，`z ← z + v·Δt`。

## 三、残差桥 (route B) 的开关（C1–C5）

对应 `manifold重构方案.md`，在 `ConditionedDepthManifoldFlow.__init__` 约第 150–153 行定义。四个开关全 `False` + 旧 config 即退化回原始「噪声起步」flow，这也是 `0710-baseline` 与 `0710-residual-bridge` 两个对照分支的由来。

| 开关 | 代号 | 含义 |
| --- | --- | --- |
| `start_from_coarse` | C1 | rollout 从粗深度隐码起步，不从噪声起步 |
| `data_coupling` | C2 | 速度场学「粗隐码 → 真值隐码」的直线路径（`z0 = condition_latent`，`target_velocity = target_latent - z0`） |
| `residual_output` | C3 | 最终输出 = `coarse_depth + decoded`（解码器只产出增量 delta） |
| `share_encoder` | C4 | 粗深度与真值共用同一编码器，保证隐码在同一流形 |
| （特征融合）| C5 | `_fuse_encoder_features` 去掉 per-image 归一化，保留绝对（metric-depth）尺度 |

## 四、损失结构

每个时间步的总损失（参见 `README_CURRENT_MAMBAFLOW_40PLUS40_DATAFLOW_20260423.md`）：

```
L_t = L_si + 0.25·L_grad + L_coarse + L_rec + L_flow + L_geo
```

| 项 | 来源 | 作用 |
| --- | --- | --- |
| `L_si` | scale-invariant loss（对 refined_depth） | 主损失 |
| `L_grad` | 多尺度梯度损失 | 边缘/结构 |
| `L_coarse` | 对粗深度的 scale-invariant 监督 | 稳定第一阶段 |
| `L_rec` | 真值 encoder→decoder 重建 L1 | 保证隐空间可逆 |
| `L_flow` | 速度场 MSE（`predicted_velocity` vs `target_velocity`） | flow matching 核心监督 |
| `L_geo` | 跨帧 smooth-L1（`predicted_delta` vs `target_delta`） | 跨帧几何一致性，仅在有前一帧 latent 时生效 |

当前残差桥配置权重（见训练 config）：`coarse_weight=1`、`reconstruction_weight=0`、`encoder_feature_scale=0`，把 flow 定位成纯精修器。

## 五、代码结构

```
spikemamba/
├── model/                          # 核心模型代码（约 2300 行）
│   ├── mamba_3d.py                 # ① 3D Mamba 编码器（ST_Mamba_Block）
│   ├── swin_transformer_3d.py      # 旧 Swin 编码器（被 Mamba 替换）
│   ├── S2DepthNet.py               # ② U-Net 解码器 + 前向总装
│   ├── manifold_flow.py            # ③ 残差桥 flow 精修（研究重点）
│   ├── model.py                    # BaseERGB2Depth 基类
│   ├── submodules.py               # ConvLayer / ResidualBlock / UpsampleConvLayer
│   ├── encoder_transformer.py      # LongSpikeStreamEncoderConv 封装
│   ├── loss.py / metric.py         # 损失与评估指标
├── trainer/spiket_trainer.py       # SpikeTTrainer，序列展开 forward_pass_sequence
├── data_loader/                    # DENSE-spike 数据集
│   ├── SpikesDENSE_dataset.py      # Synchronized / Sequence 数据集
│   └── spike_dataset.py
├── train_parallel.py               # 训练入口（DDP）
├── configs/*.json                  # 训练配置
├── scripts/                        # 消融与训练启动脚本
├── DENSE-spike/                    # 数据（符号链接）
└── runs/ , run_logs/               # 训练输出与日志
```

## 六、数据流最简摘要

```
DENSE-spike/DENSE/train/.../spike/r128/spike_*.npy
  → 读出 events，形状 [128, H, W]
  → 数据集写入 item["image"]
  → DataLoader 组成 sequence_length=2, batch_size=2
  → trainer 逐时间步展开
  → 模型 [B,128,H,W] → 3D Mamba encoder
  → 2D U-Net decoder → coarse_prediction
  → ConditionedDepthManifoldFlow → refined_depth
  → 对 depth_image 计算主损失 + 辅助损失 → loss / val_loss
```

## 七、当前训练状态（截至 2026-07-30）

- 训练：残差桥 route B manifold flow，40+40 epoch schedule。
- 输出目录 `runs/flow_reB_40plus40_20260710/`，日志 `run_logs/flow_reB_40plus40_20260710.log`。
- 对照基线 BASE best `val_loss` 0.01214；残差桥 best 为 stage2 ep7 `val_loss` 0.01293，stage2 val 稳定在约 0.0141，train 降至约 0.0010。
- 全量数据集 4990 train / 998 val，数据在 CephFS 网络盘，I/O 受限。

> 说明：本节数值来自训练监控记录，若需最新结果请以 TensorBoard / 最新日志为准。
