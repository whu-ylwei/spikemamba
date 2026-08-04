# Flow 学习用于远景/小目标深度增强的可行性研究

**日期**：2026-08-04
**背景**：承接 `FLOW_DIAGNOSIS_AND_THEORY_20260804.md` 的诊断结论——冻结主干上做残差 flow
精修**零增益**，因为 flow 的条件信息 ⊆ coarse，无法修正 coarse 已丢失的信息。本文论证一个
更有针对性的方案：**让 flow 从 spike 原始数据 / Mamba 主干中间各层输出中提取 coarse 里没有的
信息，专门增强远景与小目标深度**，并对照已有文献评估其可行性。

**本文档为文献调研+可行性论证，不改代码。**

---

## 1. 问题定义与动机

### 1.1 主干在远景/小目标上系统性退化（实证）

0320 Mamba 主干（DENSE test，224² CenterCrop）分距离档表现：

| 档(米) | abs_rel | δ<1.25 |
|---|---|---|
| _10（近） | 0.274 | 0.989 |
| _30 | 0.559 | 0.896 |
| _80 | 0.696 | 0.714 |
| _250 | 0.707 | 0.636 |
| _500（远） | 0.707 | 0.636 |

近景近乎完美（δ=0.989），远景显著退化（δ=0.636）。误差集中在远距离/小目标，三重成因：
1. **量程指数解码**：`depth = exp(reg_factor·x)`（reg_factor=5.7），远景 x→1 时 latent 微小误差被指数放大；
2. **spike 远景信息稀疏**：远处小目标在 spike 帧上占像素少、事件率低、SNR 差，主干多级下采样后被稀释；
3. **损失近景主导**：SI/L1 类损失被近景大量像素支配，远景欠拟合。

### 1.2 核心命题

> **若"远景真实结构信息"在 spike 原始流 / 主干浅层高分辨率特征中仍然存在（只是被主干深层下采样
> 或量程解码丢失），则一个能访问这些信息的生成式精修模块（flow），有可能在冻结主干（守住 coarse
> 0.595 不劣化）的前提下，定向恢复远景/小目标深度。**

可行性取决于两个问题，本文分别用文献与本项目实证回答：
- **Q1（信息存在性）**：spike/浅层特征里是否真的含 coarse 缺失的远景信息？（§4 前置探针）
- **Q2（方法有效性）**：生成式/flow 精修 + 多源条件，在文献中是否已被证明能改善困难区域深度？（§2-§3）

### 1.3 小目标 vs 远距离：同源但不对称，一套架构如何同时兼顾

这是本方案的核心张力——**两个目标都要，但最终 flow 架构是唯一的一套**。必须讲清三点。

**(1) 两者是不同的失败模式，但根因同源。**
- **远距离**=深度值大的区域，失败源于：exp(5.7·x) 量程指数放大 + spike 远景事件稀疏 + 损失近景主导。
- **小目标**=占像素少的物体（可近可远），失败源于：主干多级 2×2 下采样把小目标像素合并/抹掉 + 空间细节丢失 + 大区域主导。
- 二者重叠于"远物常表现为小目标"，但不等价（近处细杆=小目标非远；远处大墙=远非小）。
- **共同根因**：主干深层那个被下采样的 latent，同时丢掉了「高分辨率空间细节」（伤小目标）和
  「原始稀疏事件线索」（伤远距离）。coarse 在这两类区域都不可靠。

**(2) 为什么一套架构能同时覆盖两者。**
flow 要学的事在两种情况下**数学形式完全相同**：给定不可靠的 coarse + 富含高分辨率/原始信息的
条件，学 p(depth | coarse, rich_condition) 输出修正 Δ。flow **不需要"知道"自己在修小目标还是远区**，
它只需学到"哪里 coarse 不可靠、且条件里有补足信号"。两类难区共享同一个信息缺口 → 共享同一个
信息补足机制（多源条件 + 门控残差 flow）。**这就是"架构一定"能成立的根据。**

**(3) 但两者可行性不对称（最须诚实的一点）。**

| | 失败根因 | 信息可恢复性 | 可行性 |
|---|---|---|---|
| **小目标** | 主要是主干**下采样**丢细节（架构性） | **高**——细节仍在浅层/原始 spike，只是深层没用好 | **较高** |
| **远距离** | 下采样 + **spike 物理稀疏** | **部分**——下采样那部分可救，事件稀疏那部分可能物理缺失 | **存疑，取决于探针 A** |

- 小目标失败是**架构性信息丢失**，条件注入（浅层高分辨率特征/原分辨率 spike）大概率能救回。
- 远距离失败**一部分是物理性信息缺失**——远处 spike 事件本就稀疏，若原始数据远景 SNR 已太低，
  那是传感器/数据上限，再精修也无中生有不出来。故探针 A（仅从 spike 回归远景）是不可绕过的
  go/no-go：它测的正是远距离信息到底"没用好"还是"物理没有"。

**(4) "两个都要"落在损失/门控，不在架构。**
架构统一，但同时照顾两类难区靠三处设计（详见 §5.3）：
- **复合加权损失**：深度加权（抓远）+ 边界/尺寸加权（抓小目标），不能只用远景加权（会淹没近处小目标）；
- **不确定性门控是天然统一器**：小目标与远距离**都是主干高不确定区**，一个"只在低置信区精修"的
  门控无需手写判断即自动同时覆盖两者；门控图须在**高分辨率**上算，否则糊掉小目标边界；
- **条件源同供两类信息**：spike 条件编码器同时输出「高分辨率空间细节」（救小目标）与
  「事件率/时间统计」（救远距离）。

**(5) 潜在冲突（须预防）**：远景加权可能压过近处小目标（→加尺寸/边界权重补偿）；门控粒度太粗糊掉
小目标（→高分辨率门控）；两类都放开 Δ 后可能拟合远景 GT 噪声（→近景强制 Δ→0 + 鲁棒损失）。

---

## 2. 文献支撑（一）：Flow/扩散用于深度估计与精修

> 注：以下引用的标题/年份/arXiv 以正式发表为准，落地成论文前需逐条核对。

**Flow matching / rectified flow 基础**
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023（arXiv:2210.02747）——
  连续归一化流的仿真无关训练目标，本项目 velocity_field 回归即基于此。
- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*,
  ICLR 2023（arXiv:2209.03003）——直线化 coarse→GT 路径（data coupling）的理论依据。

**Flow/扩散用于单目深度估计**
- Gui et al., *DepthFM: Fast Monocular Depth Estimation with Flow Matching*, 2024
  （arXiv:2403.13788）——本项目两阶段（固定 latent 流形→flow 精修）的直接范式来源。
- Ke et al., *Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth
  Estimation*, CVPR 2024（arXiv:2312.02145）——扩散先验做深度，证明生成式模型在深度上的有效性。
- Saxena et al., *The Surprising Effectiveness of Diffusion Models for Optical Flow and Monocular
  Depth Estimation* (DDVM), NeurIPS 2023（arXiv:2306.01923）——本项目 L_flow 用 smooth-L1
  （L1>L2）的依据。
- Duan et al., *DDP: Diffusion-Based Dense Prediction*（arXiv:2303.17559）——条件扩散做稠密预测。

**这些工作对本方案的意义**：证明"生成式精修 + 条件"在深度这种病态/歧义任务上能有效建模；
DepthFM 的两阶段（先固定流形再 flow）正是本项目 latentfix 的蓝本。缺口在于它们的条件通常是
RGB 图像本身，而本项目的 coarse 已丢信息——故需 §3 的多源条件补足。

---

## 3. 文献支撑（二）：多源条件 / 中间特征 / 事件相机 / 不确定性门控

**特征/多源条件注入**
- Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet),
  ICCV 2023（arXiv:2302.05543）——外部条件旁路注入生成模型的成熟范式，对应本项目把
  spike/浅层特征作为额外条件旁路喂入 velocity_field。
- Hariharan et al., *Hypercolumns for Object Segmentation and Fine-grained Localization*,
  CVPR 2015（arXiv:1411.5752）——多层特征拼接保留细粒度/高分辨率信息，支撑"用浅层特征补远景细节"。

**Mamba / 状态空间模型视觉主干**
- Zhu et al., *Vision Mamba: Efficient Visual Representation Learning with Bidirectional State
  Space Model* (Vim), ICML 2024（arXiv:2401.09417）。
- Liu et al., *VMamba: Visual State Space Model*, NeurIPS 2024（arXiv:2401.10166）——
  本项目主干（双向 Mamba + U-Net 解码）的架构族。

**事件/脉冲相机深度估计**
- Hidalgo-Carrió et al., *Learning Monocular Dense Depth from Events* (E2Depth), 3DV 2020
  （arXiv:2010.08350）——事件流单目稠密深度，证明事件/spike 原始流可直接驱动深度学习。
- Zhu et al., *Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion*,
  CVPR 2019——事件相机深度/运动联合学习。
- （spike 相机方向）本项目 DENSE-spike 数据与 spike→depth 任务，raw spike 作为条件的动机源。

**不确定性引导 / 置信度门控精修**
- Kendall & Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*,
  NeurIPS 2017（arXiv:1703.04977）——逐像素不确定性估计，支撑 §5.3 的置信度门控残差。
- Poggi et al., *On the Uncertainty of Self-Supervised Monocular Depth Estimation*, CVPR 2020
  （arXiv:2005.06209）——深度不确定性建模，指导"只在低置信区精修"。

**这些工作对本方案的意义**：ControlNet/hypercolumn 提供多源条件注入的工程范式；事件相机深度
证明 raw 事件流含可用深度信息（支撑探针 A 的假设）；不确定性门控提供"只修远景不扰近景"的机制。

---

## 4. 前置探针（可行性 gate，最先做）

在投入任何架构改动前，必须先回答 Q1——否则一切无意义。

### 4.1 探针 A：spike 远景信息存在性
训一个**只从 raw spike 直接回归深度**的轻量网络（不经过冻结主干），单看远景档（_250/_500）
abs_rel 能否低于主干 coarse 的 0.707。
- **若能** → spike 里确有主干没用好的远景信息 → 方案有上限空间，继续。
- **若不能** → spike 远景信息本身不足 → 是**传感器/数据物理上限**，任何后处理都触不到，
  应转向改数据采集或接受天花板。**这是 go/no-go 的硬判据。**

### 4.2 探针 B：主干中间层信息定位
对冻结主干**各层输出**分别接一个线性 probe 回归远景深度，看哪一层的远景信息最丰富。
预期：**浅层（高分辨率、下采样少）**保留更多远景小目标细节，深层已被抽象/稀释。
这决定 §5 的条件该从哪一层引出。

### 4.3 探针 C：远景 GT 可靠性
统计 DENSE 远景（>某阈值）的 GT 有效像素比例与噪声水平。若远景 GT 本身稀疏/噪声大，
"增强"会退化成"拟合噪声"，需相应设计鲁棒损失或降低对远景 GT 的信任。

---

## 5. 架构方案：多源条件 flow 定向精修

### 5.1 总体（主干仍冻结，守住 coarse 0.595）

```
  raw spike (128,224,224) ─┬─→ [冻结] Mamba主干 ─┬─ 浅层特征 F_shallow (高分辨率) ┐
                           │                     └─ coarse_depth ──→ E(coarse)=cond_latent ┐
                           └─→ [新·可训] spike条件编码器 C_ψ ──→ spike_cond ┐              │
                                                                            ▼              ▼
                        velocity_field( z_t, [ cond_latent ⊕ spike_cond ⊕ proj(F_shallow) ], t )
                                                                            │
                                                     refined = coarse + gate ⊙ Δ
```

三条信息源，互补覆盖"coarse 缺失的远景信息"：
1. **cond_latent**：coarse 深度编码（现有，保证与 coarse 一致性）；
2. **spike_cond**：新增 `C_ψ` 从 raw spike 提取的高分辨率/事件率统计特征（探针 A 验证其价值）；
3. **proj(F_shallow)**：冻结主干浅层高分辨率特征投影（探针 B 定位，零额外主干开销，
   对应现有 `encoder_feature_scale` 机制——当前设为 0 即未启用，可复用打开）。

### 5.2 具体条件设计（spike 侧，按信息增量）
- (a) **事件统计图**：逐像素 spike 计数 / 首末事件时间 / 时间质心 —— 显式暴露远景稀疏事件；
- (b) **原分辨率 spike 特征**：小 CNN 在 224²（不下采样）提特征，保留远景小目标空间细节；
- (c) **多时间窗聚合**：短窗（快速远景运动）+ 长窗（静态结构）多路 spike。

### 5.3 定向到远景（损失 + 门控）
让 flow **只在远处发力、不扰动近景**（近景 coarse 已近最优）：
1. **远景加权损失**：对远景像素（GT 深度大 / x→1）放大 refined 监督权重
   （与现有 `far_downweight_gamma` 反向，新增 `far_upweight`）；
2. **近景残差正则**：近景加 `‖Δ‖²` 惩罚，强制近处 Δ→0，只放开远处（空间选择性残差）；
3. **不确定性门控**：`refined = coarse + gate ⊙ Δ`，gate 由主干/flow 输出的逐像素置信度驱动，
   Δ 只在低置信区（远景/小目标）生效——文献中"uncertainty-guided refinement"范式。

### 5.4 相对现有代码的最小改动映射（供后续实现，非本次）
- `model/manifold_flow.py::_ConditionalVelocityField`：条件通道从 `cond_latent` 扩为拼接
  `[cond_latent ⊕ spike_cond ⊕ proj(F_shallow)]`（现已接受 condition_latent，扩通道即可）；
- 新增 `C_ψ`（spike 条件编码器）模块 + 前向里从 raw spike 计算；
- `encoder_feature_scale`（现有、当前=0）：打开以接入浅层特征，或新增浅层专用投影；
- 新增 `far_upweight` / 近景 `‖Δ‖²` / gate 三个损失开关；
- 训练仍复用 latentfix 两阶段：冻主干+冻AE，只训 `C_ψ`+velocity_field+gate（守住 coarse）。

---

## 6. 可行性评估与风险

### 6.1 支持可行的论据
- **文献先例**：flow/扩散做深度（§2）已证明生成式精修在困难/歧义区域有效；多源/特征条件与
  不确定性门控（§3）是成熟范式。方法层面非空想。
- **诊断已定位靶点**：远景是主干确定的系统性弱区（§1.1），有明确优化目标。
- **不劣化保证**：冻结主干 + 门控残差，最坏情况 Δ→0 退化为 coarse（0.595），不会重蹈
  0731/0801 联合训练污染主干的覆辙（见 `FLOW_DIAGNOSIS_AND_THEORY_20260804.md` §B）。

### 6.2 风险 / 可能证伪
1. **spike 远景信息不足**（最大风险）：若探针 A 失败，是数据/传感器物理上限，方案无解。
   **必须先过探针 A。**
2. **远景 GT 噪声**：DENSE 远景 GT 可能稀疏/不可靠，监督信号差。
3. **exp 解码放大**：远景米空间误差被指数放大，训练宜在 log 空间或线性空间 L1 直接监督远景。
4. **显存/算力**：`C_ψ` 在 224² 原分辨率跑，激活远大于 latent 空间，显存压力大
   （参考 batch 化经验，见 [[reB-stage2-batching-speedup-0804]]）。
5. **收益不确定**：即便信息存在，flow 精修带来的远景 δ 提升幅度未知，需实验定。

### 6.3 与"直接改主干"路线的取舍
若探针 A/B 显示 spike/浅层远景信息充足但主干没用好，**更简单的路线是直接改主干**
（远景加权重训、更高输入分辨率、多尺度分支），无需引入 flow 复杂度。flow 方案的价值在于
**主干架构/算力受限、但生成式精修能榨出额外信息**的情形。决策依赖探针结果。

---

## 7. 代码修改初步方案（仅设计，本次不改）

以下映射到现有代码的具体锚点，供后续实现。**当前不动任何代码。** 所有改动遵循现有约定：
config 开关默认关=旧行为，向后兼容。

### 阶段 0 — 探针（先做，不碰主 flow）
- **探针 A**（go/no-go）：新脚本 `scripts/diag_spike_farfield_probe_20260805.py`。
  一个轻量 CNN 直接吃 raw spike（`item` 里的 spike tensor，主干输入前），只回归深度，
  单独评 `_250/_500` 档 abs_rel/δ，对比 coarse 0.707。复用 `evaluation_DENSE.py` 同口径。
- **探针 B**（中间层信息定位）：在 `S2DepthNet.forward`（`model/S2DepthNet.py:223` `encoded_xs = self.encoder(...)`
  之后）挂 hook 取各层输出，各接一个 linear probe 回归远景深度，看哪层远景信息最丰富。
  产出：选定接入 flow 的浅层索引。

### 阶段 1 — spike 条件编码器 C_ψ（新模块）
- **新增** `model/manifold_flow.py`：`class _SpikeConditionEncoder(nn.Module)`。
  - 输入：raw spike（`num_bins_rgb`×224×224，见 `S2DepthNet.py:191 expected_channels`）
    或其事件统计（逐像素计数/首末事件时间/时间质心，在 dataset 或 forward 里预计算）。
  - 输出：与 latent 同空间分辨率（56×56 级、`latent_channels`=96）的条件图 `spike_cond`。
  - 结构：几层 stride-conv 下采样到 latent 分辨率的小 CNN；为救小目标，保留一路
    **高分辨率 skip** 或用较浅下采样。
- **config 开关**：`manifold_flow.spike_condition = {enabled, in_stat_mode, out_channels}`，默认 `enabled=false`。

### 阶段 2 — velocity_field 扩条件通道（最小侵入改动）
- **锚点**：`_ConditionalVelocityField.__init__`（`model/manifold_flow.py:109`）当前
  `in_channels = latent_channels*2 + 1`（= z_t ⊕ cond_latent ⊕ time_map）。
- **改法**：扩为 `latent_channels*2 + 1 + spike_cond_ch (+ shallow_ch)`，`forward` 里把
  `spike_cond`、可选 `proj(F_shallow)` 一起 `torch.cat` 进条件。
- **浅层特征接入**：复用现有 `_fuse_encoder_features`（`manifold_flow.py`，`encoder_feature_scale`
  当前=0 即关闭）——打开并接探针 B 选定层，或新增浅层专用 1×1 投影。
- `ConditionedDepthManifoldFlow.__init__`（`:199` velocity_field 构造处）相应传入新通道数。
- 前向 `forward(coarse_depth, encoder_features, ...)`（`:346`）需新增入参 `spike_raw`/`spike_cond`，
  并从 `S2DepthNet.forward`（`:241` 调用处）把 spike 透传进来。

### 阶段 3 — 复合定向损失（改 loss 组装，同时压两类难区）
- **锚点**：`manifold_flow.py` 的 `_weighted_metric_l1`（现有，含 `far_gamma` 远景**降**权）
  与 `forward` 里 `L_refined_metric`/`L_delta_reg` 组装处。
- **新增权重**（config，默认关）：
  - `far_upweight_gamma`：远景**加**权 `w=exp(+γ·target)`（与现有 far_downweight 反向）→抓远距离；
  - `edge_weight`：边界/梯度幅值加权（`|∇GT|` 大处加权）→抓小目标；
  - `near_delta_reg`：近景（target 小）区域对 `‖Δ‖²` 加重惩罚，强制近处 Δ→0→只放开难区。
- 复合：`L = far_upweight·metric + edge_weight·metric_at_edges + near_delta_reg·‖Δ‖²_near`。

### 阶段 4 — 不确定性门控残差（统一覆盖两类难区）
- **锚点**：`manifold_flow.py:352-358` 的 `refined_depth = coarse_depth + decoded`。
- **改法**：`refined = coarse + gate ⊙ Δ`，`gate∈[0,1]` 逐像素。
  - gate 来源（二选一）：① velocity_field/新小头输出的置信度（sigmoid）；
    ② 主干输出方差的代理。
  - **门控图须在高分辨率**（decode 回 224² 后再 gate，或 gate 上采样），否则糊小目标边界。
- **config**：`manifold_flow.residual_gate = {enabled, source}`，默认关（=当前无门控行为）。

### 训练流程（复用现有两阶段，守住 coarse）
- 仍用 latentfix 两阶段：**冻结主干 + 冻结 AE**（守住 coarse 0.595，避免 0731/0801 联合训污染）。
- stage-2 可训集合从 `velocity_field + residual_decoder` 扩为
  `+ C_ψ + gate头 + 浅层投影`。runner 复用 `run_stage2_batched_20260804.sh`（batch 化 + host-RAM
  安全设置，见 [[reB-stage2-batching-speedup-0804]]），仅换 config。
- 复用 stage-1 AE `ep29` 快照，无需重训 stage-1。

### 评测（防档位陷阱）
- 复用 `scripts/diag_export_coarse_refined_20260804.py`（已能同时导 coarse/refined）+
  `evaluation_DENSE.py --crop_ymax224 --clip1000 --reg5.7`。
- **判据锁定同一档**：主看 `_250/_500` 远景档 refined vs coarse 的 δ/abs_rel 增量，
  兼看 total 档不退化。避免 `FLOW_DIAGNOSIS_AND_THEORY_20260804.md` §A.2 的档位混淆。

### 改动风险控制
- 每个 config 开关默认关=完全旧行为，可单独开启做消融（隔离 spike_cond / 浅层 / 门控 / 各损失的贡献）。
- 主干始终冻结→最坏情况 Δ→0 或 gate→0，退化为 coarse（0.595），不劣化。

---

## 8. 结论与下一步（仅规划）

**可行性判断**：方法层面有充分文献支撑且靶点明确，但**成败取决于 §4 探针 A 的信息存在性判据**，
这是不可绕过的前置 gate。

**推荐路线**：
1. **先做探针 A**（spike→远景深度回归）——go/no-go 硬判据；
2. 过则做探针 B（中间层信息定位）；
3. 再实现 §5 的多源条件 flow（冻主干、门控残差、远景加权），对照 coarse 只看远景档增益；
4. 全程锁定 evaluation_DENSE 同一档位对比，避免档位陷阱（见 `FLOW_DIAGNOSIS_AND_THEORY_20260804.md` §A.2）。

**本文档为文献调研 + 可行性论证，未改任何代码。实施前须先过探针 A。**

