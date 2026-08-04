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

## 7. 结论与下一步（仅规划）

**可行性判断**：方法层面有充分文献支撑且靶点明确，但**成败取决于 §4 探针 A 的信息存在性判据**，
这是不可绕过的前置 gate。

**推荐路线**：
1. **先做探针 A**（spike→远景深度回归）——go/no-go 硬判据；
2. 过则做探针 B（中间层信息定位）；
3. 再实现 §5 的多源条件 flow（冻主干、门控残差、远景加权），对照 coarse 只看远景档增益；
4. 全程锁定 evaluation_DENSE 同一档位对比，避免档位陷阱（见 `FLOW_DIAGNOSIS_AND_THEORY_20260804.md` §A.2）。

**本文档为文献调研 + 可行性论证，未改任何代码。实施前须先过探针 A。**

