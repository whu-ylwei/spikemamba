# Manifold Flow 用于 CV 的架构综述与分析

> 生成日期：2026-07-09
> 关联文档：`manifold反思.md`（当前实现问题定位）、`manifold改进方向.md`（落地方案）、`ABLATION_RESULTS_manifold_flow_20260708.md`（实验证据）
> 目的：系统梳理 "manifold flow" 在 CV 中的几条技术脉络与代表架构，厘清概念边界，最后回接到本项目 `ConditionedDepthManifoldFlow` 的定位与选型建议。
> 说明：因网络限制无法逐篇抓取全文，本综述基于各论文的摘要/公开描述 + 该领域已确立的方法论共识撰写；每条主张附出处，细节以原文为准。

---

## 0. 一个必须先厘清的术语问题

"Manifold flow" 在 CV/生成文献里并不是单一模型，而是**两条独立源头**共用了"流形 + 流"这个词，含义不同：

1. **流形学习流（Manifold-learning Flow，M-flow 一脉）**：把"流"当作**可逆映射**，同时学**数据流形本身**和流形上的**密度**。关注"数据其实在低维流形上，我要把这个流形显式建出来"。源头：Brehmer & Cranmer, *Flows for simultaneous manifold learning and density estimation*（M-flow, 2020）。
2. **黎曼流匹配（Riemannian Flow Matching, RFM 一脉）**：流形是**已知/给定**的（球面、SO(3)、环面……），在其上用**流匹配**学速度场，沿**测地线**插值传输。关注"我已知几何，如何在其上做流匹配"。源头：Chen & Lipman, *Riemannian Flow Matching on General Geometries*（2023）。

再叠加第三条工程脉络：

3. **潜空间流（Latent Flow）**：不显式谈"流形"，但用**预训练冻结的 VAE** 把图像压到低维 latent，在 latent 里跑（rectified）flow。工业级代表：SD3 *Scaling Rectified Flow Transformers*（2024）；深度领域代表：DepthFM（2024）。这条其实是 M-flow 思想的**实用近似**——VAE latent 就是"固定住的流形坐标"。

> **本项目的类名 `ConditionedDepthManifoldFlow` 混用了这三者的词汇，但架构上哪一条都没做完整**（见第 6 节）。理解上面三条边界，是判断它该往哪走的前提。

---

## 1. 脉络一：M-flow（流形学习流）—— manifold flow 的正统定义

**核心思想**（Brehmer & Cranmer, 2020，arXiv:2003.13913）：
真实数据（图像等）不充满整个高维像素空间，而是集中在一个低维流形附近（manifold hypothesis）。M-flow 用一个可逆变换把环境空间坐标拆成两组：
- **on-manifold 坐标**（低维，$u$）：描述"在流形上的位置"；
- **off-manifold 坐标**（余下维度，$v$）：描述"偏离流形多远"，生成时被**置零/投影**。

编码 = 把数据投影到流形（丢掉 off-manifold 分量）；解码 = 从流形坐标重建回环境空间。密度只在低维 on-manifold 坐标上估计。

**关键的两阶段训练**（对本项目最有借鉴意义的一点）：
- **阶段 M（manifold）**：用**重建损失**先把流形几何学对（编码→投影→解码要能重建数据）；
- **阶段 D（density）**：**固定住流形**，再在其上学密度/流。

论文明确指出**不能同时优化两者**——若一边挪动流形一边估计密度，密度目标会把流形往"好估计"的方向拽偏，得到退化解。后续 FOM（Fixed-manifold）变体干脆用**已知固定流形**只做密度。

> **对本项目的直接教训**：manifold flow 的第一原则就是"**先固定流形，再在其上流**"。本项目三个编解码器与 flow、主任务**一次性联合训练**（`manifold反思.md` 第三节第 3 条），正是 M-flow 论文警告要避免的"移动流形上估密度"。

出处：[Flows for simultaneous manifold learning and density estimation (M-flow, 2020)](https://arxiv.org/abs/2003.13913)；后续 [Multi-chart flows (2021)](https://arxiv.org/html/2505.24665v2)、[Canonical normalizing flows (2023)](https://arxiv.org/html/2310.12743) 讨论了单图无法覆盖非平凡拓扑、需多图/正则化等改进。

---

## 2. 脉络二：Riemannian Flow Matching（流形上的流匹配）

**核心思想**（Chen & Lipman, 2023，arXiv:2302.03660）：
当流形几何**已知**时（不需要学），把欧氏 flow matching 推广到黎曼流形：
- 条件路径用**测地线插值** $x_t = \exp_{x_0}(t\,\log_{x_0}(x_1))$（直线的流形版本），而非欧氏直线；
- 目标速度场沿测地线方向；训练回归它，**无需散度计算、在简单几何上 simulation-free**；
- 用 **premetric**（一个随距离单调的标量）构造目标向量场，泛化到一般几何。

它同样是 **data-to-data 的条件路径**（从一个具体 $x_0$ 到一个具体 $x_1$），这一点和欧氏 flow matching 的 rectified 版一致。

**对 CV 的意义**：图像/深度的"流形"通常**未知**，所以纯 RFM 不能直接套；但它给出的原则——**在正确的几何上沿最短路径传输**——正是 latent flow（脉络三）用"VAE latent 空间"去近似的东西。近期 [Riemannian MeanFlow (2026)](https://arxiv.org/abs/2602.07744) 进一步把它压到**一步生成**。

出处：[Riemannian Flow Matching on General Geometries (2023)](https://ar5iv.labs.arxiv.org/html/2302.03660)、[Matching Normalizing Flows and Probability Paths on Manifolds (2022)](https://ar5iv.labs.arxiv.org/html/2207.04711)。

---

## 3. 脉络三：Latent Flow —— CV 里真正落地的形态

这是"manifold flow 精神"在 CV 里最成功的工程化，共同套路：

**先固定流形（冻结 VAE），再在 latent 里做 flow matching。**

- **SD3 / Scaling Rectified Flow Transformers**（Esser et al., 2024，arXiv:2403.03206）：在**冻结的 VAE latent** 空间用 rectified flow + DiT（Transformer）主干做文生图。它印证了两件事：(a) latent 空间（=固定流形坐标）是 flow 的正确舞台；(b) rectified 的**直线路径 + 少步采样**在大规模上成立。
- **DepthFM**（Gui et al., 2024，arXiv:2403.13788）：把深度估计建成 **image latent → depth latent 的直接传输**（data-to-data 耦合），在 SD 的**冻结 VAE** latent 里跑 flow matching，1–2 步出图。**没有噪声端**——两端都是有意义的数据分布。图像 latent 既作起点又作条件。

> **对本项目**：DepthFM 是与本任务最接近的正例。它的三个关键选择恰是本项目反着做的：起点=图像 latent（非噪声）、latent=预训练冻结 VAE（非从零联合训）、耦合=data-to-data（非独立噪声）。见 `manifold反思.md` 对照表。

出处：[Scaling Rectified Flow Transformers (SD3, 2024)](https://arxiv.org/abs/2403.03206)、[DepthFM (2024)](https://arxiv.org/html/2403.13788v2)。

---

## 4. 脉络四：桥模型 / 精修范式 —— "从数据出发而非从噪声出发"

这条脉络对**"精修一个已有的 coarse 预测"**这个具体场景最对口，理论上回答了本项目"为什么不该从噪声起步"。

- **InDI（Direct Iteration）**（Delbracio & Milanfar, 2023，arXiv:2303.11435）：图像恢复不必"从噪声反复去噪"，可以**直接在退化图与干净图之间做小步迭代**，起点就是退化图。
- **I2SB（Image-to-Image Schrödinger Bridge）**（Liu et al., 2023，arXiv:2302.05872）：明确"**不从随机噪声生成，而是直接学两个给定分布（退化↔干净）之间的桥**"。原文强调"退化图是重建干净图的**结构性先验**"。
- **TryOn-Refiner**（ICCV 2025）：把精修"从 noise-to-image 范式改成直接把 coarse 映射到 refined 的 flow mapping"，源分布是 coarse 不是高斯噪声（`manifold反思.md` 已引）。
- 近期 [Residual Diffusion Bridge (2026)](https://arxiv.org/html/2510.23116) 进一步把桥建成**残差**形式（学干净−退化的增量），与"锚定 coarse 做残差"完全一致。

**共同结论**：当你已经有一个信息量很大的起点（退化图 / coarse 深度），正确做法是**在起点↔目标之间建桥/流，并以残差方式锚定起点**，而不是丢掉它从噪声重生。

出处：[InDI (2023)](https://arxiv.org/abs/2303.11435)、[I2SB (2023)](https://ar5iv.labs.arxiv.org/abs/2302.05872)、[Residual Diffusion Bridge (2026)](https://arxiv.org/html/2510.23116)。

---

## 5. 脉络五：为什么"起点在流形外"会出问题（理论旁证）

几篇分析型工作解释了"从噪声起步 + 少步积分"在流形数据上的困难：

- **Flow Matching is Adaptive to Manifold Structures**（arXiv:2602.22486）：flow 在数据集中于低维流形时训练更稳，但这依赖于**目标端在流形上**；起点在流形外时，速度场要先把样本"拉回流形"再"沿流形移动"，是两个阶段。
- **两阶段本质**（*Two-Stage Nature via Oracle Velocity*, arXiv:2512.02826）：flow 的速度场天然分**早期导航（跨模式定全局布局）**与**后期精修（贴近具体样本）**两段。从噪声起步必须付出"早期导航"这段成本；若起点已接近目标（coarse），这段几乎可省。
- **memorize/generalize 分析**（arXiv:2410.23594）：off-subspace（偏离数据子空间）分量在积分中衰减、on-subspace 分量泛化——**起点离流形越远，越多算力浪费在把 off-manifold 分量压掉**。

> **对本项目**：`_rollout_latent` 从零/噪声起步（流形外），只用 6 步欧拉，等于要求 6 步内既完成"拉回流形"又完成"贴近目标"——理论上这正是少步积分最吃力的设定。改成 coarse 起点（已在/接近流形）后，6 步几乎全用于精修，效率与精度都应改善。

出处：[Flow Matching is Adaptive to Manifold Structures (2026)](https://ar5iv.labs.arxiv.org/html/2602.22486)、[Two-Stage Nature of Flow Models (2025)](https://arxiv.org/abs/2512.02826)、[Memorize/Generalize in Data Subspaces (2024)](https://arxiv.org/html/2410.23594v1)。

---

## 6. 回接本项目：`ConditionedDepthManifoldFlow` 的定位

把五条脉络的"标准动作"与当前实现并列：

| 维度 | M-flow（脉络1） | RFM（脉络2） | Latent/DepthFM（脉络3） | 桥/精修（脉络4） | **当前实现** |
|---|---|---|---|---|---|
| 流形是否先固定 | **是**（先重建再估密度） | 已知给定 | **是**（冻结 VAE） | —（像素空间也可） | **否**（联合从零训） |
| 起点分布 | 流形坐标 | 流形上点 | 图像 latent | **退化图/coarse** | **噪声/零（流形外）** |
| 耦合方式 | — | data-to-data 测地线 | **data-to-data** | **data-to-data** | **独立噪声↔latent** |
| 是否锚定输入 | — | — | 条件输入 | **是（残差桥）** | **否**（`blend_alpha=0`） |
| 尺度处理 | 保结构 | 保几何 | VAE 保信息 | 保输入信息 | **逐图归一化去尺度** |

**判断**：当前架构挂着"manifold flow"的名，但：
- 没做到脉络 1 的核心前提（先固定流形）；
- 没有脉络 2/3/4 一致要求的 **data-to-data 耦合**（它用独立噪声耦合）；
- 违背脉络 4/5 的"从数据出发、锚定输入"（它从噪声起、默认替换 coarse）。

即"取了每条脉络里最不利于本任务的一半拼起来"，与消融中 FLOW 打不过 BASE 的结果自洽。

---

## 7. 选型建议：两条自洽路线（择一）

**路线 A｜latent manifold flow 正统（贴近 DepthFM，若要保留"流形"叙事）**
1. 先用重建损失把深度自编码器（`target_encoder`+`target_decoder`）训好并**冻结**——满足 M-flow"先固定流形"。
2. flow 在冻结 latent 里做 **coarse latent → GT latent** 的 data-to-data 传输（起点=coarse 的 latent，非噪声）。
3. 去掉逐图归一化，保尺度。
- 优点：叙事自洽、可少步；缺点：需分阶段训练 + 改代码。

**路线 B｜桥式残差精修（贴近 I2SB/InDI/TryOn-Refiner，最省、最稳，推荐先试）**
1. 直接在**特征/像素空间**建 coarse↔GT 的桥，起点=coarse，**输出=coarse + Δ（残差锚定）**。
2. 不必显式造"流形"，避开"移动流形"陷阱。
3. 下界安全：Δ→0 时退化为 coarse，天然不劣于 BASE。
- 优点：概念最简、下界安全、与实验证据最贴合；缺点：放弃"manifold"叙事（但换来可用性）。

> 无论哪条，`manifold改进方向.md` 第 2 节的纯 config 实验（`blend_alpha↑`、`encoder_feature_scale=0` 保尺度）都应**先跑**，用最低成本验证"锚定 coarse + 保尺度"能否让 FLOW ≥ BASE，再决定投入 A 或 B 的代码改动。

---

## 参考文献

**Manifold-learning flow（脉络1）**
- Flows for simultaneous manifold learning and density estimation (M-flow) — https://arxiv.org/abs/2003.13913
- Multi-chart flows — https://arxiv.org/html/2505.24665v2
- Canonical normalizing flows for manifold learning — https://arxiv.org/html/2310.12743

**Riemannian flow matching（脉络2）**
- Riemannian Flow Matching on General Geometries — https://ar5iv.labs.arxiv.org/html/2302.03660
- Matching Normalizing Flows and Probability Paths on Manifolds — https://ar5iv.labs.arxiv.org/html/2207.04711
- Riemannian MeanFlow — https://arxiv.org/abs/2602.07744

**Latent / rectified flow（脉络3）**
- Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3) — https://arxiv.org/abs/2403.03206
- DepthFM: Fast Monocular Depth Estimation with Flow Matching — https://arxiv.org/html/2403.13788v2
- Flow Straight and Fast: Rectified Flow (Liu et al., 2022) — https://arxiv.org/abs/2209.03003

**桥模型 / 精修范式（脉络4）**
- InDI: An Alternative to Denoising Diffusion for Image Restoration — https://arxiv.org/abs/2303.11435
- Image-to-Image Schrödinger Bridge (I2SB) — https://ar5iv.labs.arxiv.org/abs/2302.05872
- Residual Diffusion Bridge Model for Image Restoration — https://arxiv.org/html/2510.23116
- Conditional Rectified-flow-based TryOn Refiner (ICCV 2025) — https://iccv.thecvf.com/virtual/2025/poster/1085

**流形结构与 flow 的理论分析（脉络5）**
- Flow Matching is Adaptive to Manifold Structures — https://ar5iv.labs.arxiv.org/html/2602.22486
- Revealing the Two-Stage Nature of Flow-based Diffusion Models — https://arxiv.org/abs/2512.02826
- How Do Flow Matching Models Memorize and Generalize in Sample Data Subspaces? — https://arxiv.org/html/2410.23594v1
