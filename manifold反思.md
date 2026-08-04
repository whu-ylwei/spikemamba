# Manifold Flow 反思：当前架构为何不合理

> 生成日期：2026-07-09
> 分析对象：`model/manifold_flow.py`（`ConditionedDepthManifoldFlow`）+ `model/S2DepthNet.py` 集成
> 依据：消融实验结果（`ABLATION_RESULTS_manifold_flow_20260708.md`）+ CV 领域 flow matching / rectified flow 用于深度与精修的文献

---

## 一、实验先说明了什么

同管线 test 评估（`eval_test_manifold_flow_20260708.py`，112×112 resize + `model/metric.py`，n=998）：

| 指标 | 旧 ep70（完整训练，无 flow） | BASE 消融（ep15 子采样，无 flow） |
|------|------|------|
| test_loss ↓ | 0.13517 | **0.03976** |
| scale_invariant_error ↓ | 0.07716 | **0.008046** |
| rms_linear ↓ | 0.41763 | **0.30002** |

消融最佳 epoch（双方均在 ep15 达到 best val_loss）对比：

| 指标 | BASE (flow off) @ep15 | FLOW (flow on) @ep15 |
|------|------|------|
| val_loss ↓ | **0.012142** | 0.013717 |
| scale_invariant_error ↓ | **0.003501** | 0.006147 |

**结论**：主干（Mamba 编码器 → UNet 解码器）产出的 coarse 深度本身是好的；flow 头在监控指标（val_loss / scale-invariant）上是**净负贡献**。这不是"没调好"，而是结构性问题。（注：FLOW 那次训练因挂载故障崩在 ep19，但在双方共同最优的 ep15 已落后，故与训练不足无关。）

---

## 二、文献里 flow 用于深度/精修时怎么做

- **DepthFM**（AAAI'25）：把深度估计建成 **image 分布 → depth 分布的直接传输**（data-to-data 耦合），学一条近直线的速度场；在**预训练冻结的 VAE latent** 里做，1–2 步即可出图。关键：两端都是有意义的数据分布，不是"噪声 → 深度"。
- **TryOn-Refiner**（ICCV'25）：明确把精修"**从 noise-to-image 范式改成直接把 coarse 映射到 refined 的 flow mapping**"。源分布就是 coarse，不是高斯噪声。
- **Rectified Flow** 系：data-dependent coupling 让路径变直，少步数即可积分；独立噪声耦合的路径是弯的，欧拉少步会有明显积分误差。

**共识**：深度/精修任务的 flow 应在两个数据分布之间流动（coarse→GT 或 image→depth），并锚定 coarse。

---

## 二·补充、三种架构详解

统一底座：三者都属于 **flow matching / rectified flow** 家族。共同思路是在两个分布之间学一个**速度场** $v_\theta(x_t, t)$，训练时沿起点 $x_0$ 与终点 $x_1$ 的**直线插值** $x_t=(1-t)x_0 + t\,x_1$ 采样，回归目标速度 $x_1-x_0$；推理时从起点出发用 ODE 积分（欧拉几步）走到终点。**区别只在于"两端是什么分布"以及"在哪个空间里流"。**

### 1. DepthFM —— image latent → depth latent（data-to-data）
- **架构**：基于 Stable Diffusion 的预训练 U-Net + **冻结 VAE**，全程在 SD 的 **latent 空间**做，非像素空间。
- **流的两端**：起点 $x_0$ = 图像经冻结 VAE 编码的 latent；终点 $x_1$ = 深度图经同一 VAE 编码的 latent。是 image latent → depth latent 的直接传输，**没有噪声端**。图像 latent 同时作为 conditioning 拼进 U-Net。
- **为什么快**：起点已与终点高度相关（不是随机噪声），data-dependent 配对使路径近直，**1–2 步欧拉**即可。
- **对本项目启示**：本实现起点是噪声/零、latent 从零联合训练；DepthFM 起点是图像 latent、latent 用冻结预训练 VAE。两个关键点都反着做了。

### 2. TryOn-Refiner —— coarse → refined（精修范式）
- **架构**：一个 **conditional rectified flow 精修器**，接在已产出 coarse 结果的主模型之后。
- **流的两端**：起点 $x_0$ = coarse 图（主模型粗输出）；终点 $x_1$ = refined 图（GT/高质量目标）。论文明确把精修"从 noise-to-image 范式改成直接把 coarse 映射到 refined 的 flow mapping"，源分布是 coarse，**不是高斯噪声**。coarse 同时作为条件。
- **为什么合理**：coarse 已接近 refined，传输路径短，flow 只需学"从粗到精的增量修正"，而非从零凭空生成。
- **对本项目启示**：这正是"深度精修"该有的样子——应 coarse→GT 精修并锚定 coarse，而非扔掉 coarse 从噪声重生。本实现 `blend_alpha=0` 默认完全替换 coarse，方向相反。

### 3. Rectified Flow（底层方法论）—— 让路径变直
- **它不是具体网络，而是一套训练准则**，回答"怎么把 flow 路径学直、从而少步积分"。
- **核心**：学速度场使其对齐配对样本 $(x_0,x_1)$ 之间的直线方向 $x_1-x_0$；**coupling（配对）方式决定路径弯直**：
  - **独立耦合**（$x_0$ 随机噪声、$x_1$ 数据，二者独立采样）→ 不同样本对的直线交叉，平均速度场把路径压弯，欧拉少步误差大；
  - **data-dependent coupling**（$x_0,x_1$ 相关，如 coarse↔GT、image↔depth）→ 路径不易交叉、更直，少步即可。
- **Reflow**：用上一轮训好的模型重新生成配对 $(x_0,\hat{x}_1)$ 再训一遍，迭代把路径拉直，直到 1 步可积。
- **对本项目启示**：`L_flow` 用的正是独立耦合（`noise` 与 `target_latent` 独立），路径弯，却只用 6 步欧拉——rectified flow 明确指出这是会积分不准的组合。要么换 data coupling，要么做 reflow。

### 三者对照 + 当前实现
| | 起点 $x_0$ | 终点 $x_1$ | 空间 | latent 来源 | coupling |
|---|---|---|---|---|---|
| **DepthFM** | 图像 latent | 深度 latent | 冻结 VAE latent | 预训练冻结 | data-dependent |
| **TryOn-Refiner** | coarse 图 | refined 图 | 像素/latent | — | data-dependent (coarse↔GT) |
| **Rectified Flow** | 任意（推荐相关） | 任意 | 任意 | — | 强调 data coupling / reflow |
| **当前实现** | **噪声/零** | 自造 depth latent | 自造未冻结 latent | **从零联合训练** | **独立噪声耦合** |

**一句话**：flow 应在两个"有意义且相关"的数据分布之间流动，起点尽量接近终点，latent 空间要稳定（冻结）。当前实现在起点（噪声 vs coarse）、latent 稳定性（联合训 vs 冻结）、coupling（独立 vs data-dependent）三点上全部选了会让 flow 失效的那一侧，与消融中 FLOW 打不过 BASE 的结果一致。

---

## 二·补充二、manifold flow 与 flow matching / rectified flow 的区别

这两个词不是对立关系，而是**不同维度的概念**，本项目的类名 `ConditionedDepthManifoldFlow` 恰恰把两者各取了会失效的一半。

### 概念本义
- **flow matching / rectified flow**：指**怎么训一个连续变换（ODE）**——学速度场 $v_\theta(x_t,t)$，沿直线插值回归目标速度 $x_1-x_0$，推理时积分。它是**训练/采样范式**，不规定"在什么空间流"。rectified flow 是其变体，额外强调把路径拉直（data coupling / reflow）。
- **manifold flow**：专指**在低维流形上做的流**。核心主张：真实数据（图像、深度图）躺在一个低维流形上，因此先用（通常预训练冻结的）编解码器把数据映到流形坐标（latent），**在流形 latent 空间里做流**，再解码回去。代表工作 $\mathcal{M}$-flow（Brehmer & Cranmer, 2020）。

所以：**flow matching = 用什么目标训流（方法论）；manifold flow = 在哪里流，强调先降到低维流形（空间选择）。** 一个正经的 "manifold flow for depth" 应是：冻结的编解码器定义流形 → 在流形 latent 里用 flow matching 传输。DepthFM 就接近这个理想形态（冻结 VAE latent = 流形，flow matching 在里面跑）。

### 落到当前架构：名义 manifold，两头都残缺
- **借了 manifold flow 的"latent 空间"外壳**：有 `condition_encoder`/`target_encoder`/`target_decoder`，把深度降维到 latent 再流再解码，形式上像"在流形上流"。
- **但流形本身是坏的**：manifold flow 的前提是流形要先固定住（预训练冻结的 VAE，稳定、有意义的低维几何）。本实现三个编解码器从零、跟主任务联合训练、`L_rec` 权重仅 0.1——流形一边被定义一边被流过，是"流沙上的流形"，违背 manifold flow 最核心的前提。
- **训练目标是 flow matching，但用了最差的耦合**：`L_flow` 是 noise↔target_latent 的独立耦合，rectified flow 明确说这会把路径压弯，却只积 6 步欧拉。
- **起点又不在流形上**：`_rollout_latent` 从噪声/零起步——噪声不在深度流形上。manifold flow 的精神是"在流形内部移动"，本实现却从流形外的随机点往里跳。

### 对照表
| | 文献 manifold flow | 文献 flow matching / rectified flow | 当前实现 |
|---|---|---|---|
| 关注点 | 数据在低维流形上，先固定流形再流 | 怎么训速度场、怎么拉直路径 | 名义 manifold，实则两者都残缺 |
| latent/流形 | **冻结、预训练**、稳定 | 不规定 | **从零联合训练**、不稳定 |
| 起点 | 流形内的点 | 任意（rectified 建议相关） | **噪声/零**（流形外） |
| 耦合 | — | data coupling 才拉得直 | **独立噪声耦合** |

**结论**：类名叫 "manifold flow"，但既没满足 manifold flow 的前提（冻结稳定流形），也没用好 flow matching 的最佳实践（data coupling），是"两个范式各取会失效的一半拼起来"——这正是第三节第 1、3、7 条问题的根源。自洽的两条路线二选一：
- **走 manifold flow 正统**：先用重建预训练好深度自编码器并**冻结**得到稳定流形，再在流形 latent 里做 flow。
- **走 flow matching / rectified 正统（更贴合"精修"，推荐）**：直接 coarse→GT 的 data coupling，锚定 coarse 做残差修正，甚至可先不进 latent、在像素/特征空间流。

---

## 三、当前架构不合理的地方（按影响排序）

### 1. 流向搞反了：noise→depth，而不是 coarse→GT（首要病根）
`_rollout_latent`（`manifold_flow.py:204-215`）从 `torch.zeros_like` / 小噪声起步，6 步欧拉积分，coarse 预测只当 conditioning。最终 `refined_depth = target_decoder(refined_latent)` 是**从零重新生成**的，恰恰丢掉了实验证明最靠谱的东西（coarse）。文献做法应是 coarse↔GT 的 data coupling，路径近直、能继承 coarse 质量。

### 2. 输出完全替换 coarse（`blend_alpha=0` 默认）
`manifold_flow.py:222-223`：默认不融合，coarse 对输出零贡献。精修理应是**残差 / 锚定**在 coarse 上（refined = coarse + Δ），而非另起炉灶。

### 3. latent 空间从零联合训练、是移动靶、且几乎不受约束
`condition_encoder` / `target_encoder` / `target_decoder` 都随主任务从头训（`:148-150`）；flow-matching 的终点 `target_latent` 每步都在变（非平稳回归目标），且 `target_latent.detach()` 只用于跨帧、当步匹配无 stop-grad；重建约束 `L_rec` 权重仅 **0.1**。对比 DepthFM 用的是**冻结的预训练 VAE**。自造 latent 又不锁死，等于在流沙上做 flow matching。

### 4. condition 与 target 用两套独立编码器
`condition_encoder(coarse)` 与 `target_encoder(GT)` 产出**不同 latent 空间**，但 flow 在其一里积分、再用 `target_decoder` 解码。conditioning 的几何和目标 latent 的几何对不齐。

### 5. 逐图归一化摧毁绝对尺度
`_normalize_feature_statistics`（`:45-48`，在 `:190` 被用于条件特征）逐图减均值除标准差——**正好抹掉度量深度需要的全局尺度 / 偏移**，与观测到的 scale-invariant 变差方向一致。

### 6. 训练/推理分布不一致（exposure bias）
训练时 `target_decoder` 主要被 `L_rec` 监督去解码 `target_latent`（GT 编码器输出）；推理时没有 GT，`refined_latent` 是从噪声 rollout 出来的——**解码器推理时看到的 latent 分布，和它训练时主要学会解码的分布不是一个**，且无 self-conditioning 弥合。

### 7. 独立耦合 + 少步积分
`L_flow` 用 (noise, target) 独立耦合做 flow matching（`:244-254`），路径弯曲，却只用 6 步欧拉。文献靠 data coupling / reflow 把路径拉直才敢用少步。

---

## 四、根因归纳

第 1、2 条是主因，其余是放大器。当前设计把一个**已经不错的判别式 coarse 预测**，硬塞进一个**从噪声重生 + 自造未冻结 latent + 逐图去尺度**的生成式管线，还默认不锚定 coarse。文献里能 work 的 flow-for-depth 恰恰相反：data-to-data、锚定 coarse、冻结 latent。所以 FLOW 在监控指标上打不过 BASE 是架构决定的，不是训练不足。

---

## 五、改进方向（据文献）

1. **改流向**：flow 从 coarse（或 image 特征）流向 GT 的 data coupling，而非 noise→depth。
2. **锚定 coarse**：输出改为残差精修 `refined = coarse + Δ`，或 `blend_alpha > 0`。
3. **冻结 latent**：latent 编解码器先用重建预训练再冻结（或直接用预训练 VAE），别和主任务从零联合训。
4. **去掉逐图归一化**：保留全局尺度 / 偏移信息。
5. **统一 latent 空间**：condition 与 target 共用同一编码器 / latent 几何。
6. **拉直路径**：data-dependent coupling / reflow，才配得上少步欧拉积分。

---

## 参考文献

- DepthFM: Fast Monocular Depth Estimation with Flow Matching — https://arxiv.org/html/2403.13788v2
- Conditional Rectified-flow-based TryOn Refiner (ICCV 2025) — https://iccv.thecvf.com/virtual/2025/poster/1085
- Bridging Discriminative and Generative Image Restoration via Rectified Flow — https://arxiv.org/abs/2604.19680
- Improving the Training of Rectified Flows — https://arxiv.org/html/2405.20320
