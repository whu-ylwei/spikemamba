# 残差桥深度模型优化方向（文献支撑版）2026-07-31

> 目标：缩小残差桥（route B, flow matching）与充分训练基线的 abs_rel 差距。
> 当前 224² CenterCrop 同口径成绩（centercrop224 stage1 ep3 best，**欠训**）：
> abs_rel **1.900** / δ<1.25 **0.592** / RMS_lin 213 / SILog 0.615。
> 参考基线（同数据集 DENSE，Town10 测试）：E2Depth abs_rel **0.220** / δ 0.724；
> EReFormer(DTR) abs_rel **0.172** / δ 0.747。
> 本文每条方向都标注了来源论文与可迁移性，非空想。

## 参考文献

| 简称 | 标题 | arXiv | 与本项目关系 |
|---|---|---|---|
| DepthFM | Fast Monocular Depth Estimation with Flow Matching | 2403.13788 | **最直接对标**：flow matching 深度，数据耦合、少步推理、辅助损失 |
| E2Depth | Learning Monocular Dense Depth from Events | 2010.08350 | DENSE 数据集原始基线，log-depth 归一化与损失的出处 |
| EReFormer/DTR | Event-based Monocular Dense Depth with Recurrent Transformers | 2212.02791 | DENSE SOTA，给出 Town10 同口径数字与架构改进 |
| DDVM | Surprising Effectiveness of Diffusion Models for Flow & Depth | 2306.01923 | 生成式深度的训练技巧（infilling、L1、step-unroll） |

---

## A. 立刻可做（config 级，零/低架构风险）

### A1. log-depth 解码量程 clip1000/α5.7（相对 E2Depth，非相对 0320）
- **诊断修正（见文末「口径诊断」）**：0320 与残差桥**训练/评测都用 clip1000/reg5.7**，两者量程完全一致——所以量程**不是** 0320 对比里 abs_rel 差 3× 的原因（那条差距来自训练充分度 A2 与架构）。
- **仍成立的点**：相对 E2Depth 原文（Dmax=80m/α=3.7）本项目全线用 clip1000/α5.7，`exp(5.7·(x-1))·1000` 在远景把 normalized 小误差指数放大 → 与「δ 尚可但 abs_rel 大」的现象一致。DENSE 是 CARLA 车载，有效深度多在 80m 内，1000m 量程浪费了大量 normalized 分辨率在几乎不出现的远景上。
- **动作**：统计训练 depth 的 p98 分位；若绝大多数 <80m，训练+评测统一改 **clip 80 / α 3.7**（E2Depth 口径）重训一版。注意这会改变归一化，**必须重训**，不能只换评测参数（换评测参数会造成 train/eval 解码不一致）。
- 判据：与 E2Depth/EReFormer 严格同口径（clip80/α3.7/224²）下重训后的 abs_rel。

### A2. 充分训练（当前数字虚高的首要原因）
- **现状**：centercrop224 stage1 **ep3 就 best，ep4-10 过拟合平台**，10+10ep 远不够。
- **文献依据**：E2Depth/EReFormer 均为充分训练的收敛模型；本架构历史需 ~80ep 才到 val0.0129。
- **动作**：按 40+40ep（含 lr MultiStepLR 衰减）从头训 centercrop224，配 `weight_decay`（现为 0）1e-4~1e-2、batch_size 提到 4（224² 显存允许）。这才是架构在 224² 口径的真实上限。

### A3. flow 损失改 L1（DDVM 明确结论）
- **现状**：`manifold_flow.py:324` L_flow 用 `F.mse_loss`（L2）。
- **文献依据**：DDVM 消融 NYU：**L1 vs L2 → REL 0.075 vs 0.085, δ1 0.944 vs 0.932**，L1 全面更好。
- **动作**：velocity 回归损失 L2→L1（或 smooth L1）。一行改动，低风险。

### A4. 提高线性空间直接监督权重
- **现状**：`refined_metric_weight=0.1`（`manifold_flow.py:300`，线性空间 L1）。主损失仍 scale-invariant 主导（si_n_lambda=0.5），绝对尺度弱监督。
- **文献依据**：DepthFM/DDVM 都对最终深度做直接回归监督；abs_rel 是绝对量，需要绝对监督。
- **动作**：refined_metric_weight 试 **0.3~0.5**。注意与 A1 联动——若 A1 修好量程，此权重效果更干净。

---

## B. 中等改动（需改模型/训练循环）

### B1. 数据依赖耦合的直线路径（DepthFM 头号结构性收益）
- **现状**：本项目已有 `data_coupling=true, start_from_coarse=true`（z0=coarse latent），方向与 DepthFM 一致 ✓。但 `rollout_noise_std=0.0`。
- **文献依据**：DepthFM 对起点 x0 加**小幅余弦调度高斯噪声**（仅 x0，不加到 condition），平滑基分布、经验有效；其数据耦合是相对 naïve-FM 的最大结构性提升。
- **动作**：给 z0/coarse latent 加小噪声（rollout_noise_std 小值起步，如 0.05），做消融。低成本。

### B2. 表面法向一致性辅助损失（DepthFM 式7）
- **文献依据**：DepthFM `L = L_FM + λ·L_SN`；SN loss = 从 x_t 一步 push 到 x̂_{t→1} 解码深度，估法向后与 GT 法向比 `‖n̂-n‖₁ + ‖1-n̂ᵀn‖₁`，Sobel 边缘置信掩码，**时间调度 t_s=0.2 后才加权**。消融 δ1 +0.6%（NYU）、+0.6%（DIODE），稳定小幅提升。
- **动作**：对 refined_depth 加基于深度梯度的法向一致性损失，权重小、后段时间步加权。中等实现量。

### B3. 少步 ODE 推理 + 集成（DepthFM）
- **现状**：flow_steps=10, heun。
- **文献依据**：DepthFM **1 NFE 即可**，视频用 2 步 + 5 集成。若本项目直线性足够，10 步是浪费；步数需求高恰恰说明耦合/直线性不足（回到 B1）。
- **动作**：评测时扫 flow_steps ∈ {1,2,4,10}，若少步不掉点则减少；掉点则说明路径不直，优先修 B1。附带用 ensemble（多次采样取中位）提稳。

### B4. 无效深度像素的 infilling + 掩码损失（DDVM 头号收益）
- **文献依据**：DDVM 对稀疏/空洞 GT 做 **NN 插值填补后再加噪，但只在有效像素反传**，是其最大单项增益。
- **动作**：查 DENSE depth GT 是否有 inf/nan 空洞（`_sanitize_depth` 已 mask，但 flow 目标 latent 由 `target_encoder(clean_target)` 编码，空洞被填 0 会污染 bridge 目标）。若有空洞，改成插值填补后编码、只在有效区反传。中等收益，取决于 GT 质量。

---

## C. 大改动（架构级，回报高但工程量大）

### C1. cross-attention 跳连融合替代 sum skip（EReFormer STF）
- **现状**：config `skip_type=sum`。
- **文献依据**：EReFormer STF（cross-attention skip fusion）相对 ADD **@10/20/30m 提升 4.1/5.1/3.6%**。
- **动作**：encoder-decoder 跳连从相加改 cross-attention。工程量大。

### C2. 带遗忘门的循环状态（EReFormer GRViT）
- **文献依据**：EReFormer 的 GRViT 用**可学习 update gate**，明确优于「注意力门」和「残差 carry」（残差 carry 更差，旧时序信息无法遗忘）。
- **现状**：残差桥的循环状态传递方式需核对（`prev_latent_state`）。若是纯残差 carry，引入门控可能有收益。
- **动作**：循环状态更新加 update gate。工程量大。

---

## 推荐执行顺序

1. **A2 + weight_decay + batch 提升，40+40 充分训练** — 诊断确认这是 0320 对比里差距的首要来源（口径已同），最高优先级。
2. **A1（clip80/α3.7 统一 E2Depth 口径）+ 重训** — 仅对「相对 E2Depth 论文」的对比有意义；对 0320 对比无影响（量程本就相同）。
3. **A3（flow 损失 L1）+ A4（refined_metric 权重↑）** — 零/低成本，随 A2 一起进 config。
4. **B1（起点加噪）、B3（少步扫描）** — 消融，判断路径直线性。
5. **B2（法向损失）、B4（infilling）** — 若前面到瓶颈再上。
6. **C1/C2 架构改动** — 长期方向，前面榨干后再考虑。

## 关键提醒（口径）
- 本项目 112²(resize) 与 224²(centercrop) 口径**不可互比**；与 E2Depth/EReFormer 比必须用 **224² CenterCrop + 相同 clip/α**，且注意它们报的是 346×260 全分辨率 + 80m clip + 10/20/30m 分段 avg-abs-err，与本项目当前评测协议仍有差异，直接比 abs_rel 数值前需先统一量程（见 A1）。

---

## 口径诊断（0320 MambaSSM vs 残差桥 224²，2026-07-31 逐项查证）

**结论：残差桥 centercrop224 与 0320 已严格同口径，此前担心的「口径偏离」不成立。** 逐项核对：

| 项 | 0320 (test_DENSE.py) | 残差桥 centercrop224 | 是否一致 |
|---|---|---|---|
| 输入映射 | `CenterCrop(224)`（`utils/data_augmentation.py:78` `i=round((260-224)/2)=18, j=round((346-224)/2)=61`，纯切片） | ablation `_center_crop_chw`（同为 `round((H-Ht)/2)`，i=18/j=61） | ✓ 数学等价 |
| 训练 clip_distance | config=**1000** | config=**1000** | ✓ |
| 训练 reg_factor | 数据集默认 **5.7** | 数据集默认 **5.7** | ✓ |
| 深度归一化 | `1+log(clip(d,0,1000)/1000)/5.7`（`SpikesDENSE_dataset.py:269-279`） | 同一数据集同一公式 | ✓ |
| 评测解码 | `evaluation_DENSE` `exp(5.7·(x-1))·1000` | 同脚本同参数（--clip1000 --reg5.7） | ✓ |
| 导出 npy | (1,224,224)，值域 0.004~1.018 | (1,224,224)，值域 -0.002~1.078 | ✓ 空间/值域一致 |
| crop_ymax | 260（默认，但数据高 224<260 不裁） | 224（=全高不裁） | ✓ 等效 |

- **唯一实质差异**：0320 训练用 `RandomCrop(224)` 增广（`train_parallel.py:196-197`），残差桥 centercrop 无随机增广——这是训练策略差异，非评测口径差异，不影响 abs_rel 可比性，但残差桥缺增广可能加剧过拟合（呼应 A2）。
- **推翻的旧假设**：此前 A1 猜「残差桥 clip1000 偏离 0320」——查实两者都是 clip1000/α5.7，量程完全相同。所以残差桥 224² abs_rel 1.900 vs 0320 的 0.597 的差距，**不来自口径/量程**，而来自：①训练严重不足（10ep vs 0320 充分训练，A2）②架构本身（flow matching 残差桥 vs 直接回归）③缺训练增广。
- **112² 数字的不可比仍然成立**：precision-resume ep56(2.300)、scratch ep3(2.408) 是 resize→112² 口径，与 224² 的 1.900 不可比（分辨率+视场不同），但它们彼此间 clip/α 也都是 1000/5.7。
