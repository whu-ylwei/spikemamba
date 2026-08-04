# Manifold Flow 重构方案：架构设计 + 代码改法

> 生成日期：2026-07-09
> 关联：`manifold反思.md`（问题）、`manifold_cv架构综述.md`（文献选型）、`manifold改进方向.md`（纯 config 前置实验）
> 适配基线：**BiVMamba spike→depth**，coarse 主干已被实验证明可靠。
> 选型：采用 `manifold_cv架构综述.md` 第 7 节的**路线 B（桥式残差精修，I2SB/InDI/TryOn-Refiner 一脉）** 为主线，附路线 A（latent manifold flow 正统）的差异点。
> 前置：本方案属**改代码**级别。动手前应先跑完 `manifold改进方向.md` 第 2 节的纯 config 实验（`blend_alpha↑` + `encoder_feature_scale=0`）；只有确认纯 config 无法让 FLOW ≥ BASE，才投入本方案。

---

## 第一部分：修改后的架构

### 1.1 一句话概括
把 flow 头从"**从噪声重生一张深度图**"改成"**在 coarse 深度与 GT 之间建一座 data-to-data 的桥，flow 只学 coarse→GT 的残差修正**"。coarse 是起点也是锚，flow 至多做增量，最坏退化为 coarse（下界安全）。

### 1.2 数据流对比

**改造前（现状）**：
```
coarse_depth ──condition_encoder──► condition_latent ──┐
                                                        │ (仅作条件)
noise/zeros ──► latent_state ──rollout(6步Euler)──► refined_latent ──target_decoder──► refined_depth
GT_depth ──target_encoder──► target_latent ──(独立噪声耦合)──► L_flow
输出 = refined_depth（blend_alpha=0 时完全替换 coarse）
```
问题：起点在流形外（噪声）、独立耦合、输出不锚定 coarse、latent 联合从零训（见 `manifold反思.md`）。

**改造后（路线 B）**：
```
                    ┌─────────────── 共享深度编码器 E ───────────────┐
coarse_depth ──E──► z_coarse ─┐                                      │
GT_depth     ──E──► z_gt    ──┤                                      │
                              │                                      │
  起点 z0 = z_coarse ─────────┤                                      │
  终点 z1 = z_gt      ────────┤  沿直线 z_t=(1-t)z0+t·z1              │
                              ▼                                      │
       velocity_field(z_t, cond=z_coarse, t) ──回归──► (z1 - z0)  [L_flow, data coupling]
                              │                                      │
  推理: z0=z_coarse, 积分N步 ─► z_refined ──D──► delta_or_depth ─────┘
输出 = coarse_depth + Δ    （残差锚定；Δ 由 flow 产生）
```

### 1.3 五处关键变化（对齐文献标准动作）

| # | 变化 | 从 → 到 | 文献依据 |
|---|---|---|---|
| C1 | flow 起点 | 噪声/零 → **z_coarse** | I2SB/InDI：从数据出发（综述脉络4） |
| C2 | flow 耦合 | 独立(noise,z_gt) → **data(z_coarse,z_gt)**，目标速度 `z_gt-z_coarse` | rectified data coupling（脉络3） |
| C3 | 输出形态 | 替换/blend → **coarse + Δ 残差** | Residual Diffusion Bridge / TryOn-Refiner（脉络4） |
| C4 | 编码器 | condition/target 两套 → **共享一个 E** | 起终点须在同一流形几何（脉络1/3） |
| C5 | 尺度 | 逐图归一化 → **移除**（或 encoder_feature_scale=0 绕过） | 度量深度需保尺度（反思第5条） |

### 1.4 与路线 A 的差异（若要"流形"正统）
路线 A 在 C1–C5 之外，额外要求**分两阶段训练**：阶段一只用重建损失把共享编码器 E + 解码器 D 训成稳定 autoencoder 并**冻结**（M-flow "先固定流形"），阶段二再训 velocity_field。路线 B 不冻结、端到端联合训（但因残差锚定，联合训的风险已大幅降低）。**建议先做路线 B**，若 latent 仍不稳再升级到 A 的两阶段。

### 1.5 接口契约（必须保持，否则 S2DepthNet 会崩）
`ConditionedDepthManifoldFlow.forward(...)` 必须仍返回 4 元组
`(refined_depth, auxiliary_losses:dict, auxiliary_outputs:dict, next_latent_state)`，
且 `auxiliary_outputs` 仍含 `coarse_prediction` 与 `coarse_weight`（`S2DepthNet.py:241-246` 与 driver `run_sequence` `:104-110` 依赖它们）。本方案不改这两个消费端，只改 flow 内部。

---

## 第二部分：代码层面怎么改

所有改动集中在 `model/manifold_flow.py` 的 `ConditionedDepthManifoldFlow` 类，**不动** `S2DepthNet.py` 与两个 driver 脚本（接口契约见 1.5）。下面按"改哪个方法、改成什么、为什么"逐条给出。行号基于当前文件。

### 改动 0：新增 config 开关（向后兼容，默认走新架构可关）
在 `__init__`（`:129-155`）末尾追加读取，便于 A/B 对照且不破坏旧 config：
```python
# 新增（放在 __init__ 内，loss_weights 之后）
self.residual_output   = bool(config.get("residual_output", True))   # C3: 残差锚定
self.data_coupling     = bool(config.get("data_coupling", True))     # C2: coarse→gt 耦合
self.share_encoder     = bool(config.get("share_encoder", True))     # C4: 共享编码器
self.start_from_coarse = bool(config.get("start_from_coarse", True)) # C1: 起点用 coarse
```
> 全设 False 即退回原行为，方便消融。

### 改动 1（C4）：共享深度编码器
现状 `:148-149` 两套独立编码器：
```python
self.condition_encoder = _DepthDownsampleEncoder(1, base_channels, latent_channels)
self.target_encoder    = _DepthDownsampleEncoder(1, base_channels, latent_channels)
```
改为可共享：
```python
self.condition_encoder = _DepthDownsampleEncoder(1, base_channels, latent_channels)
if self.share_encoder:
    self.target_encoder = self.condition_encoder      # 同一实例，z_coarse 与 z_gt 同几何
else:
    self.target_encoder = _DepthDownsampleEncoder(1, base_channels, latent_channels)
```
> 作用：C2 的直线插值 `z_coarse→z_gt` 要求两端在同一 latent 空间，否则"直线"没有几何意义。

### 改动 2（C5）：移除逐图归一化
现状 `_fuse_encoder_features`（`:190`）对条件特征做 `_normalize_feature_statistics`，逐图去均值除标准差、破坏绝对尺度。
- **最省改法**：config 设 `encoder_feature_scale: 0.0`，`:158` 直接 return，跳过整条融合（纯 config，已在 `manifold改进方向.md`）。
- **保留特征融合但去归一化**：删掉 `:190` 那一行 `aligned_feature = _normalize_feature_statistics(aligned_feature)`，仅保留通道对齐 `_match_feature_channels`。
> 二选一。若想让 flow 仍吃编码器多尺度特征，用后者；若只靠 coarse latent 做条件（更干净），用前者。

### 改动 3（C1+C3）：重写 `_rollout_latent` 起点与残差解码
现状 `_rollout_latent`（`:204-215`）从零/噪声起步、返回 `refined_latent`，随后 `:221` 无条件 `target_decoder`。
改成起点可选 coarse，并把"残差 vs 全量"下放到 forward：
```python
def _rollout_latent(self, condition_latent):
    if self.start_from_coarse:                 # C1: 起点=coarse latent（在流形上）
        latent_state = condition_latent
    elif self.rollout_noise_std > 0:
        latent_state = torch.randn_like(condition_latent) * self.rollout_noise_std
    else:
        latent_state = torch.zeros_like(condition_latent)

    step_size = 1.0 / float(self.flow_steps)
    for step in range(self.flow_steps):
        # data coupling 下速度已是 coarse→gt 方向，t 从 0 累加
        time_value = step * step_size if self.start_from_coarse else (step + 0.5) * step_size
        velocity = self.velocity_field(latent_state, condition_latent, time_value)
        latent_state = latent_state + velocity * step_size
    return latent_state
```
> 起点为 coarse 时，flow 从流形上出发，6 步几乎全用于精修（综述脉络5的两阶段论据）。

### 改动 4（C2+C3）：重写 `forward` 的耦合与输出
现状 `forward`（`:217-280`）：`:220-221` 解码得 refined_depth，`:222-223` blend；`:244-254` 用独立噪声耦合算 L_flow。分三处改。

**4a｜输出改残差锚定**（替换 `:220-223`）：
```python
refined_latent = self._rollout_latent(condition_latent)
decoded = self.target_decoder(refined_latent)
if self.residual_output:
    refined_depth = coarse_depth + decoded          # C3: coarse + Δ，Δ=decoded
else:
    refined_depth = decoded
    if self.blend_alpha > 0.0:                       # 保留旧 blend 兼容
        refined_depth = self.blend_alpha * coarse_depth + (1.0 - self.blend_alpha) * refined_depth
```
> 残差模式下 decoder 学的是"深度增量"。初始化时 decoder 输出接近 0 即退化为 coarse，天然下界安全。可选：给 `target_decoder.head` 的权重做小初始化以强化这一点（改 `_DepthUpsampleDecoder`，非必需）。

**4b｜L_flow 改 data coupling**（替换 `:244-254`）：
```python
if self.data_coupling:
    z0 = condition_latent                            # 起点=coarse latent
else:
    z0 = torch.randn_like(target_latent)             # 旧行为：独立噪声
time_value = torch.rand(
    target_latent.shape[0], 1, 1, 1,
    device=target_latent.device, dtype=target_latent.dtype)
latent_state   = (1.0 - time_value) * z0 + time_value * target_latent   # 直线插值
target_velocity = target_latent - z0                 # C2: coarse→gt 方向
predicted_velocity = self.velocity_field(latent_state, condition_latent, time_value)
if self.loss_weights["flow"] > 0.0:
    auxiliary_losses["L_flow"] = self.loss_weights["flow"] * F.mse_loss(
        predicted_velocity, target_velocity)
```
> 关键：`z0` 与 `target_latent` 现在**都来自共享编码器 E**（改动1），路径 `z_gt - z_coarse` 是同一空间里的直线，短且不易交叉，少步欧拉即准。

**4c｜残差模式下的重建约束**（`:236-242` 附近，可选强化）：
残差模式下 `target_decoder` 语义变成"解码增量"，而 `L_rec`（`:236` 用 `target_decoder(target_latent)` 重建 GT）语义与之冲突。二选一：
- 简单：残差模式下把 `reconstruction_weight` 设 0（config），关掉 L_rec；
- 严格：残差模式下 L_rec 改为约束 `coarse + target_decoder(z_gt) ≈ GT`，与主输出同构。
> 推荐先用简单版跑通，再视 latent 稳定性决定是否加严格版。

### 改动 5：`_ConditionalVelocityField` 无需改结构
它 `forward(latent_state, condition_latent, time_value)` 的输入拼接维度是 `latent_channels*2 + 1`（`:94`），改造后 `latent_state` 与 `condition_latent` 仍都是 `latent_channels` 通道，**接口不变**，不用动。这是本方案能低风险落地的原因之一。

---

## 第三部分：训练与验证

### 3.1 路线 B（推荐先做，端到端联合训）
无需改 driver。改完 `manifold_flow.py` + 相应 config 后直接：
```bash
cd /root/shared-nvme/spikemamba
python3 scripts/ablation_manifold_flow_20260708.py --flow on --epochs 40 --out runs/ablation/flow_reB
python3 scripts/eval_test_manifold_flow_20260708.py --flow on --ckpt runs/ablation/flow_reB/model_best.pth.tar
```
对照 `runs/ablation/base/`（BASE best val_loss 0.01214 @ep15），门槛：**FLOW ≥ BASE**。

对应 config（`model.manifold_flow`）建议：
```jsonc
"residual_output": true, "data_coupling": true, "share_encoder": true, "start_from_coarse": true,
"encoder_feature_scale": 0.0,        // 或删归一化行后保留 1.0
"coarse_weight": 1.0, "reconstruction_weight": 0.0,   // 残差模式先关 L_rec
"flow_weight": 0.1, "geometry_weight": 0.05, "flow_steps": 6
```

### 3.2 路线 A（若 B 下 latent 仍不稳，升级为两阶段）
需在 driver 之外加一段预训练（改代码/加脚本）：
1. 阶段一：仅用 GT 深度，最小化 `‖D(E(GT)) − GT‖`，训练共享 E + D；存权重。
2. 阶段二：加载并 `requires_grad_(False)` 冻结 E、D，只训 `velocity_field`。
> 这一步才真正满足 M-flow "先固定流形"，但成本高，非首选。

### 3.3 分步验证（隔离每个变化的贡献）
用改动 0 的开关做消融，逐个打开，各跑一轮与 BASE 比：
| 实验 | 开关组合 | 验证问题 |
|---|---|---|
| E0 | 全 False（+旧 config） | 复现现状 FLOW（回归基线） |
| E1 | start_from_coarse=True | 只换起点，够不够 |
| E2 | E1 + data_coupling=True | 加 data 耦合的增益 |
| E3 | E2 + residual_output=True | 残差锚定的增益（预期这里 ≥ BASE） |
| E4 | E3 + share_encoder=True | 共享编码器是否再涨 |

---

## 第四部分：风险与回退

- **接口契约**：改动全在 flow 内部，`forward` 4 元组签名与 `auxiliary_outputs` 的 `coarse_prediction`/`coarse_weight` 保持不变（1.5），S2DepthNet 与 driver 无需改。
- **回退**：改动 0 的开关全设 False + 旧 config 即完全复原；建议改前 `git` 存一份或复制 `manifold_flow.py` 备份。
- **checkpoint 兼容**：共享编码器（改动1）会改变参数结构，旧 flow checkpoint 无法直接加载——但 BASE/coarse 主干权重不受影响，flow 头本就要重训。
- **数值**：残差模式下 `refined = coarse + Δ`，若 Δ 初期过大可能盖过 coarse；如遇不稳，减小 `flow_weight` 或对 decoder head 做小初始化。
- **L_rec 语义冲突**：见改动 4c，残差模式务必先关或改写 L_rec，否则两个目标打架。

---

## 附：改动清单速查（全部在 `model/manifold_flow.py`）
| 改动 | 位置 | 类型 |
|---|---|---|
| 0 新增开关 | `__init__` `:141` 后 | 加 4 行 config 读取 |
| 1 共享编码器 | `:148-149` | 改 target_encoder 赋值 |
| 2 去逐图归一化 | `:190`（或 config `encoder_feature_scale=0`） | 删 1 行 / 纯 config |
| 3 起点+rollout | `_rollout_latent` `:204-215` | 重写起点分支 |
| 4a 残差输出 | `forward` `:220-223` | 改输出组合 |
| 4b data coupling | `forward` `:244-254` | 改耦合与目标速度 |
| 4c L_rec 兼容 | `forward` `:236-242`/config | 关或改写 L_rec |
| 5 velocity_field | — | 不改 |

---

## 第五部分：未来改动登记册（Change Registry）

按优先级 P0（核心，必做）→ P3（探索性）分层记录**所有已识别的候选改动**，含依赖、代码位置、预期收益、风险、验证方式。每条给一个稳定编号（CR-xx）便于后续追踪与勾选。

### 图例
- **状态**：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成 · `[!]` 已否决/搁置
- **层级**：config（纯配置）· flow（改 `model/manifold_flow.py`）· net（改 `model/S2DepthNet.py`）· driver（改训练/评估脚本）· loss（改 `model/loss.py`）· data（改 data_loader）
- **依赖**：必须先完成的 CR 编号

---

### P0 — 核心重构（本方案第二部分已详述）

| ID | 状态 | 改动 | 层级 | 依赖 | 收益 / 风险 |
|----|------|------|------|------|-------------|
| CR-01 | [ ] | 新增 4 个行为开关 | flow(`__init__`) | — | 收益：可消融、可回退；风险：无 |
| CR-02 | [ ] | 共享深度编码器 E | flow(`:148`) | CR-01 | 收益：起终点同几何；风险：改参数结构，旧 flow ckpt 失效 |
| CR-03 | [ ] | 去逐图归一化 / `encoder_feature_scale=0` | flow(`:190`) 或 config | — | 收益：保绝对尺度；风险：丢多尺度条件信息 |
| CR-04 | [ ] | 起点 noise→coarse | flow(`_rollout_latent`) | CR-01 | 收益：从流形内出发；风险：需配合 CR-05 |
| CR-05 | [ ] | L_flow 改 data coupling | flow(`forward:244`) | CR-02,CR-04 | 收益：路径变直、少步准；风险：需与起点一致 |
| CR-06 | [ ] | 残差输出 coarse+Δ | flow(`forward:220`) | CR-01 | 收益：下界安全；风险：与 L_rec 语义冲突(见 CR-07) |
| CR-07 | [ ] | 残差模式下关/改写 L_rec | flow(`forward:236`)/config | CR-06 | 收益：消除目标冲突；风险：latent 约束变弱 |

---

### P1 — 稳定性与训练机制（核心跑通后立即考虑）

| ID | 状态 | 改动 | 层级 | 依赖 | 说明 |
|----|------|------|------|------|------|
| CR-10 | [ ] | 深度 AE 两阶段预训练+冻结（路线A） | driver+flow | CR-02 | M-flow "先固定流形" 正统；阶段一重建预训练 E/D，阶段二冻结只训 velocity_field |
| CR-11 | [ ] | decoder head 小初始化（残差稳） | flow(`_DepthUpsampleDecoder.head`) | CR-06 | 让 Δ 初期≈0，强化下界安全；`head` 权重 ×0.01 或 zero-init |
| CR-12 | [ ] | flow 头 warmup（先训主干再开 flow） | driver | — | 前 K epoch `flow_weight=0` 只训 coarse，再线性升权；避免早期 coarse 太差污染 flow |
| CR-13 | [ ] | 梯度隔离：coarse 分支对 flow stop-grad | flow(`forward`) | — | `condition_encoder(coarse.detach())` 可选，防 flow 梯度扰动主干；需实验取舍 |
| CR-14 | [ ] | L_geo 跨帧一致性重估 | flow(`forward:263`) | CR-06 | 残差模式下 delta 定义变了，跨帧几何项需同步改写或关闭 |
| CR-15 | [ ] | seq_len>2 的状态传递审查 | flow/net | — | `prev_latent_state` 在残差+共享编码器下的语义需重新核对 |

---

### P2 — 采样与推理效率

| ID | 状态 | 改动 | 层级 | 依赖 | 说明 |
|----|------|------|------|------|------|
| CR-20 | [ ] | 减少 flow_steps（6→2~4） | config | CR-05 | data coupling 路径直后可减步提速；先验证精度不掉 |
| CR-21 | [ ] | reflow 拉直（1-2 步可积） | driver+flow | CR-05,CR-20 | 用训好模型重生成 (z_coarse, 预测) 配对再训一轮 |
| CR-22 | [ ] | 高阶积分器（Heun/RK2 代替 Euler） | flow(`_rollout_latent`) | — | 同步数更准；每步两次 velocity 前向，成本翻倍 |
| CR-23 | [ ] | 推理步数与训练解耦并做步数消融 | driver | CR-20 | 记录 steps∈{1,2,4,6,8} 的精度-耗时曲线 |
| CR-24 | [ ] | 蒸馏到单步（MeanFlow 风格） | flow+driver | CR-21 | 探索性；参考 Riemannian MeanFlow |

---

### P3 — 表达力与条件化（探索性，收益不确定）

| ID | 状态 | 改动 | 层级 | 依赖 | 说明 |
|----|------|------|------|------|------|
| CR-30 | [ ] | 时间嵌入替代 time_map 平铺 | flow(`_ConditionalVelocityField`) | — | 当前把标量 t 平铺成 map 拼接；改 sinusoidal/MLP 嵌入更表达 |
| CR-31 | [ ] | 编码器多尺度特征作条件（保尺度版） | flow(`_fuse_encoder_features`) | CR-03 | 去归一化后重新引入 spike 编码器特征，FiLM/加法注入 |
| CR-32 | [ ] | velocity_field 加深/加宽 | config(`flow_resblocks`,`flow_hidden_channels`) | — | 纯 config 可试；先确认瓶颈在容量再加 |
| CR-33 | [ ] | 不确定度输出（预测 Δ 的方差） | flow(`head`)+loss | CR-06 | 深度估计常配 aleatoric 不确定度；需改 loss |
| CR-34 | [ ] | 在更高分辨率 latent 做 flow | flow(编码器下采样倍数) | CR-02 | 当前 3 次 stride2 下采样，latent 很粗；权衡显存 |
| CR-35 | [ ] | 条件注入 spike 时序特征（非仅coarse） | flow+net | CR-31 | 让 flow 看到原始 spike 上下文而非只有 coarse |

---

### 记录规范（后续维护本登记册时遵循）
1. **动手即改状态**：开始一条改动时把 `[ ]`→`[~]`，完成→`[x]`，附一句结果（如"E3 达到 val 0.0118 < BASE"）。
2. **新想法追加**：新增改动分配下一个 CR 编号，勿复用；注明来源（论文/实验现象）。
3. **否决要留痕**：不做的改动标 `[!]` 并写一句原因，避免以后重复讨论。
4. **每条实验完更新**：在对应 CR 行末尾追加"实验结果"链接或一句结论，与 `runs/ablation/` 目录对应。
5. **依赖先行**：动某 CR 前先确认其"依赖"列的 CR 已完成。

---

## 第六部分：一次完整的推进路线（把上面串起来）

建议按此顺序推进，每步都以"是否 ≥ BASE"为门禁：

1. **纯 config 探底**（无代码）：`manifold改进方向.md` 第 2 节 → 确认是否已足够。
2. **P0 核心重构**：CR-01→CR-02→CR-03→CR-04→CR-05→CR-06→CR-07，用 E0–E4 消融逐个验证（第三部分 3.3）。
3. **达标后稳定化**：视训练曲线取 CR-11/CR-12/CR-13。
4. **若 latent 仍不稳**：升级 CR-10 两阶段冻结（路线 A）。
5. **提速**：CR-20→CR-21，出步数-精度曲线（CR-23）。
6. **打磨/探索**：按需从 P3 挑选，每次只改一项并消融。

> 贯穿始终的纪律：**一次只改一个变量 + 与 BASE 同管线对比 + 更新本登记册**。避免多改动叠加导致无法归因（这正是当前 flow 头当初的教训）。

---

## 参考文献
- InDI: An Alternative to Denoising Diffusion for Image Restoration — https://arxiv.org/abs/2303.11435
- Image-to-Image Schrödinger Bridge (I2SB) — https://ar5iv.labs.arxiv.org/abs/2302.05872
- Residual Diffusion Bridge Model for Image Restoration — https://arxiv.org/html/2510.23116
- DepthFM: Fast Monocular Depth Estimation with Flow Matching — https://arxiv.org/html/2403.13788v2
- Conditional Rectified-flow-based TryOn Refiner (ICCV 2025) — https://iccv.thecvf.com/virtual/2025/poster/1085
