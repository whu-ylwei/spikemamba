# Manifold Flow 落地改进方案（不改代码，仅调 config）

> 生成日期：2026-07-09
> 前置：`manifold反思.md`（问题定位）、`ABLATION_RESULTS_manifold_flow_20260708.md`（实验证据）
> 适配基线：**BiVMamba spike→depth**。主干 coarse 深度已被实验证明可靠（BASE 优于 FLOW 与旧 ep70），改进须围绕并保护 coarse。
> **硬约束（已核对代码）**：本文档所有"立即可做"项都通过**编辑一个 config JSON** 实现，不动任何 `.py`。凡是 config 做不到、必须改代码的，单独归到第 5 节，不与可落地项混淆。

---

## 0. 落地入口与一条关键约束

驱动脚本 `scripts/ablation_manifold_flow_20260708.py`：
- 第 34 行 **硬编码** `CONFIG = "configs/train_s2d_spiketransformer_mambaflow_bidivim_seq2_40ep_20260423.json"`；
- `main()`（第 152-158 行）**没有 `--config` 参数**。

**推论**：不改代码时，唯一能改变 flow 行为的方式是**就地编辑上面这个 config 文件**（JSON 配置不是代码）。用新副本无效——脚本只读这个固定路径。因此下文操作 = 备份原 config → 编辑同名文件 → 用现有脚本跑。

---

## 1. 哪些旋钮"纯 config 就能生效"（附代码证据链）

下列键均由代码用 `config.get(...)` 读取，改 config 即改行为，无需碰 `.py`：

| config 键（`model.manifold_flow.*`） | 读取处 | 生效处 | 作用 |
|---|---|---|---|
| `blend_alpha` | `manifold_flow.py:138` | `:222-223` `refined = α·coarse + (1-α)·flowout` | **锚定 coarse**，α→1 时输出≈coarse |
| `coarse_weight` | `:142` → `aux_out["coarse_weight"]` (`:228`) | driver `run_sequence` `:105-107` 加 `cw·L_coarse` | 加强主干 coarse 监督 |
| `reconstruction_weight` | `:143` | `:238-242` (`L_rec`) | 约束 target AE 重建，稳定 latent |
| `flow_weight` | `:144` | `:251-254` (`L_flow`) | flow-matching 项权重 |
| `geometry_weight` | `:145` | `:263-273` (`L_geo`) | 跨帧几何一致 |
| `flow_steps` | `:135` | `:210-214` 欧拉积分步数 | 采样步数 |
| `rollout_noise_std` | `:139` | `:205-206` 起点噪声强度 | 起点噪声/零 |
| `encoder_feature_scale` | `:140` | `:158`（==0 直接 return）/`:202` | **=0 时整条 encoder 特征融合被跳过** |

driver 侧确认：aux losses 在 `run_sequence` 第 108-110 行**无条件累加**（权重已在 `manifold_flow.py` 内乘好），`L_coarse` 在第 104-107 行按 `coarse_weight` 加。**所以这些权重调 config 全部真实进 backward。**

### 关键发现：`encoder_feature_scale=0` 可无代码关闭"逐图归一化"
`manifold反思.md` 指出 `_normalize_feature_statistics`（`:45-48`）逐图去均值除标准差、破坏度量深度的绝对尺度。它只在 `_fuse_encoder_features` 内被调用（`:190`），而该函数开头（`:158`）：
```python
if self.encoder_feature_scale == 0.0 or encoder_features is None:
    return condition_latent      # 直接返回，跳过归一化与融合
```
**因此在 config 里设 `encoder_feature_scale: 0.0`，即可不改代码地绕过这条破坏尺度的路径。** 代价是 flow 不再吃编码器多尺度特征、只靠 coarse latent 做条件——对"精修"目标反而更干净。

---

## 2. 立即可跑的 config 改动集（S1，纯 config）

目标：把 flow 从"噪声重生、替换 coarse"扭成"强锚定 coarse 的轻修正 + 稳定 latent + 保尺度"，先拿到**不劣于 BASE** 的下界。

原始值 → 建议值（键路径 `model.manifold_flow.*`）：

```jsonc
{
  "enabled": true,               // 保持
  "blend_alpha": 0.0,            // → 0.85   强锚定 coarse：输出以 coarse 为主
  "coarse_weight": 0.5,         // → 1.0    coarse 与主 loss 同权，护住主干
  "reconstruction_weight": 0.1, // → 0.5    让 target AE 更快稳定（缓解移动靶）
  "flow_weight": 0.1,           // → 0.05   独立耦合路径本就弯，先降权少添乱
  "geometry_weight": 0.05,      // → 0.05   保持
  "flow_steps": 6,              // → 6      保持（未改耦合前不减步）
  "rollout_noise_std": 0.0,     // → 0.0    保持零起点（噪声起点更糟）
  "encoder_feature_scale": 1.0, // → 0.0    新增键，关闭逐图归一化路径，保尺度
  "latent_channels": 96, "base_channels": 48, "flow_hidden_channels": 128, "flow_resblocks": 2  // 结构参数，不动
}
```

> `encoder_feature_scale` 当前 config 里没有该键（代码默认 1.0）。**新增**这个键到 JSON 属于改配置，不是改代码。

**S1 的预期**：`blend_alpha=0.85` 使输出 ≈ coarse，flow 至多做 15% 修正，几乎不可能劣于 BASE；同时 `encoder_feature_scale=0` 恢复尺度。若此配置下 FLOW 仍打不过 BASE，说明连轻修正都有害，证据更强，指向必须做第 5 节的结构改动。

### 变体（消融式，各自单跑对比）
- **A｜纯锚定**：只改 `blend_alpha=0.85`、`coarse_weight=1.0`，其余保持 → 看锚定本身能否救回。
- **B｜A + 保尺度**：A 基础上加 `encoder_feature_scale=0.0` → 隔离"逐图归一化"的影响。
- **C｜B + 降 flow 项**：B 基础上 `flow_weight=0.05`、`reconstruction_weight=0.5` → 看 aux loss 配比。

逐个变体单独训，能干净地归因每个旋钮的贡献。

---

## 3. 操作步骤（不改任何 .py）

```bash
cd /root/shared-nvme/spikemamba
# 1) 备份原 config（唯一被脚本读取的文件）
cp configs/train_s2d_spiketransformer_mambaflow_bidivim_seq2_40ep_20260423.json \
   configs/train_s2d_spiketransformer_mambaflow_bidivim_seq2_40ep_20260423.json.bak

# 2) 按第 2 节编辑该 JSON 的 model.manifold_flow 段（手动或工具编辑，属改配置非改码）

# 3) 跑 FLOW 臂（与 BASE 同管线、同种子）。建议从稳定本地路径跑，
#    避免此前 FLOW 崩在 ep19 的 /home/dev/.local 挂载故障复现。
python3 scripts/ablation_manifold_flow_20260708.py --flow on --epochs 40 --out runs/ablation/flow_S1

# 4) 复用现有 test 评估，与 BASE 同口径对比（n=998）
python3 scripts/eval_test_manifold_flow_20260708.py --flow on \
   --ckpt runs/ablation/flow_S1/model_best.pth.tar
```

> BASE 结果已在 `runs/ablation/base/`（best val_loss 0.01214 @ep15）与其 test 数字，直接作对照，不必重跑。

---

## 4. 通过门槛与判读

对每个变体，用 `val_loss` 与 `scale_invariant_error` 对齐 BASE：

| 结果 | 解读 | 下一步 |
|---|---|---|
| FLOW ≥ BASE（更优或持平） | 锚定/保尺度救回了 flow | 逐步减小 `blend_alpha`，找 flow 真正增益的拐点 |
| FLOW 仅在 rms_linear/median 更好、SI 仍差 | flow 修了绝对尺度但伤了相对结构 | 保留 `encoder_feature_scale=0`，微调 flow 项权重 |
| 即便 `blend_alpha=0.85` 仍全面劣于 BASE | config 层已到极限，问题在结构 | 进入第 5 节结构改动 |

---

## 5. Config 做不到、必须改代码的改进（另列，勿混淆）

以下方向来自 `manifold反思.md`，但**无法通过 config 实现**，需改 `.py`，仅作后续规划记录：

1. **起点 noise→coarse**：`_rollout_latent`（`:204-215`）硬编码从 `zeros_like`/噪声起步。改成 `latent_state = condition_latent` 需改代码。
2. **`L_flow` 改 data coupling**：`forward`（`:244-254`）硬编码 `noise` 与 `target_latent` 独立耦合。改为 `(condition_latent, target_latent)` 耦合、目标速度 `target_latent - condition_latent` 需改代码。
3. **残差输出 `refined = coarse + Δ`**：当前只有 blend（线性混合），无残差分支，需改 `forward`。
4. **共享深度编码器**：`__init__`（`:148-149`）硬编码 `condition_encoder` 与 `target_encoder` 两套独立参数。共享需改代码。
5. **深度 AE 重建预训练 + 冻结**：需分阶段训练脚本与 `requires_grad=False`，driver 当前一次性联合训。
6. **reflow / 减步数**：依赖 2 的 data coupling 先落地。

优先级：**先把第 2 节 config 变体跑完拿到证据**；只有当 `blend_alpha=0.85` 仍打不过 BASE，才投入第 5 节的代码改动，且按 1→3→4→5 顺序（起点与残差最关键）。

---

## 6. 与 config 键的最终对应速查

| 目标 | config 键 | 改法 | 层级 |
|---|---|---|---|
| 锚定 coarse | `blend_alpha` | 0.0→0.85 | 纯 config ✅ |
| 护住主干 | `coarse_weight` | 0.5→1.0 | 纯 config ✅ |
| 稳定 latent | `reconstruction_weight` | 0.1→0.5 | 纯 config ✅ |
| 保绝对尺度 | `encoder_feature_scale` | 新增=0.0 | 纯 config ✅ |
| 少添乱 | `flow_weight` | 0.1→0.05 | 纯 config ✅ |
| 起点换 coarse | — | 无对应键 | 需改代码 ❌ |
| data coupling | — | 无对应键 | 需改代码 ❌ |
| 残差输出 | — | 无对应键 | 需改代码 ❌ |

---

## 参考文献
- DepthFM: Fast Monocular Depth Estimation with Flow Matching — https://arxiv.org/html/2403.13788v2
- Conditional Rectified-flow-based TryOn Refiner (ICCV 2025) — https://iccv.thecvf.com/virtual/2025/poster/1085
- Flow Straight and Fast: Rectified Flow (Liu et al., 2022) — https://arxiv.org/abs/2209.03003
