# 方案：分离双解码器 + 冻结主干，只训 flow 残差（loss 对应改法）2026-08-03

> 承接 `BUG_residual_vs_reconstruction_conflict_20260803.md`（残差 vs 绝对重建的语义冲突）。
> 目标：既解决冲突，又复用已训好的粗深度主干，只训练 flow 残差部分。
> 本文只做设计，不改代码。相关：[[reB-scaleB-fix-training-0803]]、`manifold反思.md`、`FIX_DIRECTIONS_20260803.md`。

---

## 一、核心思路

1. **分离解码器**：flow 内部把"重建 GT 用的解码器"和"输出残差 Δ 用的解码器"拆成两个独立模块，消除一处解码器承担互斥语义的冲突。
2. **冻结主干**：加载之前训好的主干权重（Mamba encoder + UNet decoder + pred），冻结，不再更新——它已能产出好的 coarse，无需重训（也回应"每次 scratch 重训主干很浪费"的问题）。
3. **只训 flow 残差**：可训练参数只剩 flow 模块（condition_encoder / target_encoder / target_decoder / velocity_field），学"在稳定 coarse 上加多少 Δ"。

---

## 二、当前模块边界（已核对代码）

主干（`model/S2DepthNet.py`，产出 coarse_prediction）：
- `self.encoder`（Mamba 3D，L61）、`self.resblocks`（L99）、`self.decoders`（L107）、`self.pred`（L114）。
- L224 `coarse_prediction = self.forward_decoder(encoded_xs)`。

flow（`self.manifold_flow`，L89 = `ConditionedDepthManifoldFlow`），内部：
- `condition_encoder`：coarse → z0（残差桥里 share_encoder=True 时 target_encoder 复用它）。
- `target_encoder`：GT → z1。
- `target_decoder`：**当前一处三用**（解码 refined_latent 出 Δ、解码 E(GT) 做 L_rec、解码 target_reconstruction）。← 冲突根源。
- `velocity_field`：速度场。

---

## 三、方案 A：解码器分离（推荐，彻底解决冲突）

### 结构改动（设计，不落地）
在 `ConditionedDepthManifoldFlow.__init__` 拆成两个解码器：
- **`recon_decoder`**：只服务重建/流形定义，语义 = 绝对深度。`D_rec(E(GT)) ≈ GT`。
- **`residual_decoder`**：只服务 rollout 输出，语义 = 残差增量。`D_res(refined_latent) = Δ`，`refined = coarse + Δ`。

两个解码器**参数独立、不共享**，各自的收敛目标不再矛盾：
- `D_rec(E(GT)) → GT`（L_rec 满足）。
- `D_res(E(GT)) → GT − coarse`（refined_metric 满足）。
- 二者是不同函数，可同时成立。冲突消除。

### 隐空间由谁定义
- 流形（latent 几何）由 **encoder + recon_decoder** 通过重建预训练定义并稳定（`E`/`D_rec` 是一对自编码器）。
- `residual_decoder` 只是"从 latent 读出增量"的读出头，不参与定义流形，可与 flow 一起训。

---

## 四、冻结策略

### 冻结（requires_grad=False）
- **主干全部**：`encoder`(Mamba) + `resblocks` + `decoders` + `pred`。coarse 固定。
- **可选**：`condition_encoder`/`target_encoder` + `recon_decoder`（若走"先重建预训练自编码器再冻结"的正统 manifold flow，见方案 B）。

### 训练（requires_grad=True）
- 最小版：只训 `velocity_field` + `residual_decoder`。
- 中版：加上 `condition_encoder`/`target_encoder`（让 latent 适配 flow，但主干 coarse 不变）。

### 主干权重从哪来
- 优先加载 0320 充分收敛主干（若结构兼容），否则加载残差桥已有 run 的主干部分。**需先核对 state_dict 键名匹配**（S2DepthNet 主干模块名 vs ckpt），这是落地前必做的一步。

---

## 五、loss 对应改法（关键）

冻结主干后，凡是"监督/依赖 coarse 质量"的项都应关掉或降权，只保留驱动 flow 残差的项。

| loss 项 | 当前(scaleB) | 改后 | 理由 |
|---|---|---|---|
| **L_si**（对 refined） | w=1, n_lambda=0.25 | **保留**，w=1 | 主损失，监督最终 refined 深度 |
| **L_grad**（对 refined） | w=0.25 | 保留 w=0.25 | 边缘结构 |
| **L_coarse**（对 coarse 的 SI） | coarse_weight=1 | **关掉 → 0** | coarse 主干已冻结，再监督它无梯度可用、纯浪费；且它本是"稳第一阶段"的项，主干固定后无意义 |
| **L_rec**（recon_decoder 重建 GT） | w=0.3（冲突） | 方案A：**保留 w>0**，作用于**独立的 recon_decoder**，不再冲突；方案B：若 AE 已预训练冻结则 **关掉**（已由预训练保证） | 定义/稳定流形 |
| **L_refined_metric**（对 refined 线性L1） | w=0.4 | **保留并可提到 0.4~0.6** | 绝对尺度直接监督，现在是主要"教 flow 学 Δ"的信号之一 |
| **L_flow**（速度场 smooth_l1） | w=0.5 | **保留 w=0.5** | flow matching 核心，学 z0→z1 方向 |
| **L_geo**（跨帧 Δ 一致） | w=0.05 | 保留 | 时序一致 |
| **L_delta_reg**（Δ² 正则） | w=0.01 | 保留或略降 | 抑制 Δ 外溢 |

### 要点
1. **L_coarse 必须关（→0）**：主干冻结后 coarse 是常量输入，对其求 SI loss 不产生任何可用梯度（coarse 参数不更新），保留只是白算。这是冻结主干后**最需要改的一项**。
2. **总损失重心转移到 refined 侧**：`L_si(refined) + L_refined_metric + L_flow` 成为三大主力，共同教 `residual_decoder`+`velocity_field` 学"在固定 coarse 上加正确的 Δ"。
3. **data_coupling 保持**：`z0=E(coarse)`、`z1=E(GT)`、`target_velocity=z1−z0` 不变。coarse 虽冻结，其 latent 仍是 flow 起点。
4. **L_rec 的解码器指向**：方案A 里 L_rec 必须改成用 `recon_decoder`（`D_rec(E(GT))≈GT`），而 rollout 输出改用 `residual_decoder`。这是分离的核心代码点。

### 冻结主干后的总损失（方案A，示意）
```
L = L_si(refined) + 0.25·L_grad(refined)
    + 0.0·L_coarse                      # 关
    + w_rec·|D_rec(E(GT)) − GT|         # 独立 recon_decoder，不冲突
    + w_metric·|refined − GT|_lin       # refined = coarse + D_res(refined_latent)
    + w_flow·smooth_l1(v_pred, z1−z0)
    + w_geo·L_geo + w_delta·L_delta_reg
```

---

## 六、方案 B（更正统，可选）：预训练+冻结自编码器，残差解码器另训

- 先用重建损失单独预训练 `encoder+recon_decoder`（GT→latent→GT），存 ckpt。
- 主训练时冻结 `encoder(Mamba主干)`、`condition/target_encoder`、`recon_decoder`；只训 `velocity_field`+`residual_decoder`。
- 此时 latent 流形完全固定（DepthFM 式），flow 在稳定流形上学直线路径，`L_rec` 可关（已由预训练保证）。
- 收益最干净，但要多一个预训练阶段 + 两段脚本。

---

## 七、落地前必做的核对（下一步，仍不改代码前先查）
1. **主干 state_dict 键名**：确认 0320 或残差桥 ckpt 的主干键能对上当前 S2DepthNet（`encoder.*`/`resblocks.*`/`decoders.*`/`pred.*`），决定能否直接 load 冻结。
2. **residual/recon 解码器拆分点**：`ConditionedDepthManifoldFlow.forward` 里 L285（rollout 解码）与 L320（重建解码）分别改指向新模块。
3. **optimizer 只收可训练参数**：ablation 脚本 L266 `Adam(model.parameters())` 需改成 `filter(lambda p: p.requires_grad, model.parameters())`，否则冻结无效（冻结参数进 Adam 仍会因 weight_decay/动量被动更新）。← 冻结方案的常见坑，必改。
4. **lr 可调低**：只训 flow 小模块，lr 可用 5e-5 甚至更低，避免之前 scratch+1e-4 的尖峰。

---

## 八、预期与判据
- **消除冲突**：L_rec 不再卡 floor，能真正收敛到低值（recon_decoder 干净重建）。
- **病灶A**：主干冻结 = coarse 稳定，若 flow 残差能持续降低 refined 的 val/abs_rel 到 ep10+ 仍改善，说明 flow 头在稳定流形上确实学到可泛化精修。
- **病灶B**：refined_metric 加重 + 主干固定，abs_rel 是否 < 1.900（224² 同口径对比 0320 的 0.597）。
- 规范评测：export→evaluation_DENSE --crop_ymax224 --clip1000 --reg5.7。
