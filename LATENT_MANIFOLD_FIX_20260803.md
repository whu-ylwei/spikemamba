# Latent 流形修复方案：先钉牢流形，再训 flow（DepthFM 式两阶段）2026-08-03

> 承接 [[reB-scaleB-fix-training-0803]]、`BUG_residual_vs_reconstruction_conflict_20260803.md`、
> `PLAN_split_decoder_freeze_backbone_20260803.md`、`manifold反思.md`、`FIX_DIRECTIONS_20260803.md`。
> 目标：修掉「病灶A（天花板高度）」的真正根因——latent 流形自造且不锁死——用正统
> latent flow matching 范式（先固定自编码器/流形，再在其上训 flow）。

---

## 一、问题复述：latent 流形到底是什么、病在哪

flow 不在像素空间做，而在 `encoder` 压出的低维 latent（`96×28×28`，来自 `_DepthDownsampleEncoder`
三次 stride2）里做欧拉/Heun 积分，最后 `decoder` 解回 `224×224`。

「latent 流形」= 所有合法深度图编码后 latent 聚成的低维连续子集（一张揉皱的纸）。
flow matching 要求积分路径始终待在纸面上，否则 decoder 拿到非法 latent 解出垃圾。
纸的形状由 **encoder + decoder 这对自编码器**定义，靠**重建约束 `L_rec`** 才良态。

**当前代码三个病症**（对应 `model/manifold_flow.py`）：

1. **回归目标在动（病症1）**：`target_velocity = target_encoder(GT) - z0`（L337）是速度场监督信号，
   但 `target_encoder` 每步都在更新 → 同一 GT 的 latent 终点每步都变，速度场在追移动的靶。
2. **纸没钉牢（病症2）**：唯一让 latent 良态的 `L_rec`（L322）权重仅 0.1，encoder/decoder 主要
   被 velocity_field 顺带塑形，纸扭曲、有褶皱空洞。
3. **train/inference 分布不匹配（病症3，最致命）**：decoder 训练时主要解码 `E(GT)`（L320，被 L_rec 监督），
   推理时却要解码 rollout 积分出的 `refined_latent`（L285，带积分误差+噪声，不在 E(GT) 那片区域）→
   推理时 decoder 面对没专门学过的 latent → 输出崩坏。

**现象印证**：5 轮训练一致 val 在 ep3-7 触底后平坦、train 续降 1-2 数量级 = 泛化早饱和 + 纯记忆。
**冻主干（S2DepthNet 那侧）治不了**：三病全在 flow 内部那套独立 encoder/decoder/velocity_field 上。

---

## 二、正统修法：DepthFM / LDM 范式 —— 先钉牢流形，再画路径

参考做法（见第六节文献）：**先**把自编码器（encoder + decoder）单独训到重建收敛，
把 latent 几何**固定死**；**再冻结** AE，只在这张不动的良态纸上学 flow。

分三阶段：

### 阶段 0（前提，已完成）：decoder 分离
`model/manifold_flow.py` 已加 `split_decoders` 开关（默认 False = 旧行为向后兼容）：
- `recon_decoder`：`D_rec(E(GT)) ≈ GT`（绝对深度，服务 L_rec，与 encoder 一起构成待冻结的 AE）。
- `residual_decoder`：`D_res(refined_latent) = Δ`（残差增量读出头，随 flow 训，不参与定义流形）。

消除了「一个 decoder 承担残差/绝对两种互斥语义」的冲突（`BUG_..._conflict` 那个 bug）。
`freeze_autoencoder=True` 时只冻 `encoder(condition/target)+recon_decoder`，
`velocity_field + residual_decoder` 保持可训（已冒烟验证）。

### 阶段 1：AE 预训练，固定 latent 流形
把 `encoder`（share_encoder=True 时 condition/target 同实例）+ `recon_decoder` 当自编码器，
用**大权重 L_rec** 在深度图上训到重建收敛（判据：val 重建 L1 收敛且平稳）。
- 训练目标：`min ||D_rec(E(depth)) - depth||`（masked L1）。
- 训练数据：GT 深度 **和** coarse 深度都喂（关键，见下）。
- **为什么 coarse 也要喂**：推理时 encoder 实际吃的是 coarse（`condition_encoder(coarse_depth)`），
  若 AE 只在 GT 上训，encoder 对 coarse 分布欠编码（病症2/3 残留）。让 AE 同时重建 GT 和 coarse，
  使 `E` 在两个分布上都良态，z_coarse 和 z_gt 落在同一片良态纸上。
- 产出：`ae_pretrained.pth.tar`（encoder + recon_decoder 权重）。

### 阶段 2：冻结主干 + 冻结 AE，小 lr 训 flow 20 epoch
1. 加载 0320 主干权重（`bivmamba_best/checkpoint/model_best_ep70.pth.tar`，已验证 306 key
   零 shape 冲突可完整装进 224² 残差桥主干）→ **冻结** encoder(Mamba)+resblocks+decoders+pred，
   coarse 变固定高质量输入。
2. 加载阶段 1 的 `ae_pretrained` → `freeze_autoencoder=True` 冻结 encoder+recon_decoder。
3. 只训 `velocity_field + residual_decoder`，**小 lr（5e-5）× 20 epoch**。
4. 此时：病症1消（E(GT) 靶固定）、病症2消（纸被大权重 L_rec 钉牢并冻结）、
   病症3大幅缓解（recon_decoder 冻结且在整片区域训过；residual_decoder 专职读残差）。

---

## 三、损失配置（阶段 2）

- `L_flow`（smooth_l1，已默认）：velocity_field 回归 `E(GT)-z_coarse`。**主训练信号**。
- `L_refined_metric`（w 提高到 ~0.4）：`residual_decoder` 出的 refined 对 GT 的 224² 线性空间 L1，
  对齐 abs_rel/RMS（病灶B）。
- `L_rec`：AE 已冻结，此项恒定，可关（w=0）省算力；或留小权重仅作监控。
- `reconstruction_weight` 与 residual 冲突问题已由 split_decoders 解决，可安全共存。
- `si_n_lambda=0.25`（病灶B：保留部分绝对尺度惩罚）。
- 主干 L_coarse：主干已冻结，coarse 固定，coarse_weight 可保留（对 flow 无梯度，仅数值）。

---

## 四、口径与对比

- **224² CenterCrop**（`preproc.mode=centercrop, size=[224,224]`），与 0320 严格同口径可比。
- 规范评测：`export_reB_predictions → evaluation_DENSE --crop_ymax224 --clip_distance 1000 --reg_factor 5.7`。
- 对比基线：0320 MambaSSM best（224² 口径）abs_rel 0.597 / δ 0.662；
  之前残差桥 224² 最优 ep3 abs_rel 1.900 / δ 0.592。

---

## 五、判据（是否真的修好了 latent 病）

1. **病灶A响应**：val 触底 epoch 是否后移、train-val 劈叉是否收窄（对比历史 ep3-7 触底平坦）。
   若 val 平台被打破 → latent 固定确实解开了泛化瓶颈。
2. **病灶B响应**：规范评测 abs_rel 是否 < 1.900、δ 不退化。
3. 训练日志 abs_rel 因 `metric.py` bug 不可信，看 rmse + 规范评测。
4. **消融对照**：与「阶段2 但不做阶段1 AE 预训练（AE 随机初始化后冻结）」对比，
   隔离「AE 预训练」这一变量的贡献。

---

## 六、参考文献与范式依据

- **DepthFM (2024)**：单目深度的 flow matching，用**冻结的预训练 VAE** latent，flow 只在固定 latent 空间做。
  → 本方案阶段1+2 的直接依据（先固定 AE 再训 flow）。
- **Latent Diffusion Models / Stable Diffusion (Rombach 2022)**：先训 autoencoder 固定 latent，
  再在其上训生成模型；两阶段解耦。→ 阶段划分依据。
- **Rectified Flow (Liu 2022) / Flow Matching (Lipman 2023)**：直线路径 + data coupling；
  z0=z_coarse、z1=E(GT) 的 coupling 已在代码（L330）。
- **DDVM (Saxena 2023)**：depth 回归中 L1 > L2 → L_flow 用 smooth_l1（已默认生效）。

**可选优化（若实现时验证有效，追加记录到本文档）**：
- self-conditioning：训练时也把 rollout latent 喂给 decoder 并监督，直接弥合病症3的分布 gap。
- latent 归一化（LDM 的 scaling factor）：AE 预训练后统计 latent std，flow 前除以它，使 latent ~N(0,1) 尺度。
- 若 AE 预训练重建 L1 下不去，考虑加 KL 或 VQ 正则（VAE/VQ-VAE），但先试无正则的确定性 AE。

---

## 七、涉及文件（实现清单）

- `model/manifold_flow.py`：split_decoders / freeze_autoencoder（已落地阶段0）。
- 阶段1 脚本：`scripts/pretrain_ae_manifold_20260803.py`（新增）。
- 阶段2 脚本：复用 `scripts/ablation_manifold_flow_20260708.py`，加 `--freeze-backbone` /
  `--backbone-ckpt` / `--ae-ckpt` 参数（新增）。
- config：`configs/train_..._residual_bridge_latentfix_20260803.json`（split_decoders=true,
  freeze_autoencoder=true, preproc=centercrop224）。
- runner：`scripts/run_latentfix_20260803.sh`（阶段1 预训练 → 阶段2 小 lr 20ep）。
