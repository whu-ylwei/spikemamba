# 残差桥修复方向（两个病灶诊断闭环版）2026-08-03

> 承接 `manifold反思.md`（架构反思）与 `OPTIMIZATION_DIRECTIONS_20260731.md`（文献支撑的方向清单）。
> 本文只做一件事：把"很快收敛→瓶颈"的现象，收敛到**两个可验证的病灶**，并给出**先诊断后动手**的修复顺序。
> 本轮不改代码，仅整理方向。

---

## 〇、现象复盘（数据事实）

5 轮训练（0710 40+40、centercrop224 10+10、centercrop224 40+40、scratch precision、precision resume）呈现**同一模式**：

- **val 在 ep3–7 触底后完全平坦**，之后几十个 epoch 对泛化零贡献。
- **train 同期继续降 1–2 个数量级**（0710：train 0.0179→0.00082，是 val 的 1/17）。
- 结论：不是"收敛"，是**泛化能力极早饱和，剩余训练全在记忆训练集（过拟合）**。

消融锚点：**BASE（无 flow）best val 0.01214 < FLOW（有 flow）0.01371** —— 精修头是净负贡献。
224² 同口径对比 0320：**δ<1.25 逼近（0.592 vs 0.662），abs_rel 差 3.2×（1.900 vs 0.597）**。

补充事实（0801 run 实际状态，2026-08-03 核实）：`runs/flow_reB_centercrop224_40plus40_20260731` 名为 40+40，**实际 stage1 只跑到 ep27 异常中断，stage2 从未启动**（日志无 STAGE1 done / 无 STAGE2，model_last 时间戳=ep27）。best=ep4（val0.01045）。即"40+40"实为"27+0"，且 best 落在极早的 ep4，与上述模式一致。

---

## 一、病灶 A：flow 精修头学不到"可泛化的深度结构"

### 诊断
- flow-on 打不过 flow-off；val ep3–7 触底而 train 续降 → 额外容量全进记忆，没进泛化。
- 根因（`model/manifold_flow.py`）：flow 头把文献中"能 work 的四条件"全选到失效侧：
  1. 流向反了：`_rollout_latent` 从噪声/零起步（noise→depth），coarse 只当 conditioning，最终从零重生。
  2. 输出替换 coarse：`blend_alpha=0`，coarse 对输出零贡献，非残差锚定。
  3. latent 是从零联合训练的移动靶：三个编解码器随主任务训，`target_latent` 非平稳，`L_rec` 权重仅 0.1。
  4. 独立耦合 + 少步欧拉：`L_flow` noise↔target 独立耦合（路径弯），却只积几步。
- 文献对照：DepthFM（冻结 VAE latent + data coupling）、TryOn-Refiner（coarse→refined 残差锚定）、Rectified Flow（data coupling 拉直路径）——四项全反。

### 结论
flow 头**在结构上就学不到可泛化的深度**，不是训练不足。它决定了**天花板高度**（泛化 ep3–7 饱和）。

---

## 二、病灶 B：绝对尺度（米制深度）几乎没有梯度去学

### 诊断
- δ 逼近而 abs_rel 差 3× = "相对顺序学到了、绝对距离没学到"的教科书信号。
- 根因：
  1. 主损失尺度不变（scale-invariant，si_n_lambda 旧1.0/新0.5）——奖励形状对，不奖励量程对。
  2. 条件特征 per-image 归一化（`_normalize_feature_statistics`）抹掉全局尺度/偏移。
  3. 直接监督绝对深度的项太弱（`refined_metric_weight=0.1`）。
  4. 量程浪费：clip1000/α5.7，`exp(5.7·(x-1))·1000` 把分辨率耗在几乎不出现的远景（DENSE 多 <80m），远景小误差被指数放大。
- 佐证：0730 精度增强（si_n_lambda 1.0→0.5 + 加 refined_metric_weight）已让**所有绝对指标小幅全面改善**（abs_rel 2.465→2.300），δ 持平 → 一松尺度不变约束、一加绝对监督，绝对指标就响应，只是幅度小。

### 结论
即便 A 修好、val 能续降，尺度监督缺失也会让 abs_rel 卡在高位。它决定**天花板的绝对位置偏移**。

---

## 三、两病灶关系
- **A 定天花板高度**（泛化早饱和 → "很快收敛进瓶颈"的直接成因）。
- **B 定天花板绝对位置**（相对对、绝对差 → abs_rel 卡高位）。
- 两个必须都动才能真正突破。

---

## 四、修复顺序（先诊断切分，再结构改动）

> 原则：先做低风险、能快速切分病灶是否响应的改动；再做抬天花板的结构性改动。本轮不实施，仅排序。

### 阶段 1 —— 病灶 B 快速切分（低风险，config/单行级，先做）
- **B-1** `L_flow` L2→L1（`F.mse_loss`→smooth L1）。DDVM 消融明确 L1 优。
- **B-2** `refined_metric_weight` 0.1→0.3~0.5，提高线性空间绝对监督。
- **B-3** 去掉/关掉条件特征 per-image 归一化（`_normalize_feature_statistics`），保留绝对尺度。
- **B-4**（可选，需重训）量程 clip1000/α5.7 → clip80/α3.7（E2Depth 口径）。改归一化必须重训，不能只换评测参数。
- **判据**：同口径 abs_rel 是否较 1.900 下降、δ 不退化。若响应 → 证实病灶 B，继续；若不动 → 病灶 B 权重仍不够或被 A 压制。

### 阶段 2 —— 病灶 A 结构改动（抬天花板，后做）
- **A-1** flow 头改 coarse→GT 的 data coupling（`z0=condition_latent`，`target_velocity=target_latent-z0`），起点落在流形内而非噪声。
- **A-2** 残差锚定：`blend_alpha>0` 或输出严格 `refined=coarse+Δ`，让 flow 只学增量。
- **A-3** latent 稳定化：深度自编码器先重建预训练再**冻结**，或提高 `L_rec` 权重，别让流形当移动靶。
- **A-4**（消融）起点加小噪声（rollout_noise_std 0.05）、少步 ODE 扫描（flow_steps∈{1,2,4,10}）验证路径直线性。
- **判据**：flow-on 是否**反超** flow-off（BASE 0.01214）；val 平台是否被打破（触底 epoch 后移、train/val 劈叉收窄）。

### 阶段 3 —— 充分训练 + 正则（在 A/B 见效后）
- 40+40 真正跑满（0801 那轮实为 27+0，需重跑），配 `weight_decay`(现0)1e-4~1e-2、batch 提到 4、加训练增广（残差桥 centercrop 无随机增广，加剧过拟合）。
- 降 scratch 起始 lr（1e-4 对已优架构偏高，早期 train 尖峰 ep02=0.07485 为证）。

---

## 五、验证口径（不可踩坑）
- 112²(resize) 与 224²(centercrop) **不可互比**；与 0320 / E2Depth / EReFormer 比必须 224² CenterCrop + 相同 clip/α。
- 改归一化量程（B-4）必须重训，train/eval 解码参数须一致。
- 训练日志 `abs_rel` 因 `metric.py` bug 不可信（分母近零爆表），以规范评测（export→evaluation_DENSE）和 rmse 为准。
- 参考目标：0320 MambaSSM abs_rel 0.597 / δ 0.662；E2Depth 0.220 / 0.724；EReFormer 0.172 / 0.747。

---

## 六、代码事实校正（2026-08-03 通读 `model/manifold_flow.py` 全文后）

> 前几节的病灶 A 描述部分基于早期截断阅读，通读全代码后修正如下。**残差桥分支的 C1–C5 已把大部分"范式反了"的问题修掉**，真正残留的病灶更聚焦。

已经修好的（C1–C5 全开，config 也全开，非病灶）：
- **流向**：`start_from_coarse=True`（L178,243），起点是 coarse latent，不是噪声。
- **耦合**：`data_coupling=True`（L176,311-320），走 coarse→GT 直线，`z0=condition_latent`、`target_velocity=target_latent-z0`。
- **输出**：`residual_output=True`（L175,270-272），`refined=coarse+decoded`，残差锚定，非替换。
- **编码器**：`share_encoder=True`（L177,183），coarse 与 GT 共用编码器，同一流形。
- **per-image 归一化**：`_fuse_encoder_features` 已删（L226 C5），且 `encoder_feature_scale=0` 时 L194 直接 return，编码器特征根本没融进来；`_normalize_feature_statistics`（L62）函数仍在但**未被调用**。

真正残留的病灶（据此重新聚焦）：
- **病灶 A 残留-1（最关键）**：latent 流形**无重建约束**。`reconstruction_weight=0` → `L_rec`（L305-309）权重 0 → target_encoder/decoder 定义的流形是"移动靶"，无任何重建监督。这是"流沙上做 flow"的直接根源。
- **病灶 A 残留-2**：`L_flow` 用 L2（`F.mse_loss`, L324），DDVM 消融证明 L1 更好。
- **病灶 B**：主要在损失侧——si_n_lambda（尺度不变）、`refined_metric_weight=0.1` 偏弱、量程 clip1000/α5.7。特征侧的归一化已不是问题。

---

## 七、阶段 1 与阶段 2 的精确 diff（不落地，仅列出）

### 阶段 1 —— 病灶 B 快速切分（低风险）

**diff 1-A：`L_flow` L2→smooth L1**（`model/manifold_flow.py:322-325`）
```diff
             if self.loss_weights["flow"] > 0.0:
                 auxiliary_losses["L_flow"] = (
-                    self.loss_weights["flow"] * F.mse_loss(predicted_velocity, target_velocity)
+                    self.loss_weights["flow"] * F.smooth_l1_loss(predicted_velocity, target_velocity)
                 )
```

**diff 1-B：新建 config，提高绝对监督 + 松开完全尺度不变**（不改代码）
```diff
  // configs/train_..._residual_bridge_scaleB_20260803.json (基于 precision config)
-    "refined_metric_weight": 0.1,
+    "refined_metric_weight": 0.4,
-    "si_n_lambda": 0.5,
+    "si_n_lambda": 0.25,
```

**diff 1-C（可选，需重训）**：数据集量程 clip1000/α5.7 → clip80/α3.7（E2Depth 口径）。train/eval 必须同参数，单独一版对照，不与 1-A/1-B 混。

### 阶段 2 —— 病灶 A 结构改动（抬天花板）

**diff 2-A（最关键，纯 config）**：打开已实现但被设 0 的重建约束
```diff
  // config
-    "reconstruction_weight": 0.0,
+    "reconstruction_weight": 0.3,
```

**diff 2-B：latent 自编码器可冻结开关**（`__init__` 末尾 L191 后，需配套预训练脚本）
```diff
+        self.freeze_autoencoder = bool(config.get("freeze_autoencoder", False))
+        if self.freeze_autoencoder:
+            for p in self.condition_encoder.parameters():
+                p.requires_grad_(False)
+            if not self.share_encoder:
+                for p in self.target_encoder.parameters():
+                    p.requires_grad_(False)
+            for p in self.target_decoder.parameters():
+                p.requires_grad_(False)
```

**diff 2-C：起点加小噪声**（`_rollout_latent`, L241-243；`start_from_coarse=True` 时当前不看 rollout_noise_std，需补一行）
```diff
         if self.start_from_coarse:
             # C1: start on the manifold at the coarse latent; the flow only refines.
             latent_state = condition_latent
+            if self.rollout_noise_std > 0:
+                latent_state = latent_state + torch.randn_like(latent_state) * self.rollout_noise_std
```
```diff
  // config
-    "rollout_noise_std": 0.0,
+    "rollout_noise_std": 0.05,
```

**diff 2-D：少步 ODE 扫描**（纯评测侧，扫 flow_steps∈{1,2,4,10}，零代码改动）

### 推荐切分顺序
1. **先纯 config**（2-A 的 `reconstruction_weight=0.3` + 1-B 的 `refined_metric_weight=0.4`/`si_n_lambda=0.25`）跑一版，零代码改动即可切分两病灶贡献，看 val 平台是否被打破、abs_rel 是否响应。
2. 有响应再上 **1-A**（单行 L1）与 **2-C**（起点加噪一行 + config）。
3. 前面到瓶颈再上 **2-B**（冻结流形，需预训练脚本）与 **1-C**（量程重训）。

---

## 参考文献（同 0731 清单）
- DepthFM (2403.13788)、E2Depth (2010.08350)、EReFormer/DTR (2212.02791)、DDVM (2306.01923)、Improving Rectified Flows (2405.20320)。
