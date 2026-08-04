# scaleB 实验的语义冲突缺陷定位（残差输出 vs 绝对重建）2026-08-03

> 触发：0803 scaleB 病灶切分实验运行到 ep5 时，经代码逐行核对发现 `L_rec`（reconstruction）与 `residual_output` 残差路径存在**语义冲突**，共享同一个 `target_decoder` 导致两个损失项要求解码器输出相互矛盾的东西。
> 结论：scaleB 当前配置对"验证病灶A（流形重建锚定）"是**无效实验**；已于 16:24 后停止训练（kill runner 14523 + main 14540，GPU 已释放）。
> 相关：[[reB-scaleB-fix-training-0803]]、`FIX_DIRECTIONS_20260803.md`、`manifold反思.md`。

## 一、缺陷本质

`model/manifold_flow.py` 的 `forward` 里，**同一个 `self.target_decoder` 被用在两条语义相反的路径上，且无 stop-gradient**：

| 路径 | 代码位置 | 解码器输入 | 期望输出 |
|---|---|---|---|
| 残差输出 | L285→289 | `refined_latent` | `decoded = Δ`，最终 `refined = coarse + Δ` |
| 重建监督 | L320→325 | `target_latent = E(GT)` | `reconstructed ≈ GT`（L_rec 要求） |

data_coupling（L330,336,337）：`z0=E(coarse)`、`z1=E(GT)`、`target_velocity=z1-z0`，L_flow 把 rollout 终点推向 `z1=E(GT)`。

**渐近地** `refined_latent → E(GT)`，两条路径解码器输入收敛到同一点。设 `y = D(E(GT))`：
- `refined_metric`(w=0.4) 要 `coarse + y ≈ GT` → **y = GT − coarse**
- `L_rec`(w=0.3) 要 `y ≈ GT` → **y = GT**

确定性解码器同一输入只能输出一个 `y`，除非 `coarse≡0` 否则二者不可兼得。

## 二、真实收敛结果（不是 coarse+GT）

L1 损失下 `y` 最小化 `0.4·|y−(GT−coarse)| + 0.3·|y−GT|`（加权中位数）。因 **0.4>0.3**：
- 解码器实际学 **`y → GT − coarse`（残差语义胜出）**，深度输出大致正确。
- **`L_rec` 收敛不到 0，卡在 floor ≈ 0.3·mean|coarse|**（冒烟测得 L_rec≈0.097 就是这个 floor，非在向 0 收敛）。
- 所以 **L_rec 没起到"锚定流形"作用**，只是与残差解码器对抗的、带非零下限的张力项，反而给训练加噪 —— **很可能是 ep4 train 尖峰(0.013→0.072)的推手之一**。
- ∴ 不会变成 `coarse+GT`(那要 y=GT 胜出)，而是残差语义胜出、重建损失被架空。

## 三、推广：`residual_output=True` 与任何"绝对重建约束"都不兼容

- 残差要 `D(E(GT))=GT−coarse`，绝对重建要 `D(E(GT))=GT`，恒冲突。
- **同样会坑 diff 2-B（冻结预训练自编码器）**：冻结的 AE 满足 `D(E(GT))=GT`，接上残差 → `refined=coarse+GT`，更糟。
- ∴ FIX_DIRECTIONS_20260803 里 diff 2-A（残差保留+开 rec）、2-B（残差保留+冻结AE）的设计**都有此缺陷**，是设计失误。

## 四、两条自洽路线（二选一，不能既残差又重建）

1. **残差路线（scaleB 之前的现状）**：`reconstruction_weight=0`，解码器自由学 delta，`D(E(GT))≈GT−coarse` 自洽。缺点：流形无重建约束(=病灶A本身)，但不冲突。
2. **绝对路线（DepthFM 式正统 manifold flow）**：`residual_output=False`，解码器输出绝对深度，`refined=D(refined_latent)→GT`，此时 L_rec 和冻结AE 才自洽。锚定靠 flow 起点 `z0=E(coarse)`+`L_coarse`，非残差加法。

## 五、scaleB 三个改动的自洽性复盘

scaleB 相对 centercrop224 基线改了三个权重 + 一处代码：
- `si_n_lambda 0.5→0.25`（病灶B）：**自洽**，与残差无关。
- `refined_metric_weight 0.1→0.4`（病灶B）：**自洽**，就是残差语义那侧。
- `reconstruction_weight 0→0.3`（病灶A）：**冲突源**，废项+加噪。
- `L_flow` L2→smooth_l1：**自洽**，全局改动。

∴ scaleB 的病灶B部分信号仍可用，但**病灶A结论无效**，且 rec 项污染了整体训练稳定性。

## 六、下一步（拆成两个自洽实验）

- **实验A（病灶B单独版）**：`reconstruction_weight` 退回 0，只留 si_n_lambda=0.25 + refined_metric=0.4 + L_flow smooth_l1。当前残差代码即自洽，纯验证绝对尺度监督对 abs_rel 的影响。
- **实验B（病灶A绝对路线版）**：`residual_output=false` + `reconstruction_weight>0`(+可选冻结AE)，走 DepthFM 正统，验证"稳定流形锚定"是否打破 flow 头早熟饱和。
- 两实验都要 224² CenterCrop 口径，规范评测(export→evaluation_DENSE --crop_ymax224 --clip1000 --reg5.7)对比 0320(abs_rel0.597/δ0.662)。

## 七、"是否真有问题"——冲突的绑定性核实（重要细节）

一个反驳点：训练期 refined_metric 用的解码器输入是 **rollout 结果 `refined_latent`**（从 z0=E(coarse) 积分而来），而 L_rec 用的是 **`E(GT)`**——**两者训练期并不相同**，所以冲突不是一开始就绑定的。

逐阶段分析：
- **训练早期**：flow 未学好，`refined_latent ≠ E(GT)`，两条路径解码器输入不同。解码器**可以**同时满足二者（对 rollout 输入输出 GT−coarse、对 E(GT) 输入输出 GT），此时无硬冲突，但两个目标在相邻输入上把解码器往相反方向拉，损害其平滑性/一致性。
- **训练收敛期**：flow 的训练目标(L_flow)正是把 `refined_latent` 推向 `E(GT)`。随着 flow 变好，两个输入**逐渐重合**，冲突从"软张力"变成"硬矛盾"，`L_rec` 被顶到 floor≈0.3·mean|coarse|。

∴ **结论：确有问题，且是"随 flow 收敛而增强"的隐蔽缺陷**——flow 学得越好，rec 与 residual 越打架。不是立即报错型 bug，而是把训练往一个自相矛盾的不动点上拉，污染稳定性(ep4 尖峰)并使 L_rec 失去设计意图。判断成立。

## 八、教训归因：不是提示词问题，是架构理解不到位（AI 侧）

**这是 AI(助手)的错误，根因是对架构的理解不够透彻，与用户提示词设置无关。**

- **不是提示词问题**：用户的指令链条清晰合理——先诊断、写方向文档、列 diff、再编码训练，每步都给了确认。用户没有任何误导性表述。相反，**是用户用一步纯数学推导(D(E(GT)) 同时要等于 GT 又要等于 GT−coarse)当场抓出了这个缺陷**，说明用户对架构的理解比助手当时更透彻。
- **是助手的架构理解缺陷**：我在 `FIX_DIRECTIONS` 里提 diff 2-A(残差保留+开 rec)、2-B(残差保留+冻结AE)时，**没有推演"加了 L_rec 后解码器的不动点是什么"**。我只做了 shape 层面的冒烟测试(能跑通、三个 loss 都非零)，就当作验证通过——但**冒烟测试只能证明"能运行"，证明不了"优化目标自洽"**。这是把"代码不报错"误当成"设计正确"的典型失误。
- **具体知识盲点**：没有意识到 `residual_output` 语义(D 输出增量)与 `reconstruction` 语义(D 输出绝对值)在共享解码器时互斥。这是 manifold flow / 残差精修架构的一个基本约束，我应在提改动前就推清楚。

**给未来的行动准则**：
1. 引入任何新损失项前，先写出它在**收敛不动点**上对被共享模块(尤其解码器)的要求，检查与已有损失项是否可同时满足，再动代码。
2. 冒烟测试只验证"可运行"；**目标自洽性必须靠符号推导**，不能靠 loss 数值非零来判断(非零可能正是 floor)。
3. 共享权重的模块(此处 target_decoder 一处三用)是冲突高发区，改动时优先审查。
