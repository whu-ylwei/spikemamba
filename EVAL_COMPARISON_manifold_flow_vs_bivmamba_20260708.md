# 评估对比:Manifold Flow vs 纯 BiMamba(bivmamba_best)

**整理日期**:2026-07-08
**数据集**:DENSE test（CARLA 合成,`clip_distance=1000`,spike voxel r128）
**评估口径**:所有绝对深度指标由归一化预测反投影到米制后计算,按深度上限分档。

## 参评模型

三者均为双向 Mamba 编码器 + U-Net 解码器（`S2DepthTransformerUNetConv`,当前架构,d_model=1536,含 `mamba_bwd`）,唯一区别在 **是否启用 Manifold Flow** 与 **训练序列长度**。

| 标识 | Manifold Flow | seq_len | best epoch | val_loss | checkpoint |
|---|---|---|---|---|---|
| **bivmamba_best** | 关闭 | 1 | 70 | 0.01411 | `bivmamba_best/checkpoint/model_best_ep70.pth.tar` |
| MambaFlow seq2 | 开启 | 2 | 41 | 0.02709 | `runs/train/train_s2d_MambaFlow_bidivim_seq2_ep40_continue40_halflr_20260423_bs2/checkpoints/model_best.pth.tar` |
| MambaFlow seq4 | 开启 | 4 | 57 | 0.03105 | `runs/train/train_s2d_MambaFlow_bidivim_seq4_ep40_continue40_halflr_20260502_bs1/checkpoints/model_best.pth.tar` |

> **val_loss 不可跨臂比较**:Flow 臂的 val_loss 含 coarse/rec/flow/geo 辅助损失,口径与纯 BiMamba 不同。以下 test 集统一指标才是可比项。

## 一、归一化主指标（DENSE test，`total metrics`）

指标顺序 `[mse, abs_rel_diff, scale_invariant_error, median_error, +2]`。

| 模型 | mse ↓ | scale_inv_error ↓ | median_error ↓ |
|---|---|---|---|
| bivmamba_best | 0.02114 | 0.01445 | **0.02263** |
| MambaFlow seq2 | 0.01764 | 0.01174 | 0.03218 |
| **MambaFlow seq4** | **0.01686** | **0.01091** | 0.03025 |

> `abs_rel_diff` 归一化口径下被远景无效点放大到几十~上千（bivmamba 48.7 / seq2 1005.9 / seq4 310.2）,无参考价值,以下面米制分档为准。

## 二、绝对深度指标（米制，全深度汇总）

| 模型 | abs_rel ↓ | RMS_linear(m) ↓ | SILog ↓ | median(m) ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
|---|---|---|---|---|---|---|---|
| **bivmamba_best** | **0.593** | 168.9 | 0.470 | **9.369** | 0.658 | **0.754** | 0.808 |
| MambaFlow seq2 | 0.880 | 156.3 | 0.385 | 9.753 | **0.669** | 0.758 | **0.815** |
| MambaFlow seq4 | 0.780 | **147.9** | **0.355** | 9.690 | 0.618 | 0.742 | 0.803 |

## 三、分深度档 δ<1.25（逐像素命中率，越高越好）

| 深度档 | bivmamba_best | MambaFlow seq2 | MambaFlow seq4 |
|---|---|---|---|
| **≤10m** | **0.982** | 0.962 | 0.880 |
| ≤20m | **0.935** | 0.911 | 0.829 |
| ≤30m | **0.886** | 0.850 | 0.783 |
| ≤80m | **0.710** | 0.685 | — |
| ≤250m | **0.634** | 0.624 | — |
| ≤500m | **0.634** | 0.623 | — |

## 四、分深度档 abs_rel_diff（越低越好）

| 深度档 | bivmamba_best | MambaFlow seq2 | MambaFlow seq4 |
|---|---|---|---|
| ≤10m | **0.253** | 0.387 | 0.335 |
| ≤20m | **0.354** | 0.529 | 0.470 |
| ≤30m | **0.536** | 0.818 | — |
| ≤80m | **0.688** | 1.114 | — |

## 结论

**Manifold Flow 的收益是分裂的：改善全局尺度不变类指标，但损害逐像素深度命中率。**

1. **mse / scale_invariant_error / SILog / RMS：Flow 胜。** seq4 最优（mse 0.0169、scale_inv 0.0109、SILog 0.355、RMS 147.9m）。Flow 精化头压低了整体尺度不变误差,序列越长越明显。

2. **median_error / δ<1.25 / abs_rel：纯 BiMamba 胜,尤其近景。**
   - ≤10m δ<1.25：bivmamba **0.982** > seq2 0.962 > seq4 0.880
   - ≤10m abs_rel：bivmamba **0.253** < seq4 0.335 < seq2 0.387
   - median_error：bivmamba **0.0226** < flow 0.030~0.032

3. **seq_len 越长分裂越强**：seq4 把 mse/scale_inv 压到最低,却把近景 δ 拉到最低（0.880）。长序列让模型更擅长优化尺度不变损失,却牺牲逐像素命中率。

**取舍**：
- 关注全局尺度一致性（mse/scale-invariant/SILog）→ 选 **MambaFlow seq4**。
- 关注实际逐像素深度准确度（δ<1.25、median、近景 abs_rel）→ 选 **bivmamba_best**,近景 δ<1.25=0.982 三者最高。

## 数据来源

- bivmamba_best：`bivmamba_best/logs/eval/eval_DENSE_ep70_full_metrics.log`、`test_DENSE_ep70.log`
- MambaFlow seq2：`run_logs/test/eval_mambaflow_bidivim_seq2_stage2_best_ep41_20260423.log`、`test_...log`
- MambaFlow seq4：`runs/eval/test_mambaflow_seq4_best_20260507/eval_metrics.log`、`test_stdout.log`

## 注意事项

- 单次运行,单种子,无误差棒。
- 仅 DENSE 合成数据；跨数据集（SCAPE-spike / dense_spike）评估另见 `run_logs/*scape_spike*`、`*dense_spike*`。
- seq4 的 ≤80/250/500m 档 δ 与 abs_rel 未在其 eval log 中完整输出,故对应单元格留空。
