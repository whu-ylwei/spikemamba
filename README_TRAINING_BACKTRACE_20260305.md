# SpikeMamba 训练问题回溯 README（2026-03-05）

> 目的：给下一次 Codex 对话直接接续用。  
> 范围：本次在 `shared-nvme/spikemamba` 上的训练排障、验证与实验结论。  
> 最后更新：2026-03-05（UTC）

## 0. 当前上下文

- 项目路径：`/root/shared-nvme/spikemamba`
- 主配置：`configs/train_s2d_spiketransformer_rtx5090_20260305.json`
- 主问题运行目录：`s2d_checkpoints/train_s2d_SpikeTransformer_rtx5090_20260305_20260305_114546`
- 从头小实验目录：`s2d_checkpoints/train_s2d_SpikeTransformer_fromscratch_check_20260305`
- 关键 trainer 文件：`trainer/spiket_trainer.py`

---

## 1. 问题总览（状态 + 处理）

| ID | 问题 | 状态 | 处理结果 |
|---|---|---|---|
| P01 | 训练目录已存在导致启动退出 | 已解决 | `train_parallel.py` 增加自动重命名（时间戳） |
| P02 | NumPy 2.0 移除 `np.alltrue` 导致数据加载报错 | 已解决 | 数据集代码改为 `np.all` |
| P03 | 预览阶段 `KeyError: 'image'` | 已解决 | preview 增加 `events0/events*` 回退键 |
| P04 | `UnboundLocalError: step` | 已解决 | 统一使用 `preview_step`，避免未定义变量 |
| P05 | loss 汇总写法错误（含 `no_grad` + 错误自增）导致梯度链断裂/统计错误 | 已解决 | 修正 `calculate_total_batch_loss` 逻辑 |
| P06 | 每个 key 的 loss dict 复用同一对象（alias bug） | 已解决 | 改为每个 key 独立 dict |
| P07 | `val_loss` 跨 epoch 完全相同 | 已定位（旧 ckpt 不可恢复） | 根因是模型输出饱和后梯度为 0，参数不再更新 |
| P08 | 继续训练旧 checkpoint 无效果 | 已确认（不可恢复） | `epoch008/016/024` 参数完全一致，旧 checkpoint 已塌缩 |
| P09 | 从头训练可启动但会快速塌缩（梯度后期归零） | 未彻底解决（有缓解） | `lr=8e-4` 易塌缩；`lr=1e-4` 小步实验 20 step 未出现全零梯度 |
| P10 | 当前配置下 dataloader 只返回 `depth_image`，无 `events/image`（输入目标泄漏） | 未解决（高优先级） | `baseline='s'` + `every_x_rgb_frame=1` 组合下，模型实际吃 `depth_image` |
| P11 | 对“上次为什么退出”的误判 | 已澄清 | 上次主训练在 `epochs=30` 正常结束，不是异常退出 |

---

## 2. 关键证据（可复核）

### E01. 旧 checkpoint 参数未更新

对比目录：

- `checkpoint-epoch008-loss-0.2039.pth.tar`
- `checkpoint-epoch016-loss-0.2041.pth.tar`
- `checkpoint-epoch024-loss-0.2039.pth.tar`

结果：

- `state_dict` 差异：`max_abs_diff=0`（166/166 tensor 全相同）
- optimizer step 递增：`5000 -> 10000 -> 15000`（说明循环在跑，但有效更新为 0）

### E02. `val_loss` 完全不变

TensorBoard 标量（主问题运行目录）：

- `val_loss` count=6，unique=1，值恒为 `0.22367234528064728`

### E03. 旧 checkpoint 梯度全 0

1 step 检验（加载 `epoch024`）：

- `pred_min=max=mean=0.0`
- `pred_zero_frac=1.0`
- `nonzero_grad_tensors=0/166`
- `delta_max=0.0`

### E04. 从头训练在原 lr 下快速塌缩

随机初始化 + `lr=8e-4`：

- 前几步有梯度，约 step 8 后 `nonzero_grad_tensors=0/166`
- 输出接近全 0，后续 loss 基本停滞

### E05. 降低 lr 的缓解效果（仅小实验）

随机初始化 + `lr=1e-4`（20 steps smoke）：

- 全程 `nonzero_grad_tensors=166/166`
- 未出现“全零梯度”塌缩
- 但这是 smoke 级验证，未做长程训练与最终精度确认

### E06. 数据输入异常（高优先级）

当前主配置（`baseline='s'`, `every_x_rgb_frame=1`）下，取一个 batch 的 key：

- 仅有 `['depth_image']`

意味着当前模型前向优先级会走：

- `item['image']` 不存在
- `item['events']` 不存在
- 回退使用 `item['depth_image']` 作为输入

即：训练变成“用深度 GT 预测深度 GT”，不符合论文设定与任务定义。

---

## 3. 与论文（Spike Transformer, ECCV 2022）对比得到的梯度结论

论文关键点（用于解释梯度问题）：

- 监督目标是 **对数深度表征**（不是把输出强行压到 sigmoid 饱和边界）
- 使用 scale-invariant loss，训练数据输入是事件/脉冲表征，不是深度 GT 本身

当前实现偏差：

1. 模型尾部固定 `sigmoid`，当 logit 走向大负值时输出逼近 0，梯度近零。  
2. 当前数据流在配置组合下把 `depth_image` 当输入，训练目标与输入语义错误。  
3. 在 `lr=8e-4` 下更容易把输出推入饱和区，形成“输出全零 -> 梯度全零 -> 参数不动”。

结论：

- 目前“梯度下降到 0”的主因不是正常收敛，而是 **输出饱和 + 输入语义异常 + 学习率偏激** 的组合导致训练塌缩。

---

## 4. 已修改文件（本次排障）

- `trainer/spiket_trainer.py`
  - 修复 total loss 累加写法（去掉错误 `no_grad`/自增逻辑）
  - 修复 per-key loss dict 共享引用问题
  - 修复 preview `image` 缺失回退
  - 修复 preview `step` 未定义

（历史阶段还包含 `data_loader/*.py`, `train_parallel.py`, `model/*.py` 的多项修复，见 git 记录）

---

## 5. 未解决问题（必须特殊标注）

### U01. 数据输入语义错误（最高优先级，未解决）

- 现状：训练 batch 只有 `depth_image`，无事件输入。
- 影响：训练目标被“泄漏”为输入，实验结论不可信，且容易诱发异常训练动力学。
- 建议下一步：
  1. 明确 `baseline='s'` 的预期 key（应包含 spikes/events）。
  2. 修正 `SpikesDENSE_dataset.py` 在 `every_x_rgb_frame=1` 时的键构造逻辑。
  3. 在模型前向前加断言：训练时禁止只含 `depth_image` 输入。

### U02. 长程稳定训练策略未最终定版（未解决）

- 现状：`lr=1e-4` 仅 smoke 级验证更稳定，未跑完整 epoch 级收敛。
- 建议下一步：
  1. 在修正数据输入后，再做 `lr`/梯度裁剪/输出头设计的系统对比。
  2. 将“梯度非零比例 + 输出分布 + logit 范围”纳入每 N step 监控。

---

## 6. 下次对话建议的起手检查（最短路径）

1. 先验证 dataloader key（必须看到 spikes/events，不可只有 `depth_image`）。  
2. 再做 20-step 梯度健康检查（`nonzero_grad_tensors` 不应坍塌到 0）。  
3. 最后才做长程训练与 resume。  

可直接复用的核查项：

- checkpoint 参数是否变化（跨 epoch tensor diff）
- TensorBoard 中 `val_loss` unique 值个数
- 单步 `delta_max` 与 `nonzero_grad_tensors`

---

## 7. 重要路径索引

- 主配置：`/root/shared-nvme/spikemamba/configs/train_s2d_spiketransformer_rtx5090_20260305.json`
- 主问题 ckpt：`/root/shared-nvme/spikemamba/s2d_checkpoints/train_s2d_SpikeTransformer_rtx5090_20260305_20260305_114546`
- 从头 smoke ckpt：`/root/shared-nvme/spikemamba/s2d_checkpoints/train_s2d_SpikeTransformer_fromscratch_check_20260305`
- 主日志：`/root/shared-nvme/spikemamba/logs/train_opt_20260305_114542.log`
- 从头日志：`/root/shared-nvme/spikemamba/logs/train_fromscratch_check_20260305_135133.log`
- trainer 修复文件：`/root/shared-nvme/spikemamba/trainer/spiket_trainer.py`

