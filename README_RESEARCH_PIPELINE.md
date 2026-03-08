# SpikeMamba Research README

> 面向科研复现的完整链路文档：从原始数据组织到训练结果与评估指标。  
> 适用代码版本：当前仓库 `spikemamba_publish`（训练入口为 `train_parallel.py`）。

---

## 1. 摘要（Abstract）

本项目面向 **脉冲相机单目深度估计（Monocular Depth Estimation for Spiking Camera）**，核心目标是在时间同步的脉冲序列与深度监督下，训练端到端网络预测深度图。当前实现采用：

- 数据侧：时间戳严格对齐的 `spike + depth (+ optional rgb)` 样本组织；
- 模型侧：`Mamba3DEncoder + UNet-style decoder`；
- 优化侧：`Scale-Invariant loss` 为主，支持 `Multi-Scale Gradient loss`；
- 评估侧：先导出预测，再离线计算度量，便于审计与复现实验。

---

## 2. 任务定义（Problem Formulation）

给定脉冲体输入序列 \(X\)（按时间 bins 编码），预测归一化对数深度图 \(\hat{Y}\)。训练监督来自标签 \(Y\)：

1. 原始深度 \(D\) 裁剪到 \([0, d_{clip}]\)；
2. 归一化 \(D_n = D / d_{clip}\)；
3. 对数映射 \(Y = 1 + \log(D_n) / r\)；
4. 再裁剪到 \([0,1]\)。

其中 \(r\) 为 `reg_factor`（默认 5.7）。

> 训练时网络直接回归对数空间标签 \(Y\)，评估时再反变换回米制深度。

---

## 3. 端到端链路总览（Raw Data -> Result）

```text
Raw Dataset (DENSE / Outdoor-Spike)
    |
    |  [仓库外] 传感器数据与脉冲体生成
    v
Structured Files (timestamps + spike_*.npy + depth_*.npy + optional rgb)
    |
    |  data_loader/SpikesDENSE_dataset.py
    |  - 时间戳同步
    |  - 深度log映射
    |  - 序列构造与增强
    v
Sequence Batch (L x dict)
    |
    |  model/S2DepthNet.py
    |  model/encoder_transformer.py
    |  model/mamba_3d.py
    v
Predicted Log-Depth
    |
    |  trainer/spiket_trainer.py
    |  - loss backward
    |  - optimizer step
    |  - checkpoint/tensorboard
    v
Checkpoint + TensorBoard + Validation Loss
    |
    |  test_DENSE.py
    |  evaluation_DENSE.py
    v
Depth Metrics (AbsRel / SILog / RMS / Delta etc.)
```

---

## 4. 数据层（Data Layer）

## 4.1 目录约定（训练）

训练配置默认使用：

- `DENSE/train`
- `DENSE/validation`

每个子序列目录应包含：

- `spike/r128/timestamps.txt`
- `spike/r128/spike_XXXXXXXXXX.npy`
- `depth/data/timestamps.txt`
- `depth/data/depth_XXXXXXXXXX.npy`
- 可选 `rgb/frames/frame_XXXXXXXXXX.png`

对应实现：

- `data_loader/spike_dataset.py`
- `data_loader/SpikesDENSE_dataset.py`

## 4.2 时间同步逻辑

对每个脉冲体时间戳 `event_timestamp`，在深度时间戳中查找首个 `>=` 的深度帧索引，要求时间差足够小（代码中阈值为 `1e-5` 量级）。  
这样保证了监督与输入在时间上严格配对。

## 4.3 样本粒度

单个训练样本是一个长度为 `L=sequence_length` 的序列（列表），每个元素为字典，典型键包括：

- `events{k}`：脉冲体（C x H x W）
- `image`：给模型的主输入（在 baseline=`s` 时也可为 events）
- `depth_image` 或 `depth_events{k}`：监督深度标签

## 4.4 标签变换（关键）

在数据加载阶段执行：

1. `frame = clip(frame, 0, clip_distance)`
2. `frame = frame / clip_distance`
3. `frame = 1 + log(frame) / reg_factor`
4. `frame = clip(frame, 0, 1)`

这决定了模型学习的是 **归一化对数深度**，不是原始米制深度。

## 4.5 数据增强

训练集默认：

- `RandomRotationFlip(0.0, p_hflip=0.5, p_vflip=0.0)`
- `RandomCrop(224)`

验证/测试默认：

- `CenterCrop(224)`

---

## 5. 方法（Method / Model）

## 5.1 输入规范化（Input Canonicalization）

`S2DepthTransformerUNetConv.forward()` 支持：

- 4D: `(B, C, H, W)`
- 5D: `(B, 1, T, H, W)` 或旧格式 `(B, T, 1, H, W)`

最终统一为 `(B, 1, T, H, W)` 送入 3D 编码器。

## 5.2 3D Mamba 编码器

`Mamba3DEncoder` 结构：

- `Conv3d patch_embed` 进行时空 patch embedding；
- 每个 stage 由若干 `ST_Mamba_Block` 堆叠；
- stage 间只下采样空间维（`stride=(1,2,2)`），保持时间分辨率；
- 支持 `mamba_ssm` 和 `fallback` 后端。

## 5.3 时序块投影（Temporal Chunk Projection）

`LongSpikeStreamEncoderConv` 将 3D 特征按时间维 chunk，再通过 `1x1 Conv2d` 投影并拼接，得到供 2D 解码器使用的多尺度特征。

## 5.4 解码器与输出头

解码端为 UNet 风格：

- 残差块 + 上采样模块 + skip connection；
- 输出层 `ConvLayer(..., out=1)`；
- 可选 `output_activation = identity/sigmoid/softplus`（当前配置默认 `identity`）。

---

## 6. 优化目标（Training Objective）

## 6.1 主损失：Scale-Invariant Loss

令误差 \(e = \hat{Y} - Y\)，则：

\[
L_{si} = \mathbb{E}[e^2] - \lambda (\mathbb{E}[e])^2
\]

对应实现：`model/loss.py::scale_invariant_loss`。

## 6.2 辅助损失：Multi-Scale Gradient Loss（可开关）

对多尺度池化后的误差图做空间梯度，累加绝对值作为边缘一致性约束。  
对应实现：`model/loss.py::MultiScaleGradient`。

## 6.3 优化器与调度

默认配置（`configs/train_s2d_spiketransformer_rtx5090_20260305.json`）：

- Optimizer: `Adam(lr=1e-4, weight_decay=0)`
- Scheduler: `ExponentialLR(gamma=0.5, freq=100 epochs)`
- Epochs: `30`
- Batch Size: `8`

---

## 7. 训练系统实现（Engineering of Training）

## 7.1 启动入口

```bash
CUDA_VISIBLE_DEVICES=0 python train_parallel.py \
  --config configs/train_s2d_spiketransformer_rtx5090_20260305.json \
  --datafolder /path/to/dataset_root
```

可选多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_parallel.py \
  --config configs/train_s2d_spiketransformer_rtx5090_20260305.json \
  --datafolder /path/to/dataset_root \
  --multiprocessing_distributed
```

## 7.2 训练循环

每个 batch：

1. 读取序列 `sequence`；
2. `forward_pass_sequence` 累计每步预测与损失；
3. `loss.backward()`；
4. 可选梯度裁剪 `clip_grad_norm_`；
5. `optimizer.step()`；
6. epoch 结束后写 TensorBoard、保存 checkpoint。

## 7.3 输出产物

`save_dir/name` 下生成：

- `config.json`
- `checkpoint-epochXXX-loss-XXXX.pth.tar`
- `model_best.pth.tar`（按 `monitor=val_loss`）
- `tensorboard/` 事件文件

---

## 8. 从模型到结果（Inference & Evaluation）

## 8.1 推理导出

```bash
CUDA_VISIBLE_DEVICES=0 python test_DENSE.py \
  --path_to_model /path/to/model_best.pth.tar \
  --output_path /path/to/output_dir \
  --data_folder /path/to/DENSE/test_or_validation
```

导出内容（核心）：

- `output_dir/npy/image/depth_XXXXXXXXXX.npy`（预测）
- `output_dir/ground_truth/npy/depth_image/frame_XXXXXXXXXX.npy`（GT）
- 以及可视化图、视频素材目录。

注意：实现中为了让时序状态稳定，每个序列的前两帧默认不写入导出目录。

## 8.2 指标评估

```bash
python evaluation_DENSE.py \
  --target_dataset /path/to/output_dir/ground_truth/npy/depth_image \
  --predictions_dataset /path/to/output_dir/npy/image \
  --clip_distance 1000 \
  --reg_factor 5.7 \
  --output_folder /path/to/eval_vis
```

评估脚本会先把 log 深度反变换到米制深度，再计算：

- `abs_rel_diff`
- `squ_rel_diff`
- `RMS_linear`
- `RMS_log`
- `SILog`
- `mean_depth_error`
- `median_diff`
- `delta<1.25`, `delta<1.25^2`, `delta<1.25^3`

并支持按深度阈值（10/20/30/80/250/500）分桶统计。

---

## 9. 关键配置字段（What Matters Most）

按影响优先级建议优先关注：

1. 数据与标签定义  
   `clip_distance`, `reg_factor`, `spike_folder`, `depth_folder`, `baseline`
2. 时序建模容量  
   `num_bins_rgb`, `swin_patch_size[0]`, `swin_depths`, `base_num_channels`
3. Mamba后端  
   `mamba_backend`, `mamba_require_ssm`, `mamba_d_state/d_conv/expand`
4. 优化稳定性  
   `lr`, `batch_size`, `grad_clip_norm`, `loss_weights`

---

## 10. 可复现协议（Reproducibility Protocol）

## 10.1 环境

```bash
pip install -r requirements.txt
# 或服务端环境
pip install -r requirements.server.txt
```

Mamba 后端：

```bash
pip install mamba-ssm causal-conv1d
```

若环境不兼容，可先在 config 中设置：

```json
{
  "mamba_backend": "fallback",
  "mamba_require_ssm": false
}
```

## 10.2 建议的确定性设置（建议补充）

当前代码只设置了 `torch.manual_seed(111)`，且 `cudnn.benchmark=True`。  
若要发表级复现，建议额外统一：

- `random.seed(...)`
- `np.random.seed(...)`
- `torch.cuda.manual_seed_all(...)`
- `torch.backends.cudnn.deterministic=True`
- 固定 DataLoader worker seed

---

## 11. 常见问题与诊断（FAQ / Failure Modes）

1. **输入维度报错（4D/5D）**  
   检查模型输入是否最终为 `(B,1,T,H,W)`。

2. **评估结果明显异常**  
   优先核对 `clip_distance` 与 `reg_factor` 是否与训练一致。

3. **mamba_ssm 导入失败**  
   使用 fallback 后端先跑通，再回切 `mamba_ssm`。

4. **checkpoint 覆盖或路径冲突**  
   训练脚本已对同名运行自动追加 UTC 时间戳后缀。

---

## 12. 代码地图（Code Map）

- 训练入口：`train_parallel.py`
- Trainer：`trainer/spiket_trainer.py`
- 模型主结构：`model/S2DepthNet.py`
- 3D 编码器：`model/encoder_transformer.py`, `model/mamba_3d.py`
- 损失与指标：`model/loss.py`, `model/metric.py`
- 数据集：`data_loader/SpikesDENSE_dataset.py`, `data_loader/spike_dataset.py`
- 推理：`test_DENSE.py`
- 评估：`evaluation_DENSE.py`

---

## 13. 最小复现实验（Quick Repro Checklist）

1. 准备符合目录规范的数据（含 `timestamps.txt`、`spike_*.npy`、`depth_*.npy`）。  
2. 安装依赖，确认 `torch.cuda.is_available()`。  
3. 用默认配置训练至收敛，得到 `model_best.pth.tar`。  
4. 使用 `test_DENSE.py` 导出预测 `npy`。  
5. 使用 `evaluation_DENSE.py` 计算并记录指标。  
6. 固化 config 与随机种子，存档 `config.json + tensorboard + checkpoint` 以备论文复核。

---

如果你希望把这份文档进一步升级成“论文附录风格”（加入公式编号、实验表格模板、消融实验模板、结果记录表头），可以在此文件基础上继续扩展。
