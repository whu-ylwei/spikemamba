# SpikeMamba 架构说明（shared-nvme 版本）

本文档基于 `/root/shared-nvme/spikemamba` 当前代码整理，目标是说明项目的分层结构、端到端数据流、关键耦合点和扩展入口。

---

## 1. 项目定位

这是一个典型的科研训练型代码库，核心任务是：

- 输入：脉冲相机事件体（spike voxel）与时间对齐深度监督
- 输出：单目深度预测图
- 形态：离线训练 + 离线推理 + 离线评估（非在线服务架构）

核心入口：

- 训练：`train_parallel.py`
- 推理：`test_DENSE.py`
- 评估：`evaluation_DENSE.py`

---

## 2. 架构分层

## 2.1 编排层（Orchestration）

文件：`train_parallel.py`

职责：

- 读取配置（`--config`）
- 构建 train/validation 数据集与 DataLoader
- 初始化模型、loss、metrics、trainer
- 启动单卡或分布式训练

特征：

- 通过配置驱动实验；
- 使用 `eval(...)` 动态装配模型/损失/指标；
- 支持 checkpoint resume 和初始权重加载。

## 2.2 数据层（Data Layer）

文件：

- `data_loader/spike_dataset.py`
- `data_loader/SpikesDENSE_dataset.py`

职责：

- 读取 `spike_*.npy` 与 `timestamps.txt`
- spike/depth 时间戳对齐
- 构造序列样本（长度 `sequence_length`）
- 深度标签映射：`clip -> normalize -> log space`
- 数据增强（随机翻转/裁剪、中心裁剪）

典型样本结构是 `list[dict]`，字典键包含：

- 输入键：`events{k}`、`image`
- 监督键：`depth_image`、`depth_events{k}`

## 2.3 模型层（Model Layer）

文件：

- `model/S2DepthNet.py`
- `model/encoder_transformer.py`
- `model/mamba_3d.py`
- `model/model.py`

主干结构：

1. `S2DepthTransformerUNetConv`
2. `LongSpikeStreamEncoderConv`
3. `Mamba3DEncoder`
4. UNet 风格 2D decoder + 预测头

关键实现点：

- 输入被规整为 5D 张量 `(B, 1, T, H, W)`；
- 3D 编码器按时间维建模，并仅做空间下采样；
- 时序 chunk 特征经 `1x1 Conv2d` 投影后送入解码器。

## 2.4 训练层（Training Layer）

文件：

- `base/base_trainer.py`
- `trainer/spiket_trainer.py`

职责：

- 优化器/调度器管理
- 训练-验证 epoch 循环
- loss 组合（SI loss + 可选梯度损失 + 可选MSE）
- checkpoint 与 TensorBoard 记录

训练流程：

1. batch 级序列前向
2. 聚合 loss
3. backward + grad clip
4. optimizer step
5. epoch 结束后验证和保存

## 2.5 评估层（Evaluation Layer）

文件：

- `test_DENSE.py`
- `evaluation_DENSE.py`

两阶段流程：

1. 推理脚本导出预测与 GT（npy/png）
2. 评估脚本读取导出结果计算指标

常见指标包含：

- `abs_rel_diff`
- `squ_rel_diff`
- `RMS_linear`
- `SILog`
- `delta` 系列阈值指标

---

## 3. 端到端数据流

```text
配置(JSON)
  -> train_parallel.py
  -> Dataset(时间戳对齐 + 标签映射)
  -> DataLoader(sequence batch)
  -> S2DepthTransformerUNetConv
  -> loss backward + optimizer
  -> checkpoint/tensorboard
  -> test_DENSE.py 导出预测
  -> evaluation_DENSE.py 输出指标
```

---

## 4. 配置驱动与关键文件

主要配置位于 `configs/`，常用包括：

- `train_s2d_spiketransformer.json`
- `train_s2d_spiketransformer_rtx5090_20260305.json`
- `train_s2d_spiketransformer_mambassm_strict_20260306_124138.json`

配置控制三类参数：

1. 数据参数：路径、裁剪距离、batch、workers、baseline
2. 模型参数：`num_bins`、patch/depth、输出头、（可选）mamba相关字段
3. 训练参数：loss、optimizer、scheduler、epochs、日志策略

---

## 5. 当前版本的关键架构特征

1. `mamba_ssm` 采用严格依赖路径  
   `model/mamba_3d.py` 直接导入 `mamba_ssm.modules.mamba_simple.Mamba`，导入失败会抛错终止训练。

2. 输出监督头以 `image` 为主  
   训练配置普遍使用 `loss_composition: "image"`，预测字典也默认以 `"image"` 作为主头。

3. 评估采用“导出后评估”  
   有利于审计和复核，但会产生较大中间文件目录（`output*/` 风格）。

4. 实验资产较集中  
   目录中包含 `runs/train/`、`runs/eval/`、`logs/train/`、`reports/analysis/`、`wheels/`，便于追踪但仓库较重。

---

## 6. 扩展建议（在不破坏主链路前提下）

优先扩展点：

1. 新模型：在 `model/` 新增类并修改 config 的 `arch`
2. 新损失：扩展 `model/loss.py` 并在 config 中切换
3. 新指标：扩展 `model/metric.py` 并加入 `metrics`
4. 新数据源：在 `data_loader/` 新建 Dataset，并替换 `data_loader.*.type`

优先治理点：

1. 减少 `eval(...)` 动态装配，改为显式注册表；
2. 完整设置随机种子与 deterministic 策略，增强可复现性；
3. 进一步分离“训练逻辑”和“可视化/导出逻辑”以降低耦合。

---

## 7. 一句话总结

`shared-nvme` 版本的 SpikeMamba 是一个以实验效率为核心的深度学习研究代码架构：  
链路完整、可快速迭代，但在类型安全、复现治理与脚本解耦方面仍有工程化提升空间。

---

## 8. 详细数据流文档

关于“从磁盘输入到模型前向、损失反传”的逐步说明，请查看：

- `README_DATA_FLOW.md`
