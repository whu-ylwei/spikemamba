# SpikeMamba 项目架构 README

本文档聚焦代码架构，不展开实验结果与指标细节。目标是回答三个问题：

1. 项目由哪些层组成；
2. 数据和控制流如何贯穿训练/测试；
3. 哪些位置最关键、最容易改出问题。

---

## 1. 架构总览

这是一个典型的“科研训练脚本型”项目，核心路径为：

`配置 -> 数据集 -> 模型 -> 训练器 -> checkpoint/tensorboard -> 推理导出 -> 评估脚本`

主要入口：

- 训练入口：`train_parallel.py`
- 推理入口：`test_DENSE.py`
- 评估入口：`evaluation_DENSE.py`

---

## 2. 目录与模块职责

### 2.1 核心目录

- `configs/`：实验配置（超参、数据路径、模型参数）
- `data_loader/`：数据读取、时序对齐、标签预处理
- `model/`：网络结构、编码器、损失与指标
- `trainer/`：训练与验证循环
- `base/`：`BaseModel`、`BaseTrainer` 抽象基类
- `utils/`：增强、工具函数、可视化辅助

### 2.2 关键文件

- `train_parallel.py`：实验编排器（组装所有模块）
- `data_loader/SpikesDENSE_dataset.py`：序列样本主逻辑
- `model/S2DepthNet.py`：主网络（Mamba3D encoder + UNet decoder）
- `model/encoder_transformer.py`：3D特征到2D解码特征的桥接
- `model/mamba_3d.py`：Mamba时空编码器
- `trainer/spiket_trainer.py`：前反向、日志、验证
- `test_DENSE.py`：预测导出（npy/png）
- `evaluation_DENSE.py`：离线指标计算

---

## 3. 分层架构

## 3.1 实验编排层（Orchestration）

`train_parallel.py` 负责：

- 读取 JSON 配置；
- 构建 train/validation 数据集与 DataLoader；
- 按 `arch/loss/metrics` 动态实例化对象；
- 初始化单卡或分布式训练；
- 创建 `SpikeTTrainer` 并启动训练。

## 3.2 数据层（Data）

`SpikesDENSE_dataset.py` 负责：

- `spike` 与 `depth` 时间戳对齐；
- 构造序列样本（长度 `sequence_length`）；
- 数据增强（随机翻转、随机裁剪、中心裁剪）；
- 深度标签变换：`clip -> normalize -> log mapping`。

`spike_dataset.py` 负责：

- 底层 `spike_*.npy` 与 `timestamps.txt` 读取；
- spike tensor 归一化；
- 输出基础事件体张量。

## 3.3 模型层（Model）

`S2DepthTransformerUNetConv`：

- 输入规整为 5D：`(B,1,T,H,W)`；
- 编码器：`LongSpikeStreamEncoderConv`（内部用 `Mamba3DEncoder`）；
- 解码器：UNet 风格上采样 + skip connection；
- 输出：单通道深度图（默认 `output_activation=identity`）。

`mamba_3d.py`：

- `ST_Mamba_Block` 做时空 token 建模；
- 支持 `mamba_ssm` 或 `fallback` 后端；
- stage 间只下采样空间，不下采样时间。

## 3.4 训练层（Training）

`BaseTrainer` 提供：

- optimizer / scheduler 初始化；
- checkpoint 保存与恢复；
- tensorboard 写入。

`SpikeTTrainer` 实现：

- 序列级前向 `forward_pass_sequence`；
- 组合损失（SI loss + 可选梯度损失）；
- train/validation epoch 循环；
- 预览图、直方图、梯度统计记录。

## 3.5 评估层（Evaluation）

采用两阶段：

1. `test_DENSE.py` 导出预测与GT文件；
2. `evaluation_DENSE.py` 读取导出结果并计算指标。

优点是评估与训练解耦，方便复核与复现实验。

---

## 4. 训练数据流（关键链路）

1. `train_parallel.py` 从配置读取数据路径与参数；
2. `SequenceSynchronizedFramesSpikesDENSEDataset` 产出 `sequence[list[dict]]`；
3. `DataLoader` batch 后传给 `SpikeTTrainer`；
4. `SpikeTTrainer.forward_pass_sequence` 调用模型前向；
5. `model/S2DepthNet.py` 输出 `predictions_dict["image"]`；
6. 计算损失并反传；
7. epoch 结束保存 checkpoint 和 tensorboard。

---

## 5. 配置驱动点（Config-Driven Design）

配置文件控制三类参数：

- 数据参数：路径、裁剪距离、batch、worker、baseline
- 模型参数：`num_bins`、patch、depths、mamba后端与维度
- 训练参数：loss、lr、scheduler、epochs、grad clip、日志策略

代表配置：

- `configs/train_s2d_spiketransformer.json`
- `configs/train_s2d_spiketransformer_rtx5090_20260305.json`

---

## 6. 架构优点与风险

## 6.1 优点

- 链路完整：训练、推理、评估闭环齐全；
- 配置化程度较高，改实验方便；
- 模型与数据逻辑已经拆分为独立模块。

## 6.2 风险

- `eval(...)` 动态装配较多，类型安全和可审计性弱；
- 训练脚本职责较重，易持续膨胀；
- 复现性配置（随机种子、确定性）仍可进一步系统化。

---

## 7. 建议的扩展入口

优先从以下位置扩展，不易破坏主链路：

1. 新模型：在 `model/` 增加类，并在 config 中替换 `arch`
2. 新损失：在 `model/loss.py` 增加函数，更新 config 的 `loss.type`
3. 新指标：在 `model/metric.py` 增加函数，加入 `metrics` 列表
4. 新数据源：在 `data_loader/` 新增 Dataset，替换 `data_loader.*.type`

---

## 8. 一句话总结

该项目架构本质上是一个“以实验效率为优先”的深度学习研究框架：  
分层清晰、可快速迭代，适合论文实验推进；若面向长期工程化，需要继续加强类型约束、可复现治理和模块边界。
