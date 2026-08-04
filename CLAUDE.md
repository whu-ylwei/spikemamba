# SpikeMamba Project Instructions

## Project Overview
SpikeMamba is a spike-based depth estimation model combining spiking neural networks with Mamba (state-space models) as the encoder backbone, using a U-Net style decoder.

## Architecture
- **Encoder**: `model/mamba_3d.py` — `Mamba3DEncoder` with `ST_Mamba_Block` stages
- **Decoder**: `model/S2DepthNet.py` — `S2DepthTransformerUNetConv`
- **Submodules**: `model/submodules.py` — `ResidualBlock` and other building blocks

## Key Design Decision: Residual → MHC

**All residual (skip) connections in the Mamba encoder blocks should use Multi-Head Cross-attention (MHC) instead of simple addition.**

### What this means
In `ST_Mamba_Block` (`model/mamba_3d.py`), the two residual additions:
```python
x = x + x1   # after Mamba SSM
x = x + x2   # after FFN
```
are replaced by `MHCResidual` — a learned multi-head cross-attention module where:
- **query** = the main stream `x`
- **key/value** = the branch output (`x1` or `x2`)

This allows the model to selectively attend to the branch features rather than blindly adding them.

### Implementation
`MHCResidual` is defined in `model/mamba_3d.py`. It uses `nn.MultiheadAttention` with `num_heads=8` (or fewer for small dims), operating on flattened spatial-temporal tokens.

## 关键须知：`spatial_resolution` 是死字段，实际分辨率看 transform

**`config["model"]["spatial_resolution"]`（如 0320 config 里的 `[112, 112]`）在本仓库不喂数据，容易误导。**

- 该字段仅在 `train_parallel.py` / `test_DENSE.py` 的 `if use_phased_arch:` 分支里被用来造 dummy input。0320 主干（`bivmamba_best/checkpoint/model_best_ep70.pth.tar`）内嵌 config `use_phased_arch=False`，该分支从不执行，字段被忽略。
- **实际数据口径由数据增强 transform 决定**：`train_parallel.py` 用 `RandomCrop(224)`(train)/`CenterCrop(224)`(val)，dataset `scale_factor=1`（无下采样），把 260×346 raw 裁成 **224²** 喂主干。
- **结论：0320 主干实际是 224² CenterCrop 训的，不是 112²。** 主干是 Mamba+conv、无 pos-embed，分辨率无关，但特征尺度按 224² crop 场景统计调出。
- 残差桥（route B）实验必须用 `preproc={"mode":"centercrop","size":[224,224]}` 才与 0320 同口径（coarse 在分布内）；用 resize→112² 会让主干 OOD、coarse 变差。别再因为看到 `spatial_resolution=[112,112]` 就以为要喂 112²。

## Coding Conventions
- PyTorch, Python 3.x
- Model files live in `model/`
- Training configs in `configs/` (JSON)
- Do not add docstrings or comments to unchanged code
- Keep changes minimal — only modify what is necessary
