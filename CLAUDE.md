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

## Coding Conventions
- PyTorch, Python 3.x
- Model files live in `model/`
- Training configs in `configs/` (JSON)
- Do not add docstrings or comments to unchanged code
- Keep changes minimal — only modify what is necessary
