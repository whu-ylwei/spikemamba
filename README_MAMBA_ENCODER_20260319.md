# Mamba Encoder Notes

## Scope

This note summarizes the current SpikeMamba encoder path discussed on 2026-03-19:

- how the current Mamba encoder serializes the input
- whether the current Mamba block is unidirectional or bidirectional
- whether the current architecture contains BN or LN
- whether the current serialization order is reasonable

The discussion below refers to the current Mamba-based path, not the older Swin implementation.

## Current Config Context

Current training config uses:

- `num_bins_rgb = 128`
- `spatial_resolution = [112, 112]`
- `swin_depths = [2, 2, 6]`
- `swin_num_heads = [3, 6, 12]`
- `swin_patch_size = [32, 2, 2]`
- `norm = "none"`

Important: the config field names still use the historical `swin_*` naming, but the actual backbone now is `Mamba3DEncoder`.

Relevant files:

- `configs/train_s2d_spiketransformer_mambassm_spatialseq_120ep_20260318.json`
- `model/S2DepthNet.py`
- `model/encoder_transformer.py`
- `model/mamba_3d.py`
- `data_loader/SpikesDENSE_dataset.py`

## 1. Where the Encoder Input Comes From

For the current spike baseline (`baseline == 's'` or `baseline == 'e'`), the dataset writes:

- `item['image'] = events['events']`

So the model input is the spike/event bin tensor, not an RGB image.

Typical input shape before entering the encoder:

```text
[B, 128, 112, 112]
```

Here:

- `128` is the temporal bin dimension
- `112 x 112` is the spatial size

## 2. How the Input Is Reshaped Before Mamba

In `S2DepthNet.forward()`, the model first standardizes the input to 5D:

```text
[B, 128, 112, 112] -> [B, 1, 128, 112, 112]
```

So the encoder sees:

```text
[B, C, T, H, W] = [B, 1, 128, 112, 112]
```

This means:

- the spike bins are treated as the temporal axis `T`
- the input channel count for the 3D backbone is `1`

## 3. Patch Embedding Before Sequence Modeling

Before entering the Mamba blocks, the encoder applies:

```text
Conv3d(in_chans=1, out_chans=96, kernel_size=(32,2,2), stride=(32,2,2))
```

So the input is patchified as:

```text
[B, 1, 128, 112, 112] -> [B, 96, 4, 56, 56]
```

Interpretation:

- temporal axis: `128 / 32 = 4`
- height: `112 / 2 = 56`
- width: `112 / 2 = 56`

At this point the representation is still a 3D feature grid, not yet a flat 1D sequence.

## 4. Where the Actual Serialization Happens

The real serialization happens inside `ST_Mamba_Block`.

### 4.1 Layout Conversion

The block first permutes:

```text
[B, C, T, H, W] -> [B, H, W, T, C]
```

### 4.2 Spatial Folding

Then each local `2 x 2` spatial neighborhood is folded into the channel dimension:

```text
[B, H, W, T, C] -> [B, T * (H/2) * (W/2), 4C]
```

This is the actual Mamba input sequence.

For stage 0:

```text
[B, 96, 4, 56, 56] -> [B, 3136, 384]
```

because:

```text
3136 = 4 * 28 * 28
384 = 4 * 96
```

For later stages, spatial resolution decreases but temporal length stays at `4`.

## 5. What Order the Sequence Actually Uses

The current order is not random flattening.

Because the code permutes to make `T` the leading flattened dimension before the final reshape, the effective ordering is:

```text
(t0, all spatial positions) -> (t1, all spatial positions) -> (t2, all spatial positions) -> ...
```

So the current serialization is best understood as:

- time-major at the coarse block level
- space-major inside each time slice
- with local `2 x 2` spatial context already packed into the token channel

This is important:

- the model does not simply build a pure temporal chain
- it also does not put the same spatial position across all times next to each other

## 6. Is the Current Mamba Unidirectional or Bidirectional?

Current implementation is unidirectional.

Reason:

- the model imports `mamba_ssm.modules.mamba_simple.Mamba`
- each `ST_Mamba_Block` calls `self.mamba(x1)` only once
- there is no reverse scan branch
- there is no forward/reverse fusion

So this is not a bidirectional Mamba encoder.

Important distinction:

- `backward` in the library source refers to gradient backpropagation
- it does not mean a reverse-direction sequence branch

## 7. Does the Current Architecture Have BN or LN?

### 7.1 What Is Active in the Current Training Path

Yes, the current active architecture contains `LayerNorm`.

In the Mamba encoder, every `ST_Mamba_Block` has:

- `norm1 = nn.LayerNorm(dim)`
- `norm2 = nn.LayerNorm(dim)`

So the encoder definitely uses LN.

### 7.2 What Is Not Active

The decoder-side conv/residual modules support:

- `BatchNorm2d`
- `InstanceNorm2d`

But the current config sets:

```text
norm = "none"
```

Therefore, in the current training path:

- encoder: has LN
- decoder / residual conv path: no BN, no IN

### 7.3 Whole Repo vs Current Path

The repository still contains the older Swin code, and that code includes both BN and LN in some places.

But that is not the path used by the current Mamba encoder.

## 8. Is This Serialization Order Reasonable?

Short answer: yes, but it is a compromise and has a clear bias toward spatial modeling.

### Why it is reasonable

1. After patch embedding, the temporal length is only `T = 4`, so pure long-range temporal modeling is limited anyway.
2. Dense depth estimation is highly sensitive to spatial structure, so preserving strong spatial organization is useful.
3. Folding a `2 x 2` neighborhood into the channel dimension gives each token local spatial context before Mamba processes the sequence.

### Main limitation

Because the current Mamba is unidirectional and the sequence order is:

```text
(t0, all space) -> (t1, all space) -> ...
```

the same spatial location across neighboring times is not adjacent in the serialized token stream.

That means:

- temporal continuity is not the easiest relation for the model to capture
- the encoder is likely biased toward learning spatial structure first
- temporal dependence must be carried through longer sequence distance

### Practical conclusion

For the current setup, this design is workable and not obviously wrong.

However, if the goal is to strengthen temporal modeling, this is probably not the best ordering.

Possible stronger alternatives include:

- making tokens from the same spatial location across time adjacent
- applying temporal Mamba and spatial Mamba separately
- using explicit bidirectional Mamba
- alternating temporal-first and spatial-first serialization across blocks

## 9. Decoder-Side Handling

The encoder keeps temporal length `T = 4` across stages after patch embedding.

Before sending features to the decoder, `LongSpikeStreamEncoderConv` does the following for each stage:

1. split the feature map along the temporal axis into 4 chunks
2. reshape each chunk from `[B, C, 1, H, W]` to `[B, C, H, W]`
3. apply a `1 x 1 Conv2d` to each chunk
4. concatenate the projected chunks along the channel axis

So the decoder ultimately receives 2D multi-scale feature maps, not explicit 3D spatiotemporal tensors.

## 10. Shape Summary

Current main path:

```text
dataset events
  [B, 128, 112, 112]
    -> S2DepthNet reshape
  [B, 1, 128, 112, 112]
    -> Conv3d patch embed (32,2,2)
  [B, 96, 4, 56, 56]
    -> ST_Mamba_Block serialization
  [B, 3136, 384]
    -> Mamba
  [B, 3136, 384]
    -> restore 3D feature map
  [B, 96, 4, 56, 56]
    -> multi-stage spatial downsampling
    -> split temporal chunks and project to 2D
    -> decoder
    -> depth output
```

## 11. Final Takeaways

- The current Mamba encoder treats spike bins as the temporal axis.
- Real sequence serialization happens inside each `ST_Mamba_Block`, not at the raw input level.
- Current Mamba is unidirectional, not bidirectional.
- Current active path contains `LayerNorm` in the encoder.
- Current active decoder/residual path does not use BN because `norm = "none"`.
- The current serialization order is reasonable for spatially dominated depth estimation, but it is not the most temporally explicit design.
