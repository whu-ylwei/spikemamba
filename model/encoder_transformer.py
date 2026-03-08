import torch
import torch.nn as nn

from .mamba_3d import Mamba3DEncoder


class LongSpikeStreamEncoderConv(nn.Module):
    def __init__(
        self,
        patch_size=(32, 2, 2),
        in_chans=1,
        temporal_bins=128,
        embed_dim=96,
        depths=[2, 2, 6],
        num_heads=[3, 6, 12],
        patch_norm=False,
        out_indices=(0, 1, 2),
        frozen_stages=-1,
        new_version=3,
        mamba_backend="mamba_ssm",
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_require_ssm=False,
    ):
        super(LongSpikeStreamEncoderConv, self).__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.temporal_bins = temporal_bins
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Number of temporal chunks used by the decoder-side 2D projections.
        self.num_blocks = max(1, temporal_bins // patch_size[0])

        self.num_encoders = len(self.depths)
        self.out_channels = [self.embed_dim * (2 ** i) for i in range(self.num_encoders)]

        self.backbone = Mamba3DEncoder(
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            depths=self.depths,
            out_indices=self.out_indices,
            mamba_backend=mamba_backend,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_require_ssm=mamba_require_ssm,
        )

        self.patches_T = self.num_blocks

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_encoders):
            conv_layer_i = nn.ModuleList()
            for _ in range(self.num_blocks):
                conv_layer_i.append(
                    nn.Conv2d(self.out_channels[i], self.out_channels[i] // self.num_blocks, 1)
                )
            self.conv_layers.append(conv_layer_i)

    def forward(self, inputs):
        # inputs: [B, C, T, H, W]
        if inputs.ndim != 5:
            raise ValueError(f"Expected 5D input [B,C,T,H,W], got shape {tuple(inputs.shape)}")

        features = self.backbone(inputs)

        outs = []
        for i in range(self.num_encoders):
            out_layer_i = []
            features_i = features[i].chunk(self.num_blocks, dim=2)
            if len(features_i) != self.num_blocks:
                raise RuntimeError(
                    f"Unexpected temporal chunks: got {len(features_i)}, expected {self.num_blocks}. "
                    "Check temporal input shape and patch_size."
                )

            B, C, T, H, W = features_i[0].shape
            for k in range(self.num_blocks):
                feature_k = features_i[k].reshape(B, -1, H, W)
                out_k = self.conv_layers[i][k](feature_k)
                out_layer_i.append(out_k)
            out_i = torch.cat(out_layer_i, dim=1)
            outs.append(out_i)

        return outs
