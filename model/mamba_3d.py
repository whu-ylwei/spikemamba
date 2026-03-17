import torch
import torch.nn as nn
try:
    # Import from modules path to avoid optional top-level dependencies.
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception as exc:
    raise ImportError(
        "mamba_ssm is required and fallback is disabled. "
        "Please install a GPU-compatible mamba_ssm build before training."
    ) from exc


class ST_Mamba_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba_dim = dim * 4
        self.mamba = Mamba(self.mamba_dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    @staticmethod
    def _fold_spatial_tokens(x):
        """
        Convert [B, H, W, T, C] into [B, T * H/2 * W/2, 4C]
        by folding each 2x2 spatial neighborhood into the channel axis.
        """
        B, H, W, T, C = x.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(
                f"Spatial folding requires even H and W, got H={H}, W={W}"
            )

        H2, W2 = H // 2, W // 2
        x = x.reshape(B, H2, 2, W2, 2, T, C)
        x = x.permute(0, 5, 1, 3, 2, 4, 6).contiguous()
        x = x.reshape(B, T * H2 * W2, 4 * C)
        return x, (H, W, T, C)

    @staticmethod
    def _unfold_spatial_tokens(x, shape):
        """
        Convert [B, T * H/2 * W/2, 4C] back into [B, H, W, T, C].
        """
        H, W, T, C = shape
        B = x.shape[0]
        H2, W2 = H // 2, W // 2

        x = x.reshape(B, T, H2, W2, 2, 2, C)
        x = x.permute(0, 2, 4, 3, 5, 1, 6).contiguous()
        x = x.reshape(B, H, W, T, C)
        return x

    def forward(self, x):
        # x: [B, C, T, H, W] -> [B, H, W, T, C]
        x = x.permute(0, 3, 4, 2, 1).contiguous()

        # Fold 2x2 spatial neighborhoods into the channel axis so Mamba
        # sees a sequence of length T * H/2 * W/2 with embedding size 4C.
        x1 = self.norm1(x)
        x1, folded_shape = self._fold_spatial_tokens(x1)
        x1 = self.mamba(x1)
        x1 = self._unfold_spatial_tokens(x1, folded_shape)

        # back to [B, C, T, H, W]
        x1 = x1.permute(0, 4, 3, 1, 2).contiguous()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x + x1

        # FFN
        x2 = self.norm2(x.permute(0, 2, 3, 4, 1))
        x2 = self.mlp(x2)
        x2 = x2.permute(0, 4, 1, 2, 3).contiguous()
        x = x + x2

        return x


class MambaStage(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.blocks = nn.Sequential(*[ST_Mamba_Block(dim) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)


class Mamba3DEncoder(nn.Module):
    """
    drop-in replacement for SwinTransformer3D
    """
    def __init__(self, patch_size, in_chans, embed_dim, depths, out_indices=(0,1,2)):
        super().__init__()
        self.out_indices = out_indices
        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.stages = nn.ModuleList()
        self.channels = []

        dim = embed_dim
        for i, d in enumerate(depths):
            self.channels.append(dim)
            self.stages.append(MambaStage(dim, d))
            dim *= 2

        # Keep temporal resolution; only downsample spatial dimensions.
        self.downsamples = nn.ModuleList([
            nn.Conv3d(
                self.channels[i],
                self.channels[i] * 2,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2)
            )
            for i in range(len(depths) - 1)
        ])

    def forward(self, x):
        """
        x: B,C,H,W  ? after patch_embed ? B,C,T,H,W
        """
        x = self.patch_embed(x)
        features = []

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                features.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        return features
