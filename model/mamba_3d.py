import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except Exception:
    class Mamba(nn.Module):
        """Fallback when mamba_ssm is unavailable."""

        def __init__(self, dim):
            super().__init__()
            self.proj_in = nn.Linear(dim, dim)
            self.act = nn.GELU()
            self.proj_out = nn.Linear(dim, dim)

        def forward(self, x):
            return self.proj_out(self.act(self.proj_in(x)))


class ST_Mamba_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        # x: B, C, T, H, W  ? convert to B,H,W,T,C
        x = x.permute(0, 3, 4, 2, 1)      # B,H,W,T,C
        B,H,W,T,C = x.shape

        # --- Mamba on Temporal dimension ---
        x1 = self.norm1(x)
        x1 = x1.reshape(B*H*W, T, C)
        x1 = self.mamba(x1)
        x1 = x1.reshape(B, H, W, T, C)

        # back to B,C,T,H,W
        x1 = x1.permute(0, 4, 3, 1, 2)

        x = x.permute(0, 4, 3, 1, 2)     # same shape as x1
        x = x + x1                       # residual

        # FFN
        x2 = self.norm2(x.permute(0,2,3,4,1))
        x2 = self.mlp(x2)
        x2 = x2.permute(0,4,1,2,3)
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
