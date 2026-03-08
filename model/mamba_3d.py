import inspect
import warnings

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba as MambaSSM
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:
    MambaSSM = None
    _MAMBA_IMPORT_ERROR = exc


class FallbackMamba(nn.Module):
    """Fallback block used when mamba_ssm is unavailable."""

    def __init__(self, dim):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj_out(self.act(self.proj_in(x)))


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _init_mamba_with_compat(dim, d_state, d_conv, expand):
    if MambaSSM is None:
        raise ImportError("mamba_ssm is not available") from _MAMBA_IMPORT_ERROR

    kwargs = {
        "d_model": dim,
        "d_state": d_state,
        "d_conv": d_conv,
        "expand": expand,
    }

    try:
        signature = inspect.signature(MambaSSM.__init__)
        accepts_varkw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not accepts_varkw:
            valid_keys = {k for k in signature.parameters if k != "self"}
            kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    except (TypeError, ValueError):
        signature = None

    try:
        return MambaSSM(**kwargs)
    except TypeError as first_error:
        legacy_kwargs = dict(kwargs)
        legacy_kwargs.pop("d_model", None)
        try:
            return MambaSSM(dim, **legacy_kwargs)
        except TypeError as second_error:
            raise TypeError(
                "Unable to initialize mamba_ssm.Mamba. "
                "Checked kwargs-style and legacy positional constructors."
            ) from second_error


def build_mamba_module(dim, mamba_config=None):
    mamba_config = mamba_config or {}

    backend = str(mamba_config.get("mamba_backend", "mamba_ssm")).strip().lower()
    require_ssm = _to_bool(mamba_config.get("mamba_require_ssm", False))
    d_state = int(mamba_config.get("mamba_d_state", 16))
    d_conv = int(mamba_config.get("mamba_d_conv", 4))
    expand = int(mamba_config.get("mamba_expand", 2))

    if backend not in ("mamba_ssm", "fallback"):
        raise ValueError(
            f"Unsupported mamba_backend='{backend}'. "
            "Use 'mamba_ssm' or 'fallback'."
        )

    if backend == "fallback":
        return FallbackMamba(dim)

    if MambaSSM is None:
        install_hint = (
            "Install with: pip install mamba-ssm causal-conv1d "
            "(match your torch/cuda versions)."
        )
        if require_ssm:
            raise ImportError(
                "mamba_backend is set to 'mamba_ssm' but mamba_ssm import failed. "
                f"{install_hint}"
            ) from _MAMBA_IMPORT_ERROR
        warnings.warn(
            "mamba_ssm import failed; using fallback Mamba block. "
            "Set model.mamba_require_ssm=true to enforce hard failure. "
            + install_hint
        )
        return FallbackMamba(dim)

    return _init_mamba_with_compat(
        dim=dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )


class ST_Mamba_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, mamba_config=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = build_mamba_module(dim, mamba_config=mamba_config)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        # x: B, C, T, H, W -> B, H, W, T, C
        x = x.permute(0, 3, 4, 2, 1)
        b, h, w, t, c = x.shape

        x1 = self.norm1(x)
        x1 = x1.reshape(b * h * w, t, c)
        x1 = self.mamba(x1)
        x1 = x1.reshape(b, h, w, t, c)
        x1 = x1.permute(0, 4, 3, 1, 2)

        x = x.permute(0, 4, 3, 1, 2)
        x = x + x1

        x2 = self.norm2(x.permute(0, 2, 3, 4, 1))
        x2 = self.mlp(x2)
        x2 = x2.permute(0, 4, 1, 2, 3)
        x = x + x2
        return x


class MambaStage(nn.Module):
    def __init__(self, dim, depth, mamba_config=None):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ST_Mamba_Block(dim, mamba_config=mamba_config) for _ in range(depth)]
        )

    def forward(self, x):
        return self.blocks(x)


class Mamba3DEncoder(nn.Module):
    """Drop-in replacement for SwinTransformer3D with Mamba blocks."""

    def __init__(
        self,
        patch_size,
        in_chans,
        embed_dim,
        depths,
        out_indices=(0, 1, 2),
        mamba_backend="mamba_ssm",
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_require_ssm=False,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.patch_embed = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.mamba_config = {
            "mamba_backend": mamba_backend,
            "mamba_d_state": mamba_d_state,
            "mamba_d_conv": mamba_d_conv,
            "mamba_expand": mamba_expand,
            "mamba_require_ssm": mamba_require_ssm,
        }

        self.stages = nn.ModuleList()
        self.channels = []

        dim = embed_dim
        for depth in depths:
            self.channels.append(dim)
            self.stages.append(MambaStage(dim, depth, mamba_config=self.mamba_config))
            dim *= 2

        # Keep temporal resolution; only downsample spatial dimensions.
        self.downsamples = nn.ModuleList(
            [
                nn.Conv3d(
                    self.channels[i],
                    self.channels[i] * 2,
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                )
                for i in range(len(depths) - 1)
            ]
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.patch_embed(x)
        features = []

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                features.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        return features
