import torch
import torch.nn as nn
import torch.nn.functional as F

from model.submodules import ConvLayer, ResidualBlock, UpsampleConvLayer


def _masked_l1(prediction, target, mask):
    if mask is None:
        return F.l1_loss(prediction, target)
    valid = mask.expand_as(target)
    if valid.any():
        return torch.abs(prediction[valid] - target[valid]).mean()
    return prediction.new_zeros(())


def _sanitize_depth(depth):
    valid_mask = torch.isfinite(depth)
    clean_depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return clean_depth, valid_mask


def _match_feature_channels(feature, target_channels):
    current_channels = feature.shape[1]
    if current_channels == target_channels:
        return feature

    if current_channels > target_channels:
        if current_channels % target_channels == 0:
            group_size = current_channels // target_channels
            b, _, h, w = feature.shape
            return feature.reshape(b, target_channels, group_size, h, w).mean(dim=2)

        b, _, h, w = feature.shape
        pooled = F.adaptive_avg_pool1d(
            feature.permute(0, 2, 3, 1).reshape(b * h * w, 1, current_channels),
            target_channels,
        )
        return pooled.reshape(b, h, w, target_channels).permute(0, 3, 1, 2)

    repeat_times = (target_channels + current_channels - 1) // current_channels
    return feature.repeat(1, repeat_times, 1, 1)[:, :target_channels, :, :]


def _normalize_feature_statistics(feature):
    mean = feature.mean(dim=(2, 3), keepdim=True)
    std = feature.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
    return (feature - mean) / std


class _DepthDownsampleEncoder(nn.Module):
    def __init__(self, in_channels, base_channels, latent_channels, norm=None):
        super().__init__()
        mid_channels = max(base_channels * 2, latent_channels // 2)
        self.stem = ConvLayer(in_channels, base_channels, 3, stride=2, padding=1, norm=norm)
        self.res1 = ResidualBlock(base_channels, base_channels, norm=norm)
        self.down2 = ConvLayer(base_channels, mid_channels, 3, stride=2, padding=1, norm=norm)
        self.res2 = ResidualBlock(mid_channels, mid_channels, norm=norm)
        self.down3 = ConvLayer(mid_channels, latent_channels, 3, stride=2, padding=1, norm=norm)
        self.res3 = ResidualBlock(latent_channels, latent_channels, norm=norm)

    def forward(self, x):
        x = self.res1(self.stem(x))
        x = self.res2(self.down2(x))
        x = self.res3(self.down3(x))
        return x


class _DepthUpsampleDecoder(nn.Module):
    def __init__(self, latent_channels, base_channels, norm=None):
        super().__init__()
        mid_channels = max(base_channels * 2, latent_channels // 2)
        low_channels = max(base_channels, latent_channels // 4)
        out_channels = max(base_channels // 2, 16)

        self.up1 = UpsampleConvLayer(latent_channels, mid_channels, 3, padding=1, norm=norm)
        self.res1 = ResidualBlock(mid_channels, mid_channels, norm=norm)
        self.up2 = UpsampleConvLayer(mid_channels, low_channels, 3, padding=1, norm=norm)
        self.res2 = ResidualBlock(low_channels, low_channels, norm=norm)
        self.up3 = UpsampleConvLayer(low_channels, out_channels, 3, padding=1, norm=norm)
        self.res3 = ResidualBlock(out_channels, out_channels, norm=norm)
        self.head = ConvLayer(out_channels, 1, 1, activation=None, norm=norm)

    def forward(self, x):
        x = self.res1(self.up1(x))
        x = self.res2(self.up2(x))
        x = self.res3(self.up3(x))
        return self.head(x)


class _ConditionalVelocityField(nn.Module):
    def __init__(self, latent_channels, hidden_channels, num_resblocks=2, norm=None):
        super().__init__()
        in_channels = latent_channels * 2 + 1
        self.stem = ConvLayer(in_channels, hidden_channels, 3, padding=1, norm=norm)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels, hidden_channels, norm=norm) for _ in range(num_resblocks)]
        )
        self.head = ConvLayer(hidden_channels, latent_channels, 3, padding=1, activation=None, norm=norm)

    def forward(self, latent_state, condition_latent, time_value):
        if torch.is_tensor(time_value):
            if time_value.ndim == 0:
                time_map = torch.full(
                    (latent_state.shape[0], 1, latent_state.shape[2], latent_state.shape[3]),
                    float(time_value.item()),
                    dtype=latent_state.dtype,
                    device=latent_state.device,
                )
            else:
                time_map = time_value.to(dtype=latent_state.dtype, device=latent_state.device)
                if time_map.shape[-2:] != latent_state.shape[-2:]:
                    time_map = time_map.expand(-1, -1, latent_state.shape[2], latent_state.shape[3])
        else:
            time_map = torch.full(
                (latent_state.shape[0], 1, latent_state.shape[2], latent_state.shape[3]),
                float(time_value),
                dtype=latent_state.dtype,
                device=latent_state.device,
            )
        x = torch.cat([latent_state, condition_latent, time_map], dim=1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class ConditionedDepthManifoldFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        latent_channels = int(config.get("latent_channels", 96))
        base_channels = int(config.get("base_channels", max(32, latent_channels // 2)))
        hidden_channels = int(config.get("flow_hidden_channels", max(96, latent_channels)))
        flow_resblocks = int(config.get("flow_resblocks", 2))
        flow_steps = int(config.get("flow_steps", 6))

        self.flow_steps = max(1, flow_steps)
        self.blend_alpha = float(config.get("blend_alpha", 0.0))
        self.rollout_noise_std = float(config.get("rollout_noise_std", 0.0))
        self.encoder_feature_scale = float(config.get("encoder_feature_scale", 1.0))
        self.loss_weights = {
            "coarse": float(config.get("coarse_weight", 0.5)),
            "reconstruction": float(config.get("reconstruction_weight", 0.1)),
            "flow": float(config.get("flow_weight", 0.1)),
            "geometry": float(config.get("geometry_weight", 0.05)),
        }

        # Residual-bridge refactor switches (route B; see manifold重构方案.md).
        # All False + legacy config reproduces the original noise-based flow behaviour.
        self.residual_output = bool(config.get("residual_output", True))      # C3: coarse + delta anchor
        self.data_coupling = bool(config.get("data_coupling", True))          # C2: coarse -> gt coupling
        self.share_encoder = bool(config.get("share_encoder", True))          # C4: single shared encoder
        self.start_from_coarse = bool(config.get("start_from_coarse", True))  # C1: rollout starts at coarse latent

        self.condition_encoder = _DepthDownsampleEncoder(1, base_channels, latent_channels)
        if self.share_encoder:
            # Same instance so z_coarse and z_gt live on the same latent manifold (C4).
            self.target_encoder = self.condition_encoder
        else:
            self.target_encoder = _DepthDownsampleEncoder(1, base_channels, latent_channels)
        self.target_decoder = _DepthUpsampleDecoder(latent_channels, base_channels)
        self.velocity_field = _ConditionalVelocityField(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            num_resblocks=flow_resblocks,
        )

    def _fuse_encoder_features(self, condition_latent, encoder_features):
        if self.encoder_feature_scale == 0.0 or encoder_features is None:
            return condition_latent

        if torch.is_tensor(encoder_features):
            encoder_features = [encoder_features]

        target_hw = condition_latent.shape[-2:]
        target_channels = condition_latent.shape[1]
        fused_encoder_latent = None
        num_valid_features = 0

        for feature in encoder_features:
            if feature is None:
                continue
            if not torch.is_tensor(feature):
                raise TypeError(
                    f"Expected encoder feature tensor, got {type(feature)} in manifold flow conditioning."
                )
            if feature.ndim != 4:
                raise ValueError(
                    f"Expected 4D encoder feature map [B,C,H,W], got shape {tuple(feature.shape)}."
                )

            aligned_feature = feature.to(dtype=condition_latent.dtype, device=condition_latent.device)
            if aligned_feature.shape[-2:] != target_hw:
                aligned_feature = F.interpolate(
                    aligned_feature,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned_feature = _match_feature_channels(aligned_feature, target_channels)
            # C5: drop per-image normalization to preserve absolute (metric-depth) scale.

            if fused_encoder_latent is None:
                fused_encoder_latent = aligned_feature
            else:
                fused_encoder_latent = fused_encoder_latent + aligned_feature
            num_valid_features += 1

        if fused_encoder_latent is None:
            return condition_latent

        fused_encoder_latent = fused_encoder_latent / float(num_valid_features)
        return condition_latent + self.encoder_feature_scale * fused_encoder_latent

    def _rollout_latent(self, condition_latent):
        if self.start_from_coarse:
            # C1: start on the manifold at the coarse latent; the flow only refines.
            latent_state = condition_latent
        elif self.rollout_noise_std > 0:
            latent_state = torch.randn_like(condition_latent) * self.rollout_noise_std
        else:
            latent_state = torch.zeros_like(condition_latent)

        step_size = 1.0 / float(self.flow_steps)
        for step in range(self.flow_steps):
            # With data coupling the velocity already points coarse->gt, so t accumulates from 0.
            time_value = step * step_size if self.start_from_coarse else (step + 0.5) * step_size
            velocity = self.velocity_field(latent_state, condition_latent, time_value)
            latent_state = latent_state + velocity * step_size
        return latent_state

    def forward(self, coarse_depth, encoder_features=None, target_depth=None, prev_latent_state=None):
        condition_latent = self.condition_encoder(coarse_depth)
        condition_latent = self._fuse_encoder_features(condition_latent, encoder_features)
        refined_latent = self._rollout_latent(condition_latent)
        decoded = self.target_decoder(refined_latent)
        if self.residual_output:
            # C3: anchor to coarse; the decoder produces a depth increment (delta).
            refined_depth = coarse_depth + decoded
        else:
            refined_depth = decoded
            if self.blend_alpha > 0.0:
                refined_depth = self.blend_alpha * coarse_depth + (1.0 - self.blend_alpha) * refined_depth

        auxiliary_losses = {}
        auxiliary_outputs = {
            "coarse_prediction": coarse_depth,
            "coarse_weight": self.loss_weights["coarse"],
            "encoder_feature_scale": self.encoder_feature_scale,
        }

        target_latent = None
        if target_depth is not None:
            clean_target, valid_mask = _sanitize_depth(target_depth)
            target_latent = self.target_encoder(clean_target)
            reconstructed_target = self.target_decoder(target_latent)

            if self.loss_weights["reconstruction"] > 0.0:
                auxiliary_losses["L_rec"] = (
                    self.loss_weights["reconstruction"]
                    * _masked_l1(reconstructed_target, clean_target, valid_mask)
                )

            if self.data_coupling:
                # C2: straight path from the coarse latent to the GT latent.
                z0 = condition_latent
            else:
                z0 = torch.randn_like(target_latent)
            time_value = torch.rand(
                target_latent.shape[0], 1, 1, 1, device=target_latent.device, dtype=target_latent.dtype
            )
            latent_state = (1.0 - time_value) * z0 + time_value * target_latent
            target_velocity = target_latent - z0
            predicted_velocity = self.velocity_field(latent_state, condition_latent, time_value)
            if self.loss_weights["flow"] > 0.0:
                auxiliary_losses["L_flow"] = (
                    self.loss_weights["flow"] * F.mse_loss(predicted_velocity, target_velocity)
                )

            auxiliary_outputs["target_reconstruction"] = reconstructed_target

        previous_target_latent = None
        previous_pred_latent = None
        if isinstance(prev_latent_state, dict):
            previous_target_latent = prev_latent_state.get("target_latent")
            previous_pred_latent = prev_latent_state.get("pred_latent")
        if (
            target_latent is not None
            and previous_target_latent is not None
            and previous_pred_latent is not None
            and self.loss_weights["geometry"] > 0.0
        ):
            predicted_delta = refined_latent - previous_pred_latent
            target_delta = target_latent - previous_target_latent
            auxiliary_losses["L_geo"] = (
                self.loss_weights["geometry"] * F.smooth_l1_loss(predicted_delta, target_delta)
            )

        next_latent_state = {
            "pred_latent": refined_latent.detach(),
            "target_latent": target_latent.detach() if target_latent is not None else None,
        }

        return refined_depth, auxiliary_losses, auxiliary_outputs, next_latent_state
