import torch.nn as nn
import torch
import torch.nn.functional as F

from model.model import BaseERGB2Depth
from model.encoder_transformer import LongSpikeStreamEncoderConv
from model.submodules import ResidualBlock, ConvLayer, UpsampleConvLayer


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


def identity(x1, x2=None):
    return x1


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)



class S2DepthTransformerUNetConv(BaseERGB2Depth):
    def __init__(self, config):
        super(S2DepthTransformerUNetConv, self).__init__(config)
        assert self.base_num_channels % 48 == 0

        self.depths=[int(i) for i in config["swin_depths"]]
        self.num_encoders = len(self.depths)
        self.num_heads=[int(i) for i in config["swin_num_heads"]]
        self.patch_size=[int(i) for i in config["swin_patch_size"]]
        self.out_indices=[int(i) for i in config["swin_out_indices"]]
        self.ape=config["ape"]
        try:
            self.num_v = config["new_v"]
        except KeyError:
            self.num_v = 0
        self.mamba_backend = str(config.get("mamba_backend", "mamba_ssm"))
        self.mamba_d_state = int(config.get("mamba_d_state", 16))
        self.mamba_d_conv = int(config.get("mamba_d_conv", 4))
        self.mamba_expand = int(config.get("mamba_expand", 2))
        self.mamba_require_ssm = _as_bool(config.get("mamba_require_ssm", False))

        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders-1)
        output_activation = str(config.get("output_activation", "identity")).lower()
        if output_activation in ("identity", "none", "linear"):
            self.output_activation = None
        elif output_activation == "sigmoid":
            self.output_activation = torch.sigmoid
        elif output_activation == "softplus":
            self.output_activation = F.softplus
        else:
            raise ValueError(
                f"Unsupported output_activation='{output_activation}'. "
                "Use one of: identity, sigmoid, softplus."
            )
        self.output_scale = float(config.get("output_scale", 1.0))
        self.output_shift = float(config.get("output_shift", 0.0))
        # self.num_channel_spikes = config["num_channel_spikes"]
        self.num_output_channels = 1

        print('----- ', self.num_heads)
        self.encoder = LongSpikeStreamEncoderConv(
            patch_size=self.patch_size,
            # Treat spike bins as temporal dimension for 3D encoder.
            in_chans=1,
            temporal_bins=self.num_bins_rgb,
            embed_dim=self.base_num_channels,
            depths=self.depths,
            num_heads=self.num_heads,
            out_indices=self.out_indices,
            new_version=self.num_v,
            mamba_backend=self.mamba_backend,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            mamba_require_ssm=self.mamba_require_ssm,
        )

        self.UpsampleLayer = UpsampleConvLayer

        if self.skip_type == 'sum':
            self.apply_skip_connection = skip_sum
        elif self.skip_type == 'concat':
            self.apply_skip_connection = skip_concat
        elif self.skip_type == 'no_skip' or self.skip_type is None:
            self.apply_skip_connection = identity
        else:
            raise KeyError('Could not identify skip_type, please add "skip_type":'
                           ' "sum", "concat" or "no_skip" to config["model"]')

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()
    
    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))
    

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels // 2 if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)

    def forward_decoder(self, super_states):
        # last superstate is taken as input for decoder.
        if not bool(self.baseline) and self.state_combination == "convlstm":
            x = super_states[-1][0]
        else:
            x = super_states[-1]
        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            if i == 0:
                x = decoder(x)
                # print(x.shape)
            else:
                if not bool(self.baseline) and self.state_combination == "convlstm":
                    x = decoder(self.apply_skip_connection(x, super_states[self.num_encoders - i - 1][0]))
                else:
                    x = decoder(self.apply_skip_connection(x, super_states[self.num_encoders - i - 1]))
                    # print(x.shape)
            # x = decoder(x)

        # Use a configurable output head to avoid hard sigmoid saturation.
        logits = self.pred(x)
        if self.output_activation is None:
            img = logits
        else:
            img = self.output_activation(logits)
        if self.output_scale != 1.0 or self.output_shift != 0.0:
            img = img * self.output_scale + self.output_shift

        return img

    def _select_spike_tensor(self, item):
        if "depth_image" in item and not any(k in item for k in ("image", "events", "events0")):
            raise RuntimeError(
                "Input dictionary only contains 'depth_image' and no spike/event tensor. "
                "This leaks supervision target into model input and breaks training."
            )

        for key in ("image", "events", "events0"):
            tensor = item.get(key)
            if torch.is_tensor(tensor):
                return tensor.to(self.gpu, non_blocking=True)

        for key in sorted(item.keys()):
            if key.startswith("events") and torch.is_tensor(item[key]):
                return item[key].to(self.gpu, non_blocking=True)

        for key, value in item.items():
            if key.startswith("depth"):
                continue
            if torch.is_tensor(value):
                return value.to(self.gpu, non_blocking=True)

        raise ValueError(f"No usable spike/event tensor found. Available keys: {list(item.keys())}")

    def forward(self, item, prev_super_states, prev_states_lstm):
        # def forward(self, spike_tensor, prev_states=None):
        """
        :param item: N x C x H x W tensor or dict containing the image data
        :return: predicted depth logits/map tensor of size N x 1 x H x W.
        """

        predictions_dict = {}

        if isinstance(item, dict):
            spike_tensor = self._select_spike_tensor(item)
        elif torch.is_tensor(item):
            spike_tensor = item.to(self.gpu, non_blocking=True)
        else:
            raise TypeError(f"Unsupported input type: {type(item)}. Expected dict or tensor.")

        expected_channels = self.num_bins_rgb

        if spike_tensor.ndim == 4:
            actual_channels = spike_tensor.shape[1]
            if actual_channels != expected_channels:
                if actual_channels == 1 and expected_channels > 1:
                    spike_tensor = spike_tensor.repeat(1, expected_channels, 1, 1)
                elif actual_channels > expected_channels:
                    spike_tensor = spike_tensor[:, :expected_channels, :, :]
                else:
                    repeat_times = expected_channels // actual_channels
                    remainder = expected_channels % actual_channels
                    spike_tensor = spike_tensor.repeat(1, repeat_times, 1, 1)
                    if remainder > 0:
                        additional_channels = spike_tensor[:, :remainder, :, :]
                        spike_tensor = torch.cat([spike_tensor, additional_channels], dim=1)
                    spike_tensor = spike_tensor[:, :expected_channels, :, :]
            spike_tensor = spike_tensor.unsqueeze(1)
        elif spike_tensor.ndim == 5:
            # Accept both (B,1,T,H,W) and legacy (B,T,1,H,W).
            if spike_tensor.shape[1] == expected_channels and spike_tensor.shape[2] == 1:
                spike_tensor = spike_tensor.permute(0, 2, 1, 3, 4).contiguous()
            elif spike_tensor.shape[1] != 1 or spike_tensor.shape[2] != expected_channels:
                raise ValueError(
                    f"Unsupported 5D spike tensor shape {tuple(spike_tensor.shape)}; "
                    f"expected (B,1,{expected_channels},H,W) or (B,{expected_channels},1,H,W)."
                )
        else:
            raise ValueError(
                f"Unsupported spike tensor rank {spike_tensor.ndim}; expected 4D or 5D input."
            )

        encoded_xs = self.encoder(spike_tensor)
        prediction = self.forward_decoder(encoded_xs)
        predictions_dict["image"] = prediction

        return predictions_dict, {'image': None}, prev_states_lstm
