# Standalone X3D Temporal Shift Backbone - No MMCV Dependency
# Pure PyTorch implementation for easy migration
# Original: mmaction/models/backbones/x3dTemporalshift.py

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# ============== Helper Classes (Replace MMCV) ==============

class Swish(nn.Module):
    """Swish activation function (equivalent to SiLU in PyTorch 1.7+)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvModule(nn.Module):
    """
    A conv block that bundles conv/norm/activation layers.
    Replaces mmcv.cnn.ConvModule
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=False,
                 conv_cfg=None,  # ignored, always Conv3d
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()

        # Conv layer
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

        # Norm layer
        self.bn = None
        if norm_cfg is not None:
            self.bn = nn.BatchNorm3d(out_channels)

        # Activation layer
        self.activation = None
        if act_cfg is not None:
            act_type = act_cfg.get('type', 'ReLU')
            if act_type == 'ReLU':
                self.activation = nn.ReLU(inplace=act_cfg.get('inplace', False))
            elif act_type == 'Swish':
                self.activation = Swish()
            elif act_type == 'SiLU':
                self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# ============== Helper Functions (Replace MMCV init) ==============

def kaiming_init(module, mode='fan_out', nonlinearity='relu'):
    """Kaiming initialization"""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def constant_init(module, val):
    """Constant initialization"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0)

def normal_init(module, mean=0, std=0.01):
    """Normal initialization"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0)


# ============== SE Module ==============

class SEModule(nn.Module):
    """Squeeze-and-Excitation Module"""

    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.bottleneck = self._round_width(channels, reduction)
        self.fc1 = nn.Conv3d(channels, self.bottleneck, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(self.bottleneck, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _round_width(width, multiplier, min_width=8, divisor=8):
        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


# ============== BlockX3D ==============

class BlockX3D(nn.Module):
    """BlockX3D with Temporal Shift Module (TSM)

    Structure: TSM -> 1x1 Conv (expand) -> 3x3 DW Conv -> SE -> Swish -> 1x1 Conv (project) -> Residual
    """

    def __init__(self,
                 inplanes,
                 planes,
                 outplanes,
                 spatial_stride=1,
                 downsample=None,
                 se_ratio=None,
                 use_swish=True,
                 use_sta=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 with_cp=False,
                 fold_div=8):  # TSM fold division
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.outplanes = outplanes
        self.spatial_stride = spatial_stride
        self.downsample = downsample
        self.se_ratio = se_ratio
        self.use_swish = use_swish
        self.with_cp = with_cp
        self.fold_div = fold_div

        # 1x1 expand conv
        self.conv1 = ConvModule(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # 3x3 depthwise conv (spatial only)
        self.conv2 = ConvModule(
            in_channels=planes,
            out_channels=planes,
            kernel_size=(1, 3, 3),
            stride=(1, self.spatial_stride, self.spatial_stride),
            padding=(0, 1, 1),
            groups=planes,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.swish = Swish()

        # 1x1 project conv
        self.conv3 = ConvModule(
            in_channels=planes,
            out_channels=outplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if self.se_ratio is not None:
            self.se_module = SEModule(planes, self.se_ratio)

        self.relu = nn.ReLU(inplace=True)

    def temporal_shift(self, x, fold_div=8):
        """Temporal Shift Module - Zero parameters, zero FLOPs

        Shifts 1/fold_div channels forward, 1/fold_div backward, rest unchanged
        """
        n, c, t, h, w = x.size()
        fold = c // fold_div

        out = torch.zeros_like(x)
        out[:, :fold, :-1] = x[:, :fold, 1:]              # shift left (future -> current)
        out[:, fold:2*fold, 1:] = x[:, fold:2*fold, :-1]  # shift right (past -> current)
        out[:, 2*fold:, :] = x[:, 2*fold:, :]             # no shift

        return out

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            # Apply TSM first
            out = self.temporal_shift(x, self.fold_div)

            # Inverted bottleneck
            out = self.conv1(out)        # expand
            out = self.conv2(out)        # depthwise

            if self.se_ratio is not None:
                out = self.se_module(out)  # SE attention

            out = self.swish(out)
            out = self.conv3(out)        # project

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity  # residual connection
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        return out


# ============== X3DTemporalShift Backbone ==============

class X3DTemporalShift(nn.Module):
    """X3D Backbone with Temporal Shift Module

    Pure PyTorch implementation without MMCV dependency.

    Args:
        gamma_w (float): Global channel width expansion factor. Default: 1.
        gamma_b (float): Bottleneck channel width expansion factor. Default: 1.
        gamma_d (float): Network depth expansion factor. Default: 1.
        pretrained (str | None): Path to pretrained model. Default: None.
        in_channels (int): Input channels. Default: 3.
        num_stages (int): Number of stages. Default: 4.
        spatial_strides (tuple): Spatial strides for each stage. Default: (2, 2, 2, 2).
        se_style (str): SE module style, 'half' or 'all'. Default: 'half'.
        se_ratio (float): SE reduction ratio. Default: 1/16.
        fold_div (int): TSM fold division ratio. Default: 8.

    Input shape: (B, C, T, H, W) - e.g., (1, 3, 16, 224, 224)
    Output shape: (B, feat_dim, T, H/32, W/32) - e.g., (1, 432, 16, 7, 7)
    """

    def __init__(self,
                 gamma_w=1.0,
                 gamma_b=1.0,
                 gamma_d=1.0,
                 pretrained=None,
                 in_channels=3,
                 num_stages=4,
                 stage_blocks=None,
                 spatial_strides=(2, 2, 2, 2),
                 frozen_stages=-1,
                 se_style='half',
                 se_ratio=1/16,
                 use_swish=True,
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 fold_div=8,
                 **kwargs):
        super().__init__()

        self.gamma_w = gamma_w
        self.gamma_b = gamma_b
        self.gamma_d = gamma_d
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = 24
        self.num_stages = num_stages
        self.spatial_strides = spatial_strides
        self.frozen_stages = frozen_stages
        self.se_style = se_style
        self.se_ratio = se_ratio
        self.use_swish = use_swish
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.fold_div = fold_div

        # Calculate stage blocks
        self.stage_blocks = stage_blocks
        if self.stage_blocks is None:
            self.stage_blocks = [1, 2, 5, 3]
            self.stage_blocks = [self._round_repeats(x, self.gamma_d) for x in self.stage_blocks]

        # Apply gamma_w to base channels
        self.base_channels = self._round_width(self.base_channels, self.gamma_w)

        self.stage_blocks = self.stage_blocks[:num_stages]
        self.layer_inplanes = self.base_channels

        # Config dicts for ConvModule
        self.conv_cfg = dict(type='Conv3d')
        self.norm_cfg = dict(type='BN3d', requires_grad=True)
        self.act_cfg = dict(type='ReLU', inplace=True)

        # Build stem layer
        self._make_stem_layer()

        # Build residual layers
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            inplanes = self.base_channels * 2 ** i
            planes = int(inplanes * self.gamma_b)

            res_layer = self._make_res_layer(
                self.layer_inplanes,
                inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride
            )
            self.layer_inplanes = inplanes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # Final conv
        self.feat_dim = self.base_channels * 2 ** (len(self.stage_blocks) - 1)
        self.conv5 = ConvModule(
            self.feat_dim,
            int(self.feat_dim * self.gamma_b),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.feat_dim = int(self.feat_dim * self.gamma_b)

    @staticmethod
    def _round_width(width, multiplier, min_depth=8, divisor=8):
        if not multiplier:
            return width
        width *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(width + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    @staticmethod
    def _round_repeats(repeats, multiplier):
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _make_stem_layer(self):
        """Stem: spatial conv (no temporal conv as TSM handles temporal)"""
        self.conv1_s = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _make_res_layer(self, layer_inplanes, inplanes, planes, blocks, spatial_stride=1):
        """Build a residual layer"""
        downsample = None
        if spatial_stride != 1 or layer_inplanes != inplanes:
            downsample = ConvModule(
                layer_inplanes,
                inplanes,
                kernel_size=1,
                stride=(1, spatial_stride, spatial_stride),
                padding=0,
                bias=False,
                norm_cfg=self.norm_cfg,
                act_cfg=None)

        # SE style: half means every other block has SE
        use_se = [i % 2 == 0 for i in range(blocks)] if self.se_style == 'half' else [True] * blocks

        layers = []
        # First block (with downsample)
        layers.append(BlockX3D(
            layer_inplanes,
            planes,
            inplanes,
            spatial_stride=spatial_stride,
            downsample=downsample,
            se_ratio=self.se_ratio if use_se[0] else None,
            use_swish=self.use_swish,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            with_cp=self.with_cp,
            fold_div=self.fold_div))

        # Remaining blocks
        for i in range(1, blocks):
            layers.append(BlockX3D(
                inplanes,
                planes,
                inplanes,
                spatial_stride=1,
                se_ratio=self.se_ratio if use_se[i] else None,
                use_swish=self.use_swish,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                with_cp=self.with_cp,
                fold_div=self.fold_div))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        """Freeze stages for fine-tuning"""
        if self.frozen_stages >= 0:
            self.conv1_s.eval()
            for param in self.conv1_s.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        """Initialize weights"""
        if isinstance(self.pretrained, str):
            logging.info(f'Loading pretrained model from: {self.pretrained}')
            state_dict = torch.load(self.pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            # Remove 'backbone.' prefix if exists
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_state_dict[k[9:]] = v
                else:
                    new_state_dict[k] = v
            self.load_state_dict(new_state_dict, strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BlockX3D):
                        if m.conv3.bn is not None:
                            constant_init(m.conv3.bn, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        Returns:
            Feature tensor of shape (B, feat_dim, T, H', W')
        """
        x = self.conv1_s(x)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        x = self.conv5(x)
        return x

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()


# ============== Test ==============

if __name__ == '__main__':
    # Test the standalone backbone
    model = X3DTemporalShift(
        gamma_w=1,
        gamma_b=2.25,
        gamma_d=2.2,
        se_style='half',
        fold_div=8
    )
    model.init_weights()

    # Input: (B, C, T, H, W)
    input_tensor = torch.randn(1, 3, 16, 224, 224)
    output = model(input_tensor)

    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output channels (feat_dim): {model.feat_dim}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
