# Standalone X3D Temporal Shift Pose Backbone - No MMCV Dependency
# Pure PyTorch implementation for easy migration
# Original: mmaction/models/backbones/x3dTShiftPose.py

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== Helper Classes (Replace MMCV) ==============

class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvModule(nn.Module):
    """Conv + BN + Activation module (replaces mmcv.cnn.ConvModule)"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

        self.bn = nn.BatchNorm3d(out_channels) if norm_cfg is not None else None

        self.activation = None
        if act_cfg is not None:
            act_type = act_cfg.get('type', 'ReLU')
            if act_type == 'ReLU':
                self.activation = nn.ReLU(inplace=act_cfg.get('inplace', False))
            elif act_type == 'Swish':
                self.activation = Swish()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# ============== Helper Functions ==============

def kaiming_init(module):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def constant_init(module, val):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0)


# ============== SE Module ==============

class SEModule(nn.Module):
    """Squeeze-and-Excitation Module"""

    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.bottleneck = self._round_width(channels, reduction)
        self.fc1 = nn.Conv3d(channels, self.bottleneck, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(self.bottleneck, channels, kernel_size=1)
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
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return module_input * x


# ============== BlockX3D for Pose ==============

class BlockX3DPose(nn.Module):
    """BlockX3D with TSM for Pose stream

    Key difference from RGB: fold_div=4 (more aggressive temporal shift)
    """

    def __init__(self,
                 inplanes,
                 planes,
                 outplanes,
                 spatial_stride=1,
                 downsample=None,
                 se_ratio=None,
                 use_swish=True,
                 norm_cfg=None,
                 act_cfg=None,
                 fold_div=4):  # Pose uses fold_div=4 (RGB uses 8)
        super().__init__()

        self.downsample = downsample
        self.se_ratio = se_ratio
        self.fold_div = fold_div

        # 1x1 expand
        self.conv1 = ConvModule(
            inplanes, planes,
            kernel_size=1, stride=1, padding=0, bias=False,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 3x3 depthwise (spatial only)
        self.conv2 = ConvModule(
            planes, planes,
            kernel_size=(1, 3, 3),
            stride=(1, spatial_stride, spatial_stride),
            padding=(0, 1, 1),
            groups=planes,
            bias=False,
            norm_cfg=norm_cfg, act_cfg=None)

        self.swish = Swish() if use_swish else nn.Identity()

        # 1x1 project
        self.conv3 = ConvModule(
            planes, outplanes,
            kernel_size=1, stride=1, padding=0, bias=False,
            norm_cfg=norm_cfg, act_cfg=None)

        if self.se_ratio is not None:
            self.se_module = SEModule(planes, self.se_ratio)

        self.relu = nn.ReLU(inplace=True)

    def temporal_shift(self, x):
        """TSM with fold_div=4 (shift 25% forward, 25% backward, 50% unchanged)"""
        n, c, t, h, w = x.size()
        fold = c // self.fold_div

        out = torch.zeros_like(x)
        out[:, :fold, :-1] = x[:, :fold, 1:]              # shift left
        out[:, fold:2*fold, 1:] = x[:, fold:2*fold, :-1]  # shift right
        out[:, 2*fold:, :] = x[:, 2*fold:, :]             # no shift

        return out

    def forward(self, x):
        identity = x

        # TSM first
        out = self.temporal_shift(x)

        out = self.conv1(out)
        out = self.conv2(out)

        if self.se_ratio is not None:
            out = self.se_module(out)

        out = self.swish(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


# ============== X3DTemporalShiftPose Backbone ==============

class X3DTemporalShiftPose(nn.Module):
    """X3D Pose Backbone with Temporal Shift Module

    Pure PyTorch implementation without MMCV dependency.

    Key differences from RGB backbone:
    - fold_div=4 (RGB uses 8): More aggressive temporal modeling for sparse pose
    - conv1_stride=1 by default: Preserve spatial resolution for small heatmaps
    - Fewer stages (default 3): Smaller network for pose

    Args:
        gamma_w (float): Width multiplier. Default: 1.
        gamma_b (float): Bottleneck multiplier. Default: 2.25.
        gamma_d (float): Depth multiplier. Default: 2.2.
        in_channels (int): Input channels (17 for body joints). Default: 17.
        base_channels (int): Base channel number. Default: 24.
        num_stages (int): Number of stages. Default: 3.
        stage_blocks (tuple): Blocks per stage. Default: (5, 11, 7).
        spatial_strides (tuple): Spatial strides. Default: (2, 2, 2).
        conv1_stride (int): Stem conv stride. Default: 1.
        fold_div (int): TSM fold division. Default: 4.

    Input shape: (B, 17, T, H, W) - e.g., (1, 17, 48, 56, 56)
    Output shape: (B, feat_dim, T, H', W') - e.g., (1, 216, 48, 7, 7)
    """

    def __init__(self,
                 gamma_w=1.0,
                 gamma_b=2.25,
                 gamma_d=2.2,
                 pretrained=None,
                 in_channels=17,
                 base_channels=24,
                 num_stages=3,
                 stage_blocks=(5, 11, 7),
                 spatial_strides=(2, 2, 2),
                 conv1_stride=1,
                 frozen_stages=-1,
                 se_style='half',
                 se_ratio=1/16,
                 use_swish=True,
                 norm_eval=False,
                 zero_init_residual=True,
                 fold_div=4,
                 **kwargs):
        super().__init__()

        self.gamma_w = gamma_w
        self.gamma_b = gamma_b
        self.gamma_d = gamma_d
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = self._round_width(base_channels, gamma_w)
        self.num_stages = num_stages
        self.stage_blocks = stage_blocks[:num_stages]
        self.spatial_strides = spatial_strides
        self.conv1_stride = conv1_stride
        self.frozen_stages = frozen_stages
        self.se_style = se_style
        self.se_ratio = se_ratio
        self.use_swish = use_swish
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.fold_div = fold_div

        self.norm_cfg = dict(type='BN3d')
        self.act_cfg = dict(type='ReLU', inplace=True)

        # Build stem
        self._make_stem_layer()

        # Build stages
        self.res_layers = []
        self.layer_inplanes = self.base_channels

        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            inplanes = self.base_channels * 2 ** i
            planes = int(inplanes * self.gamma_b)

            res_layer = self._make_res_layer(
                self.layer_inplanes, inplanes, planes, num_blocks,
                spatial_stride=spatial_stride)

            self.layer_inplanes = inplanes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # Final conv
        self.feat_dim = self.base_channels * 2 ** (len(self.stage_blocks) - 1)
        self.conv5 = ConvModule(
            self.feat_dim,
            int(self.feat_dim * self.gamma_b),
            kernel_size=1, stride=1, padding=0, bias=False,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
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

    def _make_stem_layer(self):
        """Stem layer with configurable stride"""
        self.conv1_s = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=(1, 3, 3),
            stride=(1, self.conv1_stride, self.conv1_stride),
            padding=(0, 1, 1),
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _make_res_layer(self, layer_inplanes, inplanes, planes, blocks, spatial_stride=1):
        """Build residual layer"""
        downsample = None
        if spatial_stride != 1 or layer_inplanes != inplanes:
            downsample = ConvModule(
                layer_inplanes, inplanes,
                kernel_size=1,
                stride=(1, spatial_stride, spatial_stride),
                padding=0, bias=False,
                norm_cfg=self.norm_cfg, act_cfg=None)

        use_se = [i % 2 == 0 for i in range(blocks)] if self.se_style == 'half' else [True] * blocks

        layers = []
        # First block
        layers.append(BlockX3DPose(
            layer_inplanes, planes, inplanes,
            spatial_stride=spatial_stride,
            downsample=downsample,
            se_ratio=self.se_ratio if use_se[0] else None,
            use_swish=self.use_swish,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            fold_div=self.fold_div))

        # Remaining blocks
        for i in range(1, blocks):
            layers.append(BlockX3DPose(
                inplanes, planes, inplanes,
                spatial_stride=1,
                se_ratio=self.se_ratio if use_se[i] else None,
                use_swish=self.use_swish,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                fold_div=self.fold_div))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
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
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BlockX3DPose):
                    if m.conv3.bn is not None:
                        constant_init(m.conv3.bn, 0)

        if isinstance(self.pretrained, str):
            logging.info(f'Loading pretrained model from: {self.pretrained}')
            state_dict = torch.load(self.pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_state_dict[k[9:]] = v
                else:
                    new_state_dict[k] = v
            self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        """
        Args:
            x: Pose heatmap volume of shape (B, 17, T, H, W)
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
    # Test the standalone pose backbone
    model = X3DTemporalShiftPose(
        gamma_d=1,
        in_channels=17,
        base_channels=24,
        num_stages=3,
        stage_blocks=(5, 11, 7),
        spatial_strides=(2, 2, 2),
        conv1_stride=1,
        fold_div=4
    )
    model.init_weights()

    # Input: (B, 17joints, T, H, W)
    input_tensor = torch.randn(1, 17, 48, 56, 56)
    output = model(input_tensor)

    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output channels (feat_dim): {model.feat_dim}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
