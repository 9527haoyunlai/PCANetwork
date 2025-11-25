"""
X3D Temporal Shift Pose Backbone for EPAM-Net
Pure PyTorch implementation without mmcv dependencies
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容相对导入和绝对导入
try:
    from .utils import ConvModule, Swish, build_activation_layer, kaiming_init, constant_init
except ImportError:
    from utils import ConvModule, Swish, build_activation_layer, kaiming_init, constant_init


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module

    Args:
        channels (int): Number of input channels
        reduction (float): Reduction ratio for bottleneck
    """
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


class BlockX3DPose(nn.Module):
    """
    BlockX3D 3D building block for X3D Pose with Temporal Shift Module

    Args:
        inplanes (int): Number of channels for the input
        planes (int): Number of channels for intermediate features
        outplanes (int): Number of channels for the output
        spatial_stride (int): Spatial stride. Default: 1
        downsample (nn.Module | None): Downsample layer. Default: None
        se_ratio (float | None): SE reduction ratio. Default: None
        use_swish (bool): Whether to use swish activation. Default: True
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
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.outplanes = outplanes
        self.spatial_stride = spatial_stride
        self.downsample = downsample
        self.se_ratio = se_ratio
        self.use_swish = use_swish
        self.sta = use_sta

        self.conv1 = ConvModule(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # Depthwise convolution
        self.conv2 = ConvModule(
            in_channels=planes,
            out_channels=planes,
            kernel_size=(1, 3, 3),
            stride=(1, self.spatial_stride, self.spatial_stride),
            padding=(0, 1, 1),
            groups=planes,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.swish = Swish() if self.use_swish else nn.Identity()

        self.conv3 = ConvModule(
            in_channels=planes,
            out_channels=outplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if self.se_ratio is not None:
            self.se_module = SEModule(planes, self.se_ratio)

        if self.sta:
            pass  # Placeholder for spatial-temporal attention

        self.relu = build_activation_layer(act_cfg)

    def temporal_shift(self, x, fold_div=4, inplace=False):
        """
        Temporal Shift operation for pose stream

        Note: fold_div=4 is smaller than RGB (8) because pose has fewer channels

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, T, H, W)
            fold_div (int): Fold division factor. Default: 4
            inplace (bool): Whether to perform inplace operation. Default: False

        Returns:
            torch.Tensor: Shifted tensor of shape (N, C, T, H, W)
        """
        n, c, t, h, w = x.size()
        fold = c // fold_div

        if inplace:
            raise NotImplementedError("Inplace temporal shift not implemented")
        else:
            out = torch.zeros_like(x)
            out[:, :fold, :-1] = x[:, :fold, 1:]  # Shift left
            out[:, fold:2*fold, 1:] = x[:, fold:2*fold, :-1]  # Shift right
            out[:, 2*fold:, :] = x[:, 2*fold:, :]  # No shift
        return out

    def forward(self, x):
        """Forward pass"""
        identity = x
        x_shifted = self.temporal_shift(x)
        out = self.conv1(x_shifted)
        out = self.conv2(out)

        if self.se_ratio is not None:
            out = self.se_module(out)

        out = self.swish(out)
        out = self.conv3(out)

        if self.sta:
            pass  # Placeholder for STA module

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class X3DTemporalShiftPose(nn.Module):
    """
    X3D Temporal Shift Backbone for Pose stream

    This backbone processes skeleton heatmaps (17 joints) with temporal shift
    module for efficient spatiotemporal feature extraction.

    Args:
        gamma_w (float): Global channel width expansion factor. Default: 1.0
        gamma_b (float): Bottleneck channel width expansion factor. Default: 2.25
        gamma_d (float): Network depth expansion factor. Default: 2.2
        in_channels (int): Channel num of input features (17 for skeleton joints). Default: 17
        base_channels (int): Base number of channels. Default: 24
        num_stages (int): Number of stages. Default: 3
        stage_blocks (tuple): Number of blocks per stage. Default: (5, 11, 7)
        spatial_strides (tuple): Spatial strides of residual blocks. Default: (2, 2, 2)
        conv1_stride (int): Stride for first conv layer. Default: 1
        se_style (str): SE module style ('half' or 'all'). Default: 'half'
        se_ratio (float | None): SE reduction ratio. Default: 1/16

    Input Shape:
        - (N, 17, T, H, W) where T=48, H=W=56 (typical)

    Output Shape:
        - (N, feat_dim, T, H', W') where T=48, H'=W'=7 (typical)
          feat_dim = base_channels * gamma_w * gamma_b * (2^(num_stages-1))
    """
    def __init__(self,
                 gamma_w=1.0,
                 gamma_b=2.25,
                 gamma_d=2.2,
                 in_channels=17,
                 base_channels=24,
                 num_stages=3,
                 stage_blocks=(5, 11, 7),
                 spatial_strides=(2, 2, 2),
                 conv1_stride=1,
                 se_style='half',
                 se_ratio=1/16,
                 use_swish=True,
                 use_sta=False):
        super().__init__()
        self.gamma_w = gamma_w
        self.gamma_b = gamma_b
        self.gamma_d = gamma_d
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.stage_blocks = stage_blocks
        self.conv1_stride = conv1_stride

        # Apply channel width multiplier
        self.base_channels = self._round_width(self.base_channels, self.gamma_w)

        print('Pose backbone stage_blocks:', self.stage_blocks)

        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.spatial_strides = spatial_strides
        assert len(spatial_strides) == num_stages

        self.se_style = se_style
        assert self.se_style in ['all', 'half']
        self.se_ratio = se_ratio
        self.use_swish = use_swish
        self.use_sta = use_sta

        self.conv_cfg = dict(type='Conv3d')
        self.norm_cfg = dict(type='BN3d', requires_grad=True)
        self.act_cfg = dict(type='ReLU', inplace=True)

        self.block = BlockX3DPose
        self.stage_blocks = self.stage_blocks[:num_stages]
        self.layer_inplanes = self.base_channels

        # Build stem layer
        self._make_stem_layer()

        # Build residual layers
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            inplanes = self.base_channels * 2 ** i
            planes = int(inplanes * self.gamma_b)

            res_layer = self.make_res_layer(
                self.block,
                self.layer_inplanes,
                inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                se_style=self.se_style,
                se_ratio=self.se_ratio,
                use_swish=self.use_swish,
                use_sta=self.use_sta,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg)

            self.layer_inplanes = inplanes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # Final conv layer
        self.feat_dim = self.base_channels * 2 ** (len(self.stage_blocks) - 1)
        self.conv5 = ConvModule(
            self.feat_dim,
            int(self.feat_dim * self.gamma_b),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.feat_dim = int(self.feat_dim * self.gamma_b)

    @staticmethod
    def _round_width(width, multiplier, min_depth=8, divisor=8):
        """Round width of filters based on width multiplier"""
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
        """Round number of layers based on depth multiplier"""
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def make_res_layer(self,
                       block,
                       layer_inplanes,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       se_style='half',
                       se_ratio=None,
                       use_swish=True,
                       use_sta=False,
                       norm_cfg=None,
                       act_cfg=None,
                       conv_cfg=None):
        """Build residual layer"""
        downsample = None
        if spatial_stride != 1 or layer_inplanes != inplanes:
            downsample = ConvModule(
                layer_inplanes,
                inplanes,
                kernel_size=1,
                stride=(1, spatial_stride, spatial_stride),
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)

        use_se = [False] * blocks
        if se_style == 'all':
            use_se = [True] * blocks
        elif se_style == 'half':
            use_se = [i % 2 == 0 for i in range(blocks)]
        else:
            raise NotImplementedError

        layers = []
        layers.append(
            block(
                layer_inplanes,
                planes,
                inplanes,
                spatial_stride=spatial_stride,
                downsample=downsample,
                se_ratio=se_ratio if use_se[0] else None,
                use_swish=use_swish,
                use_sta=use_sta,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg))

        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    inplanes,
                    spatial_stride=1,
                    se_ratio=se_ratio if use_se[i] else None,
                    use_swish=use_swish,
                    use_sta=use_sta,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg))

        return nn.Sequential(*layers)

    def _make_stem_layer(self):
        """Construct the stem layer"""
        print("Pose backbone conv1_stride:", self.conv1_stride)
        self.conv1_s = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=(1, 3, 3),
            stride=(1, self.conv1_stride, self.conv1_stride),
            padding=(0, 1, 1),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

        # Zero-init residual blocks
        for m in self.modules():
            if isinstance(m, BlockX3DPose):
                if hasattr(m.conv3, 'norm') and m.conv3.norm is not None:
                    constant_init(m.conv3.norm, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input pose heatmap tensor of shape (N, 17, T, H, W)

        Returns:
            torch.Tensor: Output feature tensor of shape (N, feat_dim, T, H', W')
        """
        x = self.conv1_s(x)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        x = self.conv5(x)
        return x


if __name__ == '__main__':
    # Test Pose backbone
    model = X3DTemporalShiftPose(
        gamma_d=1,
        in_channels=17,
        base_channels=24,
        num_stages=3,
        stage_blocks=(5, 11, 7),
        spatial_strides=(2, 2, 2),
        conv1_stride=1
    )
    model.init_weights()

    # Typical input: batch=2, channels=17 (joints), frames=48, height=width=56
    input_tensor = torch.rand(2, 17, 48, 56, 56)
    output = model(input_tensor)
    print(f"Pose Backbone output shape: {output.shape}")
    print(f"Output feature dimension: {model.feat_dim}")
    # Expected output: (2, 216, 48, 7, 7) for default config
