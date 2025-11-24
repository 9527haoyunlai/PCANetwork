"""
Attention modules for EPAM-Net
Pure PyTorch implementation without mmcv dependencies
"""
import torch
import torch.nn as nn


class ChannelPool(nn.Module):
    """Channel pooling: concatenates max and mean along channel dimension"""
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    """Basic 3D convolution block with optional batch norm and ReLU"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CBAMSpatialEfficientTemporalAttention(nn.Module):
    """
    CBAM-based Spatial and Efficient Temporal Attention Module

    This module generates spatial-temporal attention maps from pose features
    to guide RGB feature learning.

    Args:
        attention_type (str): Type of attention mechanism. 'nested' for nested
                              spatial-temporal attention, 'serial' for serial attention.
                              Default: 'nested'
        kernel_size (int): Kernel size for spatial convolution. Default: 7

    Input Shape:
        - x: (N, C, T, H, W) - Pose feature tensor
        - x_rgb (optional): (N, C_rgb, T, H, W) - RGB feature tensor (only used when attention_type='serial')

    Output Shape:
        - (N, 1, T, H, W) - Spatial-temporal attention maps

    Example:
        >>> attention = CBAMSpatialEfficientTemporalAttention(attention_type='nested')
        >>> pose_feat = torch.randn(2, 216, 16, 7, 7)
        >>> attention_map = attention(pose_feat)
        >>> print(attention_map.shape)  # torch.Size([2, 1, 16, 7, 7])
    """
    def __init__(self, attention_type='nested', kernel_size=7):
        super().__init__()
        self.attention_type = attention_type
        k = kernel_size

        # Channel pooling layer
        self.compress = ChannelPool()

        # Spatial attention: reduces channel from 2 to 1
        self.spatial = BasicConv(2, 1, (1, k, k), stride=1, padding='same', relu=False)

        # Temporal attention components
        self.gap_nested = nn.AdaptiveAvgPool3d((None, 1, 1))  # Pool spatial dims, keep temporal
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  # Global average pooling
        self.conv_temporal_attention = nn.Conv1d(1, 1, kernel_size=k,
                                                  padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_rgb=None):
        """
        Forward pass

        Args:
            x (torch.Tensor): Pose features of shape (N, C, T, H, W)
            x_rgb (torch.Tensor, optional): RGB features, only used when attention_type='serial'

        Returns:
            torch.Tensor: Attention maps of shape (N, 1, T, H, W)
        """
        bs, c, t, h, w = x.shape

        # Step 1: Generate spatial attention
        x_compress = self.compress(x)  # (N, 2, T, H, W)
        x_out = self.spatial(x_compress)  # (N, 1, T, H, W)
        spatial_attention = self.sigmoid(x_out).view(bs, 1, t, h, w)

        if self.attention_type == 'nested':
            # Nested: Apply temporal attention on spatial attention
            x = self.gap_nested(spatial_attention)  # (N, 1, T, 1, 1)
            x = self.conv_temporal_attention(x.view(bs, 1, t))  # (N, 1, T)
            temporal_attention = self.sigmoid(x).view(bs, 1, t, 1, 1)
            return spatial_attention * temporal_attention
        else:
            # Serial: Apply RGB features with spatial attention then temporal attention
            x_rgb_scaled = x_rgb * spatial_attention
            x_rgb = x_rgb_scaled.transpose(1, 2)  # (N, T, C, H, W)
            x_rgb = self.gap(x_rgb)  # (N, T, 1, 1, 1)
            x_rgb = self.conv_temporal_attention(x_rgb.view(bs, 1, t))  # (N, 1, T)
            x_rgb = self.sigmoid(x_rgb.view(bs, t))
            return x_rgb.view(bs, 1, t, 1, 1)


class SpatialTemporalAttention(nn.Module):
    """
    Spatial-Temporal Attention Module

    Alternative attention mechanism that jointly models spatial and temporal attention.

    Args:
        channels (int): Number of input channels. Default: 256
        temporal_attention_type (str): Type of temporal attention. Default: 'global'

    Input Shape:
        - x: (N, C, T, H, W) - Input feature tensor

    Output Shape:
        - (N, 1, T, H, W) - Spatial-temporal attention maps
    """
    def __init__(self, channels=256, temporal_attention_type='global'):
        super().__init__()
        print(f"SpatialTemporalAttention temporal_attention_type: {temporal_attention_type}")
        self.inter_channels = 16
        self.temporal_attention_type = temporal_attention_type

        self.conv_ch_compress = nn.Conv3d(channels, 1, (1, 3, 3), padding='same')
        self.conv_spatial_attention = nn.Conv3d(1, 1, (1, 7, 7), padding='same')
        self.gap = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc1 = nn.Linear(in_features=16, out_features=self.inter_channels, bias=False)
        self.fc2 = nn.Linear(in_features=self.inter_channels, out_features=16, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input features of shape (N, C, T, H, W)

        Returns:
            torch.Tensor: Attention maps of shape (N, 1, T, H, W)
        """
        bs, c, t, h, w = x.shape

        # Spatial attention
        x_ch_compressed = torch.relu(self.conv_ch_compress(x))
        x = self.conv_spatial_attention(x_ch_compressed)
        spatial_attention = self.sigmoid(x).view(bs, 1, t, h, w)

        # Temporal attention
        pooled_spatial_attention = self.gap(spatial_attention)
        x = torch.relu(self.fc1(pooled_spatial_attention.view(bs, t)))
        x = self.fc2(x)
        temporal_attention = self.sigmoid(x).view(bs, 1, t, 1, 1)

        return spatial_attention * temporal_attention


if __name__ == '__main__':
    # Test CBAM attention
    attention = CBAMSpatialEfficientTemporalAttention(attention_type='nested')
    input_tensor = torch.randn(2, 216, 16, 7, 7)
    output = attention(input_tensor)
    print(f"CBAM Attention output shape: {output.shape}")
    assert output.shape == (2, 1, 16, 7, 7), "Output shape mismatch"

    # Test Spatial-Temporal attention
    st_attention = SpatialTemporalAttention(channels=256)
    input_tensor2 = torch.randn(2, 256, 16, 7, 7)
    output2 = st_attention(input_tensor2)
    print(f"Spatial-Temporal Attention output shape: {output2.shape}")
    assert output2.shape == (2, 1, 16, 7, 7), "Output shape mismatch"

    print("All tests passed!")
