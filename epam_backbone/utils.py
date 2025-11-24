"""
Utility functions to replace mmcv dependencies with pure PyTorch implementations
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvModule(nn.Module):
    """
    A conv block that bundles conv/norm/activation layers.
    Replacement for mmcv.cnn.ConvModule

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Kernel size
        stride (int or tuple): Stride. Default: 1
        padding (int or tuple or str): Padding. Default: 0
        dilation (int or tuple): Dilation. Default: 1
        groups (int): Groups. Default: 1
        bias (bool): Whether to use bias. Default: True
        conv_cfg (dict): Config dict for convolution layer. Default: dict(type='Conv3d')
        norm_cfg (dict): Config dict for normalization layer. Default: None
        act_cfg (dict): Config dict for activation layer. Default: dict(type='ReLU')
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()

        # Build conv layer
        conv_type = conv_cfg.get('type', 'Conv3d')
        if conv_type == 'Conv3d':
            self.conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )
        else:
            raise NotImplementedError(f'Conv type {conv_type} not implemented')

        # Build norm layer
        self.norm = None
        if norm_cfg is not None:
            norm_type = norm_cfg.get('type', 'BN3d')
            if norm_type == 'BN3d':
                self.norm = nn.BatchNorm3d(out_channels)
            else:
                raise NotImplementedError(f'Norm type {norm_type} not implemented')

        # Build activation layer
        self.act = None
        if act_cfg is not None:
            self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


def build_activation_layer(act_cfg):
    """
    Build activation layer from config dict

    Args:
        act_cfg (dict): Config dict for activation layer

    Returns:
        nn.Module: Activation layer
    """
    act_type = act_cfg.get('type', 'ReLU')
    inplace = act_cfg.get('inplace', False)

    if act_type == 'ReLU':
        return nn.ReLU(inplace=inplace)
    elif act_type == 'Swish':
        return Swish()
    elif act_type == 'SiLU':  # SiLU is equivalent to Swish
        return nn.SiLU(inplace=inplace)
    else:
        raise NotImplementedError(f'Activation type {act_type} not implemented')


def kaiming_init(module, mode='fan_out', nonlinearity='relu'):
    """
    Initialize module parameters with Kaiming initialization

    Args:
        module (nn.Module): Module to initialize
        mode (str): 'fan_in' or 'fan_out'. Default: 'fan_out'
        nonlinearity (str): Nonlinearity function. Default: 'relu'
    """
    if hasattr(module, 'weight') and module.weight is not None:
        init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, 0)


def constant_init(module, val):
    """
    Initialize module parameters with constant value

    Args:
        module (nn.Module): Module to initialize
        val (float): Constant value
    """
    if hasattr(module, 'weight') and module.weight is not None:
        init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, 0)


def normal_init(module, mean=0, std=1):
    """
    Initialize module parameters with normal distribution

    Args:
        module (nn.Module): Module to initialize
        mean (float): Mean of normal distribution. Default: 0
        std (float): Standard deviation. Default: 1
    """
    if hasattr(module, 'weight') and module.weight is not None:
        init.normal_(module.weight, mean=mean, std=std)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, 0)


def load_checkpoint(model, checkpoint_path, strict=True, map_location='cpu'):
    """
    Load checkpoint from file

    Args:
        model (nn.Module): Model to load checkpoint
        checkpoint_path (str): Path to checkpoint file
        strict (bool): Whether to strictly enforce key matching. Default: True
        map_location (str or dict): Map location. Default: 'cpu'

    Returns:
        dict: Checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict)

    return checkpoint
