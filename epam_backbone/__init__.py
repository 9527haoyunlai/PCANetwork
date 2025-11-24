"""
EPAM-Net Standalone Backbone Module
Pure PyTorch implementation without mmcv dependencies
"""
from .epam_backbone import EPAMBackbone
from .x3d_temporal_shift_rgb import X3DTemporalShift
from .x3d_temporal_shift_pose import X3DTemporalShiftPose
from .attention_module import CBAMSpatialEfficientTemporalAttention

__all__ = [
    'EPAMBackbone',
    'X3DTemporalShift',
    'X3DTemporalShiftPose',
    'CBAMSpatialEfficientTemporalAttention'
]

__version__ = '1.0.0'
