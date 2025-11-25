"""
EPAM-Net Standalone Backbone Module
Pure PyTorch implementation without mmcv dependencies
"""
import sys
import os

# 确保当前目录在路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .epam_backbone import EPAMBackbone
    from .x3d_temporal_shift_rgb import X3DTemporalShift
    from .x3d_temporal_shift_pose import X3DTemporalShiftPose
    from .attention_module import CBAMSpatialEfficientTemporalAttention
except ImportError:
    from epam_backbone import EPAMBackbone
    from x3d_temporal_shift_rgb import X3DTemporalShift
    from x3d_temporal_shift_pose import X3DTemporalShiftPose
    from attention_module import CBAMSpatialEfficientTemporalAttention

__all__ = [
    'EPAMBackbone',
    'X3DTemporalShift',
    'X3DTemporalShiftPose',
    'CBAMSpatialEfficientTemporalAttention'
]

__version__ = '1.0.0'
