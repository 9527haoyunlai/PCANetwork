# Standalone X3D Temporal Shift Backbones
# Pure PyTorch implementation - No MMCV Dependency
# Compatible with any PyTorch version >= 1.7

from .x3d_temporal_shift_standalone import X3DTemporalShift
from .x3d_temporal_shift_pose_standalone import X3DTemporalShiftPose

__all__ = ['X3DTemporalShift', 'X3DTemporalShiftPose']

# ============== Usage Example ==============
"""
# Import the standalone backbones
from standalone_backbones import X3DTemporalShift, X3DTemporalShiftPose

# RGB Backbone
rgb_backbone = X3DTemporalShift(
    gamma_w=1, gamma_b=2.25, gamma_d=2.2,
    se_style='half', fold_div=8
)
rgb_backbone.init_weights()

# Pose Backbone
pose_backbone = X3DTemporalShiftPose(
    gamma_d=1, in_channels=17, base_channels=24,
    num_stages=3, stage_blocks=(5, 11, 7),
    spatial_strides=(2, 2, 2), conv1_stride=1, fold_div=4
)
pose_backbone.init_weights()

# Forward pass
rgb_input = torch.randn(1, 3, 16, 224, 224)   # RGB video
pose_input = torch.randn(1, 17, 48, 56, 56)   # Pose heatmaps

rgb_feats = rgb_backbone(rgb_input)   # [B, 432, 16, 7, 7]
pose_feats = pose_backbone(pose_input) # [B, 216, 48, 7, 7]
"""
