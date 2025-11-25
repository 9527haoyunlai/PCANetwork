"""
EPAM-Net Backbone: Integrated RGB-Pose Dual-Stream with Attention Fusion
Pure PyTorch implementation without mmcv dependencies
"""
import torch
import torch.nn as nn

# 兼容相对导入和绝对导入
try:
    from .x3d_temporal_shift_rgb import X3DTemporalShift
    from .x3d_temporal_shift_pose import X3DTemporalShiftPose
    from .attention_module import CBAMSpatialEfficientTemporalAttention
    from .utils import kaiming_init, constant_init
except ImportError:
    from x3d_temporal_shift_rgb import X3DTemporalShift
    from x3d_temporal_shift_pose import X3DTemporalShiftPose
    from attention_module import CBAMSpatialEfficientTemporalAttention
    from utils import kaiming_init, constant_init


class EPAMBackbone(nn.Module):
    """
    EPAM-Net Backbone: Efficient Pose-driven Attention-guided Multimodal Network

    This backbone extracts multimodal features from RGB videos and skeleton pose sequences,
    using pose-driven attention to guide RGB feature learning. The architecture consists of:
    1. RGB Stream: X3D with Temporal Shift for efficient 3D CNN
    2. Pose Stream: X3D with Temporal Shift processing skeleton heatmaps
    3. Attention Module: CBAM-based spatial-temporal attention for feature fusion

    Args:
        num_classes (int): Number of action classes. Default: 60
        rgb_pretrained (str | None): Path to pretrained RGB model. Default: None
        pose_pretrained (str | None): Path to pretrained Pose model. Default: None
        attention_type (str): Type of attention ('CBAM_spatial_efficient_temporal',
                             'spatial_temporal', or 'self_attention').
                             Default: 'CBAM_spatial_efficient_temporal'
        freeze_rgb (bool): Whether to freeze RGB backbone. Default: False
        freeze_pose (bool): Whether to freeze Pose backbone. Default: False
        return_both_streams (bool): If True, returns both RGB and Pose features separately.
                                   If False, returns fused features. Default: True

    Input:
        - rgb_videos: (N, 3, 16, 224, 224) - RGB video frames
        - pose_heatmaps: (N, 17, 48, 56, 56) - Skeleton pose heatmaps (17 joints)

    Output:
        If return_both_streams=True:
            - rgb_features: (N, 432, 16, 7, 7) - RGB stream features
            - pose_features: (N, 216, 48, 7, 7) - Pose stream features
        If return_both_streams=False:
            - fused_features: (N, 432, 16, 7, 7) - Attention-guided RGB features

    Example:
        >>> backbone = EPAMBackbone(num_classes=60, attention_type='CBAM_spatial_efficient_temporal')
        >>> backbone.init_weights()
        >>> rgb = torch.randn(2, 3, 16, 224, 224)
        >>> pose = torch.randn(2, 17, 48, 56, 56)
        >>> rgb_feat, pose_feat = backbone(rgb, pose)
        >>> print(f"RGB features: {rgb_feat.shape}, Pose features: {pose_feat.shape}")
    """

    def __init__(self,
                 num_classes=60,
                 rgb_pretrained=None,
                 pose_pretrained=None,
                 attention_type='CBAM_spatial_efficient_temporal',
                 freeze_rgb=False,
                 freeze_pose=False,
                 return_both_streams=True):
        super().__init__()

        self.num_classes = num_classes
        self.rgb_pretrained = rgb_pretrained
        self.pose_pretrained = pose_pretrained
        self.attention_type = attention_type
        self.return_both_streams = return_both_streams

        # RGB Backbone: X3D with Temporal Shift
        self.rgb_backbone = X3DTemporalShift(
            gamma_w=1,
            gamma_b=2.25,
            gamma_d=2.2,
            in_channels=3,
            use_sta=False,
            se_style='half'
        )

        # Pose Backbone: X3D with Temporal Shift for skeleton heatmaps
        self.pose_backbone = X3DTemporalShiftPose(
            gamma_d=1,
            in_channels=17,
            base_channels=24,
            num_stages=3,
            stage_blocks=(5, 11, 7),
            spatial_strides=(2, 2, 2),
            conv1_stride=1
        )

        # Attention Module
        self.attention_module = None
        if attention_type == 'CBAM_spatial_efficient_temporal':
            self.attention_module = CBAMSpatialEfficientTemporalAttention(attention_type='nested')
        elif attention_type == 'spatial_temporal':
            from .attention_module import SpatialTemporalAttention
            self.attention_module = SpatialTemporalAttention(channels=216)
        elif attention_type == 'self_attention':
            raise NotImplementedError("Self-attention not implemented in standalone version")
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        print(f"EPAM Backbone initialized with attention type: {attention_type}")
        print(f"RGB feature dim: {self.rgb_backbone.feat_dim}")
        print(f"Pose feature dim: {self.pose_backbone.feat_dim}")

        # Freeze backbones if specified
        if freeze_rgb:
            self._freeze_backbone(self.rgb_backbone)
            print("RGB backbone frozen")
        if freeze_pose:
            self._freeze_backbone(self.pose_backbone)
            print("Pose backbone frozen")

    def _freeze_backbone(self, backbone):
        """Freeze all parameters in a backbone"""
        for param in backbone.parameters():
            param.requires_grad = False

    def init_weights(self):
        """
        Initialize weights for all modules

        If pretrained paths are provided, loads them. Otherwise, uses default initialization.
        """
        # Initialize RGB backbone
        if self.rgb_pretrained:
            print(f"Loading RGB pretrained weights from: {self.rgb_pretrained}")
            checkpoint = torch.load(self.rgb_pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Filter out keys that don't belong to backbone
            backbone_state = {}
            for k, v in state_dict.items():
                if 'rgb_backbone' in k or 'backbone' in k:
                    new_key = k.replace('rgb_backbone.', '').replace('backbone.', '')
                    backbone_state[new_key] = v
                elif not any(x in k for x in ['cls_head', 'fc', 'classifier']):
                    backbone_state[k] = v

            self.rgb_backbone.load_state_dict(backbone_state, strict=False)
            print("RGB pretrained weights loaded successfully")
        else:
            self.rgb_backbone.init_weights()

        # Initialize Pose backbone
        if self.pose_pretrained:
            print(f"Loading Pose pretrained weights from: {self.pose_pretrained}")
            checkpoint = torch.load(self.pose_pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Filter out keys that don't belong to backbone
            backbone_state = {}
            for k, v in state_dict.items():
                if 'pose_backbone' in k or 'backbone' in k:
                    new_key = k.replace('pose_backbone.', '').replace('backbone.', '')
                    backbone_state[new_key] = v
                elif not any(x in k for x in ['cls_head', 'fc', 'classifier']):
                    backbone_state[k] = v

            self.pose_backbone.load_state_dict(backbone_state, strict=False)
            print("Pose pretrained weights loaded successfully")
        else:
            self.pose_backbone.init_weights()

        # Initialize attention module
        for m in self.attention_module.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv1d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                constant_init(m, 1)

    def forward(self, rgb_videos, pose_heatmaps):
        """
        Forward pass through EPAM backbone

        Args:
            rgb_videos (torch.Tensor): RGB video tensor of shape (N, 3, T_rgb, H, W)
                                      Typical: (N, 3, 16, 224, 224)
            pose_heatmaps (torch.Tensor): Pose heatmap tensor of shape (N, 17, T_pose, H_p, W_p)
                                         Typical: (N, 17, 48, 56, 56)

        Returns:
            If return_both_streams=True:
                tuple: (rgb_features, pose_features)
                    - rgb_features: Attention-guided RGB features (N, 432, 16, 7, 7)
                    - pose_features: Pose features (N, 216, 48, 7, 7)
            If return_both_streams=False:
                torch.Tensor: Fused RGB features (N, 432, 16, 7, 7)
        """
        # Extract features from both streams
        rgb_feats = self.rgb_backbone(rgb_videos)  # (N, 432, 16, 7, 7)
        pose_feats = self.pose_backbone(pose_heatmaps)  # (N, 216, 48, 7, 7)

        # Temporal alignment: downsample pose features from 48 to 16 frames
        # Sample every 3rd frame to match RGB temporal dimension
        time_strided_inds = [i for i in range(0, 48, 3)]
        time_strided_pose_feats = torch.index_select(
            pose_feats, 2, torch.tensor(time_strided_inds, device=pose_feats.device)
        )  # (N, 216, 16, 7, 7)

        # Generate attention maps from pose features
        attention_maps = self.attention_module(time_strided_pose_feats)  # (N, 1, 16, 7, 7)

        # Apply attention to RGB features
        rgb_attended_feats = rgb_feats * attention_maps  # (N, 432, 16, 7, 7)

        # Use skip connection for robust feature learning
        rgb_fused = rgb_feats + rgb_attended_feats  # (N, 432, 16, 7, 7)

        if self.return_both_streams:
            return rgb_fused, pose_feats
        else:
            return rgb_fused

    def get_feature_dims(self):
        """
        Get output feature dimensions

        Returns:
            dict: Dictionary containing feature dimensions
        """
        return {
            'rgb_channels': self.rgb_backbone.feat_dim,
            'pose_channels': self.pose_backbone.feat_dim,
            'rgb_spatial': 7,  # After downsampling from 224x224
            'pose_spatial': 7,  # After downsampling from 56x56
            'rgb_temporal': 16,  # Number of RGB frames
            'pose_temporal': 48,  # Number of pose frames
        }


if __name__ == '__main__':
    # Test EPAM Backbone
    print("Testing EPAM Backbone...")

    # Create backbone
    backbone = EPAMBackbone(
        num_classes=60,
        attention_type='CBAM_spatial_efficient_temporal',
        return_both_streams=True
    )
    backbone.init_weights()

    # Create dummy inputs
    rgb_videos = torch.randn(2, 3, 16, 224, 224)
    pose_heatmaps = torch.randn(2, 17, 48, 56, 56)

    # Forward pass
    print("\nForward pass...")
    rgb_features, pose_features = backbone(rgb_videos, pose_heatmaps)

    print(f"\nOutput shapes:")
    print(f"  RGB features: {rgb_features.shape}")
    print(f"  Pose features: {pose_features.shape}")

    print(f"\nFeature dimensions:")
    dims = backbone.get_feature_dims()
    for key, value in dims.items():
        print(f"  {key}: {value}")

    print("\nTest passed!")
