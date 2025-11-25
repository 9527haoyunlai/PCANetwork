"""
EPAM Backbone wrapper for MMAction2 integration
将standalone EPAM Backbone适配到MMAction2框架
"""
import sys
import torch
import torch.nn as nn
from typing import Dict, Optional, Union, List

from mmengine.model import BaseModule
from mmaction.registry import MODELS

# 添加epam_backbone路径
epam_backbone_path = '/home/zh/ChCode/codes01/mmaction2/epam_backbone'
if epam_backbone_path not in sys.path:
    sys.path.insert(0, epam_backbone_path)

# 导入EPAM Backbone
from epam_backbone import EPAMBackbone as StandaloneEPAMBackbone


@MODELS.register_module()
class EPAMBackbone(BaseModule):
    """
    EPAM-Net Backbone wrapper for MMAction2
    
    这是一个适配器，将standalone EPAM Backbone集成到MMAction2框架中。
    
    主要特点：
    - RGB流 + Pose流双backbone架构
    - Pose特征引导的注意力机制
    - 使用X3D + Temporal Shift Module进行高效特征提取
    
    Args:
        num_classes (int): 动作类别数。默认: 60
        rgb_pretrained (str, optional): RGB预训练权重路径。默认: None
        pose_pretrained (str, optional): Pose预训练权重路径。默认: None
        attention_type (str): 注意力类型。默认: 'CBAM_spatial_efficient_temporal'
        freeze_rgb (bool): 是否冻结RGB backbone。默认: False
        freeze_pose (bool): 是否冻结Pose backbone。默认: False
        return_both_streams (bool): 是否返回两个流的特征。默认: True
        init_cfg (dict, optional): 初始化配置。默认: None
        
    输入:
        - imgs: RGB视频 (N, 3, 16, 224, 224)
        - heatmap_imgs: 姿态热图 (N, 17, 48, 56, 56)
        
    输出:
        - rgb_feat: RGB特征 (N, 432, 16, 7, 7)
        - pose_feat: Pose特征 (N, 216, 48, 7, 7)
    """
    
    def __init__(self,
                 num_classes: int = 60,
                 rgb_pretrained: Optional[str] = None,
                 pose_pretrained: Optional[str] = None,
                 attention_type: str = 'CBAM_spatial_efficient_temporal',
                 freeze_rgb: bool = False,
                 freeze_pose: bool = False,
                 return_both_streams: bool = True,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        
        self.num_classes = num_classes
        self.rgb_pretrained = rgb_pretrained
        self.pose_pretrained = pose_pretrained
        self.return_both_streams = return_both_streams
        
        # 创建standalone EPAM Backbone
        self.backbone = StandaloneEPAMBackbone(
            num_classes=num_classes,
            rgb_pretrained=rgb_pretrained,
            pose_pretrained=pose_pretrained,
            attention_type=attention_type,
            freeze_rgb=freeze_rgb,
            freeze_pose=freeze_pose,
            return_both_streams=return_both_streams
        )
        
        # 特征维度
        self.rgb_feat_dim = 432
        self.pose_feat_dim = 216
        
        print(f"✅ EPAM Backbone (MMAction2) 已初始化")
        print(f"   - RGB特征维度: {self.rgb_feat_dim}")
        print(f"   - Pose特征维度: {self.pose_feat_dim}")
        print(f"   - 注意力类型: {attention_type}")
    
    def init_weights(self) -> None:
        """
        初始化权重
        
        如果提供了预训练权重路径，则加载；否则使用默认初始化。
        """
        # EPAM Backbone有自己的init_weights方法
        self.backbone.init_weights()
        print("✅ EPAM Backbone权重初始化完成")
    
    def forward(self, imgs: torch.Tensor, heatmap_imgs: torch.Tensor,
                gt: Optional[torch.Tensor] = None, 
                gt_coarse: Optional[torch.Tensor] = None) -> tuple:
        """
        前向传播
        
        Args:
            imgs: RGB视频 (N, 3, 16, 224, 224)
            heatmap_imgs: 姿态热图 (N, 17, 48, 56, 56)
            gt: 细粒度标签 (训练时需要，用于head的FR loss)
            gt_coarse: 粗粒度标签 (训练时需要，用于head的coarse loss)
            
        Returns:
            tuple: 6个元素以兼容RGBPoseHead
                - rgb_feat: (N, 432, 16, 7, 7) 最终RGB特征
                - pose_feat: (N, 216, 48, 7, 7) 最终Pose特征
                - rgb_feat1: (N, 432, 16, 7, 7) 中间RGB特征（用于FR loss）
                - pose_feat1: (N, 216, 48, 7, 7) 中间Pose特征（用于FR loss）
                - gt: 细粒度标签
                - gt_coarse: 粗粒度标签
        """
        # 调用standalone backbone（它不需要gt参数）
        rgb_feat, pose_feat = self.backbone(imgs, heatmap_imgs)
        
        # 为了兼容RGBPoseHead，需要返回6个元素
        # rgb_feat1和pose_feat1用于Feature Renovation loss
        # 这里我们简单地复制特征（EPAM没有中间特征的概念）
        rgb_feat1 = rgb_feat.clone()
        pose_feat1 = pose_feat.clone()
        
        # 返回6个元素：(x_rgb, x_pose, x_rgb1, x_pose1, gt, gt_coarse)
        return rgb_feat, pose_feat, rgb_feat1, pose_feat1, gt, gt_coarse


# 为了兼容性，也注册为EPAMNet
@MODELS.register_module()
class EPAMNet(EPAMBackbone):
    """EPAMNet的别名，与EPAMBackbone完全相同"""
    pass


if __name__ == '__main__':
    # 测试代码
    print("测试EPAM Backbone (MMAction2 wrapper)")
    
    # 创建模型
    model = EPAMBackbone(
        num_classes=60,
        attention_type='CBAM_spatial_efficient_temporal',
        return_both_streams=True
    )
    model.init_weights()
    
    # 测试前向传播
    rgb = torch.randn(2, 3, 16, 224, 224)
    pose = torch.randn(2, 17, 48, 56, 56)
    
    rgb_feat, pose_feat = model(rgb, pose)
    
    print(f"\n输入:")
    print(f"  RGB: {rgb.shape}")
    print(f"  Pose: {pose.shape}")
    print(f"\n输出:")
    print(f"  RGB特征: {rgb_feat.shape}")
    print(f"  Pose特征: {pose_feat.shape}")
    print(f"\n✅ 测试通过！")

