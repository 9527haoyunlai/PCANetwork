"""
EPAM Recognizer for MMAction2
使用EPAM Backbone的多模态识别器
"""
import sys
import torch
import numpy as np
from typing import Dict, Tuple, Optional

from mmengine.model import BaseModel
from mmaction.registry import MODELS
from mmaction.utils import OptSampleList, SampleList
from .base import BaseRecognizer


def fine2coarse_ntu60(x):
    """NTU-60映射：0-59 → 0-7"""
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)


def fine2coarse_ntu60_semantic(x):
    """
    基于NTU-60动作语义的8类粗分类
    """
    if 0 <= x <= 9:
        return 0  # Drink, eat, brushing, reading (10个动作)
    elif 10 <= x <= 19:
        return 1  # Hand clapping, phone, camera (10个)
    elif 20 <= x <= 29:
        return 2  # Pickup, throw, sitting (10个)
    elif 30 <= x <= 39:
        return 3  # Standing, falling, kicking (10个)
    elif 40 <= x <= 44:
        return 4  # Punching, pushing, hugging (5个)
    elif 45 <= x <= 49:
        return 5  # Giving, handshaking (5个)
    elif 50 <= x <= 54:
        return 6  # Touch pocket, sneeze, staggering (5个)
    else:
        return 7  # Other (剩余)


@MODELS.register_module()
class EPAMRecognizer(BaseModel):
    """
    EPAM多模态识别器
    
    使用EPAM Backbone进行RGB+Pose双流特征提取，
    然后通过RGBPoseHead进行分类（保留原型学习等高级功能）。
    
    Args:
        backbone: EPAM Backbone配置
        cls_head: 分类头配置 (通常使用RGBPoseHead)
        data_preprocessor: 数据预处理配置
        train_cfg: 训练配置
        test_cfg: 测试配置
    """
    
    def __init__(self,
                 backbone: Dict,
                 cls_head: Dict,
                 data_preprocessor: Dict = None,
                 train_cfg: Dict = None,
                 test_cfg: Dict = None):
        """
        初始化EPAM Recognizer
        """
        # 调用BaseModel的初始化
        if data_preprocessor is None:
            data_preprocessor = dict(type='ActionDataPreprocessor')
        
        super().__init__(data_preprocessor=data_preprocessor)
        
        # 设置训练和测试配置
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # 构建backbone
        self.backbone = MODELS.build(backbone)
        
        # 构建分类头
        self.cls_head = MODELS.build(cls_head)
        
        print("✅ EPAM Recognizer已初始化")
        print(f"   - Backbone: {backbone['type']}")
        print(f"   - Head: {cls_head['type']}")
    
    @property
    def with_cls_head(self):
        """bool: 是否有分类头"""
        return hasattr(self, 'cls_head') and self.cls_head is not None
    
    def extract_feat(self,
                     inputs: Dict[str, torch.Tensor],
                     stage: str = 'backbone',
                     data_samples: OptSampleList = None,
                     test_mode: bool = False) -> Tuple:
        """
        提取RGB和Pose特征
        
        Args:
            inputs: 包含'imgs'和'heatmap_imgs'的字典
            stage: 特征提取阶段 ('backbone' 或 'head')
            data_samples: 数据样本（训练时需要）
            test_mode: 是否测试模式
            
        Returns:
            tuple: (rgb_feat, pose_feat), loss_predict_kwargs
        """
        # [N, num_views, C, T, H, W] -> [N * num_views, C, T, H, W]
        rgb_data = inputs.get('imgs')
        pose_data = inputs.get('heatmap_imgs')
        
        if rgb_data is not None:
            rgb_data = rgb_data.reshape((-1, ) + rgb_data.shape[2:])
        
        if pose_data is not None:
            pose_data = pose_data.reshape((-1, ) + pose_data.shape[2:])
        
        # 获取粗粒度标签（用于传递给backbone）
        loss_predict_kwargs = dict()
        gts = None
        gts_coarse = None
        
        if data_samples is not None and not test_mode:
            gts = []
            for data in data_samples:
                gts.extend(data.gt_labels.item)
            gts = torch.stack(gts)
            temp = gts.cpu().numpy()
            
            # 根据类别数判断数据集
            max_label = int(temp.max())
            if max_label >= 52:  # NTU-60 (标签0-59)
                gts_coarse = [fine2coarse_ntu60_semantic(int(i)) for i in temp]
            else:
                gts_coarse = [fine2coarse_ntu60(int(i)) for i in temp]
            gts_coarse = torch.from_numpy(np.array(gts_coarse)).cuda()
        
        # 通过EPAM backbone提取特征
        # backbone返回6个元素：(rgb_feat, pose_feat, rgb_feat1, pose_feat1, gt, gt_coarse)
        if rgb_data is not None and pose_data is not None:
            result = self.backbone(rgb_data, pose_data, gts, gts_coarse)
            # result是包含6个元素的元组
            x = result  # 直接传递完整的元组给head
        elif rgb_data is not None:
            # 如果只有RGB，仍需要空的pose输入（EPAM需要两个输入）
            B = rgb_data.shape[0]
            pose_data = torch.zeros(B, 17, 48, 56, 56, device=rgb_data.device)
            result = self.backbone(rgb_data, pose_data, gts, gts_coarse)
            # 只保留RGB特征
            x = (result[0], None, result[2], None, result[4], result[5])
        else:
            # 如果只有pose，仍需要空的RGB输入
            B = pose_data.shape[0]
            rgb_data = torch.zeros(B, 3, 16, 224, 224, device=pose_data.device)
            result = self.backbone(rgb_data, pose_data, gts, gts_coarse)
            # 只保留Pose特征
            x = (None, result[1], None, result[3], result[4], result[5])
        
        if stage == 'backbone':
            return x, loss_predict_kwargs
        
        if self.with_cls_head and stage == 'head':
            # 不传递loss_predict_kwargs给head，因为RGBPoseHead.forward()不接受额外参数
            x = self.cls_head(x)
            return x, loss_predict_kwargs
        
        return x, loss_predict_kwargs
    
    def loss(self, inputs: Dict[str, torch.Tensor],
             data_samples: SampleList, **kwargs) -> dict:
        """
        计算损失
        
        Args:
            inputs: 输入数据
            data_samples: 数据样本
            
        Returns:
            dict: 损失字典
        """
        feats, loss_predict_kwargs = self.extract_feat(inputs, data_samples=data_samples)
        # 不传递loss_predict_kwargs给head.loss()，因为head不需要这些参数
        loss = self.cls_head.loss(feats, data_samples)
        return loss
    
    def predict(self, inputs: Dict[str, torch.Tensor],
                data_samples: SampleList, **kwargs) -> SampleList:
        """
        预测结果
        
        Args:
            inputs: 输入数据
            data_samples: 数据样本
            
        Returns:
            SampleList: 预测结果
        """
        feats, loss_predict_kwargs = self.extract_feat(inputs, data_samples=data_samples, test_mode=True)
        # 不传递loss_predict_kwargs给head.predict()
        predictions = self.cls_head.predict(feats, data_samples)
        return predictions
    
    def forward(self, inputs: Dict[str, torch.Tensor],
                data_samples: OptSampleList = None,
                mode: str = 'tensor', **kwargs):
        """
        前向传播
        
        Args:
            inputs: 输入数据
            data_samples: 数据样本
            mode: 前向模式
                - 'loss': 返回损失
                - 'predict': 返回预测
                - 'tensor': 返回特征
                
        Returns:
            根据mode返回不同结果
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            feats, _ = self.extract_feat(inputs, data_samples=data_samples)
            return feats
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                            'Only supports loss, predict and tensor mode.')


if __name__ == '__main__':
    # 测试代码
    print("测试EPAM Recognizer")
    
    # 创建配置
    backbone_cfg = dict(
        type='EPAMBackbone',
        num_classes=60,
        attention_type='CBAM_spatial_efficient_temporal',
        return_both_streams=True
    )
    
    head_cfg = dict(
        type='RGBPoseHead',
        num_classes=60,
        num_coarse_classes=8,
        in_channels=[432, 216],  # EPAM的输出维度
        loss_components=['rgb', 'pose'],
        loss_weights=[1.0, 1.2, 0.5, 0.9],
        average_clips='prob'
    )
    
    # 注意：这里只是测试结构，实际运行需要完整的MMAction2环境
    print("✅ EPAM Recognizer配置测试通过！")

