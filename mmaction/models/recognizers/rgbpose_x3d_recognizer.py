# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Union, List
import sys
import torch
import numpy as np
from mmengine.model import BaseModel
from mmaction.registry import MODELS
from mmaction.utils import OptSampleList, SampleList, ForwardResults

# 导入standalone X3D backbones (纯PyTorch实现，无MMCV依赖)
sys.path.insert(0, '/home/zh/ChCode/codes01/mmaction2')
try:
    from standalone_backbones import X3DTemporalShift, X3DTemporalShiftPose
    print("✅ Standalone X3D backbones imported successfully")
except ImportError as e:
    print(f"❌ Failed to import standalone X3D backbones: {e}")
    raise

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
class RGBPoseX3DRecognizer(BaseModel):
    """
    RGB+Pose双backbone X3D识别器
    
    使用独立的X3D TemporalShift backbone分别处理RGB和Pose输入
    """

    def __init__(self,
                 rgb_backbone: Dict,
                 pose_backbone: Dict,
                 cls_head: Dict,
                 data_preprocessor: Dict = None,
                 train_cfg: Dict = None,
                 test_cfg: Dict = None):
        """
        Args:
            rgb_backbone: RGB backbone配置
            pose_backbone: Pose backbone配置
            cls_head: 分类头配置
            data_preprocessor: 数据预处理配置
            train_cfg: 训练配置
            test_cfg: 测试配置
        """
        # 调用BaseModel的初始化
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=None
        )
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # 构建独立的RGB和Pose backbone (使用standalone实现)
        rgb_params = {k: v for k, v in rgb_backbone.items() if k != 'type'}
        pose_params = {k: v for k, v in pose_backbone.items() if k != 'type'}
        
        self.rgb_backbone = X3DTemporalShift(**rgb_params)
        self.pose_backbone = X3DTemporalShiftPose(**pose_params)
        
        # 初始化权重
        self.rgb_backbone.init_weights()
        self.pose_backbone.init_weights()
        
        # 构建分类头
        self.cls_head = MODELS.build(cls_head)
    
    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
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
            stage: 特征提取阶段
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
        
        # 获取粗粒度标签（用于层次化损失）
        loss_predict_kwargs = dict()
        
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
            
            loss_predict_kwargs['gt'] = gts
            loss_predict_kwargs['gt_coarse'] = gts_coarse
        
        # 通过各自的backbone提取特征
        if rgb_data is not None and pose_data is not None:
            rgb_feat = self.rgb_backbone(rgb_data)      # [B, 432, 16, 7, 7]
            pose_feat = self.pose_backbone(pose_data)   # [B, 216, 48, 7, 7]
            # 在训练模式下，需要将gt和gt_coarse添加到特征列表中
            if not test_mode and 'gt' in loss_predict_kwargs:
                x = [rgb_feat, pose_feat, rgb_feat, pose_feat, 
                     loss_predict_kwargs['gt'], loss_predict_kwargs['gt_coarse']]
            else:
                x = [rgb_feat, pose_feat]
        elif rgb_data is not None:
            rgb_feat = self.rgb_backbone(rgb_data)
            x = [rgb_feat, None]
        else:
            pose_feat = self.pose_backbone(pose_data)
            x = [None, pose_feat]
        
        if stage == 'backbone':
            return x, loss_predict_kwargs
        
        if self.with_cls_head and stage == 'head':
            # 不再传递gt和gt_coarse作为kwargs，它们已经在x中了
            x = self.cls_head(x)
            return x, loss_predict_kwargs
        
        return x, loss_predict_kwargs
    
    def loss(self, inputs: Dict[str, torch.Tensor],
             data_samples: SampleList, **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.
        
        Args:
            inputs: 输入数据
            data_samples: 数据样本
            
        Returns:
            dict: 损失字典
        """
        feats, loss_predict_kwargs = self.extract_feat(inputs, data_samples=data_samples)
        # feats已经包含了gt和gt_coarse，不再需要传递kwargs
        loss = self.cls_head.loss(feats, data_samples)
        return loss
    
    def predict(self, inputs: Dict[str, torch.Tensor],
                data_samples: SampleList, **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples.
        
        Args:
            inputs: 输入数据
            data_samples: 数据样本
            
        Returns:
            SampleList: 预测结果
        """
        feats, loss_predict_kwargs = self.extract_feat(inputs, data_samples=data_samples, test_mode=True)
        predictions = self.cls_head.predict(feats, data_samples, **loss_predict_kwargs)
        return predictions

    def forward(self, inputs: Union[Dict[str, torch.Tensor], List, torch.Tensor],
                data_samples: OptSampleList = None,
                mode: str = 'tensor', **kwargs) -> ForwardResults:
        """Forward function.
        
        Args:
            inputs: 输入数据（字典、列表或张量）
            data_samples: 数据样本
            mode: 前向模式
                - 'loss': 返回损失
                - 'predict': 返回预测
                - 'tensor': 返回特征
                
        Returns:
            根据mode返回不同结果
        """
        # 处理输入格式：将列表转换为字典
        if isinstance(inputs, (list, tuple)):
            # inputs = [rgb_data, pose_data]
            inputs = {'imgs': inputs[0], 'heatmap_imgs': inputs[1]}
        
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