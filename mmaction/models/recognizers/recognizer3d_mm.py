# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
import numpy as np
from mmaction.registry import MODELS
from mmaction.utils import OptSampleList
from .base import BaseRecognizer

def fine2coarse(x):
    if x<=4:
        return 0
    elif 5<=x<=10:
        return 1
    elif 11<=x<=23:
        return 2
    elif 24<=x<=31:
        return 3
    elif 32<=x<=37:
        return 4
    elif 38<=x<=47:
        return 5
    else:
        return 6

def fine2coarse_ntu60(x):
    """NTU-60映射：0-59 → 0-7"""
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)

# 或者使用语义分组（更好）：
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
class MMRecognizer3D(BaseRecognizer):
    """Multi-modal 3D recognizer model framework."""

    def extract_feat(self,
                     inputs: Dict[str, torch.Tensor],
                     stage: str = 'backbone',
                     data_samples: OptSampleList = None,
                     test_mode: bool = False) -> Tuple:
        """Extract features.

        Args:
            inputs (dict[str, torch.Tensor]): The multi-modal input data.
            stage (str): Which stage to output the feature.
                Defaults to ``'backbone'``.
            data_samples (list[:obj:`ActionDataSample`], optional): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                tuple[torch.Tensor]: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline.
        """
        # [N, num_views, C, T, H, W] ->
        # [N * num_views, C, T, H, W]
        for m, m_data in inputs.items():
            m_data = m_data.reshape((-1, ) + m_data.shape[2:])
            inputs[m] = m_data
        
        #get gt_label

        # gts=[]
        # for data in data_samples:
        #     gts.extend(data.gt_labels.item)
        # gts=torch.stack(gts)
        # temp=gts.cpu().numpy()
        # gts_coarse=[fine2coarse(i) for i in temp]
        # gts_coarse=torch.from_numpy(np.array(gts_coarse)).cuda()

        # 修改为：
        gts = []
        for data in data_samples:
            gts.extend(data.gt_labels.item)
        gts = torch.stack(gts)
        temp = gts.cpu().numpy()

        # 根据类别数判断数据集
        max_label = int(temp.max())
        if max_label >= 52:  # NTU-60 (标签0-59)
            gts_coarse = [fine2coarse_ntu60(int(i)) for i in temp]
        else:  # MA-52 (标签0-51)
            gts_coarse = [fine2coarse(int(i)) for i in temp]
        gts_coarse = torch.from_numpy(np.array(gts_coarse)).cuda()

        inputs['gt'] = gts
        inputs['gt_coarse'] = gts_coarse
        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = dict()

        x = self.backbone(**inputs)
        if stage == 'backbone':
            return x, loss_predict_kwargs

        if self.with_cls_head and stage == 'head':
            x = self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs
