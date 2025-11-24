# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Union, List
import torch

from mmengine.model import BaseDataPreprocessor, ModuleDict

from mmaction.registry import MODELS


@MODELS.register_module()
class RGBPoseDataPreprocessor(BaseDataPreprocessor):
    """
    RGB+Pose数据预处理器
    专门处理双模态输入，兼容列表和字典格式
    """

    def __init__(self, preprocessors: Dict) -> None:
        super().__init__()
        self.preprocessors = ModuleDict()
        for name, pre_cfg in preprocessors.items():
            assert 'type' in pre_cfg, (
                'Each data preprocessor should contain the key type, '
                f'but got {pre_cfg}')
            self.preprocessors[name] = MODELS.build(pre_cfg)

    def forward(self, data: Dict, training: bool = False) -> Dict:
        """
        预处理数据
        
        Args:
            data (dict): 包含'inputs'和'data_samples'的字典
            training (bool): 是否训练模式
        
        Returns:
            dict: 预处理后的数据
        """
        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        
        # 处理inputs：应该是字典格式 {'imgs': [...], 'heatmap_imgs': [...]}
        if not isinstance(inputs, dict):
            raise TypeError(f"Expected inputs to be dict, got {type(inputs)}")
        
        # 对每个模态进行预处理
        for modality, modality_data in inputs.items():
            if modality in self.preprocessors:
                preprocessor = self.preprocessors[modality]
                modality_data, data_samples = preprocessor.preprocess(
                    modality_data, data_samples, training)
                inputs[modality] = modality_data
        
        data['inputs'] = inputs
        data['data_samples'] = data_samples
        return data

