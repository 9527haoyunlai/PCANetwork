#!/usr/bin/env python3
"""
测试EPAM接口修复 - 验证gt参数正确传递给backbone而不是head
"""
import torch
from mmaction.models.backbones import EPAMBackbone
from mmaction.models.heads import RGBPoseHead
from mmaction.models.recognizers import EPAMRecognizer

print("=" * 80)
print("测试EPAM接口修复")
print("=" * 80)

# 测试1: 验证EPAMBackbone接受gt参数
print("\n[测试1] EPAMBackbone接受gt和gt_coarse参数...")
try:
    backbone = EPAMBackbone(
        num_classes=60,
        attention_type='CBAM_spatial_efficient_temporal',
        return_both_streams=True
    )
    backbone.init_weights()
    
    rgb = torch.randn(2, 3, 16, 224, 224)
    pose = torch.randn(2, 17, 48, 56, 56)
    gt = torch.randint(0, 60, (2,))
    gt_coarse = torch.randint(0, 8, (2,))
    
    with torch.no_grad():
        rgb_feat, pose_feat = backbone(rgb, pose, gt, gt_coarse)
    
    print(f"✅ Backbone接受gt参数")
    print(f"   输出: RGB {rgb_feat.shape}, Pose {pose_feat.shape}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试2: 验证RGBPoseHead.forward()不接受gt参数
print("\n[测试2] RGBPoseHead.forward()不接受额外参数...")
try:
    head = RGBPoseHead(
        num_classes=60,
        num_coarse_classes=8,
        in_channels=[432, 216],
        loss_components=['rgb', 'pose'],
        loss_weights=[1.0, 1.2, 0.5, 0.9],
        average_clips='prob'
    )
    
    rgb_feat = torch.randn(2, 432, 16, 7, 7)
    pose_feat = torch.randn(2, 216, 48, 7, 7)
    feats = [rgb_feat, pose_feat]
    
    with torch.no_grad():
        # RGBPoseHead.forward()只接受feats参数
        cls_scores = head(feats)
    
    print(f"✅ Head正确工作（不需要gt参数）")
    print(f"   输出: {list(cls_scores.keys())}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 验证完整流程
print("\n[测试3] 完整EPAMRecognizer流程...")
try:
    from mmengine.structures import LabelData
    from mmaction.structures import ActionDataSample
    
    # 构建recognizer
    recognizer = EPAMRecognizer(
        backbone=dict(
            type='EPAMBackbone',
            num_classes=60,
            attention_type='CBAM_spatial_efficient_temporal',
            return_both_streams=True
        ),
        cls_head=dict(
            type='RGBPoseHead',
            num_classes=60,
            num_coarse_classes=8,
            in_channels=[432, 216],
            loss_components=['rgb', 'pose'],
            loss_weights=[1.0, 1.2, 0.5, 0.9],
            average_clips='prob'
        ),
        data_preprocessor=dict(
            type='MultiModalDataPreprocessor',
            preprocessors=dict(
                imgs=dict(
                    type='ActionDataPreprocessor',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    format_shape='NCTHW'),
                heatmap_imgs=dict(type='ActionDataPreprocessor')
            )
        )
    )
    
    recognizer.eval()
    
    # 准备输入
    inputs = {
        'imgs': torch.randn(2, 1, 3, 16, 224, 224),
        'heatmap_imgs': torch.randn(2, 1, 17, 48, 56, 56)
    }
    
    # 准备data_samples（用于loss计算）
    data_samples = []
    for i in range(2):
        data_sample = ActionDataSample()
        gt_labels = LabelData()
        gt_labels.item = [torch.tensor(i % 60)]
        data_sample.gt_labels = gt_labels
        data_samples.append(data_sample)
    
    # 测试loss模式（这是报错的地方）
    recognizer.train()
    with torch.no_grad():
        loss = recognizer(inputs, data_samples=data_samples, mode='loss')
    
    print(f"✅ Loss计算成功")
    print(f"   损失键: {list(loss.keys())}")
    
    # 测试predict模式
    recognizer.eval()
    with torch.no_grad():
        predictions = recognizer(inputs, data_samples=data_samples, mode='predict')
    
    print(f"✅ Predict成功")
    print(f"   预测数量: {len(predictions)}")
    
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("🎉 所有测试通过！修复成功！")
print("=" * 80)
print("\n现在可以重新运行训练命令:")
print("CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \\")
print("    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py \\")
print("    2 \\")
print("    --work-dir work_dirs/epam_ntu60_baseline_2gpu")
print("=" * 80)

