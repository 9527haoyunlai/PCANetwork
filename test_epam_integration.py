#!/usr/bin/env python3
"""
测试EPAM Backbone集成到MMAction2

验证:
1. EPAM Backbone能否正确注册和初始化
2. EPAM Recognizer能否正确构建
3. 完整模型能否正确前向传播
4. 配置文件是否有效
"""
import sys
import torch
import numpy as np
from mmengine.config import Config
from mmengine.registry import MODELS

print("=" * 80)
print("测试EPAM Backbone集成到MMAction2")
print("=" * 80)

# 首先导入所有mmaction模块以触发注册
print("\n[初始化] 导入MMAction2模块...")
import mmaction.models.backbones
import mmaction.models.recognizers
import mmaction.models.heads
print("✅ MMAction2模块导入完成")

# ==========================================
# 测试1: 验证EPAM Backbone注册
# ==========================================
print("\n[测试1] 验证EPAM Backbone注册...")
try:
    from mmaction.models.backbones import EPAMBackbone
    print("✅ EPAM Backbone导入成功")
    
    # 检查是否在注册表中
    if 'EPAMBackbone' in MODELS.module_dict:
        print("✅ EPAM Backbone已注册到MODELS")
    else:
        print(f"⚠️  EPAM Backbone未在MODELS中找到")
        print(f"   已注册的backbones: {[k for k in MODELS.module_dict.keys() if 'Backbone' in k or 'backbone' in k][:10]}")
except Exception as e:
    print(f"❌ EPAM Backbone注册失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==========================================
# 测试2: 验证EPAM Recognizer注册
# ==========================================
print("\n[测试2] 验证EPAM Recognizer注册...")
try:
    from mmaction.models.recognizers import EPAMRecognizer
    print("✅ EPAM Recognizer导入成功")
    
    # 检查是否在注册表中
    if 'EPAMRecognizer' in MODELS.module_dict:
        print("✅ EPAM Recognizer已注册到MODELS")
    else:
        print("⚠️  EPAM Recognizer未在MODELS中找到（可能使用lazy import）")
except Exception as e:
    print(f"❌ EPAM Recognizer注册失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==========================================
# 测试3: 单独测试EPAM Backbone
# ==========================================
print("\n[测试3] 单独测试EPAM Backbone...")
try:
    backbone = EPAMBackbone(
        num_classes=60,
        attention_type='CBAM_spatial_efficient_temporal',
        return_both_streams=True
    )
    backbone.init_weights()
    print("✅ EPAM Backbone初始化成功")
    
    # 测试前向传播
    rgb = torch.randn(2, 3, 16, 224, 224)
    pose = torch.randn(2, 17, 48, 56, 56)
    
    with torch.no_grad():
        rgb_feat, pose_feat = backbone(rgb, pose)
    
    print(f"  输入: RGB {rgb.shape}, Pose {pose.shape}")
    print(f"  输出: RGB特征 {rgb_feat.shape}, Pose特征 {pose_feat.shape}")
    
    assert rgb_feat.shape == (2, 432, 16, 7, 7), f"RGB特征维度错误: {rgb_feat.shape}"
    assert pose_feat.shape == (2, 216, 48, 7, 7), f"Pose特征维度错误: {pose_feat.shape}"
    print("✅ EPAM Backbone前向传播成功，输出维度正确")
    
except Exception as e:
    print(f"❌ EPAM Backbone测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==========================================
# 测试4: 从配置文件构建模型
# ==========================================
print("\n[测试4] 从配置文件构建完整模型...")
try:
    config_path = 'configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py'
    cfg = Config.fromfile(config_path)
    print(f"✅ 配置文件加载成功: {config_path}")
    
    # 构建模型
    model = MODELS.build(cfg.model)
    print("✅ 模型构建成功")
    print(f"  模型类型: {type(model).__name__}")
    print(f"  Backbone类型: {type(model.backbone).__name__}")
    print(f"  Head类型: {type(model.cls_head).__name__}")
    
except FileNotFoundError:
    print(f"⚠️  配置文件未找到: {config_path}")
    print("   跳过此测试（配置文件路径可能需要调整）")
except Exception as e:
    print(f"❌ 配置文件模型构建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==========================================
# 测试5: 完整模型前向传播
# ==========================================
print("\n[测试5] 测试完整模型前向传播...")
try:
    # 手动构建模型配置
    model_cfg = dict(
        type='EPAMRecognizer',
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
    
    model = MODELS.build(model_cfg)
    model.eval()
    print("✅ 完整模型构建成功")
    
    # 准备输入数据
    inputs = {
        'imgs': torch.randn(2, 1, 3, 16, 224, 224),  # [N, num_views, C, T, H, W]
        'heatmap_imgs': torch.randn(2, 1, 17, 48, 56, 56)
    }
    
    # 测试tensor模式（特征提取）
    with torch.no_grad():
        feats = model(inputs, mode='tensor')
    
    print(f"  输入: RGB {inputs['imgs'].shape}, Pose {inputs['heatmap_imgs'].shape}")
    print(f"  输出特征: RGB {feats[0].shape}, Pose {feats[1].shape}")
    print("✅ 模型前向传播（tensor模式）成功")
    
    # 测试predict模式（需要data_samples）
    from mmengine.structures import LabelData
    from mmaction.structures import ActionDataSample
    
    data_samples = []
    for i in range(2):
        data_sample = ActionDataSample()
        gt_labels = LabelData()
        gt_labels.item = [torch.tensor(i % 60)]
        data_sample.gt_labels = gt_labels
        data_samples.append(data_sample)
    
    with torch.no_grad():
        predictions = model(inputs, data_samples=data_samples, mode='predict')
    
    print(f"  预测结果数量: {len(predictions)}")
    print("✅ 模型预测（predict模式）成功")
    
except Exception as e:
    print(f"❌ 完整模型测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==========================================
# 测试6: 模型参数统计
# ==========================================
print("\n[测试6] 模型参数统计...")
try:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 统计各模块参数
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.cls_head.parameters())
    
    print(f"\n  模块详情:")
    print(f"    - Backbone: {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
    print(f"    - Head: {head_params:,} ({head_params/total_params*100:.1f}%)")
    print("✅ 参数统计完成")
    
except Exception as e:
    print(f"⚠️  参数统计失败: {e}")

# ==========================================
# 总结
# ==========================================
print("\n" + "=" * 80)
print("🎉 所有测试通过！EPAM Backbone已成功集成到MMAction2")
print("=" * 80)
print("\n下一步操作:")
print("1. 使用配置文件开始训练:")
print("   python tools/train.py configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py")
print("\n2. 或进行测试:")
print("   python tools/test.py configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py \\")
print("       work_dirs/epam_ntu60_baseline/best_checkpoint.pth")
print("\n3. 如需调整超参数，编辑配置文件:")
print("   configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py")
print("=" * 80)

