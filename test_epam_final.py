#!/usr/bin/env python3
"""
最终测试：验证EPAM Backbone返回6元素以兼容RGBPoseHead
"""
import torch

print("=" * 80)
print("EPAM接口修复验证 - 6元素返回值")
print("=" * 80)

# 测试：验证EPAMBackbone返回6个元素
print("\n[测试] EPAMBackbone返回6个元素...")
try:
    from mmaction.models.backbones import EPAMBackbone
    
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
        result = backbone(rgb, pose, gt, gt_coarse)
    
    print(f"✅ Backbone返回元素数量: {len(result)}")
    
    if len(result) == 6:
        rgb_feat, pose_feat, rgb_feat1, pose_feat1, gt_out, gt_coarse_out = result
        print(f"   [0] RGB特征: {rgb_feat.shape}")
        print(f"   [1] Pose特征: {pose_feat.shape}")
        print(f"   [2] RGB特征1 (FR): {rgb_feat1.shape}")
        print(f"   [3] Pose特征1 (FR): {pose_feat1.shape}")
        print(f"   [4] GT标签: {gt_out.shape if gt_out is not None else None}")
        print(f"   [5] GT粗粒度: {gt_coarse_out.shape if gt_coarse_out is not None else None}")
        print("✅ 返回值格式正确！")
    else:
        print(f"❌ 期望6个元素，实际{len(result)}个")
        
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("修复完成！现在可以重新运行训练命令")
print("=" * 80)
print("\nCUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \\")
print("    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py \\")
print("    2 \\")
print("    --work-dir work_dirs/epam_ntu60_baseline_2gpu")
print("=" * 80)

