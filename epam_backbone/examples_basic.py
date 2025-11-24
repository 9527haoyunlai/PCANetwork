"""
示例1: 基础使用 - EPAM Backbone特征提取
"""
import torch
import sys
sys.path.append('..')  # 添加父目录到路径

from epam_backbone import EPAMBackbone


def basic_usage_example():
    """基础使用示例"""
    print("=" * 60)
    print("示例1: 基础使用 - EPAM Backbone特征提取")
    print("=" * 60)

    # 1. 创建backbone实例
    print("\n[1] 创建EPAM Backbone...")
    backbone = EPAMBackbone(
        num_classes=60,
        attention_type='CBAM_spatial_efficient_temporal',
        return_both_streams=True
    )

    # 2. 初始化权重
    print("[2] 初始化权重...")
    backbone.init_weights()

    # 3. 准备输入数据（模拟数据）
    print("[3] 准备输入数据...")
    batch_size = 2
    rgb_videos = torch.randn(batch_size, 3, 16, 224, 224)
    pose_heatmaps = torch.randn(batch_size, 17, 48, 56, 56)

    print(f"    RGB输入维度: {rgb_videos.shape}")
    print(f"    Pose输入维度: {pose_heatmaps.shape}")

    # 4. 前向传播
    print("[4] 前向传播...")
    backbone.eval()
    with torch.no_grad():
        rgb_features, pose_features = backbone(rgb_videos, pose_heatmaps)

    # 5. 查看输出
    print("[5] 输出特征维度:")
    print(f"    RGB特征: {rgb_features.shape}")
    print(f"    Pose特征: {pose_features.shape}")

    # 6. 获取特征维度信息
    print("[6] 特征维度信息:")
    dims = backbone.get_feature_dims()
    for key, value in dims.items():
        print(f"    {key}: {value}")

    print("\n✓ 基础使用示例完成!\n")


def usage_with_pretrained():
    """使用预训练权重的示例"""
    print("=" * 60)
    print("示例2: 使用预训练权重")
    print("=" * 60)

    # 注意：需要提供实际的预训练权重路径
    rgb_pretrained_path = '/path/to/rgb_pretrained.pth'
    pose_pretrained_path = '/path/to/pose_pretrained.pth'

    print("\n[1] 创建带预训练权重的Backbone...")
    print(f"    RGB权重路径: {rgb_pretrained_path}")
    print(f"    Pose权重路径: {pose_pretrained_path}")

    try:
        backbone = EPAMBackbone(
            rgb_pretrained=rgb_pretrained_path,
            pose_pretrained=pose_pretrained_path
        )
        backbone.init_weights()
        print("✓ 预训练权重加载成功!")
    except FileNotFoundError:
        print("⚠ 预训练权重文件未找到，跳过此示例")

    print()


def frozen_backbone_example():
    """冻结backbone的示例"""
    print("=" * 60)
    print("示例3: 冻结Backbone进行迁移学习")
    print("=" * 60)

    print("\n[1] 创建冻结的Backbone...")
    backbone = EPAMBackbone(
        freeze_rgb=True,
        freeze_pose=True
    )
    backbone.init_weights()

    # 检查参数是否真的被冻结
    print("[2] 检查参数冻结状态...")
    rgb_params_trainable = sum(p.numel() for p in backbone.rgb_backbone.parameters() if p.requires_grad)
    pose_params_trainable = sum(p.numel() for p in backbone.pose_backbone.parameters() if p.requires_grad)

    print(f"    RGB可训练参数数量: {rgb_params_trainable}")
    print(f"    Pose可训练参数数量: {pose_params_trainable}")

    if rgb_params_trainable == 0 and pose_params_trainable == 0:
        print("✓ Backbone已成功冻结!")
    else:
        print("⚠ Backbone未完全冻结")

    print()


def feature_extraction_only():
    """仅进行特征提取的示例"""
    print("=" * 60)
    print("示例4: 仅特征提取（不返回两个流）")
    print("=" * 60)

    print("\n[1] 创建只返回融合特征的Backbone...")
    backbone = EPAMBackbone(
        return_both_streams=False  # 只返回融合后的RGB特征
    )
    backbone.init_weights()

    print("[2] 前向传播...")
    rgb_videos = torch.randn(2, 3, 16, 224, 224)
    pose_heatmaps = torch.randn(2, 17, 48, 56, 56)

    backbone.eval()
    with torch.no_grad():
        fused_features = backbone(rgb_videos, pose_heatmaps)

    print(f"[3] 融合特征维度: {fused_features.shape}")
    print("✓ 特征提取完成!\n")


def gpu_usage_example():
    """GPU使用示例"""
    print("=" * 60)
    print("示例5: GPU加速")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\n⚠ CUDA不可用，跳过GPU示例")
        print()
        return

    print(f"\n[1] CUDA可用，设备数量: {torch.cuda.device_count()}")

    # 创建模型并移到GPU
    print("[2] 创建模型并移到GPU...")
    device = torch.device('cuda:0')
    backbone = EPAMBackbone().to(device)
    backbone.init_weights()

    # 创建GPU数据
    print("[3] 准备GPU数据...")
    rgb_videos = torch.randn(2, 3, 16, 224, 224).to(device)
    pose_heatmaps = torch.randn(2, 17, 48, 56, 56).to(device)

    # 前向传播
    print("[4] GPU前向传播...")
    backbone.eval()
    with torch.no_grad():
        rgb_feat, pose_feat = backbone(rgb_videos, pose_heatmaps)

    print(f"[5] 输出特征在设备: {rgb_feat.device}")
    print("✓ GPU加速示例完成!\n")


def batch_processing_example():
    """批处理示例"""
    print("=" * 60)
    print("示例6: 批处理多个视频")
    print("=" * 60)

    print("\n[1] 创建Backbone...")
    backbone = EPAMBackbone()
    backbone.init_weights()
    backbone.eval()

    # 模拟处理多个batch
    print("[2] 处理多个batch...")
    num_batches = 3
    batch_size = 4

    all_rgb_features = []
    all_pose_features = []

    with torch.no_grad():
        for i in range(num_batches):
            print(f"    处理batch {i+1}/{num_batches}...")

            rgb_videos = torch.randn(batch_size, 3, 16, 224, 224)
            pose_heatmaps = torch.randn(batch_size, 17, 48, 56, 56)

            rgb_feat, pose_feat = backbone(rgb_videos, pose_heatmaps)

            all_rgb_features.append(rgb_feat)
            all_pose_features.append(pose_feat)

    # 合并所有特征
    print("[3] 合并所有特征...")
    all_rgb_features = torch.cat(all_rgb_features, dim=0)
    all_pose_features = torch.cat(all_pose_features, dim=0)

    print(f"[4] 总RGB特征维度: {all_rgb_features.shape}")
    print(f"    总Pose特征维度: {all_pose_features.shape}")
    print("✓ 批处理示例完成!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("EPAM Backbone 使用示例集合")
    print("=" * 60 + "\n")

    # 运行所有示例
    basic_usage_example()
    usage_with_pretrained()
    frozen_backbone_example()
    feature_extraction_only()
    gpu_usage_example()
    batch_processing_example()

    print("=" * 60)
    print("所有示例运行完毕!")
    print("=" * 60)
