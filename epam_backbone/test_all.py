"""
EPAM Backbone 测试脚本
验证所有模块是否正常工作
"""
import torch
import sys

def test_imports():
    """测试所有模块导入"""
    print("=" * 60)
    print("测试1: 模块导入")
    print("=" * 60)

    try:
        from epam_backbone import EPAMBackbone
        print("✓ EPAMBackbone 导入成功")
    except Exception as e:
        print(f"✗ EPAMBackbone 导入失败: {e}")
        return False

    try:
        from epam_backbone import X3DTemporalShift
        print("✓ X3DTemporalShift 导入成功")
    except Exception as e:
        print(f"✗ X3DTemporalShift 导入失败: {e}")
        return False

    try:
        from epam_backbone import X3DTemporalShiftPose
        print("✓ X3DTemporalShiftPose 导入成功")
    except Exception as e:
        print(f"✗ X3DTemporalShiftPose 导入失败: {e}")
        return False

    try:
        from epam_backbone import CBAMSpatialEfficientTemporalAttention
        print("✓ CBAMSpatialEfficientTemporalAttention 导入成功")
    except Exception as e:
        print(f"✗ CBAMSpatialEfficientTemporalAttention 导入失败: {e}")
        return False

    print("\n✓ 所有模块导入测试通过!\n")
    return True


def test_rgb_backbone():
    """测试RGB backbone"""
    print("=" * 60)
    print("测试2: RGB Backbone")
    print("=" * 60)

    try:
        from epam_backbone import X3DTemporalShift

        print("\n创建RGB Backbone...")
        model = X3DTemporalShift(
            gamma_w=1,
            gamma_b=2.25,
            gamma_d=2.2,
            se_style='half'
        )
        model.init_weights()
        model.eval()

        print("输入数据...")
        x = torch.randn(1, 3, 16, 224, 224)
        print(f"  输入维度: {x.shape}")

        print("前向传播...")
        with torch.no_grad():
            output = model(x)

        print(f"  输出维度: {output.shape}")
        print(f"  特征通道数: {model.feat_dim}")

        assert output.shape == (1, 432, 16, 7, 7), f"输出维度错误: {output.shape}"

        print("\n✓ RGB Backbone测试通过!\n")
        return True

    except Exception as e:
        print(f"\n✗ RGB Backbone测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_pose_backbone():
    """测试Pose backbone"""
    print("=" * 60)
    print("测试3: Pose Backbone")
    print("=" * 60)

    try:
        from epam_backbone import X3DTemporalShiftPose

        print("\n创建Pose Backbone...")
        model = X3DTemporalShiftPose(
            gamma_d=1,
            in_channels=17,
            base_channels=24,
            num_stages=3,
            stage_blocks=(5, 11, 7),
            spatial_strides=(2, 2, 2),
            conv1_stride=1
        )
        model.init_weights()
        model.eval()

        print("输入数据...")
        x = torch.randn(1, 17, 48, 56, 56)
        print(f"  输入维度: {x.shape}")

        print("前向传播...")
        with torch.no_grad():
            output = model(x)

        print(f"  输出维度: {output.shape}")
        print(f"  特征通道数: {model.feat_dim}")

        assert output.shape == (1, 216, 48, 7, 7), f"输出维度错误: {output.shape}"

        print("\n✓ Pose Backbone测试通过!\n")
        return True

    except Exception as e:
        print(f"\n✗ Pose Backbone测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_attention_module():
    """测试注意力模块"""
    print("=" * 60)
    print("测试4: Attention Module")
    print("=" * 60)

    try:
        from epam_backbone import CBAMSpatialEfficientTemporalAttention

        print("\n创建Attention Module...")
        attention = CBAMSpatialEfficientTemporalAttention(attention_type='nested')

        print("输入数据...")
        x = torch.randn(2, 216, 16, 7, 7)
        print(f"  输入维度: {x.shape}")

        print("前向传播...")
        with torch.no_grad():
            output = attention(x)

        print(f"  输出维度: {output.shape}")

        assert output.shape == (2, 1, 16, 7, 7), f"输出维度错误: {output.shape}"

        print("\n✓ Attention Module测试通过!\n")
        return True

    except Exception as e:
        print(f"\n✗ Attention Module测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_epam_backbone():
    """测试完整EPAM Backbone"""
    print("=" * 60)
    print("测试5: 完整EPAM Backbone")
    print("=" * 60)

    try:
        from epam_backbone import EPAMBackbone

        print("\n创建EPAM Backbone...")
        backbone = EPAMBackbone(
            num_classes=60,
            attention_type='CBAM_spatial_efficient_temporal',
            return_both_streams=True
        )
        backbone.init_weights()
        backbone.eval()

        print("输入数据...")
        rgb_videos = torch.randn(2, 3, 16, 224, 224)
        pose_heatmaps = torch.randn(2, 17, 48, 56, 56)
        print(f"  RGB输入: {rgb_videos.shape}")
        print(f"  Pose输入: {pose_heatmaps.shape}")

        print("前向传播...")
        with torch.no_grad():
            rgb_feat, pose_feat = backbone(rgb_videos, pose_heatmaps)

        print(f"  RGB特征: {rgb_feat.shape}")
        print(f"  Pose特征: {pose_feat.shape}")

        assert rgb_feat.shape == (2, 432, 16, 7, 7), f"RGB特征维度错误: {rgb_feat.shape}"
        assert pose_feat.shape == (2, 216, 48, 7, 7), f"Pose特征维度错误: {pose_feat.shape}"

        print("\n测试特征维度信息...")
        dims = backbone.get_feature_dims()
        print(f"  RGB通道: {dims['rgb_channels']}")
        print(f"  Pose通道: {dims['pose_channels']}")

        print("\n✓ 完整EPAM Backbone测试通过!\n")
        return True

    except Exception as e:
        print(f"\n✗ 完整EPAM Backbone测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """测试梯度流"""
    print("=" * 60)
    print("测试6: 梯度流")
    print("=" * 60)

    try:
        from epam_backbone import EPAMBackbone

        print("\n创建EPAM Backbone...")
        backbone = EPAMBackbone(return_both_streams=False)
        backbone.init_weights()
        backbone.train()

        print("输入数据...")
        rgb_videos = torch.randn(2, 3, 16, 224, 224, requires_grad=True)
        pose_heatmaps = torch.randn(2, 17, 48, 56, 56, requires_grad=True)

        print("前向传播...")
        features = backbone(rgb_videos, pose_heatmaps)

        print("计算损失...")
        loss = features.mean()

        print("反向传播...")
        loss.backward()

        print("检查梯度...")
        has_grad = rgb_videos.grad is not None
        print(f"  输入是否有梯度: {has_grad}")

        if has_grad:
            print(f"  梯度范数: {rgb_videos.grad.norm().item():.6f}")

        print("\n✓ 梯度流测试通过!\n")
        return True

    except Exception as e:
        print(f"\n✗ 梯度流测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_freeze():
    """测试冻结功能"""
    print("=" * 60)
    print("测试7: Backbone冻结")
    print("=" * 60)

    try:
        from epam_backbone import EPAMBackbone

        print("\n创建冻结的Backbone...")
        backbone = EPAMBackbone(freeze_rgb=True, freeze_pose=True)
        backbone.init_weights()

        print("检查参数...")
        rgb_trainable = sum(p.numel() for p in backbone.rgb_backbone.parameters() if p.requires_grad)
        pose_trainable = sum(p.numel() for p in backbone.pose_backbone.parameters() if p.requires_grad)

        print(f"  RGB可训练参数: {rgb_trainable}")
        print(f"  Pose可训练参数: {pose_trainable}")

        assert rgb_trainable == 0, "RGB backbone未正确冻结"
        assert pose_trainable == 0, "Pose backbone未正确冻结"

        print("\n✓ Backbone冻结测试通过!\n")
        return True

    except Exception as e:
        print(f"\n✗ Backbone冻结测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_batch_sizes():
    """测试不同batch size"""
    print("=" * 60)
    print("测试8: 不同Batch Size")
    print("=" * 60)

    try:
        from epam_backbone import EPAMBackbone

        backbone = EPAMBackbone()
        backbone.init_weights()
        backbone.eval()

        batch_sizes = [1, 2, 4, 8]

        for bs in batch_sizes:
            print(f"\n测试batch size = {bs}...")
            rgb_videos = torch.randn(bs, 3, 16, 224, 224)
            pose_heatmaps = torch.randn(bs, 17, 48, 56, 56)

            with torch.no_grad():
                rgb_feat, pose_feat = backbone(rgb_videos, pose_heatmaps)

            assert rgb_feat.shape[0] == bs, f"RGB特征batch size错误"
            assert pose_feat.shape[0] == bs, f"Pose特征batch size错误"
            print(f"  ✓ batch size {bs} 通过")

        print("\n✓ 不同Batch Size测试通过!\n")
        return True

    except Exception as e:
        print(f"\n✗ 不同Batch Size测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("EPAM Backbone 完整测试套件")
    print("=" * 60 + "\n")

    results = []

    # 运行所有测试
    results.append(("模块导入", test_imports()))
    results.append(("RGB Backbone", test_rgb_backbone()))
    results.append(("Pose Backbone", test_pose_backbone()))
    results.append(("Attention Module", test_attention_module()))
    results.append(("完整EPAM Backbone", test_epam_backbone()))
    results.append(("梯度流", test_gradient_flow()))
    results.append(("Backbone冻结", test_freeze()))
    results.append(("不同Batch Size", test_batch_sizes()))

    # 输出总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:.<40} {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
