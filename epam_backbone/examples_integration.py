"""
示例2: 集成到自定义模型 - 完整的动作识别pipeline
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')

from epam_backbone import EPAMBackbone


class ActionRecognitionModel(nn.Module):
    """
    完整的动作识别模型示例
    使用EPAM Backbone进行特征提取，添加自定义分类头
    """
    def __init__(self, num_classes=60, fusion_type='sum'):
        """
        Args:
            num_classes: 动作类别数
            fusion_type: 融合方式 ('sum', 'concat', 'weighted')
        """
        super().__init__()
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        # EPAM Backbone
        self.backbone = EPAMBackbone(
            num_classes=num_classes,
            return_both_streams=True
        )

        # RGB分类头
        self.rgb_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(432, num_classes)
        )

        # Pose分类头
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(216, num_classes)
        )

        # 如果使用concat融合，需要额外的融合层
        if fusion_type == 'concat':
            self.fusion_fc = nn.Linear(num_classes * 2, num_classes)
        elif fusion_type == 'weighted':
            self.rgb_weight = nn.Parameter(torch.tensor(0.5))
            self.pose_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb_videos, pose_heatmaps):
        """
        前向传播

        Args:
            rgb_videos: (N, 3, 16, 224, 224)
            pose_heatmaps: (N, 17, 48, 56, 56)

        Returns:
            logits: (N, num_classes)
        """
        # 提取特征
        rgb_feat, pose_feat = self.backbone(rgb_videos, pose_heatmaps)

        # 分类
        rgb_logits = self.rgb_head(rgb_feat)
        pose_logits = self.pose_head(pose_feat)

        # 融合
        if self.fusion_type == 'sum':
            final_logits = rgb_logits + pose_logits
        elif self.fusion_type == 'concat':
            concat_logits = torch.cat([rgb_logits, pose_logits], dim=1)
            final_logits = self.fusion_fc(concat_logits)
        elif self.fusion_type == 'weighted':
            # 使用softmax归一化权重
            weights = F.softmax(torch.stack([self.rgb_weight, self.pose_weight]), dim=0)
            final_logits = weights[0] * rgb_logits + weights[1] * pose_logits
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        return final_logits


class TwoStreamModel(nn.Module):
    """
    两流模型示例
    分别对RGB和Pose流进行分类，然后融合
    """
    def __init__(self, num_classes=60):
        super().__init__()

        self.backbone = EPAMBackbone(return_both_streams=True)

        # RGB流完整分类器
        self.rgb_classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(432, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Pose流完整分类器
        self.pose_classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(216, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb_videos, pose_heatmaps, return_separate=False):
        """
        前向传播

        Args:
            rgb_videos: RGB输入
            pose_heatmaps: Pose输入
            return_separate: 是否返回两个流的独立预测

        Returns:
            如果return_separate=False: 融合后的logits
            如果return_separate=True: (融合logits, RGB logits, Pose logits)
        """
        # 特征提取
        rgb_feat, pose_feat = self.backbone(rgb_videos, pose_heatmaps)

        # 分类
        rgb_logits = self.rgb_classifier(rgb_feat)
        pose_logits = self.pose_classifier(pose_feat)

        # 融合（简单平均）
        fused_logits = (rgb_logits + pose_logits) / 2

        if return_separate:
            return fused_logits, rgb_logits, pose_logits
        return fused_logits


class LightweightModel(nn.Module):
    """
    轻量级模型示例
    冻结backbone，只训练分类头
    适用于数据较少或计算资源有限的场景
    """
    def __init__(self, num_classes=60):
        super().__init__()

        # 冻结的backbone
        self.backbone = EPAMBackbone(
            freeze_rgb=True,
            freeze_pose=True,
            return_both_streams=False  # 只返回融合特征
        )

        # 轻量级分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(432, num_classes)
        )

    def forward(self, rgb_videos, pose_heatmaps):
        # 提取特征（不计算梯度）
        with torch.no_grad():
            features = self.backbone(rgb_videos, pose_heatmaps)

        # 分类（计算梯度）
        logits = self.classifier(features)
        return logits


def example_training_loop():
    """训练循环示例"""
    print("=" * 60)
    print("示例: 完整的训练循环")
    print("=" * 60)

    # 创建模型
    print("\n[1] 创建模型...")
    model = ActionRecognitionModel(num_classes=60, fusion_type='sum')
    model.backbone.init_weights()

    # 创建优化器
    print("[2] 创建优化器...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 模拟训练数据
    print("[3] 模拟训练...")
    model.train()

    for epoch in range(2):  # 只训练2个epoch作为示例
        print(f"\nEpoch {epoch + 1}/2")

        for batch_idx in range(3):  # 只训练3个batch
            # 模拟数据
            rgb_videos = torch.randn(4, 3, 16, 224, 224)
            pose_heatmaps = torch.randn(4, 17, 48, 56, 56)
            labels = torch.randint(0, 60, (4,))

            # 前向传播
            logits = model(rgb_videos, pose_heatmaps)
            loss = criterion(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx == 0:
                print(f"  Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

    print("\n✓ 训练示例完成!\n")


def example_evaluation():
    """评估示例"""
    print("=" * 60)
    print("示例: 模型评估")
    print("=" * 60)

    # 创建模型
    print("\n[1] 创建模型...")
    model = TwoStreamModel(num_classes=60)
    model.backbone.init_weights()
    model.eval()

    # 模拟测试数据
    print("[2] 模拟测试...")
    num_samples = 10
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(num_samples):
            # 模拟数据
            rgb_videos = torch.randn(1, 3, 16, 224, 224)
            pose_heatmaps = torch.randn(1, 17, 48, 56, 56)
            label = torch.randint(0, 60, (1,))

            # 预测
            fused_logits, rgb_logits, pose_logits = model(
                rgb_videos, pose_heatmaps, return_separate=True
            )

            # 计算准确率
            _, predicted = torch.max(fused_logits, 1)
            total += 1
            correct += (predicted == label).item()

    accuracy = 100 * correct / total
    print(f"[3] 模拟准确率: {accuracy:.2f}%")
    print("✓ 评估示例完成!\n")


def example_feature_extraction():
    """特征提取并保存示例"""
    print("=" * 60)
    print("示例: 特征提取与保存")
    print("=" * 60)

    print("\n[1] 创建backbone...")
    backbone = EPAMBackbone(return_both_streams=True)
    backbone.init_weights()
    backbone.eval()

    # 提取特征
    print("[2] 提取特征...")
    all_rgb_features = []
    all_pose_features = []
    all_labels = []

    num_videos = 5
    with torch.no_grad():
        for i in range(num_videos):
            rgb_videos = torch.randn(1, 3, 16, 224, 224)
            pose_heatmaps = torch.randn(1, 17, 48, 56, 56)
            label = i  # 模拟标签

            rgb_feat, pose_feat = backbone(rgb_videos, pose_heatmaps)

            # 全局平均池化，转为向量
            rgb_vec = F.adaptive_avg_pool3d(rgb_feat, 1).squeeze()
            pose_vec = F.adaptive_avg_pool3d(pose_feat, 1).squeeze()

            all_rgb_features.append(rgb_vec.cpu())
            all_pose_features.append(pose_vec.cpu())
            all_labels.append(label)

    # 保存特征
    print("[3] 保存特征...")
    features_dict = {
        'rgb_features': torch.stack(all_rgb_features),
        'pose_features': torch.stack(all_pose_features),
        'labels': torch.tensor(all_labels)
    }

    # torch.save(features_dict, 'extracted_features.pth')
    print(f"    RGB特征维度: {features_dict['rgb_features'].shape}")
    print(f"    Pose特征维度: {features_dict['pose_features'].shape}")
    print("✓ 特征提取示例完成!\n")


def example_custom_fusion():
    """自定义融合策略示例"""
    print("=" * 60)
    print("示例: 自定义融合策略")
    print("=" * 60)

    class AttentionFusionModel(nn.Module):
        """使用注意力机制融合两个流的预测"""
        def __init__(self, num_classes=60):
            super().__init__()
            self.backbone = EPAMBackbone(return_both_streams=True)

            self.rgb_head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(432, num_classes)
            )

            self.pose_head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(216, num_classes)
            )

            # 注意力权重网络
            self.attention = nn.Sequential(
                nn.Linear(num_classes * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=1)
            )

        def forward(self, rgb_videos, pose_heatmaps):
            rgb_feat, pose_feat = self.backbone(rgb_videos, pose_heatmaps)

            rgb_logits = self.rgb_head(rgb_feat)
            pose_logits = self.pose_head(pose_feat)

            # 计算注意力权重
            concat_logits = torch.cat([rgb_logits, pose_logits], dim=1)
            weights = self.attention(concat_logits)  # (N, 2)

            # 加权融合
            fused_logits = (weights[:, 0:1] * rgb_logits +
                           weights[:, 1:2] * pose_logits)

            return fused_logits

    print("\n[1] 创建注意力融合模型...")
    model = AttentionFusionModel(num_classes=60)
    model.backbone.init_weights()

    print("[2] 测试前向传播...")
    rgb_videos = torch.randn(2, 3, 16, 224, 224)
    pose_heatmaps = torch.randn(2, 17, 48, 56, 56)

    model.eval()
    with torch.no_grad():
        logits = model(rgb_videos, pose_heatmaps)

    print(f"[3] 输出维度: {logits.shape}")
    print("✓ 自定义融合示例完成!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("EPAM Backbone 集成示例")
    print("=" * 60 + "\n")

    # 运行所有示例
    example_training_loop()
    example_evaluation()
    example_feature_extraction()
    example_custom_fusion()

    print("=" * 60)
    print("所有集成示例运行完毕!")
    print("=" * 60)
