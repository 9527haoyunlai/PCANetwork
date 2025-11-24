# EPAM Backbone 快速入门指南

## ✅ 已完成的工作

我已经成功将EPAM-Net的backbone部分提取并重构为独立模块，完全移除了mmcv依赖。以下是完成的内容：

### 📁 文件结构

```
epam_backbone/
├── __init__.py                      # 模块初始化，导出主要类
├── utils.py                         # 工具函数（替代mmcv）
├── attention_module.py              # 注意力模块（CBAM等）
├── x3d_temporal_shift_rgb.py        # RGB流backbone
├── x3d_temporal_shift_pose.py       # Pose流backbone
├── epam_backbone.py                 # 主干网络封装类
├── README.md                        # 详细文档
├── requirements.txt                 # 依赖列表
├── test_all.py                      # 完整测试套件
├── examples_basic.py                # 基础使用示例
└── examples_integration.py          # 集成示例
```

### 🎯 核心功能

1. **RGB Stream (X3DTemporalShift)**
   - 输入: (N, 3, 16, 224, 224)
   - 输出: (N, 432, 16, 7, 7)
   - 特点: Temporal Shift + SE Module

2. **Pose Stream (X3DTemporalShiftPose)**
   - 输入: (N, 17, 48, 56, 56)
   - 输出: (N, 216, 48, 7, 7)
   - 特点: 处理骨架热图，密集时序采样

3. **Attention Module**
   - CBAM空间-时序注意力
   - 引导RGB特征学习

4. **EPAM Backbone**
   - 统一封装，易于集成
   - 支持预训练权重加载
   - 支持backbone冻结

## 🚀 快速开始

### 1. 最简单的使用

```python
from epam_backbone import EPAMBackbone
import torch

# 创建backbone
backbone = EPAMBackbone()
backbone.init_weights()

# 准备数据
rgb = torch.randn(1, 3, 16, 224, 224)
pose = torch.randn(1, 17, 48, 56, 56)

# 提取特征
rgb_feat, pose_feat = backbone(rgb, pose)
```

### 2. 集成到你的模型

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 使用EPAM Backbone
        self.backbone = EPAMBackbone(return_both_streams=True)
        self.backbone.init_weights()

        # 添加你的分类头
        self.classifier = nn.Linear(432, num_classes)

    def forward(self, rgb, pose):
        rgb_feat, pose_feat = self.backbone(rgb, pose)

        # 池化
        rgb_vec = F.adaptive_avg_pool3d(rgb_feat, 1).flatten(1)

        # 分类
        logits = self.classifier(rgb_vec)
        return logits
```

### 3. 加载预训练权重

```python
backbone = EPAMBackbone(
    rgb_pretrained='path/to/rgb_pretrained.pth',
    pose_pretrained='path/to/pose_pretrained.pth'
)
backbone.init_weights()
```

### 4. 冻结backbone进行微调

```python
backbone = EPAMBackbone(
    freeze_rgb=True,
    freeze_pose=True
)
```

## 📊 输入输出规格

### 输入

| 流 | 维度 | 说明 |
|----|------|------|
| RGB | (N, 3, 16, 224, 224) | RGB视频，16帧 |
| Pose | (N, 17, 48, 56, 56) | 17关节热图，48帧 |

### 输出

| 流 | 维度 | 说明 |
|----|------|------|
| RGB特征 | (N, 432, 16, 7, 7) | 注意力引导的RGB特征 |
| Pose特征 | (N, 216, 48, 7, 7) | Pose流特征 |

### 关节点顺序 (COCO-17格式)

```
0: 鼻子
1-2: 左右眼
3-4: 左右耳
5-6: 左右肩
7-8: 左右肘
9-10: 左右腕
11-12: 左右髋
13-14: 左右膝
15-16: 左右踝
```

## 🔧 关键特性

### 1. 零mmcv依赖
所有mmcv功能已用纯PyTorch重写：
- `ConvModule` → 自定义ConvModule
- `kaiming_init` → torch.nn.init
- `load_checkpoint` → torch.load

### 2. 时序对齐
Pose特征(48帧)自动下采样到RGB帧数(16帧)：
```python
# 每3帧采样一次
indices = [0, 3, 6, ..., 45]
```

### 3. 注意力机制
```
Pose特征 → 空间注意力 → 时序注意力 → 注意力图
                                    ↓
RGB特征 × 注意力图 + RGB特征 → 增强的RGB特征
```

### 4. 灵活的融合策略
```python
# 返回两个流的特征
rgb_feat, pose_feat = backbone(rgb, pose)

# 或只返回融合特征
backbone = EPAMBackbone(return_both_streams=False)
fused_feat = backbone(rgb, pose)
```

## 📝 示例代码

### examples_basic.py
包含6个基础示例：
1. 基础使用
2. 使用预训练权重
3. 冻结backbone
4. 仅特征提取
5. GPU加速
6. 批处理

### examples_integration.py
包含高级集成示例：
1. 完整训练循环
2. 模型评估
3. 特征提取与保存
4. 自定义融合策略

### test_all.py
完整的测试套件，验证所有功能：
```bash
cd epam_backbone
python test_all.py
```

## 🎓 使用场景

### 场景1: 替换其他模型的backbone

```python
# 原模型
class OriginalModel:
    def __init__(self):
        self.backbone = SomeBackbone()
        self.head = Classifier()

# 替换为EPAM Backbone
class NewModel:
    def __init__(self):
        self.backbone = EPAMBackbone()  # ← 直接替换
        self.head = Classifier()
```

### 场景2: 多任务学习

```python
backbone = EPAMBackbone()

# 共享backbone，多个任务头
action_head = ActionClassifier(432)
scene_head = SceneClassifier(432)
object_head = ObjectDetector(432)
```

### 场景3: 知识蒸馏

```python
# 教师模型
teacher = EPAMBackbone()
teacher.load_state_dict(pretrained)
teacher.eval()

# 学生模型
student = LightweightModel()

# 蒸馏训练
with torch.no_grad():
    teacher_feat = teacher(rgb, pose)
student_feat = student(rgb, pose)
loss = distillation_loss(student_feat, teacher_feat)
```

## ⚠️ 注意事项

1. **内存占用**
   - RGB+Pose双流需要较多内存
   - 建议batch size: 4-8（单GPU）
   - 可冻结某个流减少内存

2. **数据预处理**
   - RGB需要归一化到ImageNet统计
   - Pose热图需要高斯分布生成
   - 详见README中的预处理部分

3. **预训练权重**
   - 原项目权重可能需要键名转换
   - 使用`strict=False`加载以忽略分类头

4. **版本兼容性**
   - PyTorch >= 1.7.0
   - 无其他第三方依赖

## 📚 更多信息

- **完整文档**: 查看 `README.md`
- **API参考**: 查看各模块的docstring
- **问题反馈**: 提交到项目Issue

## 🎉 总结

EPAM Backbone已成功提取并优化：
- ✅ 完全独立，无mmcv依赖
- ✅ 接口简洁，易于集成
- ✅ 文档完善，示例丰富
- ✅ 测试充分，稳定可靠

现在你可以直接将`epam_backbone`文件夹复制到任何项目中使用！
