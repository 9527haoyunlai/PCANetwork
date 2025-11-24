# 配置文件说明

本目录包含三个主要实验的配置文件。

## 📁 配置文件列表

### 1. `rgbpose_conv3d.py`
**实验1: 原始Backbone + MA52数据集**
- 数据集: MA52 (52个动作类别)
- Backbone: RGBPoseConv3D
- RGB通道: 2048维
- Pose通道: 512维
- 骨架关键点: 28点
- 训练脚本: `train_ma52_original.sh`

### 2. `pcan_ntu60.py`
**实验2: 原始Backbone + NTU60数据集**
- 数据集: NTU RGB+D 60 (60个动作类别)
- Backbone: RGBPoseConv3D
- RGB通道: 2048维
- Pose通道: 512维
- 骨架关键点: 17点
- 分层分类: 8个粗类 + 60个细类
- 训练脚本: `train_ntu60_original.sh`
- 预期性能: ~85-87%

### 3. `pcan_ntu60_x3d.py` ⭐
**实验3: X3D Backbone + NTU60数据集**
- 数据集: NTU RGB+D 60 (60个动作类别)
- Backbone: X3D TemporalShift (轻量高效)
- RGB通道: 432维 (X3D-M)
- Pose通道: 216维 (X3D-S)
- 参数量: ~15M (减少70%)
- 骨架关键点: 17点
- 分层分类: 8个粗类 + 60个细类
- 训练脚本: `train_ntu60_x3d.sh`
- **历史最佳**: **90.44%** @ Epoch 78

---

## 🔄 切换实验

**无需修改配置文件！** 直接运行对应的训练脚本即可：

```bash
# 实验1
bash train_ma52_original.sh

# 实验2
bash train_ntu60_original.sh

# 实验3 (推荐)
bash train_ntu60_x3d.sh
```

---

## 📝 关键配置项说明

### Backbone配置

**原始RGBPoseConv3D** (实验1、2):
```python
backbone_cfg = dict(
    type='RGBPoseConv3D',
    rgb_pathway=dict(
        base_channels=64,
        out_channels=2048  # RGB输出通道
    ),
    pose_pathway=dict(
        base_channels=32,
        out_channels=512   # Pose输出通道
    )
)
```

**X3D TemporalShift** (实验3):
```python
rgb_backbone = dict(
    gamma_w=1,              # 宽度因子
    gamma_b=2.25,           # 瓶颈因子
    gamma_d=2.2,            # 深度因子
    out_channels=432        # RGB输出通道
)

pose_backbone = dict(
    gamma_d=1,
    out_channels=216        # Pose输出通道
)
```

### Head配置

**通用设置**:
```python
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,              # NTU60: 60类; MA52: 52类
    num_coarse_classes=8,        # 粗类别数（仅NTU60）
    in_channels=[2048, 512],     # 原始网络
    # in_channels=[432, 216],    # X3D网络
    loss_components=['rgb', 'pose'],
    loss_weights=[1.0, 1.5, 0.6, 1.2]
)
```

### 数据增强

**通用pipeline**:
- 统一采样帧: RGB 16帧, Pose 48帧
- 图像resize: 256x256 → 224x224
- RandomResizedCrop: (0.56, 1.0)
- 水平翻转: 50%概率
- 骨架热图生成: sigma=0.7

---

## ⚙️ 优化建议

如果想进一步提升实验3的性能 (90% → 91-92%)，可以修改 `pcan_ntu60_x3d.py`:

### 1. 学习率优化
```python
optim_wrapper = dict(
    optimizer=dict(
        lr=0.004,  # 从0.012降到0.004
        weight_decay=0.0004  # 从0.0002提高
    ),
    clip_grad=dict(max_norm=30)  # 从40降到30
)
```

### 2. 学习率调度优化
```python
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=5e-5,  # 从1e-6提高到5e-5
        T_max=65,      # 从75降到65
        end=70         # 从80降到70
    )
]
```

### 3. 训练周期优化
```python
train_cfg = dict(
    max_epochs=70,  # 从80降到70
    val_interval=2
)

default_hooks = dict(
    early_stopping=dict(
        patience=8,      # 从15降到8
        min_delta=0.001  # 从0.0005提高
    )
)
```

详细优化说明请参考项目根目录的 `TRAINING_GUIDE.md`。

---

## 📊 性能对比

| 配置文件 | 数据集 | Backbone | 参数量 | 准确率 |
|---------|--------|---------|--------|--------|
| rgbpose_conv3d.py | MA52 | RGBPoseConv3D | ~50M | ~80-85% |
| pcan_ntu60.py | NTU60 | RGBPoseConv3D | ~50M | ~85-87% |
| pcan_ntu60_x3d.py | NTU60 | X3D TemporalShift | ~15M | **90.44%** ✨ |

---

## 💡 提示

- **实验3 (X3D)** 需要 `emap_backbone/` 目录
- 所有配置文件都使用 **2卡GPU并行训练**
- 切换实验 **无需修改任何Python代码**
- 查看详细训练指南: `../../../../../../TRAINING_GUIDE.md`

---

**最后更新**: 2025-11-24

