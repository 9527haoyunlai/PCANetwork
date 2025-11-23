# 🎯 PCAN NTU-60 冲击95%准确率完整方案

## 📊 现状分析

| 项目 | 当前状态 | 目标 | 差距 |
|------|---------|------|------|
| **验证准确率** | 89.19% | **95%+** | **+5.81%** |
| RGB分支 | 89.00% | 95%+ | +6% |
| Pose分支 | 82.50% | 90%+ | +7.5% |

**关键认知**：从89%到95%不是微调，是**质的飞跃**！需要系统性改进。

---

## 🔍 NTU-60数据集SOTA参考

| 方法 | 准确率 | 关键技术 |
|------|--------|---------|
| InfoGCN | 93.0% | 图卷积 |
| CTR-GCN | 92.4% | Channel-wise Topology |
| **PoseConv3D** | 94.1% | RGB+Pose双流 |
| **PCAN (理论上限)** | **94-95%** | 跨模态注意力 |

**结论**：95%是achievable的，但需要充分挖掘潜力！

---

## 🚀 完整优化方案（10个维度）

### **1. 训练策略：从头训练100个epoch（关键！）**

```python
# 不要从epoch 26继续！从头训练才能充分学习

# ==========================================
# 训练配置
# ==========================================
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=100,  # ← 100个epoch充分训练
    val_begin=1, 
    val_interval=1)

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', 
        lr=0.01,              # ← 从头训练用正常学习率
        momentum=0.9, 
        weight_decay=0.0003), # ← 增加正则化
    clip_grad=dict(max_norm=40, norm_type=2))

# 学习率策略：Cosine with Warm Restarts
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=1e-6,
        by_epoch=True,
        begin=5,
        end=100)
]

# 不加载任何checkpoint
load_from = None
resume = False
```

---

### **2. 数据增强：激进策略（RGB分支）**

```python
train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    
    # ← 更激进的数据增强
    dict(type='RandomResizedCrop', area_range=(0.40, 1.0)),  # 0.56→0.40
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    
    # ← 强化颜色增强
    dict(type='ColorJitter', 
         brightness=0.4,    # 0.3→0.4
         contrast=0.4, 
         saturation=0.4, 
         hue=0.15),         # 0.1→0.15
    
    # ← 新增：随机擦除（模拟遮挡）
    dict(type='RandomErasing', 
         probability=0.25,
         min_area_ratio=0.02,
         max_area_ratio=0.2),
    
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]
```

---

### **3. Loss权重：平衡双分支**

```python
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,
    num_coarse_classes=8,
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],
    loss_weights=[1.0, 1.5, 0.6, 1.2],  # ← 平衡配置
    #             ↑    ↑    ↑    ↑
    #           RGB  Pose RGB粗 Pose粗
    # RGB主分支: 1.0 (基准)
    # Pose主分支: 1.5 (适度提升，不要2.0)
    # RGB粗分类: 0.6 (辅助)
    # Pose粗分类: 1.2 (Pose需要更多层次监督)
    average_clips='prob')
```

---

### **4. 测试时增强（TTA）：提升1-2%**

```python
# test_pipeline中使用多clip
test_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=10,        # ← 10个clip取平均
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    
    # ← TTA: 多尺度测试
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

# 进一步提升：测试时翻转增强（需要自定义代码）
# 左右翻转 + 原始 = 2倍TTA
# 可以额外提升0.5-1%
```

---

### **5. 模型架构：增强跨模态融合**

```python
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,    # ← 保持，已经够大
    rgb_pathway=dict(
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        fusion_kernel=7,
        base_channels=64,  # ← 可以考虑增加到80（但会增加显存）
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1),
        with_pool2=False),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        fusion_kernel=7,
        in_channels=17,
        base_channels=32,  # ← 可以考虑增加到48
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride_s=1,
        conv1_stride_t=1,
        pool1_stride_s=1,
        pool1_stride_t=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        with_pool2=False))
```

---

### **6. 批量大小：增大提升稳定性**

```python
train_dataloader = dict(
    batch_size=24,        # ← 20→24 (如果显存够)
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        split='xsub_train',
        pipeline=train_pipeline))

# 相应调整学习率（线性缩放）
# lr = 0.01 * (24/20) = 0.012
```

---

### **7. 早停策略：更宽容**

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5,
        save_best='acc/RGBPose_1:1_top1',
        rule='greater',
        max_keep_ckpts=10),  # ← 保留更多checkpoint
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='acc/RGBPose_1:1_top1',
        patience=20,         # ← 15→20
        min_delta=0.0003))   # ← 0.0005→0.0003
```

---

### **8. Label Smoothing（可选）**

```python
# 在RGBPoseHead中添加label smoothing
# 修改 mmaction/models/heads/rgbpose_head.py

class RGBPoseHead(BaseHead):
    def __init__(self, ..., label_smooth=0.1):
        self.label_smooth = label_smooth
    
    def loss(self, ...):
        # 使用label smoothing的交叉熵
        # 可以提升0.5-1%泛化能力
```

---

### **9. Mixup/Cutmix数据增强（高级）**

```python
# 需要自定义实现
# Mixup: 混合两个样本
# Cutmix: 裁剪粘贴两个样本
# 在视频数据上实现比较复杂，但效果显著（+1-2%）

train_pipeline = [
    # ... 前面的pipeline
    dict(type='VideoMixup', alpha=0.2, prob=0.5),  # 需要自己实现
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]
```

---

### **10. 模型集成（Ensemble）：最后的杀手锏**

```python
# 训练3个模型：
# 1. PCAN (当前)
# 2. PCAN + 不同随机种子
# 3. PCAN + 不同数据增强

# 测试时融合预测：
# final_pred = 0.4 * model1 + 0.3 * model2 + 0.3 * model3
# 可以提升1-2%
```

---

## 📈 预期效果

### **各阶段目标**

| Epoch | 预期准确率 | 说明 |
|-------|----------|------|
| 10-20 | 88-90% | 基础学习 |
| 30-40 | 90-92% | 性能提升 |
| 50-60 | 92-93% | 接近收敛 |
| 70-80 | 93-94% | 精细调整 |
| **90-100** | **94-95%** | **目标达成** |

### **各优化贡献估算**

| 优化项 | 预期提升 | 难度 | 优先级 |
|--------|---------|------|--------|
| 从头训练100 epoch | +2-3% | ⭐ | 🔥🔥🔥 |
| 更强数据增强 | +1-2% | ⭐⭐ | 🔥🔥 |
| TTA (10 clips) | +1-1.5% | ⭐ | 🔥🔥 |
| 优化loss权重 | +0.5-1% | ⭐ | 🔥 |
| Label smoothing | +0.5-1% | ⭐⭐ | 🔥 |
| 增大batch size | +0.3-0.5% | ⭐ | 🔥 |
| Mixup/Cutmix | +1-2% | ⭐⭐⭐⭐ | 💡 |
| 模型集成 | +1-2% | ⭐⭐ | 💡 |
| **总计** | **+7-13%** | - | - |

**从89% → 95-96%是可行的！**

---

## 🛠️ 实施步骤

### **Phase 1: 基础优化（预期达到92-93%）**

1. ✅ 从头训练100个epoch
2. ✅ 优化loss权重
3. ✅ 增强数据增强
4. ✅ 调整学习率策略

**预计时间**: 约20小时（100 epoch × 12分钟）

---

### **Phase 2: 进阶优化（预期达到94%）**

1. ✅ 实现TTA（测试时10 clips）
2. ✅ 增大batch size
3. ✅ 优化early stopping

**预计时间**: 已包含在Phase 1

---

### **Phase 3: 高级优化（冲击95%+）**

1. 💡 实现Label Smoothing
2. 💡 实现Mixup/Cutmix
3. 💡 训练多个模型做ensemble

**预计时间**: 额外40-60小时

---

## 📝 立即可用的完整配置

我现在帮你生成一个完整的配置文件，包含上述所有基础优化（Phase 1 + Phase 2）。

**关键决策**：
- ✅ 从头训练100个epoch（不加载checkpoint）
- ✅ 激进数据增强
- ✅ 优化loss权重
- ✅ TTA (10 clips)
- ⏸️ Label Smoothing（需要修改代码，Phase 3）
- ⏸️ Mixup/Cutmix（需要修改代码，Phase 3）

---

## ⚠️ 重要提醒

### **1. 硬件要求**
- **GPU**: 至少2×A100 40GB
- **时间**: 约20小时连续训练
- **存储**: 至少50GB（checkpoints）

### **2. 风险评估**
- 100个epoch可能过拟合 → 用early stopping保护
- 数据增强太强可能降低性能 → 可以适度调整
- 从头训练可能不如epoch 26 → 但天花板更高

### **3. 备选方案**
如果Phase 1训练到50 epoch还没超过89%：
- 降低数据增强强度
- 调整loss权重
- 考虑加载epoch 26作为预训练

---

**要我现在生成完整的配置文件吗？我会创建一个针对95%目标的优化版本。**

