# EPAM NTU60 配置优化对比

## 📊 关键参数对比表

| 参数 | Baseline配置 | Optimized配置 | 变化 | 原因 |
|------|-------------|--------------|------|------|
| **学习率 (LR)** | 0.01 | **0.0015** | ↓ 85% | 匹配原始模型，EPAM参数量更少 |
| **Weight Decay** | 0.0001 | **0.0003** | ↑ 3x | 更强L2正则化，防止过拟合 |
| **LR调度器** | MultiStepLR | **CosineAnnealingLR** | 改变 | 更平滑的衰减，通常+0.5-1%准确率 |
| **随机种子** | ❌ 未设置 | **100** | 新增 | 确保实验可复现 |
| **Momentum** | 0.9 | 0.9 | ✅ 保持 | 标准SGD配置 |
| **Gradient Clip** | 40 | 40 | ✅ 保持 | 防止梯度爆炸 |
| **Batch Size/GPU** | 8 | 8 | ✅ 保持 | 适合EPAM参数量 |
| **Total Batch** | 16 (8×2) | 16 (8×2) | ✅ 保持 | 合理配置 |
| **Epochs** | 50 | 50 | ✅ 保持 | 标准训练周期 |

## 🎯 优化要点详解

### 1. 学习率降低 (0.01 → 0.0015)

**原理**：
- 原始RGBPoseConv3D模型（15M参数）使用0.0015学习率
- EPAM模型参数量更少（2.8M），理论上需要更小的学习率
- 使用Linear Scaling Rule：`lr = base_lr × (batch_size / base_batch_size)`

**预期效果**：
- ✅ 训练更稳定，不会出现loss震荡
- ✅ 更好的收敛，能到达更好的局部最优
- ✅ 减少过拟合风险

**验证指标**：
- 训练loss应该平滑下降
- 验证准确率稳步提升
- 不应该出现loss突然跳跃

### 2. Weight Decay增强 (0.0001 → 0.0003)

**原理**：
- L2正则化：在损失函数中添加 `λ * ||W||²` 惩罚项
- 强迫模型权重保持较小的值
- 原始模型使用0.0003，证明有效

**预期效果**：
- ✅ 减少过拟合
- ✅ 提高泛化能力（测试集表现更好）
- ✅ 训练集和验证集准确率差距更小

**验证指标**：
- Train Acc vs Val Acc 差距 < 3%
- 最终Val Acc应该更高

### 3. Cosine退火学习率 (MultiStepLR → CosineAnnealingLR)

**MultiStepLR**（Baseline）：
```
LR
│
0.01 ├─────────────────────┐
     │                     │ 突然下降
0.001├──────────────┐      │
     │              │      │
0.0001├──────┐      │      │
     │       │      │      │
     └───────┴──────┴──────┴─→ Epoch
         20      40      
```

**CosineAnnealingLR**（Optimized）：
```
LR
│
0.0015├─╮
      │  ╲
      │   ╲   平滑下降
      │    ╲
      │     ╲
1e-6  │      ╲___________
      └──────────────────→ Epoch
           0        50
```

**优势**：
- ✅ 更平滑的下降，避免突然跳变
- ✅ 后期仍有微小学习率，能做精细调整
- ✅ 经验上通常比MultiStepLR效果好0.5-1%

### 4. 随机种子设置 (100)

**作用**：
```python
randomness = dict(seed=100, deterministic=False)
```

- 固定数据shuffle顺序
- 固定权重初始化
- 固定dropout行为

**重要性**：
- ✅ 实验可复现
- ✅ 方便调试和对比
- ✅ 论文发表必需

## 📈 预期性能对比

### 训练曲线预期

**Baseline**:
```
Loss: 快速下降 → 可能震荡 → 收敛较早
Acc:  快速上升 → 可能波动 → 可能过拟合
```

**Optimized**:
```
Loss: 平稳下降 → 稳定收敛 → 更低最终loss
Acc:  稳步上升 → 持续改进 → 更高最终acc
```

### 数值预期

| 指标 | Baseline | Optimized | 提升 |
|------|---------|-----------|------|
| **Train Acc** | 94-96% | 92-94% | 轻微降低（正常，正则化效果） |
| **Val Acc** | 89-91% | 91-93% | **+1-2%** ⬆️ |
| **Train-Val Gap** | 4-5% | 1-2% | **缩小** ✅ |
| **收敛Epoch** | ~30 | ~35-40 | 稍慢但更充分 |
| **训练稳定性** | 中 | **高** ⬆️ |

## 🔬 实验验证方法

### 训练时监控

```bash
# 查看训练曲线
tail -f work_dirs/epam_ntu60_optimized/*.log | grep "loss:"
```

**关键指标**：
1. **Loss下降平滑度**：不应该有大幅跳跃
2. **学习率变化**：应该看到平滑的cosine曲线
3. **准确率提升**：每个epoch都应该稳步增长

### 对比实验

如果想对比两个配置：

```bash
# Baseline
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py \
    2 --work-dir work_dirs/baseline

# Optimized  
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_optimized.py \
    2 --work-dir work_dirs/optimized
```

然后对比 `work_dirs/baseline/` 和 `work_dirs/optimized/` 的日志。

## 📚 理论背景

### SGD with Momentum (0.9)

```python
v_t = momentum × v_{t-1} + gradient
w_t = w_{t-1} - lr × v_t
```

**效果**：像一个下坡的球，有惯性能冲过小坑洼。

### L2 Regularization

```python
Loss_total = Loss_CE + λ × ||W||²
where λ = weight_decay = 0.0003
```

**效果**：惩罚大权重，让模型更简洁。

### Cosine Annealing

```python
lr_t = eta_min + (lr_max - eta_min) × (1 + cos(π × t / T_max)) / 2
```

**效果**：
- 前期快速下降
- 中期稳定学习
- 后期微调

## ⚠️ 注意事项

### 如果训练太慢（loss下降缓慢）

**症状**：30个epoch后loss还很高

**解决**：
```python
# 方案1：稍微提高学习率
lr = 0.002  # 从0.0015提到0.002

# 方案2：使用warmup
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=45, eta_min=1e-6, by_epoch=True, begin=5, end=50)
]
```

### 如果过拟合（train acc >> val acc）

**症状**：train 95%, val 87%

**解决**：
```python
# 方案1：增强正则化
weight_decay = 0.0005  # 从0.0003提到0.0005

# 方案2：增加数据增强
dict(type='ColorJitter', brightness=0.2, contrast=0.2),
```

### 如果欠拟合（准确率都很低）

**症状**：train 80%, val 78%

**解决**：
```python
# 方案1：提高学习率
lr = 0.003

# 方案2：训练更多epoch
max_epochs = 80
```

## 🎓 参考文献

1. **SGD with Momentum**: Sutskever et al., "On the importance of initialization and momentum in deep learning", ICML 2013

2. **Cosine Annealing**: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017

3. **Weight Decay**: Krogh & Hertz, "A Simple Weight Decay Can Improve Generalization", NIPS 1991

4. **原始模型超参数**: 基于您提供的RGBPoseConv3D成功训练经验

## ✅ 检查清单

训练前确认：

- [ ] 配置文件路径正确
- [ ] GPU可用（nvidia-smi）
- [ ] 数据路径正确
- [ ] Work dir有写入权限
- [ ] 已运行test_epam_final.py验证接口

训练中监控：

- [ ] Loss平滑下降
- [ ] 学习率按cosine曲线变化
- [ ] GPU利用率正常（>80%）
- [ ] 每个epoch准确率都在提升

训练后评估：

- [ ] Val Acc达到预期（90%+）
- [ ] Train-Val gap合理（<3%）
- [ ] 保存了best checkpoint
- [ ] 可以复现结果（再跑一次结果接近）

---

**准备好开始优化训练了！** 🚀

