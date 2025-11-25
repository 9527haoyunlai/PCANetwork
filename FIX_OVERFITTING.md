# 过拟合问题修复方案

## 问题诊断

### 严重过拟合的证据
```
训练准确率 (Epoch 23): RGB 75%, Pose 87.5%
验证准确率 (Epoch 22): RGBPose 11.09%
差距: 75% vs 11% = 64%的巨大鸿沟！
```

### 准确率变化轨迹
```
Epoch 1:  3.26%  ✓
Epoch 3:  8.15%  ✓
Epoch 4:  12.07% ✓
Epoch 5:  13.65% ✓
Epoch 6:  17.55% ✓
Epoch 9:  18.34% ✓ ← 峰值后开始下降
Epoch 22: 11.09% ✗ ← 崩溃
```

**验证准确率在Epoch 9达到峰值18.34%，之后持续下降到11.09%**

---

## 根本原因

1. **学习率衰减过快**
   - Cosine annealing在50 epochs内衰减到接近0
   - 模型在早期学到的知识后期被"忘记"

2. **Weight Decay过大**
   - 0.0003的weight decay对4.9M参数的小模型来说可能过强
   - 限制了模型的表达能力

3. **没有Early Stopping**
   - 应该在Epoch 9就停止训练
   - 继续训练反而导致性能恶化

4. **模型容量较小**
   - EPAM只有4.9M参数
   - 可能需要更温和的训练策略

---

## 修复方案

### 核心改动（在 `epam_ntu60_fixed.py`）

| 参数 | 原值 | 修改值 | 原因 |
|------|------|--------|------|
| **学习率** | 0.0015 | **0.003** | 给模型更多探索空间 |
| **Weight Decay** | 0.0003 | **0.00005** | 减少过度正则化（降低6倍） |
| **LR周期 (T_max)** | 50 | **100** | 更缓慢的学习率衰减 |
| **Max Epochs** | 50 | **100** | 延长训练窗口 |
| **Dropout** | 0.5 | **0.6** | 增强正则化 |
| **Loss权重** | [1.0, 1.2, 0.5, 0.9] | **[1.0, 1.2, 0.3, 0.6]** | 降低特征修复权重 |

### 新增Early Stopping
```python
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='acc/RGBPose_1:1_top1',
        patience=10,  # 10个epoch不提升就停止
        rule='greater',
        min_delta=0.001
    )
]
```

---

## 训练命令

### 立即开始训练（GPU 1和2）
```bash
cd /home/zh/ChCode/codes01/mmaction2

CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_fixed.py \
    2 \
    --work-dir work_dirs/epam_ntu60_fixed
```

---

## 预期效果

1. **更稳定的学习曲线**
   - 学习率衰减更平缓，模型有更长时间适应

2. **更好的泛化能力**
   - 降低weight decay减少了过度正则化
   - Early stopping防止过度训练

3. **目标准确率**
   - 保守估计：**25-30%** (比当前18.34%峰值提升)
   - 乐观估计：**35-40%** (如果Early stopping在合适时机停止)

---

## 监控指标

训练时重点关注：

```bash
# 查看实时日志
tail -f work_dirs/epam_ntu60_fixed/*/[日期时间].log

# 关键指标
acc/RGBPose_1:1_top1  # 主要指标（越高越好）
loss                  # 训练loss（应该平稳下降）
grad_norm             # 梯度范数（应该稳定，不要爆炸）
```

---

## 如果还是过拟合怎么办？

### 更激进的修改方案

如果上述修改后仍然过拟合，可以尝试：

1. **进一步降低weight decay**
   ```python
   weight_decay=0.00001  # 从0.00005再降低5倍
   ```

2. **使用MultiStepLR替代CosineAnnealing**
   ```python
   param_scheduler = [
       dict(type='MultiStepLR',
            milestones=[30, 60, 80],
            gamma=0.1)
   ]
   ```

3. **增加数据增强**
   - 增大RandomResizedCrop的范围
   - 增加更多的数据增强技巧

4. **考虑从Kinetics预训练模型微调**
   - EPAM可能需要预训练权重

---

## 杀死当前训练并重新开始

```bash
# 1. 查找训练进程
ps aux | grep dist_train

# 2. 杀死进程
kill -9 [PID]

# 3. 清理旧的工作目录（可选）
rm -rf work_dirs/epam_ntu60_optimized_2gpu

# 4. 启动新的训练
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_fixed.py \
    2 \
    --work-dir work_dirs/epam_ntu60_fixed
```

---

## 总结

当前模型的核心问题是**严重过拟合**，表现为：
- ✅ 训练集准确率高（75-87.5%）
- ❌ 验证集准确率低（11-18%）
- ❌ 验证准确率在Epoch 9后持续下降

修复策略是通过**降低正则化强度**（降低weight decay）、**放缓学习率衰减**（延长T_max）、**增加Early Stopping**来让模型更好地泛化。

立即执行上述训练命令，预期在10-20个epoch内看到明显改善。

