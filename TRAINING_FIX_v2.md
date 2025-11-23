# 🔧 PCAN NTU-60 训练配置修正

## ❌ 上次训练失败的原因

### 1. **早停过于严格**
- Epoch 1达到88.32%后
- Epoch 2-10持续下降到85%
- 10个epoch内未超过88.32%
- **触发早停，训练终止**

### 2. **Warmup策略错误**
- 从epoch 26 (89.19%)的模型重新训练
- 使用warmup从lr=0.001开始
- **导致模型"遗忘"之前学到的知识**
- 准确率从89.19%降到88.32%

### 3. **学习率过小**
- Warmup阶段学习率太小
- 无法有效更新权重
- 训练陷入停滞

---

## ✅ 已修正的配置

### **修改1: 去掉Warmup，直接微调**
```python
# 优化器
lr=0.003  # 直接使用小学习率微调（原来0.001起步）

# 学习率调度
CosineAnnealingLR:
  - 直接从0.003开始余弦退火
  - 30个epoch降到1e-5
  - 无Warmup阶段
```

### **修改2: 放宽早停条件**
```python
early_stopping:
  patience: 10 → 15 epochs
  min_delta: 0.001 → 0.0005 (0.1% → 0.05%)
```

### **修改3: 调整训练epoch**
```python
max_epochs: 24 → 30
# 从epoch 26再训练30个epoch（共到epoch 56）
```

---

## 📊 预期训练曲线

```
Epoch  1-5:  准确率可能小幅波动 87-89%
Epoch  6-15: 稳步提升 89-90%
Epoch 16-30: 平稳收敛 90-91%
```

学习率变化：
```
Epoch  1: lr = 0.003
Epoch 10: lr = 0.0015
Epoch 20: lr = 0.0003
Epoch 30: lr = 0.00001
```

---

## 🚀 重新开始训练

### **步骤1: 清理旧训练记录**
```bash
cd /home/zh/ChCode/codes01/mmaction2

# 备份上次失败的训练
mv work_dirs/pcan_ntu60/20251122_030520 work_dirs/pcan_ntu60/failed_20251122_030520

# 清理其他临时文件（保留checkpoint）
rm work_dirs/pcan_ntu60/epoch_*.pth 2>/dev/null || true
```

### **步骤2: 启动训练**
```bash
# 双卡训练
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    2 \
    --work-dir work_dirs/pcan_ntu60

# 或使用脚本
./train_resume.sh
```

---

## 📈 监控指标

### **关键指标**
1. **学习率**: 应该从0.003开始平滑下降
2. **准确率**: Epoch 1应该≥88.5%，然后逐步提升
3. **Loss**: 应该平稳下降，不要突变

### **实时监控**
```bash
# 查看最新日志
tail -f work_dirs/pcan_ntu60/$(ls -t work_dirs/pcan_ntu60/*.log | head -1)

# 查看学习率
grep "lr:" work_dirs/pcan_ntu60/*.log | tail -20

# 查看验证准确率
grep "acc/RGBPose_1:1_top1:" work_dirs/pcan_ntu60/*.log | grep "Epoch(val)"
```

---

## ⚠️ 如果仍然不工作

### **备选方案：降低Pose权重**
如果准确率仍然下降，可能是Pose权重太高了：

```python
# 在配置文件第48行
loss_weights=[1., 1.5, 0.5, 0.8]  # 将Pose从2.0降到1.5
```

### **或者：使用更保守的学习率**
```python
# 第203行
lr=0.001  # 从0.003降到0.001
```

---

## 🎯 预期效果

| 训练阶段 | Epoch | 预期准确率 | 学习率 |
|---------|-------|----------|--------|
| 稳定期 | 1-5 | 88-89% | 0.003-0.0027 |
| 提升期 | 6-15 | 89-90% | 0.0027-0.0015 |
| 收敛期 | 16-30 | 90-91% | 0.0015-0.00001 |

---

**关键要点**:
1. ✅ 不要Warmup！直接从0.003微调
2. ✅ 放宽早停，给模型更多机会
3. ✅ 监控第1个epoch，应该≥88.5%
4. ⚠️ 如果第1个epoch就低于88%，立即停止并降低学习率

---

**生成时间**: 2025-11-22 06:00  
**修正版本**: v2 (微调模式)

