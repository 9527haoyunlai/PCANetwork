# 🚀 PCAN NTU-60 训练指南

## ✅ 配置优化已完成！

所有优化配置已经应用到 `configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py`

---

## 📋 优化内容总览

### 1️⃣ **训练稳定性优化**
- ✅ 学习率策略：`MultiStepLR` → `LinearLR + CosineAnnealingLR`
- ✅ 初始学习率：`0.015` → `0.01` (降低33%)
- ✅ 梯度裁剪：`max_norm=40` → `max_norm=20`
- ✅ 训练epoch：`30` → `50`

### 2️⃣ **Pose分支强化**
- ✅ Pose loss权重：`1.0` → `2.0` (提升100%)
- ✅ 粗分类Pose权重：`0.5` → `1.0` (提升100%)

### 3️⃣ **数据增强强化**
- ✅ RandomResizedCrop范围：`0.56-1.0` → `0.50-1.0`
- ✅ 新增ColorJitter：`brightness=0.3, contrast=0.3, saturation=0.3`

### 4️⃣ **训练管理优化**
- ✅ 早停机制：patience=10, min_delta=0.001
- ✅ Checkpoint间隔：10 → 5 epochs
- ✅ 自动保留最佳模型和最近5个checkpoint

---

## 🎯 预期效果

| 指标 | 当前(Epoch 30) | 目标(Epoch 50) | 提升 |
|------|----------------|----------------|------|
| **融合准确率** | 88.93% | >90.5% | +1.6% |
| **RGB分支** | 89.35% | >91.0% | +1.7% |
| **Pose分支** | 83.79% | >87.5% | +3.7% |

---

## 🚀 开始训练

### 方法1: 使用交互式脚本（推荐）
```bash
cd /home/zh/ChCode/codes01/mmaction2
./train_resume.sh
```
脚本会自动检测GPU并让你选择训练模式。

### 方法2: 直接命令

#### 双卡训练（推荐）
```bash
cd /home/zh/ChCode/codes01/mmaction2
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    2 \
    --work-dir work_dirs/pcan_ntu60
```

#### 单卡训练
```bash
cd /home/zh/ChCode/codes01/mmaction2
python tools/train.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    --work-dir work_dirs/pcan_ntu60
```

---

## 📊 训练监控

### 实时查看日志
```bash
# 查看最新日志
tail -f work_dirs/pcan_ntu60/$(ls -t work_dirs/pcan_ntu60/*.log | head -1)

# 查看验证准确率
watch -n 60 "grep 'acc/RGBPose_1:1_top1' work_dirs/pcan_ntu60/*.log | tail -20"
```

### 关键指标监控
- **融合准确率**: `acc/RGBPose_1:1_top1` (主要指标)
- **RGB准确率**: `acc/rgb_top1`
- **Pose准确率**: `acc/pose_top1`
- **学习率**: `lr:`
- **Loss**: `loss:`

---

## 📈 训练完成后

### 1. 查看最佳模型
```bash
ls -lh work_dirs/pcan_ntu60/best_*.pth
```

### 2. 绘制训练曲线
```bash
python tools/analysis_tools/analyze_logs.py plot_curve \
    work_dirs/pcan_ntu60/$(ls -t work_dirs/pcan_ntu60/*.log | head -1) \
    --keys acc/RGBPose_1:1_top1 acc/rgb_top1 acc/pose_top1 loss \
    --out work_dirs/pcan_ntu60/training_curve.png
```

### 3. 测试最佳模型
```bash
python tools/test.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    work_dirs/pcan_ntu60/best_acc_RGBPose_1:1_top1_epoch_XX.pth
```

---

## 🔍 学习率变化对比

### 优化前（MultiStepLR）
```
Epoch  1-10: 0.015 (固定)
Epoch 11-20: 0.0015 (↓90%) ← 导致Epoch 8崩溃和后期震荡
Epoch 21-30: 0.00015 (↓99%)
```

### 优化后（Cosine Annealing）
```
Epoch  1-5:  0.001 → 0.01 (Warmup)
Epoch  6-50: 0.01 → 0.00001 (平滑余弦退火) ← 稳定平滑
```

---

## ⚠️ 重要提示

1. **训练将从Epoch 30自动恢复**
   - 配置已设置 `resume=True`
   - 会从 `work_dirs/pcan_ntu60/epoch_30.pth` 继续

2. **早停保护**
   - 如果10个epoch内准确率提升<0.1%，会自动停止
   - 避免浪费训练时间

3. **显存要求**
   - 单卡：至少11GB
   - 双卡：batch_size=20，每卡10个样本

4. **ColorJitter初期影响**
   - 新增的颜色增强可能导致前2-3个epoch准确率略低
   - 这是正常现象，后续会恢复并超越

5. **Checkpoint自动管理**
   - 自动保留最佳模型
   - 只保留最近5个epoch的checkpoint
   - 旧checkpoint自动删除

---

## 📁 相关文件

- **配置文件**: `configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py`
- **训练脚本**: `train_resume.sh`
- **验证脚本**: `verify_config_simple.sh`
- **优化总结**: `configs/skeleton/posec3d/rgbpose_conv3d/OPTIMIZATION_SUMMARY.md`
- **训练日志**: `work_dirs/pcan_ntu60/*.log`
- **Checkpoint**: `work_dirs/pcan_ntu60/*.pth`

---

## 🆘 常见问题

### Q: 训练中断了怎么办？
A: 再次运行 `./train_resume.sh`，会自动从最新checkpoint恢复。

### Q: 想从头开始训练怎么办？
A: 修改配置文件，将 `resume = True` 改为 `resume = False`，并设置 `load_from`。

### Q: 如何调整batch size？
A: 修改配置文件中的 `train_dataloader.batch_size`，同时调整学习率（建议按比例）。

### Q: 早停太敏感/太宽松？
A: 修改 `default_hooks.early_stopping.patience`（增大=更宽松，减小=更敏感）。

---

## 📞 训练支持

如有问题，检查：
1. 日志文件中的错误信息
2. GPU显存是否充足 (`nvidia-smi`)
3. 数据路径是否正确
4. Checkpoint文件是否存在

---

**配置优化时间**: 2025-11-21  
**基于**: 30个epoch训练日志分析  
**目标**: 从88.93%提升到90.5%+  

🎉 **祝训练顺利！**

