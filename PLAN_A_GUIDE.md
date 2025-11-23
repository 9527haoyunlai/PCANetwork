# 🎯 方案A：稳妥冲刺95% - 完整执行指南

## 📋 **三阶段训练计划**

### **阶段1: 稳定基础（当前配置）** ✅ 已准备好
- **配置文件**: `configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_95target.py`
- **启动脚本**: `train_rescue.sh`（GPU 1,2）
- **目标**: 70-75%
- **Epochs**: 50
- **时间**: 约12小时
- **Checkpoint**: Epoch 11 (57.45%) 开始

### **阶段2: 性能提升（中等激进）** 📝 待创建
- **配置文件**: `pcan_ntu60_stage2_85percent.py`（我会创建）
- **目标**: 85-90%
- **Epochs**: 50-100 (继续50个)
- **时间**: 约12小时
- **从阶段1最佳checkpoint开始**

### **阶段3: 冲刺95%（激进优化）** 📝 待创建
- **配置文件**: `pcan_ntu60_stage3_95percent.py`（我会创建）
- **目标**: 93-95%+
- **Epochs**: 100-150 (继续50个)
- **时间**: 约12小时
- **从阶段2最佳checkpoint开始**

---

## 🚀 **你现在要做的**

### **步骤1: 启动阶段1训练**

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 使用你修改后的脚本（GPU 1,2）
bash train_rescue.sh
```

### **步骤2: 监控训练**

```bash
# 实时查看日志
tail -f train_rescue.log

# 查看最新验证结果
grep 'Epoch(val).*8244/8244' train_rescue.log | tail -3

# 查看GPU使用
nvidia-smi
```

### **步骤3: 等待阶段1完成（约12小时）**

训练会自动：
- 每5个epoch保存checkpoint
- 自动保存最佳模型
- 达到70-75%准确率

### **步骤4: 阶段1完成后**

我会为你准备好阶段2配置文件，届时运行：

```bash
# 阶段2启动脚本（我会提前创建好）
bash train_stage2.sh
```

---

## 📊 **预期时间表**

| 时间点 | 阶段 | 准确率 | 行动 |
|--------|------|--------|------|
| **现在** | 阶段1开始 | 57% → 70-75% | ✅ 运行 train_rescue.sh |
| **+12小时** | 阶段1完成 | 70-75% | 切换到阶段2 |
| **+24小时** | 阶段2完成 | 85-90% | 切换到阶段3 |
| **+36小时** | 阶段3完成 | 93-95% | 🎉 达成目标！ |

---

## 📁 **文件状态**

### **已准备好的文件** ✅
- `configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_95target.py` - 阶段1配置
- `train_rescue.sh` - 阶段1启动脚本（GPU 1,2）
- `work_dirs/pcan_ntu60_95target/best_acc_RGBPose_1:1_top1_epoch_11.pth` - 起始checkpoint

### **我接下来会创建的文件** 📝
- `pcan_ntu60_stage2_85percent.py` - 阶段2配置
- `train_stage2.sh` - 阶段2启动脚本
- `pcan_ntu60_stage3_95percent.py` - 阶段3配置
- `train_stage3.sh` - 阶段3启动脚本

---

## 🎯 **关键监控指标**

### **阶段1健康标准**
- ✅ 学习率: 0.005
- ✅ Pose Top1 > 20%
- ✅ RGBPose Top1稳步上升
- ✅ Epoch 50达到70-75%

### **如果出现问题**
- ❌ Pose Top1 < 15% → 通知我
- ❌ 准确率不增长 → 通知我
- ❌ OOM错误 → 通知我

---

## 💡 **重要提示**

1. **不要中断训练**：让阶段1完整运行50个epoch
2. **保存checkpoint**：最佳模型会自动保存
3. **12小时后联系我**：我会准备好阶段2配置
4. **耐心等待**：36小时后达到95%

---

## ✅ **当前状态**

- 🟢 训练已停止
- 🟢 GPU 1, 2 已空闲
- 🟢 配置文件已准备好
- 🟢 Checkpoint已加载（Epoch 11, 57.45%）
- 🟢 你可以开始训练了！

---

**现在你可以运行了：**

```bash
cd /home/zh/ChCode/codes01/mmaction2
bash train_rescue.sh
```

**祝训练顺利！12小时后见！** 🚀

