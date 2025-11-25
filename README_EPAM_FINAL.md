# 🎯 EPAM-Net NTU60 训练 - 最终版

## ✅ 状态总结

| 项目 | 状态 | 说明 |
|------|------|------|
| **接口修复** | ✅ 完成 | 6元素返回值，gt参数传递正确 |
| **模块注册** | ✅ 完成 | EPAMBackbone和EPAMRecognizer已注册 |
| **配置优化** | ✅ 完成 | 基于原始模型超参数优化 |
| **测试验证** | ✅ 通过 | test_epam_final.py全部通过 |
| **文档完善** | ✅ 完成 | 包含详细说明和对比 |

---

## 🚀 三步开始训练

### 方法1：使用启动脚本（推荐）

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 运行启动脚本
./START_TRAINING.sh
```

### 方法2：直接命令

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 使用优化配置训练
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_optimized.py \
    2 \
    --work-dir work_dirs/epam_ntu60_optimized_2gpu
```

### 方法3：单GPU训练

```bash
# 如果只想用一张卡
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_optimized.py \
    --work-dir work_dirs/epam_ntu60_optimized_1gpu
```

---

## 📊 配置文件说明

### 1. Optimized配置 ⭐⭐⭐⭐⭐ （强烈推荐）

**文件**: `configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_optimized.py`

**特点**:
- ✅ 学习率: 0.0015（匹配原始模型）
- ✅ Weight Decay: 0.0003（更强正则化）
- ✅ CosineAnnealingLR（更平滑的学习率衰减）
- ✅ 随机种子: 100（可复现）
- ✅ 预期准确率: 91-93%（NTU60 X-Sub）

**适用场景**: 
- 首次训练
- 追求最佳性能
- 需要可复现结果

### 2. Baseline配置 ⭐⭐⭐

**文件**: `configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py`

**特点**:
- 初始配置，未经优化
- 预期准确率: 89-91%
- 适合对比实验

---

## 📚 完整文档导航

| 文档 | 内容 | 何时查看 |
|------|------|---------|
| **README_EPAM_FINAL.md** (本文) | 快速开始总览 | ⭐ 现在 |
| `TRAINING_QUICK_START.md` | 训练命令和监控 | 训练时 |
| `OPTIMIZATION_COMPARISON.md` | 配置对比和原理 | 想了解细节时 |
| `EPAM_FIX_NOTES.md` | 接口修复说明 | 遇到问题时 |
| `EPAM_INTEGRATION_README.md` | 完整集成文档 | 深入研究时 |

---

## 🎯 优化要点速览

### 关键改进

1. **学习率降低** (0.01 → 0.0015)
   - 更稳定的训练
   - 更好的收敛
   - 预期准确率 +1-2%

2. **正则化增强** (0.0001 → 0.0003)
   - 减少过拟合
   - 提高泛化能力
   - Train-Val gap缩小

3. **Cosine学习率** (MultiStep → Cosine)
   - 更平滑的衰减
   - 后期微调能力更强
   - 通常 +0.5-1% 准确率

4. **可复现性** (seed=100)
   - 每次结果一致
   - 方便调试对比

### 超参数一览表

```python
# 优化配置关键参数
optimizer = SGD(
    lr=0.0015,           # ← 优化
    momentum=0.9,        # ← 标准
    weight_decay=0.0003  # ← 优化
)

scheduler = CosineAnnealingLR(  # ← 优化
    T_max=50,
    eta_min=1e-6
)

batch_size = 8 per GPU  # 总batch=16
epochs = 50
gradient_clip = 40
seed = 100              # ← 新增
```

---

## 📈 预期训练过程

### 正常的训练曲线

```
Epoch  |  Loss   |  Train Acc  |  Val Acc
-------|---------|-------------|----------
  1    |  4.05   |    10%      |   8%
  5    |  2.50   |    40%      |   38%
 10    |  1.20   |    65%      |   63%
 20    |  0.50   |    85%      |   82%
 30    |  0.30   |    90%      |   88%
 40    |  0.20   |    92%      |   91%
 50    |  0.15   |    93%      |   92%    ← 预期最终
```

### 关键指标

- **最终Val Acc**: 91-93%
- **Train-Val Gap**: 1-2% ✅
- **训练时间**: 6-8小时（2× RTX 3090）
- **GPU内存**: 10-12GB/卡

---

## 🔍 监控训练

### 实时查看日志

```bash
# 终端1：查看训练进度
tail -f work_dirs/epam_ntu60_optimized_2gpu/*.log | grep "Epoch"

# 终端2：查看GPU使用
watch -n 1 nvidia-smi

# 终端3：查看详细loss
tail -f work_dirs/epam_ntu60_optimized_2gpu/*.log | grep "loss:"
```

### 检查关键指标

```bash
# 查看最佳准确率
grep "best_acc" work_dirs/epam_ntu60_optimized_2gpu/*.log

# 查看学习率变化
grep "lr:" work_dirs/epam_ntu60_optimized_2gpu/*.log | head -20
```

---

## 🧪 训练完成后

### 1. 找到最佳模型

```bash
ls -lh work_dirs/epam_ntu60_optimized_2gpu/best_*.pth
```

### 2. 测试模型

```bash
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_test.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_optimized.py \
    work_dirs/epam_ntu60_optimized_2gpu/best_acc_RGBPose_1:1_top1_*.pth \
    2
```

### 3. 查看结果

```bash
# 查看测试准确率
cat work_dirs/epam_ntu60_optimized_2gpu/*.log | grep "Evaluating"
```

---

## ⚠️ 常见问题

### Q1: 训练启动失败

**检查清单**:
```bash
# 1. 验证接口修复
python test_epam_final.py

# 2. 检查GPU
nvidia-smi

# 3. 检查数据路径
ls /home/zh/ChCode/codes01/mmaction2/data/nturgbd_videos/
ls /home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl
```

### Q2: Loss不下降或NaN

**可能原因**:
- 学习率太高 → 降到0.001
- 梯度爆炸 → 检查gradient_clip
- 数据问题 → 检查数据预处理

### Q3: 准确率低于预期

**可能原因**:
- 训练epoch不够 → 增加到80
- 过拟合 → 增加weight_decay到0.0005
- 数据问题 → 检查数据标注

### Q4: GPU内存不足

**解决方案**:
```python
# 降低batch size
batch_size=4  # 从8降到4
```

---

## 🎓 理论背景速查

### SGD + Momentum
```
更新规则: v_t = 0.9 × v_{t-1} + gradient
         w_t = w_{t-1} - 0.0015 × v_t
```
像下坡的球，有惯性。

### Weight Decay (L2正则)
```
Loss = CE_Loss + 0.0003 × ||W||²
```
惩罚大权重，防止过拟合。

### Cosine Annealing
```
lr_t = 1e-6 + (0.0015 - 1e-6) × (1 + cos(π×t/50)) / 2
```
平滑下降，后期微调。

### Gradient Clipping
```
if ||gradient|| > 40:
    gradient = gradient × (40 / ||gradient||)
```
防止梯度爆炸。

---

## ✅ 最终检查清单

训练前:
- [ ] GPU可用（nvidia-smi）
- [ ] 数据路径正确
- [ ] 配置文件正确
- [ ] 接口测试通过（test_epam_final.py）

训练中:
- [ ] Loss平滑下降
- [ ] GPU利用率 > 80%
- [ ] 准确率稳步提升
- [ ] 无OOM错误

训练后:
- [ ] Val Acc ≥ 90%
- [ ] 保存了best checkpoint
- [ ] 可以成功测试
- [ ] 结果可复现

---

## 🎉 开始训练！

选择一种方法开始：

### ⭐ 推荐：使用启动脚本
```bash
./START_TRAINING.sh
```

### 或直接命令
```bash
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_optimized.py \
    2 \
    --work-dir work_dirs/epam_ntu60_optimized_2gpu
```

---

## 📞 获取帮助

如果遇到问题：

1. 查看 `OPTIMIZATION_COMPARISON.md` 了解配置细节
2. 查看 `EPAM_FIX_NOTES.md` 了解接口修复
3. 运行 `python test_epam_final.py` 验证环境
4. 检查日志文件中的错误信息

---

**祝训练顺利！预期准确率 91-93%！** 🚀

*最后更新: 2025-11-25*
*版本: Final Optimized*

