# 快速开始 - 30秒启动训练 ⚡

## 第一步：选择实验 (3选1)

```bash
# 实验1: MA52 + 原始Backbone (~80-85%)
bash train_ma52_original.sh

# 实验2: NTU60 + 原始Backbone (~85-87%)
bash train_ntu60_original.sh

# 实验3: NTU60 + X3D Backbone (90.44%) ⭐ 推荐
bash train_ntu60_x3d.sh
```

## 第二步：监控训练

```bash
# 实时查看日志
tail -f logs/train_ntu60_x3d.log

# 查看最新验证结果
grep "Epoch(val).*8244/8244" logs/train_ntu60_x3d.log | tail -3
```

## 第三步：查看结果

```bash
# 最佳模型位置
ls work_dirs/pcan_ntu60_x3d/best_*.pth

# 使用TensorBoard查看训练曲线
tensorboard --logdir=work_dirs/pcan_ntu60_x3d/vis_data --port=6006
```

---

## 💡 重要提示

### ✅ 切换实验无需修改代码
- 直接运行对应的训练脚本
- 每个实验完全独立
- 自动加载正确配置

### ⚠️ 训练前检查
```bash
# 检查GPU
nvidia-smi

# 检查数据集
ls data/nturgbd_videos/
ls data/skeleton/ntu60_xsub.pkl

# 检查环境
conda activate openmmlab
```

### 📚 详细文档
- **完整指南**: `TRAINING_GUIDE.md`
- **项目结构**: `PROJECT_STRUCTURE.md`
- **配置说明**: `configs/skeleton/posec3d/rgbpose_conv3d/CONFIG_README.md`

---

## 🎯 预期性能

| 实验 | 数据集 | Backbone | 训练时间 | 准确率 |
|------|--------|----------|----------|--------|
| 1 | MA52 | 原始 | ~12h | ~80-85% |
| 2 | NTU60 | 原始 | ~26h | ~85-87% |
| **3⭐** | **NTU60** | **X3D** | **~20h** | **90.44%** |

---

**开始训练吧！🚀**

