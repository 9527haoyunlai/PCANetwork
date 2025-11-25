# EPAM + NTU60 训练快速开始指南

## ✅ 修复状态

**所有接口问题已修复！** 可以正常训练。

## 🎯 推荐配置

现在有两个配置可选：

| 配置 | 文件 | 特点 | 推荐度 |
|------|------|------|--------|
| **Optimized** | `epam_ntu60_optimized.py` | 优化超参数，预期+1-2%准确率 | ⭐⭐⭐⭐⭐ |
| Baseline | `epam_ntu60_baseline.py` | 初始配置 | ⭐⭐⭐ |

**强烈推荐使用 Optimized 配置！** 详见 `OPTIMIZATION_COMPARISON.md`

## 🚀 立即开始训练（推荐）

### 使用GPU 1和2训练 - Optimized配置

```bash
cd /home/zh/ChCode/codes01/mmaction2

# ⭐ 推荐：使用优化配置
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_optimized.py \
    2 \
    --work-dir work_dirs/epam_ntu60_optimized_2gpu
```

### 或使用Baseline配置（对比实验）

```bash
# Baseline配置（如需对比）
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py \
    2 \
    --work-dir work_dirs/epam_ntu60_baseline_2gpu
```

### 监控训练

```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看训练日志（另一个终端）
tail -f work_dirs/epam_ntu60_baseline_2gpu/*.log

# 或使用tensorboard（如果启用）
tensorboard --logdir work_dirs/epam_ntu60_baseline_2gpu
```

## 📊 预期输出

训练开始时应该看到：

```
Distributed training: True
World size: 2
Distributed launcher: pytorch
...
Epoch [1/50] ...
```

如果看到以下内容说明正在正常训练：
```
Epoch [1/50][100/XXXX]  lr: x.xxxe-xx  eta: XX:XX:XX  time: x.xxx  data_time: x.xxx  
loss: x.xxx  loss_rgb: x.xxx  loss_pose: x.xxx  ...
```

## 🔍 验证修复（可选）

在训练前可以运行测试验证：

```bash
# 验证6元素返回值
python test_epam_final.py

# 预期输出
✅ Backbone返回元素数量: 6
✅ 返回值格式正确！
```

## 📁 输出文件

训练结果将保存在：

```
work_dirs/epam_ntu60_baseline_2gpu/
├── epam_ntu60_baseline.py          # 配置文件备份
├── *.log                            # 训练日志
├── *.json                           # 训练指标
├── epoch_*.pth                      # 定期checkpoint
├── best_acc_RGBPose_1:1_top1_epoch_*.pth  # 最佳模型
└── last_checkpoint                  # 最新checkpoint链接
```

## 💡 常用命令

### 从checkpoint恢复训练

```bash
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py \
    2 \
    --resume work_dirs/epam_ntu60_baseline_2gpu/latest.pth
```

### 测试训练好的模型

```bash
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_test.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py \
    work_dirs/epam_ntu60_baseline_2gpu/best_acc_RGBPose_1:1_top1_epoch_*.pth \
    2
```

### 单GPU训练（如果需要）

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py \
    --work-dir work_dirs/epam_ntu60_baseline_1gpu
```

## ⚙️ 配置调整

如果需要调整训练参数，编辑配置文件：

```python
# configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py

# 调整batch size
train_dataloader = dict(
    batch_size=8,  # 每卡batch，总batch=8*2=16
    ...
)

# 调整学习率
optim_wrapper = dict(
    optimizer=dict(
        lr=0.01,  # 初始学习率
        ...
    )
)

# 调整训练epochs
train_cfg = dict(
    max_epochs=50,  # 总epoch数
    ...
)
```

## ⚠️ 常见问题

### Q1: 端口被占用
**错误**: `Address already in use`

**解决**:
```bash
CUDA_VISIBLE_DEVICES=1,2 \
MASTER_PORT=29501 \  # 修改端口
bash tools/dist_train.sh ...
```

### Q2: GPU内存不足
**错误**: `CUDA out of memory`

**解决**: 降低batch size
```python
train_dataloader = dict(
    batch_size=4,  # 从8降到4
    ...
)
```

### Q3: 数据加载慢
**解决**: 增加workers
```python
train_dataloader = dict(
    num_workers=16,  # 增加到16
    persistent_workers=True,
    ...
)
```

## 📈 预期性能

基于EPAM-Net论文，在NTU RGB+D 60 X-Sub上：

| 指标 | 预期值 |
|------|--------|
| **Top-1准确率** | ~90-93% |
| **训练时间** | ~6-8小时 (2x RTX 3090, 50 epochs) |
| **GPU内存** | ~10-12GB per GPU |

## 📞 问题反馈

如果遇到问题：

1. 查看日志文件
2. 检查 `EPAM_FIX_NOTES.md` 了解修复细节
3. 运行 `python test_epam_final.py` 验证接口

## ✨ 修复内容

- ✅ `EPAMBackbone` 返回6个元素（兼容RGBPoseHead）
- ✅ `gt` 和 `gt_coarse` 正确传递给backbone
- ✅ 移除了错误的kwargs传递
- ✅ 所有测试通过

---

**准备好了吗？开始训练吧！** 🚀

```bash
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py \
    2 \
    --work-dir work_dirs/epam_ntu60_baseline_2gpu
```

Good luck! 🎉

