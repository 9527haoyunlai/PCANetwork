# 训练指南

本项目包含三个主要实验，使用不同的backbone和数据集进行训练。

## 📋 快速导航

- [三种实验说明](#三种实验说明)
- [快速开始](#快速开始)
- [配置文件说明](#配置文件说明)
- [训练监控](#训练监控)
- [常见问题](#常见问题)

---

## 三种实验说明

### 实验1: 原始Backbone + MA52数据集

**目标**: 在MA52数据集上训练原始双流网络

**特点**:
- 数据集: MA52 (52个动作类别)
- Backbone: RGBPoseConv3D (原始设计)
- RGB通道: 2048维
- Pose通道: 512维
- 骨架关键点: 28点

**训练脚本**: `train_ma52_original.sh`

**配置文件**: `configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py`

**预期性能**: ~80-85%

---

### 实验2: 原始Backbone + NTU60数据集

**目标**: 在NTU60数据集上训练原始双流网络

**特点**:
- 数据集: NTU RGB+D 60 (60个动作类别)
- Backbone: RGBPoseConv3D (原始设计)
- RGB通道: 2048维
- Pose通道: 512维
- 骨架关键点: 17点
- 分层分类: 8个粗类 + 60个细类

**训练脚本**: `train_ntu60_original.sh`

**配置文件**: `configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py`

**预期性能**: ~85-87%

---

### 实验3: X3D Backbone + NTU60数据集 ⭐

**目标**: 使用轻量级X3D网络突破90%准确率

**特点**:
- 数据集: NTU RGB+D 60 (60个动作类别)
- Backbone: X3D TemporalShift (轻量高效)
- RGB通道: 432维 (X3D-M)
- Pose通道: 216维 (X3D-S)
- 参数量: ~15M (减少70%)
- 分层分类: 8个粗类 + 60个细类

**训练脚本**: `train_ntu60_x3d.sh`

**配置文件**: `configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_x3d.py`

**历史最佳**: **90.44%** @ Epoch 78
- RGB分支: 83.93%
- Pose分支: 89.06%

---

## 快速开始

### 1. 环境检查

```bash
# 激活conda环境
conda activate openmmlab

# 检查GPU
nvidia-smi

# 检查数据集
ls data/ma52/raw_videos/          # MA52数据集
ls data/nturgbd_videos/           # NTU60数据集
ls data/skeleton/ntu60_xsub.pkl  # NTU60标注文件
```

### 2. 选择实验并启动训练

#### 实验1: MA52 + 原始Backbone
```bash
bash train_ma52_original.sh
```

#### 实验2: NTU60 + 原始Backbone
```bash
bash train_ntu60_original.sh
```

#### 实验3: NTU60 + X3D Backbone (推荐)
```bash
bash train_ntu60_x3d.sh
```

### 3. 后台训练 (可选)

如果想在后台运行训练：

```bash
# 实验1
nohup bash train_ma52_original.sh > logs/ma52.out 2>&1 &

# 实验2
nohup bash train_ntu60_original.sh > logs/ntu60.out 2>&1 &

# 实验3
nohup bash train_ntu60_x3d.sh > logs/x3d.out 2>&1 &
```

---

## 配置文件说明

### 核心配置文件位置

```
configs/skeleton/posec3d/rgbpose_conv3d/
├── rgbpose_conv3d.py          # 实验1配置 (MA52)
├── pcan_ntu60.py              # 实验2配置 (NTU60 原始)
└── pcan_ntu60_x3d.py          # 实验3配置 (NTU60 X3D)
```

### 切换实验不需要修改代码

**重要**: 三个实验完全独立，只需要：
1. 运行对应的训练脚本
2. 脚本会自动加载正确的配置文件
3. **无需修改任何Python代码**

### 关键配置项

#### 数据集路径
- MA52: `data/ma52/raw_videos/`
- NTU60: `data/nturgbd_videos/`
- 标注文件: `data/skeleton/ntu60_xsub.pkl`

#### 训练参数
- GPU: 1,2 (2卡并行)
- Batch size: 
  - MA52: 16 (每卡8)
  - NTU60原始: 24 (每卡12)
  - NTU60_X3D: 32 (每卡16)

#### 学习率策略
- MA52: lr=0.001, 固定
- NTU60原始: lr=0.001, CosineAnnealing
- NTU60_X3D: lr=0.012 → 1e-6, CosineAnnealing + Warmup

---

## 训练监控

### 实时查看训练日志

```bash
# 实验1
tail -f logs/train_ma52_original.log

# 实验2
tail -f logs/train_ntu60_original.log

# 实验3
tail -f logs/train_ntu60_x3d.log
```

### 查看验证结果

```bash
# 查看所有验证epoch的准确率
grep "Epoch(val).*8244/8244" logs/train_ntu60_x3d.log

# 查看最近3次验证结果
grep "Epoch(val).*8244/8244" logs/train_ntu60_x3d.log | tail -3

# 查看最佳结果
grep "best checkpoint" logs/train_ntu60_x3d.log
```

### 使用TensorBoard (可选)

```bash
# 启动TensorBoard
tensorboard --logdir=work_dirs/pcan_ntu60_x3d/vis_data --port=6006

# 在浏览器打开: http://localhost:6006
```

### GPU使用监控

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或使用gpustat
pip install gpustat
gpustat -i 1
```

---

## 常见问题

### Q1: 如何切换不同的实验？

**A**: 直接运行对应的训练脚本即可：
```bash
bash train_ma52_original.sh      # 实验1
bash train_ntu60_original.sh     # 实验2
bash train_ntu60_x3d.sh          # 实验3
```

**无需修改任何代码！**

---

### Q2: 实验3 (X3D) 需要额外依赖吗？

**A**: 需要确保 `emap_backbone/` 目录存在。训练脚本会自动检查。

---

### Q3: 训练中断后如何恢复？

**A**: 使用 `--resume` 参数：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_x3d.py \
    2 \
    --work-dir work_dirs/pcan_ntu60_x3d \
    --resume
```

---

### Q4: 如何只使用1张GPU训练？

**A**: 修改脚本中的 `CUDA_VISIBLE_DEVICES` 和GPU数量：

```bash
# 只使用GPU 1
CUDA_VISIBLE_DEVICES=1 \
bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_x3d.py \
    1 \  # 改为1
    --work-dir work_dirs/pcan_ntu60_x3d
```

注意：单卡训练需要相应调整batch size。

---

### Q5: 内存不足 (OOM) 怎么办？

**A**: 降低batch size：

1. 编辑配置文件
2. 找到 `train_dataloader` 部分
3. 减小 `batch_size` (例如从16降到8)

---

### Q6: 如何评估已训练的模型？

**A**: 使用test脚本：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
bash tools/dist_test.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_x3d.py \
    work_dirs/pcan_ntu60_x3d/best_acc_RGBPose_1:1_top1_epoch_78.pth \
    2
```

---

## 训练时间估计

| 实验 | 数据集大小 | Epochs | 单epoch时间 | 总时间 |
|------|-----------|--------|------------|--------|
| 实验1 | MA52 (~20K) | 50 | ~15分钟 | ~12小时 |
| 实验2 | NTU60 (~40K) | 80 | ~20分钟 | ~26小时 |
| 实验3 | NTU60 (~40K) | 80 | ~15分钟 | ~20小时 |

*基于2卡A100 40GB的估计*

---

## 输出文件说明

### 训练输出目录结构

```
work_dirs/
├── ma52_original/              # 实验1输出
│   ├── best_*.pth              # 最佳模型
│   ├── latest.pth              # 最新checkpoint
│   └── *.log                   # 训练日志
├── pcan_ntu60_original/        # 实验2输出
│   └── ...
└── pcan_ntu60_x3d/             # 实验3输出
    └── ...

logs/                           # 日志文件
├── train_ma52_original.log
├── train_ntu60_original.log
└── train_ntu60_x3d.log
```

---

## 性能对比

| 实验 | Backbone | 参数量 | NTU60准确率 | 训练时间 |
|------|---------|--------|-------------|---------|
| 实验2 | RGBPoseConv3D | ~50M | ~85-87% | ~26h |
| 实验3 | X3D TemporalShift | ~15M | **90.44%** ✨ | ~20h |

**结论**: X3D架构在更少参数量和更短训练时间下取得了最佳性能！

---

## 进阶优化

如果想进一步提升实验3的性能 (90% → 91-92%)：

1. **优化学习率**:
   ```python
   # 编辑 pcan_ntu60_x3d.py
   lr=0.004  # 从0.012降到0.004
   eta_min=5e-5  # 从1e-6提高到5e-5
   ```

2. **调整训练周期**:
   ```python
   max_epochs=70  # 从80降到70
   patience=8  # 从15降到8
   ```

3. **增强正则化**:
   ```python
   weight_decay=0.0004  # 从0.0002提高
   clip_grad=dict(max_norm=30)  # 从40降到30
   ```

详细配置见之前的优化建议文档。

---

## 联系与支持

如有问题，请参考：
- 项目README: `README.md`
- MMAction2文档: https://mmaction2.readthedocs.io/
- Issues: https://github.com/open-mmlab/mmaction2/issues

---

**祝训练顺利！🚀**

