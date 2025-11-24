# 项目结构说明

本文档说明了整理后的项目结构和文件组织。

## 📁 核心文件结构

```
mmaction2/
│
├── 🚀 训练脚本 (根目录)
│   ├── train_ma52_original.sh         # 实验1: MA52 + 原始Backbone
│   ├── train_ntu60_original.sh        # 实验2: NTU60 + 原始Backbone
│   └── train_ntu60_x3d.sh             # 实验3: NTU60 + X3D Backbone ⭐
│
├── 📖 文档
│   ├── TRAINING_GUIDE.md              # 主训练指南 (必读!)
│   ├── PROJECT_STRUCTURE.md           # 本文档
│   ├── README.md                      # MMAction2原始README
│   └── README_zh-CN.md                # MMAction2中文README
│
├── ⚙️ 配置文件
│   └── configs/skeleton/posec3d/rgbpose_conv3d/
│       ├── rgbpose_conv3d.py          # 实验1配置
│       ├── pcan_ntu60.py              # 实验2配置
│       ├── pcan_ntu60_x3d.py          # 实验3配置
│       └── CONFIG_README.md           # 配置文件说明
│
├── 📊 数据集
│   └── data/
│       ├── ma52/                      # MA52数据集
│       ├── nturgbd_videos/            # NTU60视频数据
│       └── skeleton/                  # 骨架标注文件
│
├── 🔧 代码模块
│   ├── mmaction/                      # 核心代码
│   │   ├── models/                    # 模型定义
│   │   │   ├── backbones/            # Backbone网络
│   │   │   ├── heads/                # 分类头
│   │   │   │   └── rgbpose_head.py   # RGB+Pose融合头
│   │   │   └── recognizers/          # 识别器
│   │   │       └── rgbpose_x3d_recognizer.py
│   │   └── ...
│   └── emap_backbone/                 # X3D Backbone (实验3专用)
│
├── 📝 日志输出
│   └── logs/
│       ├── train_ma52_original.log    # 实验1日志
│       ├── train_ntu60_original.log   # 实验2日志
│       └── train_ntu60_x3d.log        # 实验3日志
│
└── 💾 训练输出
    └── work_dirs/
        ├── ma52_original/             # 实验1输出
        ├── pcan_ntu60_original/       # 实验2输出
        └── pcan_ntu60_x3d/            # 实验3输出
            ├── best_*.pth             # 最佳模型
            ├── latest.pth             # 最新checkpoint
            └── vis_data/              # 可视化数据
```

---

## 🎯 三种实验对比

| 实验 | 脚本 | 配置文件 | 数据集 | Backbone | 性能 |
|------|------|---------|--------|----------|------|
| 实验1 | `train_ma52_original.sh` | `rgbpose_conv3d.py` | MA52 (52类) | RGBPoseConv3D | ~80-85% |
| 实验2 | `train_ntu60_original.sh` | `pcan_ntu60.py` | NTU60 (60类) | RGBPoseConv3D | ~85-87% |
| 实验3⭐ | `train_ntu60_x3d.sh` | `pcan_ntu60_x3d.py` | NTU60 (60类) | X3D TemporalShift | **90.44%** |

---

## 🚀 快速开始

### 1. 选择实验

```bash
# 查看训练指南
cat TRAINING_GUIDE.md

# 选择并运行实验
bash train_ntu60_x3d.sh  # 推荐从实验3开始
```

### 2. 监控训练

```bash
# 实时查看日志
tail -f logs/train_ntu60_x3d.log

# 查看验证结果
grep "Epoch(val).*8244/8244" logs/train_ntu60_x3d.log | tail -3
```

### 3. 查看结果

```bash
# 查看最佳模型
ls work_dirs/pcan_ntu60_x3d/best_*.pth

# 查看训练曲线（使用TensorBoard）
tensorboard --logdir=work_dirs/pcan_ntu60_x3d/vis_data
```

---

## 📋 关键文件说明

### 训练脚本
- **位置**: 项目根目录
- **命名规则**: `train_{数据集}_{backbone}.sh`
- **功能**: 
  - 环境检查
  - GPU检测
  - 自动启动训练
  - 生成日志文件

### 配置文件
- **位置**: `configs/skeleton/posec3d/rgbpose_conv3d/`
- **格式**: Python配置文件
- **包含**: 
  - 模型架构
  - 数据pipeline
  - 训练参数
  - 优化器配置

### 日志文件
- **位置**: `logs/`
- **格式**: 文本日志
- **内容**: 
  - 训练损失
  - 验证准确率
  - GPU使用情况
  - 错误信息

### 输出文件
- **位置**: `work_dirs/{实验名}/`
- **包含**: 
  - Checkpoint文件 (`.pth`)
  - 配置文件副本
  - 可视化数据
  - 训练日志

---

## 🔄 切换实验

### 无需修改代码！

三个实验完全独立，切换方式：

```bash
# 停止当前训练 (Ctrl+C)

# 运行另一个实验
bash train_ma52_original.sh     # 切换到实验1
bash train_ntu60_original.sh    # 切换到实验2
bash train_ntu60_x3d.sh         # 切换到实验3
```

每个训练脚本会：
1. 自动加载正确的配置文件
2. 使用对应的数据集
3. 输出到独立的目录
4. 生成独立的日志文件

**完全不需要修改任何Python代码！**

---

## 📊 输出文件详解

### Checkpoint文件

```
work_dirs/pcan_ntu60_x3d/
├── best_acc_RGBPose_1:1_top1_epoch_78.pth    # 最佳模型
├── epoch_10.pth                               # 第10个epoch
├── epoch_20.pth                               # 第20个epoch
└── latest.pth                                 # 最新checkpoint
```

### 日志文件

```
logs/
└── train_ntu60_x3d.log                       # 包含:
    ├── 系统信息
    ├── 配置详情
    ├── 训练过程 (每个epoch)
    ├── 验证结果 (每2个epoch)
    └── 最佳模型记录
```

### 可视化数据

```
work_dirs/pcan_ntu60_x3d/vis_data/
├── scalars.json                              # 训练曲线数据
├── config.py                                 # 配置文件副本
└── {timestamp}.json                          # 训练记录
```

---

## 🔧 常见操作

### 恢复训练

```bash
# 在训练脚本中添加 --resume
CUDA_VISIBLE_DEVICES=1,2 \
bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_x3d.py \
    2 \
    --work-dir work_dirs/pcan_ntu60_x3d \
    --resume  # 添加这个参数
```

### 测试模型

```bash
CUDA_VISIBLE_DEVICES=1,2 \
bash tools/dist_test.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_x3d.py \
    work_dirs/pcan_ntu60_x3d/best_acc_RGBPose_1:1_top1_epoch_78.pth \
    2
```

### 清理输出

```bash
# 删除某个实验的输出（谨慎！）
rm -rf work_dirs/pcan_ntu60_x3d/
rm -f logs/train_ntu60_x3d.log

# 或者只删除中间checkpoint，保留最佳模型
cd work_dirs/pcan_ntu60_x3d/
rm -f epoch_*.pth
rm -f latest.pth
# best_*.pth 会保留
```

---

## 📈 性能追踪

### 实时监控

```bash
# 方法1: tail命令
tail -f logs/train_ntu60_x3d.log

# 方法2: grep过滤
watch -n 10 "grep 'Epoch(val).*8244/8244' logs/train_ntu60_x3d.log | tail -3"

# 方法3: TensorBoard
tensorboard --logdir=work_dirs/pcan_ntu60_x3d/vis_data --port=6006
```

### 性能统计

```bash
# 查看所有验证结果
grep "Epoch(val).*8244/8244" logs/train_ntu60_x3d.log

# 查看最佳性能
grep "best checkpoint" logs/train_ntu60_x3d.log

# 查看训练时间
grep "eta:" logs/train_ntu60_x3d.log | tail -5
```

---

## 💡 重要提示

### ✅ 做到了
- ✅ 三个清晰独立的训练脚本
- ✅ 完整的训练指南文档
- ✅ 整理的配置文件目录
- ✅ 统一的日志管理
- ✅ 清晰的输出组织
- ✅ 删除所有临时文件

### ⚠️ 注意事项
- 实验3 (X3D) 需要 `emap_backbone/` 目录
- 所有实验默认使用2卡GPU (1,2)
- 确保数据集路径正确
- 训练前检查GPU可用性

### 📚 文档优先级
1. **`TRAINING_GUIDE.md`** - 训练入门必读
2. **`PROJECT_STRUCTURE.md`** - 本文档，项目结构
3. **`CONFIG_README.md`** - 配置文件详解
4. **`README.md`** - MMAction2官方文档

---

## 🆘 遇到问题？

1. **查看训练指南**: `cat TRAINING_GUIDE.md`
2. **检查日志**: `tail -f logs/train_*.log`
3. **查看配置**: 配置文件目录的 `CONFIG_README.md`
4. **GPU问题**: `nvidia-smi` 检查GPU状态

---

**项目已完成整理！祝训练顺利！🚀**

*最后更新: 2025-11-24*

