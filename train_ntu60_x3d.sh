#!/bin/bash

# ==========================================
# 实验3: X3D Backbone + NTU60数据集
# ==========================================
# 数据集：NTU RGB+D 60 (60个动作类别)
# Backbone: X3D TemporalShift (轻量高效)
# 配置文件: configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_x3d.py
# 最佳性能: 90.44% (Epoch 78)
# ==========================================

cd /home/zh/ChCode/codes01/mmaction2

# 激活conda环境
source /home/zh/anaconda3/bin/activate openmmlab

echo "=========================================="
echo "🚀 实验3: X3D Backbone + NTU60数据集"
echo "=========================================="
echo "数据集: NTU RGB+D 60 (60类动作)"
echo "Backbone: X3D TemporalShift"
echo "  - RGB通道: 432维 (X3D-M)"
echo "  - Pose通道: 216维 (X3D-S)"
echo "  - 骨架关键点: 17点"
echo "  - 参数量: ~15M (减少70%)"
echo "  - 分层分类: 8个粗类 + 60个细类"
echo "=========================================="
echo ""

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi
echo ""

# 检查emap_backbone目录
if [ ! -d "emap_backbone" ]; then
    echo "❌ 错误：找不到emap_backbone目录！"
    echo "X3D backbone需要此目录。"
    exit 1
fi

echo "✅ emap_backbone目录存在"
echo ""

# 检查配置文件
CONFIG="configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_x3d.py"
if [ ! -f "$CONFIG" ]; then
    echo "❌ 错误：找不到配置文件 $CONFIG"
    exit 1
fi

echo "✅ 配置文件: $CONFIG"
echo ""

# 检查数据集
if [ ! -d "data/nturgbd_videos" ]; then
    echo "❌ 错误：找不到NTU60数据集 (data/nturgbd_videos)"
    exit 1
fi

if [ ! -f "data/skeleton/ntu60_xsub.pkl" ]; then
    echo "❌ 错误：找不到NTU60标注文件"
    exit 1
fi

echo "✅ NTU60数据集存在"
echo ""

# 显示训练参数
echo "训练参数："
echo "  - GPU: 1,2 (2卡并行)"
echo "  - Batch size: 32 (每卡16)"
echo "  - Learning rate: 0.012 → 1e-6"
echo "  - Max epochs: 80"
echo "  - Loss权重: [1.0, 1.5, 0.6, 1.2]"
echo "  - 优化器: SGD"
echo "  - Warmup: 5 epochs"
echo "  - 学习率调度: CosineAnnealing"
echo ""

echo "⏱️  3秒后启动训练..."
sleep 3
echo ""

# 启动训练
echo "=========================================="
echo "启动训练..."
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=1,2 \
bash tools/dist_train.sh \
    $CONFIG \
    2 \
    --work-dir work_dirs/pcan_ntu60_x3d \
    2>&1 | tee logs/train_ntu60_x3d.log

echo ""
echo "✅ 训练完成！"
echo ""
echo "查看结果："
echo "  - 日志: logs/train_ntu60_x3d.log"
echo "  - checkpoint: work_dirs/pcan_ntu60_x3d/"
echo "  - 最佳模型: work_dirs/pcan_ntu60_x3d/best_*.pth"
echo ""
echo "历史最佳性能: 90.44% @ Epoch 78"
echo "  - RGB分支: 83.93%"
echo "  - Pose分支: 89.06%"
echo ""

