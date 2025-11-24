#!/bin/bash

# ==========================================
# 实验1: 原始Backbone + MA52数据集
# ==========================================
# 数据集：MA52 (52个动作类别)
# Backbone: RGBPoseConv3D (原始双流网络)
# 配置文件: configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py
# ==========================================

cd /home/zh/ChCode/codes01/mmaction2

# 激活conda环境
source /home/zh/anaconda3/bin/activate openmmlab

echo "=========================================="
echo "🚀 实验1: 原始Backbone + MA52数据集"
echo "=========================================="
echo "数据集: MA52 (52类动作)"
echo "Backbone: RGBPoseConv3D"
echo "  - RGB通道: 2048维"
echo "  - Pose通道: 512维"
echo "  - 骨架关键点: 28点"
echo "=========================================="
echo ""

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi
echo ""

# 检查配置文件
CONFIG="configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py"
if [ ! -f "$CONFIG" ]; then
    echo "❌ 错误：找不到配置文件 $CONFIG"
    exit 1
fi

echo "✅ 配置文件: $CONFIG"
echo ""

# 检查数据集
if [ ! -d "data/ma52/raw_videos" ]; then
    echo "❌ 错误：找不到MA52数据集 (data/ma52/raw_videos)"
    exit 1
fi

if [ ! -f "data/ma52/MA-52_openpose_28kp/MA52_train.pkl" ]; then
    echo "❌ 错误：找不到MA52标注文件"
    exit 1
fi

echo "✅ MA52数据集存在"
echo ""

# 显示训练参数
echo "训练参数："
echo "  - GPU: 1,2 (2卡并行)"
echo "  - Batch size: 16 (每卡8)"
echo "  - Learning rate: 0.001"
echo "  - Max epochs: 50"
echo "  - 数据增强: 标准"
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
    --work-dir work_dirs/ma52_original \
    2>&1 | tee logs/train_ma52_original.log

echo ""
echo "✅ 训练完成！"
echo ""
echo "查看结果："
echo "  - 日志: logs/train_ma52_original.log"
echo "  - checkpoint: work_dirs/ma52_original/"
echo "  - 最佳模型: work_dirs/ma52_original/best_*.pth"
echo ""

