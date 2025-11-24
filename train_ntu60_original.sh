#!/bin/bash

# ==========================================
# 实验2: 原始Backbone + NTU60数据集
# ==========================================
# 数据集：NTU RGB+D 60 (60个动作类别)
# Backbone: RGBPoseConv3D (原始双流网络)
# 配置文件: configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py
# ==========================================

cd /home/zh/ChCode/codes01/mmaction2

# 激活conda环境
source /home/zh/anaconda3/bin/activate openmmlab

echo "=========================================="
echo "🚀 实验2: 原始Backbone + NTU60数据集"
echo "=========================================="
echo "数据集: NTU RGB+D 60 (60类动作)"
echo "Backbone: RGBPoseConv3D"
echo "  - RGB通道: 2048维"
echo "  - Pose通道: 512维"
echo "  - 骨架关键点: 17点"
echo "  - 分层分类: 8个粗类 + 60个细类"
echo "=========================================="
echo ""

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi
echo ""

# 检查配置文件
CONFIG="configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py"
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
echo "  - Batch size: 24 (每卡12)"
echo "  - Learning rate: 0.001"
echo "  - Max epochs: 80"
echo "  - Loss权重: [1.0, 1.2, 0.5, 0.8]"
echo "  - 优化器: SGD"
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
    --work-dir work_dirs/pcan_ntu60_original \
    2>&1 | tee logs/train_ntu60_original.log

echo ""
echo "✅ 训练完成！"
echo ""
echo "查看结果："
echo "  - 日志: logs/train_ntu60_original.log"
echo "  - checkpoint: work_dirs/pcan_ntu60_original/"
echo "  - 最佳模型: work_dirs/pcan_ntu60_original/best_*.pth"
echo ""
echo "预期性能: ~85-87% (Top-1 Accuracy)"
echo ""

