#!/bin/bash

# ========================================
# PCAN NTU-60 冲击95%准确率训练脚本
# ========================================

cd /home/zh/ChCode/codes01/mmaction2

echo "========================================="
echo "🎯 目标: 从89% → 95%+"
echo "策略: 从头训练100个epoch"
echo "========================================="

# 配置文件
CONFIG="configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_95target.py"

# 工作目录
WORK_DIR="work_dirs/pcan_ntu60_95target"

# 清空之前的训练（可选）
# rm -rf $WORK_DIR

echo ""
echo "📋 训练配置:"
echo "  - 配置文件: $CONFIG"
echo "  - 工作目录: $WORK_DIR"
echo "  - 总Epoch: 100"
echo "  - 初始学习率: 0.01"
echo "  - Batch Size: 20"
echo "  - 数据增强: 激进模式"
echo ""

echo "⏰ 预计训练时间: 约20小时"
echo ""

read -p "确认开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "已取消训练"
    exit 1
fi

echo ""
echo "🚀 开始训练..."
echo ""

# 双卡训练
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
    $CONFIG \
    2 \
    --work-dir $WORK_DIR

echo ""
echo "========================================="
echo "✅ 训练完成!"
echo "========================================="
echo ""
echo "📊 查看最佳结果:"
echo "  grep 'best' $WORK_DIR/*.log | tail -5"
echo ""
echo "📈 查看训练曲线:"
echo "  grep 'Epoch(val)' $WORK_DIR/*.log | grep 'acc/RGBPose_1:1_top1'"
echo ""

