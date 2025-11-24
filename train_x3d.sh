#!/bin/bash

# ========================================== 
# X3D TemporalShift训练脚本
# 目标：从87% → 90-93%
# 架构创新：X3D + TemporalShift
# ==========================================

cd /home/zh/ChCode/codes01/mmaction2

# 激活conda环境
source /home/zh/anaconda3/bin/activate openmmlab

echo "=========================================="
echo "🚀 X3D TemporalShift 训练"
echo "=========================================="
echo "目标：突破87%瓶颈 → 90-93%"
echo "架构："
echo "  - RGB: X3D TemporalShift (432通道)"
echo "  - Pose: X3D TemporalShift Pose (216通道)"
echo "  - 参数量：~15M (减少70%)"
echo "  - 训练策略：80 epochs, lr=0.012"
echo "=========================================="
echo ""

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi
echo ""

# 检查emap_backbone是否存在
if [ ! -d "emap_backbone" ]; then
    echo "❌ 错误：找不到emap_backbone目录！"
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

echo "✅ 配置文件存在: $CONFIG"
echo ""

# 显示训练参数
echo "训练参数："
echo "  - 配置文件: $CONFIG"
echo "  - GPU: 1,2"
echo "  - Batch size: 12"
echo "  - Max epochs: 80"
echo "  - 初始lr: 0.012"
echo "  - Weight decay: 0.0002"
echo "  - Warmup: 5 epochs"
echo "  - 预计时间: 8-10小时"
echo ""
echo "⏱️  3秒后自动启动训练..."
sleep 3
echo ""
echo "=========================================="
echo "启动训练..."
echo "=========================================="
echo ""

# 启动训练（同时输出到控制台和文件）
( CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh $CONFIG 2 --work-dir work_dirs/pcan_ntu60_x3d ) 2>&1 | tee train_x3d.log

TRAIN_PID=$!

echo "✅ X3D训练已启动！PID: $TRAIN_PID"
echo ""
echo "=========================================="
echo "📊 监控命令"
echo "=========================================="
echo ""
echo "实时查看日志："
echo "  tail -f train_x3d.log"
echo ""
echo "查看最新验证结果："
echo "  grep 'Epoch(val).*8244/8244' train_x3d.log | tail -3"
echo ""
echo "查看GPU使用："
echo "  nvidia-smi"
echo ""
echo "停止训练："
echo "  kill $TRAIN_PID"
echo ""
echo "=========================================="
echo "🎯 预期结果"
echo "=========================================="
echo ""
echo "Epoch 10:  ~84%"
echo "Epoch 20:  ~86%"
echo "Epoch 30:  ~88%"
echo "Epoch 40:  ~89%"
echo "Epoch 60:  ~90-91%"
echo "Epoch 80:  ~91-93% ✨"
echo ""
echo "预计8-10小时完成训练"
echo "=========================================="
echo ""

