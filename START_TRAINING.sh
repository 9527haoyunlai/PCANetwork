#!/bin/bash
# ==========================================
# EPAM + NTU60 训练启动脚本
# ==========================================

echo "=========================================="
echo "EPAM-Net NTU60 训练启动"
echo "=========================================="

# 配置
GPU_IDS="1,2"              # 使用的GPU编号
NUM_GPUS=2                 # GPU数量
CONFIG="configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_optimized.py"  # ⭐ 优化配置
WORK_DIR="work_dirs/epam_ntu60_optimized_2gpu"

echo ""
echo "📋 训练配置："
echo "   - GPU: ${GPU_IDS}"
echo "   - GPU数量: ${NUM_GPUS}"
echo "   - 配置文件: ${CONFIG}"
echo "   - 输出目录: ${WORK_DIR}"
echo ""

# 检查GPU
echo "🔍 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | grep -E "^[${GPU_IDS//,/|}]"

if [ $? -ne 0 ]; then
    echo "❌ 错误：指定的GPU不可用"
    exit 1
fi

echo "✅ GPU可用"
echo ""

# 检查配置文件
if [ ! -f "${CONFIG}" ]; then
    echo "❌ 错误：配置文件不存在: ${CONFIG}"
    exit 1
fi

echo "✅ 配置文件存在"
echo ""

# 创建工作目录
mkdir -p "${WORK_DIR}"
echo "✅ 工作目录已创建: ${WORK_DIR}"
echo ""

# 显示关键超参数
echo "🎯 关键超参数（优化版）："
echo "   - 学习率: 0.0015 (匹配原始模型)"
echo "   - Weight Decay: 0.0003 (更强正则化)"
echo "   - LR调度: CosineAnnealingLR (更平滑)"
echo "   - Batch Size: 8/GPU, 总batch=16"
echo "   - Epochs: 50"
echo "   - 随机种子: 100 (可复现)"
echo ""

# 询问确认
echo "=========================================="
read -p "⚠️  确认开始训练？(y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "❌ 训练已取消"
    exit 0
fi

echo ""
echo "🚀 开始训练..."
echo "=========================================="
echo ""

# 启动训练
CUDA_VISIBLE_DEVICES=${GPU_IDS} bash tools/dist_train.sh \
    ${CONFIG} \
    ${NUM_GPUS} \
    --work-dir ${WORK_DIR}

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 训练完成！"
    echo "=========================================="
    echo ""
    echo "📁 结果保存在: ${WORK_DIR}"
    echo ""
    echo "📊 查看结果："
    echo "   - 最佳模型: ${WORK_DIR}/best_*.pth"
    echo "   - 训练日志: ${WORK_DIR}/*.log"
    echo "   - 训练曲线: ${WORK_DIR}/*.json"
    echo ""
    echo "🧪 测试模型："
    echo "   CUDA_VISIBLE_DEVICES=${GPU_IDS} bash tools/dist_test.sh \\"
    echo "       ${CONFIG} \\"
    echo "       ${WORK_DIR}/best_*.pth \\"
    echo "       ${NUM_GPUS}"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 训练失败！"
    echo "=========================================="
    echo ""
    echo "🔍 请检查："
    echo "   1. 日志文件: ${WORK_DIR}/*.log"
    echo "   2. GPU内存是否充足"
    echo "   3. 数据路径是否正确"
    echo ""
    exit 1
fi

