#!/bin/bash

# 🚀 训练拯救脚本 - 从Epoch 11重启
# 使用保守策略，避免训练崩溃

cd /home/zh/ChCode/codes01/mmaction2

# 激活conda环境
source /home/zh/anaconda3/bin/activate openmmlab

echo "======================================"
echo "从Epoch 11重启训练（57.45%）"
echo "使用保守配置，避免崩溃"
echo "======================================"

# 清理之前的残留进程
pkill -9 -f "train.py" 2>/dev/null
sleep 2

# 检查GPU状态
nvidia-smi

echo ""
echo "启动训练..."
echo ""

CUDA_VISIBLE_DEVICES=0,1 \
nohup bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_95target.py \
    2 \
    --work-dir work_dirs/pcan_ntu60_95target_rescue \
    > train_rescue.log 2>&1 &

echo "训练已启动！PID: $!"
echo ""
echo "查看日志："
echo "  tail -f train_rescue.log"
echo ""
echo "查看验证结果："
echo "  grep 'Epoch(val).*8244/8244' train_rescue.log | tail -5"
echo ""
echo "预计12小时后完成50个epoch"

