#!/bin/bash

# 🎯 阶段3训练脚本 - 冲刺95%！
# 从阶段2的85-90%继续

cd /home/zh/ChCode/codes01/mmaction2

# 激活conda环境
source /home/zh/anaconda3/bin/activate openmmlab

echo "=========================================="
echo "阶段3：冲刺95%！"
echo "从阶段2最佳checkpoint开始"
echo "=========================================="

# 获取阶段2最佳checkpoint
STAGE2_BEST=$(ls -t work_dirs/pcan_ntu60_stage2/best_acc_RGBPose_1:1_top1_epoch_*.pth 2>/dev/null | head -1)

if [ -z "$STAGE2_BEST" ]; then
    echo "错误：找不到阶段2的最佳checkpoint！"
    echo "请确保阶段2训练已完成。"
    exit 1
fi

echo "阶段2最佳checkpoint: $STAGE2_BEST"

# 提取epoch数字
EPOCH_NUM=$(echo $STAGE2_BEST | grep -oP 'epoch_\K[0-9]+')
echo "阶段2最佳Epoch: $EPOCH_NUM"

# 更新配置文件中的load_from
sed -i "s|epoch_XX|epoch_${EPOCH_NUM}|g" configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_stage3_95percent.py

# 检查GPU状态
nvidia-smi

echo ""
echo "启动阶段3训练（最后冲刺）..."
echo ""

CUDA_VISIBLE_DEVICES=1,2 \
nohup bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_stage3_95percent.py \
    2 \
    --work-dir work_dirs/pcan_ntu60_stage3_95percent \
    > train_stage3.log 2>&1 &

echo "阶段3训练已启动！PID: $!"
echo ""
echo "查看日志："
echo "  tail -f train_stage3.log"
echo ""
echo "查看验证结果："
echo "  grep 'Epoch(val).*8244/8244' train_stage3.log | tail -5"
echo ""
echo "预计12小时后达到95%！🎯"
echo ""
echo "=============================="
echo "方案A完整流程即将完成！"
echo "=============================="

