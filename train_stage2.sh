#!/bin/bash

# 🚀 阶段2训练脚本 - 冲刺90-92% (保守策略)
# 从阶段1的87%继续

cd /home/zh/ChCode/codes01/mmaction2

# 激活conda环境
source /home/zh/anaconda3/bin/activate openmmlab

echo "=========================================="
echo "阶段2：冲刺90-92% (保守策略)"
echo "从阶段1 87%的excellent成绩开始"
echo "使用lr=0.003温和微调，30 epochs"
echo "=========================================="

# 获取阶段1最佳checkpoint
STAGE1_BEST=$(ls -t work_dirs/pcan_ntu60_95target_rescue/best_acc_RGBPose_1:1_top1_epoch_*.pth 2>/dev/null | head -1)

if [ -z "$STAGE1_BEST" ]; then
    echo "错误：找不到阶段1的最佳checkpoint！"
    echo "请确保阶段1训练已完成。"
    exit 1
fi

echo "阶段1最佳checkpoint: $STAGE1_BEST"

# 提取epoch数字
EPOCH_NUM=$(echo $STAGE1_BEST | grep -oP 'epoch_\K[0-9]+')
echo "阶段1最佳Epoch: $EPOCH_NUM (87.23%)"

# 配置文件已更新为保守策略：
echo "  - lr: 0.003 (温和微调)"
echo "  - max_epochs: 30"
echo "  - loss_weights: [1.0, 1.2, 0.5, 0.9]"

# 检查GPU状态
nvidia-smi

echo ""
echo "启动阶段2训练..."
echo ""

CUDA_VISIBLE_DEVICES=1,2 \
nohup bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60_stage2_85percent.py \
    2 \
    --work-dir work_dirs/pcan_ntu60_stage2 \
    > train_stage2.log 2>&1 &

echo "阶段2训练已启动！PID: $!"
echo ""
echo "查看日志："
echo "  tail -f train_stage2.log"
echo ""
echo "查看验证结果："
echo "  grep 'Epoch(val).*8244/8244' train_stage2.log | tail -5"
echo ""
echo "预计7小时后达到90-92%"
echo "祝你休息愉快！🌙"

