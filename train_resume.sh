#!/bin/bash

###############################################################
# PCAN NTU-60 优化训练脚本
# 从Epoch 30继续训练到Epoch 50
# 配置已优化：余弦退火学习率 + 早停 + 增强数据增强
###############################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PCAN NTU-60 优化训练脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 配置路径
CONFIG="configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py"
WORK_DIR="work_dirs/pcan_ntu60"

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG${NC}"
    exit 1
fi

# 检查checkpoint
if [ ! -f "$WORK_DIR/last_checkpoint" ]; then
    echo -e "${RED}错误: 找不到last_checkpoint文件${NC}"
    exit 1
fi

LAST_CKPT=$(cat $WORK_DIR/last_checkpoint)
echo -e "${YELLOW}将从checkpoint恢复训练: $LAST_CKPT${NC}"

# 显示优化配置摘要
echo -e "\n${GREEN}优化配置摘要:${NC}"
echo "  ├─ Max Epochs: 30 → 50 (+20个epoch)"
echo "  ├─ 学习率策略: MultiStepLR → LinearLR+CosineAnnealingLR"
echo "  ├─ 初始学习率: 0.015 → 0.01"
echo "  ├─ Pose分支权重: 1.0 → 2.0"
echo "  ├─ 数据增强: 新增ColorJitter"
echo "  ├─ 早停patience: 10 epochs"
echo "  └─ Checkpoint间隔: 10 → 5 epochs"

# 检测GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo -e "\n${GREEN}检测到GPU数量: $NUM_GPUS${NC}"

# 询问用户选择训练模式
echo -e "\n${YELLOW}请选择训练模式:${NC}"
echo "  1) 双卡训练 (推荐)"
echo "  2) 单卡训练"
echo "  3) 指定GPU卡号"
read -p "请输入选择 [1-3]: " choice

case $choice in
    1)
        echo -e "${GREEN}启动双卡训练...${NC}"
        CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
            $CONFIG \
            2 \
            --work-dir $WORK_DIR \
            --cfg-options resume=True
        ;;
    2)
        echo -e "${GREEN}启动单卡训练...${NC}"
        python tools/train.py \
            $CONFIG \
            --work-dir $WORK_DIR \
            --cfg-options resume=True
        ;;
    3)
        read -p "请输入GPU卡号 (例如: 0 或 0,1): " gpu_ids
        echo -e "${GREEN}使用GPU: $gpu_ids${NC}"
        
        # 计算GPU数量
        IFS=',' read -ra GPUS <<< "$gpu_ids"
        num_gpus=${#GPUS[@]}
        
        if [ $num_gpus -gt 1 ]; then
            echo -e "${GREEN}启动多卡训练 (${num_gpus}卡)...${NC}"
            CUDA_VISIBLE_DEVICES=$gpu_ids bash tools/dist_train.sh \
                $CONFIG \
                $num_gpus \
                --work-dir $WORK_DIR \
                --cfg-options resume=True
        else
            echo -e "${GREEN}启动单卡训练...${NC}"
            CUDA_VISIBLE_DEVICES=$gpu_ids python tools/train.py \
                $CONFIG \
                --work-dir $WORK_DIR \
                --cfg-options resume=True
        fi
        ;;
    *)
        echo -e "${RED}无效选择，退出${NC}"
        exit 1
        ;;
esac

# 训练完成后的提示
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}训练完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n${YELLOW}查看最佳模型:${NC}"
echo "  ls -lh $WORK_DIR/best_*.pth"
echo -e "\n${YELLOW}查看训练日志:${NC}"
echo "  tail -100 $WORK_DIR/\$(ls -t $WORK_DIR/*.log | head -1)"
echo -e "\n${YELLOW}可视化训练曲线:${NC}"
echo "  python tools/analysis_tools/analyze_logs.py plot_curve \\"
echo "      $WORK_DIR/\$(ls -t $WORK_DIR/*.log | head -1) \\"
echo "      --keys acc/RGBPose_1:1_top1 loss \\"
echo "      --out $WORK_DIR/training_curve.png"
echo ""

