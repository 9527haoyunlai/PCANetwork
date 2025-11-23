#!/bin/bash

###############################################################
# 简单配置验证脚本（不依赖Python环境）
###############################################################

CONFIG="configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}配置文件验证${NC}"
echo -e "${GREEN}========================================${NC}\n"

if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}❌ 配置文件不存在: $CONFIG${NC}"
    exit 1
fi

echo "检查关键配置项..."
echo ""

# 检查max_epochs
if grep -q "max_epochs=50" "$CONFIG"; then
    echo -e "${GREEN}✅ max_epochs = 50${NC}"
else
    echo -e "${YELLOW}⚠️  max_epochs 可能未设置为50${NC}"
fi

# 检查学习率
if grep -q "lr=0.01" "$CONFIG"; then
    echo -e "${GREEN}✅ 初始学习率 = 0.01${NC}"
else
    echo -e "${YELLOW}⚠️  初始学习率可能不是0.01${NC}"
fi

# 检查梯度裁剪
if grep -q "max_norm=20" "$CONFIG"; then
    echo -e "${GREEN}✅ 梯度裁剪 max_norm = 20${NC}"
else
    echo -e "${YELLOW}⚠️  梯度裁剪可能不是20${NC}"
fi

# 检查学习率调度器
if grep -q "CosineAnnealingLR" "$CONFIG"; then
    echo -e "${GREEN}✅ 余弦退火学习率已配置${NC}"
else
    echo -e "${RED}❌ 未检测到CosineAnnealingLR${NC}"
fi

# 检查loss权重
if grep -q "loss_weights=\[1., 2., 0.5, 1.0\]" "$CONFIG"; then
    echo -e "${GREEN}✅ Loss权重已优化 (Pose=2.0)${NC}"
else
    echo -e "${YELLOW}⚠️  Loss权重可能未优化${NC}"
fi

# 检查resume
if grep -q "resume = True" "$CONFIG"; then
    echo -e "${GREEN}✅ Resume训练已启用${NC}"
else
    echo -e "${YELLOW}⚠️  Resume可能未启用${NC}"
fi

# 检查早停
if grep -q "early_stopping" "$CONFIG"; then
    echo -e "${GREEN}✅ 早停机制已配置${NC}"
else
    echo -e "${YELLOW}⚠️  未检测到早停配置${NC}"
fi

# 检查ColorJitter
if grep -q "ColorJitter" "$CONFIG"; then
    echo -e "${GREEN}✅ ColorJitter数据增强已添加${NC}"
else
    echo -e "${YELLOW}⚠️  未检测到ColorJitter${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}验证完成！${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${YELLOW}准备开始训练？运行:${NC}"
echo "  ./train_resume.sh"
echo ""

