#!/bin/bash

# ========================================
# 实时监控训练进度
# ========================================

WORK_DIR="work_dirs/pcan_ntu60_95target"
LOG_FILE=$(ls -t $WORK_DIR/*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "⚠️  未找到日志文件，训练可能还没开始"
    exit 1
fi

echo "========================================="
echo "📊 训练监控面板"
echo "========================================="
echo "日志文件: $LOG_FILE"
echo ""

# 提取验证准确率
echo "🎯 验证准确率趋势:"
echo "----------------------------------------"
grep "Epoch(val)" $LOG_FILE | grep "acc/RGBPose_1:1_top1" | \
    awk '{
        for(i=1;i<=NF;i++) {
            if($i ~ /Epoch\(val\)/) {
                epoch = $(i+1);
                gsub(/\[/, "", epoch);
                gsub(/\].*/, "", epoch);
            }
            if($i ~ /acc\/RGBPose_1:1_top1:/) {
                acc = $(i+1);
                printf "Epoch %3s: %s\n", epoch, acc;
            }
        }
    }'

echo ""
echo "----------------------------------------"

# 显示最佳结果
echo ""
echo "🏆 历史最佳:"
grep "best" $LOG_FILE 2>/dev/null | tail -3 || echo "  (暂无)"

echo ""
echo "========================================="
echo ""
echo "💡 提示:"
echo "  - 实时监控: tail -f $LOG_FILE"
echo "  - 查看学习率: grep 'lr:' $LOG_FILE | tail -20"
echo "  - 重新运行监控: $0"
echo ""

