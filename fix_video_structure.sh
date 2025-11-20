#!/bin/bash

cd /home/zh/ChCode/codes01/mmaction2/data/ma52/raw_videos

echo "移动视频文件到正确位置..."

# 移动文件
for dir in train val test; do
    if [ -d "$dir" ]; then
        echo "处理 $dir/ ..."
        mv $dir/*.mp4 . 2>/dev/null && echo "  ✓ $dir/ 处理完成"
        rmdir $dir 2>/dev/null
    fi
done

# 统计
total=$(ls *.mp4 2>/dev/null | wc -l)
echo ""
echo "完成！共有 $total 个视频文件"

# 验证关键文件
for sample in test0000 val0000 train0000; do
    if [ -f "${sample}.mp4" ]; then
        echo "✓ ${sample}.mp4 存在"
    fi
done
