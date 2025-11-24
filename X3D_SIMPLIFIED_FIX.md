# X3D集成遇到的问题和解决方案

## 问题根源

emap_backbone 是基于旧版本 mmcv (1.x) 开发的，与当前环境的 mmcv 2.x/mmengine 不兼容。

主要问题：
1. `digit_version` API 变更
2. `mmcv.runner` 移到了 `mmengine.runner`
3. `mmcv.cnn` 部分API移到了 `mmengine.model`
4. `mmcv.utils` 部分API变更

## 简化方案：放弃X3D，使用保守微调

鉴于X3D集成复杂度太高，建议采用**方案B：极保守微调**

### 配置修改

使用原有的 RGBPoseConv3D backbone，采用极低学习率：

```python
# 保持阶段1的loss_weights
loss_weights=[1.0, 1.5, 0.6, 1.2]

# 极低学习率
lr=0.0005

# 短期训练
max_epochs=15

# load_from 阶段1最佳checkpoint
load_from='work_dirs/pcan_ntu60_95target_rescue/best_acc_RGBPose_1:1_top1_epoch_42.pth'
resume=False
```

### 预期效果

- 起点：87.23%
- 目标：88-89%
- 时间：3-4小时
- 风险：极低

### 启动命令

```bash
# 使用修改后的pcan_ntu60_stage2_85percent.py配置
# 或创建新的保守配置
```

---

## 如果坚持使用X3D

需要完成以下工作（预计2-3小时）：

1. **修复所有 emap_backbone 文件的 mmcv 导入**
   - x3dTemporalshift.py ✅ 已修复
   - x3dTShiftPose.py ✅ 已修复
   - 其他10+个backbone文件需要修复

2. **创建完整的 mmcv_compat 兼容层**
   - ConvModule
   - Swish/SiLU
   - load_checkpoint
   - _BatchNorm
   - build_activation_layer
   - weight init functions

3. **注册到 mmaction MODELS**
   - 修改 mmaction/models/backbones/__init__.py
   - 导入并注册 X3D classes

4. **测试和调试**
   - 检查输出通道是否匹配
   - 验证数据流
   - 确认训练能正常开始

---

## 推荐：保守微调方案

时间紧迫，建议采用保守方案快速达到88-89%。

