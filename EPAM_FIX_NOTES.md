# EPAM Backbone 接口修复说明

## 问题描述

训练时出现错误：
```
TypeError: forward() got an unexpected keyword argument 'gt'
```

## 根本原因

1. **`RGBPoseConv3D`** 的 `forward()` 方法接受 `gt` 和 `gt_coarse` 参数，并返回 **6个元素**：
   ```python
   def forward(self, imgs, heatmap_imgs, gt, gt_coarse):
       ...
       return x_rgb, x_pose, x_rgb1, x_pose1, gt, gt_coarse
   ```

2. **`RGBPoseHead`** 的 `forward()` 方法期望接收这6个元素的列表：
   ```python
   def forward(self, x: List[torch.Tensor]):
       x_rgb, x_pose = x[0], x[1]  # 最终特征
       ...
       if self.training:
           x_rgb1, x_pose1 = x[2], x[3]  # 中间特征（用于FR loss）
           gt = x[4]                       # 细粒度标签
           gt_coarse = x[5]                # 粗粒度标签
   ```

3. **原始错误**: 我的 `EPAMBackbone` 只返回2个元素 `(rgb_feat, pose_feat)`，并且错误地将 `gt` 和 `gt_coarse` 通过 `loss_predict_kwargs` 传递给了 head 的 `loss()` 方法，而不是通过 `forward()` 参数。

## 修复方案

### 1. 修改 `EPAMBackbone.forward()` - 返回6个元素

**文件**: `mmaction/models/backbones/epam_backbone.py`

```python
def forward(self, imgs: torch.Tensor, heatmap_imgs: torch.Tensor,
            gt: Optional[torch.Tensor] = None, 
            gt_coarse: Optional[torch.Tensor] = None) -> tuple:
    """
    返回6个元素以兼容RGBPoseHead：
    - rgb_feat: 最终RGB特征
    - pose_feat: 最终Pose特征
    - rgb_feat1: 中间RGB特征（用于FR loss）
    - pose_feat1: 中间Pose特征（用于FR loss）
    - gt: 细粒度标签
    - gt_coarse: 粗粒度标签
    """
    rgb_feat, pose_feat = self.backbone(imgs, heatmap_imgs)
    
    # 复制特征用于FR loss（EPAM没有中间特征的概念）
    rgb_feat1 = rgb_feat.clone()
    pose_feat1 = pose_feat.clone()
    
    # 返回6个元素：(x_rgb, x_pose, x_rgb1, x_pose1, gt, gt_coarse)
    return rgb_feat, pose_feat, rgb_feat1, pose_feat1, gt, gt_coarse
```

**关键点**:
- 接受 `gt` 和 `gt_coarse` 作为参数
- 返回6个元素而不是2个
- 将特征复制一份作为 `rgb_feat1` 和 `pose_feat1`（用于FR loss）

### 2. 修改 `EPAMRecognizer.extract_feat()` - 正确传递参数

**文件**: `mmaction/models/recognizers/epam_recognizer.py`

**修改点**:

#### a) 将gt和gt_coarse传递给backbone
```python
# ✅ 正确：传递给backbone
result = self.backbone(rgb_data, pose_data, gts, gts_coarse)

# ❌ 错误（之前的做法）：放入loss_predict_kwargs传给head
# loss_predict_kwargs['gt'] = gts
# loss_predict_kwargs['gt_coarse'] = gts_coarse
```

#### b) 处理6元素的返回值
```python
# backbone返回6个元素
result = self.backbone(rgb_data, pose_data, gts, gts_coarse)
x = result  # 直接传递完整的元组给head
```

#### c) 不传递loss_predict_kwargs给head
```python
# ✅ 正确：head不接受额外的kwargs
x = self.cls_head(x)

# ❌ 错误（之前的做法）
# x = self.cls_head(x, **loss_predict_kwargs)
```

### 3. 修改 `EPAMRecognizer.loss()` - 简化调用

```python
def loss(self, inputs, data_samples, **kwargs):
    feats, loss_predict_kwargs = self.extract_feat(inputs, data_samples=data_samples)
    # 不传递loss_predict_kwargs给head.loss()
    loss = self.cls_head.loss(feats, data_samples)
    return loss
```

### 4. 修改 `EPAMRecognizer.predict()` - 简化调用

```python
def predict(self, inputs, data_samples, **kwargs):
    feats, loss_predict_kwargs = self.extract_feat(inputs, data_samples=data_samples, test_mode=True)
    # 不传递loss_predict_kwargs给head.predict()
    predictions = self.cls_head.predict(feats, data_samples)
    return predictions
```

## 数据流对比

### ❌ 修复前（错误）

```
EPAMRecognizer
    ↓
extract_feat():
    - 生成 gt, gt_coarse
    - 放入 loss_predict_kwargs
    - backbone(rgb, pose)  ← 没有传递gt
    - 返回 (rgb_feat, pose_feat)  ← 只有2个元素
    ↓
loss():
    - cls_head.loss(feats, **loss_predict_kwargs)  ← 传递gt给loss
    ↓
RGBPoseHead.loss():
    - self(feats, **kwargs)  ← 试图传递gt给forward
    ↓
RGBPoseHead.forward():
    - forward(x)  ← 不接受额外的kwargs
    ✗ TypeError: forward() got an unexpected keyword argument 'gt'
```

### ✅ 修复后（正确）

```
EPAMRecognizer
    ↓
extract_feat():
    - 生成 gt, gt_coarse
    - backbone(rgb, pose, gt, gt_coarse)  ← 传递给backbone
    - 返回 (rgb, pose, rgb1, pose1, gt, gt_coarse)  ← 6个元素
    ↓
loss():
    - cls_head.loss(feats)  ← 不传递额外参数
    ↓
RGBPoseHead.loss():
    - self(feats)  ← 不传递额外参数
    ↓
RGBPoseHead.forward():
    - forward(x)  ← x包含6个元素
    - x[0], x[1]: 最终特征
    - x[2], x[3]: 中间特征（FR loss）
    - x[4], x[5]: gt标签
    ✓ 正常工作
```

## 为什么需要6个元素？

`RGBPoseHead` 在训练时使用多个损失：

1. **分类损失** (Fine & Coarse)
   - 使用 `x[0]` (RGB) 和 `x[1]` (Pose)
   - 需要 `x[4]` (gt) 和 `x[5]` (gt_coarse)

2. **Feature Renovation Loss**
   - 使用 `x[2]` (RGB1) 和 `x[3]` (Pose1)
   - 需要 `x[4]` (gt) 和 `x[5]` (gt_coarse)

3. **Tree Loss** (层次化)
   - 需要粗粒度和细粒度预测
   - 需要 `x[4]` (gt) 和 `x[5]` (gt_coarse)

## 验证修复

运行测试脚本：
```bash
python test_epam_final.py
```

预期输出：
```
✅ Backbone返回元素数量: 6
   [0] RGB特征: torch.Size([2, 432, 16, 7, 7])
   [1] Pose特征: torch.Size([2, 216, 48, 7, 7])
   [2] RGB特征1 (FR): torch.Size([2, 432, 16, 7, 7])
   [3] Pose特征1 (FR): torch.Size([2, 216, 48, 7, 7])
   [4] GT标签: torch.Size([2])
   [5] GT粗粒度: torch.Size([2])
✅ 返回值格式正确！
```

## 重新训练

现在可以正常训练：

```bash
# 使用GPU 1和2
CUDA_VISIBLE_DEVICES=1,2 bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/epam_ntu60_baseline.py \
    2 \
    --work-dir work_dirs/epam_ntu60_baseline_2gpu
```

## 总结

**关键要点**:
1. ✅ `gt` 和 `gt_coarse` 必须作为**参数**传递给 backbone，而不是传递给 head
2. ✅ Backbone 必须返回 **6个元素**：`(rgb, pose, rgb1, pose1, gt, gt_coarse)`
3. ✅ Head 的 `forward()` 通过访问列表索引 `x[4]` 和 `x[5]` 来获取标签
4. ✅ 不要在 `cls_head.loss()` 或 `cls_head.predict()` 调用时传递额外的 kwargs

**修复文件**:
- ✅ `mmaction/models/backbones/epam_backbone.py`
- ✅ `mmaction/models/recognizers/epam_recognizer.py`

---

*修复日期: 2025-11-25*  
*测试状态: ✅ 通过*

