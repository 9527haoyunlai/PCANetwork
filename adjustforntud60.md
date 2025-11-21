# 模型在ntud60数据集上的适用性
_Exported on 11/21/2025 at 02:26:09 GMT+8 from Cursor (2.0.77)_

---

**User**

我对模型的网络结构大致了解了，现在我想请问一下，这个网络模型如果跑ntud60的数据集，可以做到吗？如果可以需要做哪些准备和调整呢，因为ntu数据集是不存在粗分类的，包括这个骨架序列，因为ntu的骨架点是17或者25个，而这个论文中使用的ma52数据集的骨架部分数据则是28个骨骼点呢，因此如果该模型跑ntu60数据集的话整体变化大不大？@mmaction2 @PACN模型解释.md 

---

**Cursor**

好问题！让我帮你分析PCAN模型如何适配NTU-RGB+D 60数据集。首先让我查看一下相关配置和代码结构。

非常好的问题！让我详细分析PCAN模型在NTU-RGB+D 60数据集上的适配方案。

## 📊 关键差异对比

| 维度 | MA-52（当前） | NTU-RGB+D 60 | 差异程度 |
|------|--------------|--------------|---------|
| **类别数** | 52类（action）+ 7类（body） | 60类（无层次） | ⚠️ 中等 |
| **骨架点数** | 28点（OpenPose） | 25点（Kinect v2） | ⚠️ 中等 |
| **层次标签** | ✓ 有粗粒度标签 | ❌ 无层次结构 | ❌ 重大 |
| **数据规模** | ~24K样本 | ~40K样本（XSub） | ✓ 无影响 |

## ✅ 可行性分析

**答案：可以，但需要适度修改**

### 修改程度估计

```
整体变化：⭐⭐⭐☆☆ (中等)

必须修改的部分：
  🔴 骨架点数：in_channels (28→25)
  🔴 类别数：num_classes (52→60)
  🔴 粗分类处理：移除或虚拟化
  🔴 TreeLoss：去掉或修改

可选修改的部分：
  🟡 原型数量：自动适配60类
  🟢 Backbone：无需修改
  🟢 双流架构：保持不变
```

## 🔧 详细适配方案

### 方案A：完全去除PCAN特性（最简单，不推荐）

这样就退化成基础的RGBPoseConv3D，失去了PCAN的优势。

### 方案B：保留PCAN，虚拟化粗分类（推荐）⭐

为NTU-60创建虚拟的粗分类标签。

### 方案C：自定义NTU的粗分类（最优，但工作量大）

根据NTU-60的动作语义定义层次结构。

---

## 📝 方案B实施指南（推荐）

### 第1步：修改配置文件

创建 `configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d_ntu60.py`：

```python
_base_ = '../../../_base_/default_runtime.py'

# ========== 修改1: Backbone配置 ==========
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        fusion_kernel=7,
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1),
        with_pool2=False),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        fusion_kernel=7,
        in_channels=25,  # ← 改为25 (NTU骨架点数)
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride_s=1,
        conv1_stride_t=1,
        pool1_stride_s=1,
        pool1_stride_t=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        with_pool2=False))

# ========== 修改2: Head配置 ==========
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← 改为60类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],
    loss_weights=[1., 1.],
    average_clips='prob')

data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    preprocessors=dict(
        imgs=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        heatmap_imgs=dict(type='ActionDataPreprocessor')))

model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    data_preprocessor=data_preprocessor)

# ========== 修改3: 数据配置 ==========
dataset_type = 'PoseDataset'
data_root = 'data/nturgbd_videos/'
ann_file = 'data/skeleton/ntu60_3d.pkl'  # 或 ntu60_2d.pkl

# NTU-60的骨架点定义（25个关节点）
left_kp = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

# ... 其他pipeline和dataloader配置类似 ...

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=3, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[10, 20],
        gamma=0.1)
]
```

### 第2步：修改Head代码以支持无粗分类模式

需要修改 `mmaction/models/heads/rgbpose_head.py`：

```python
# ========== 修改1: __init__方法（约第520行）==========
def __init__(self,
             num_classes: int,
             in_channels: Tuple[int],
             loss_cls: Dict = dict(type='CrossEntropyLoss'),
             loss_components: List[str] = ['rgb', 'pose'],
             loss_weights: Union[float, Tuple[float]] = 1.,
             dropout: float = 0.5,
             init_std: float = 0.01,
             use_coarse: bool = True,  # ← 新增：是否使用粗分类
             num_coarse_classes: int = 7,  # ← 新增：粗分类数量
             **kwargs) -> None:
    
    super().__init__(num_classes, in_channels, loss_cls, **kwargs)
    
    # ... 原有代码 ...
    
    self.fc_rgb = nn.Linear(self.in_channels[0], num_classes)
    self.fc_pose = nn.Linear(self.in_channels[1], num_classes)
    
    # 根据use_coarse决定是否创建粗分类相关模块
    self.use_coarse = use_coarse
    if self.use_coarse:
        self.fc_rgb_coarse = nn.Linear(self.in_channels[0], num_coarse_classes)
        self.fc_pose_coarse = nn.Linear(self.in_channels[1], num_coarse_classes)
        
        self.fr_coarse_rgb = ST_RenovateNet(
            2048, 8, n_class=num_coarse_classes, h_channel=128)
        self.fr_coarse_pose = ST_RenovateNet(
            512, 32, n_class=num_coarse_classes, h_channel=128)
        
        self.tree_loss_rgb = TreeLoss()
        self.tree_loss_pose = TreeLoss()
    
    # 细粒度RenovateNet（永远需要）
    self.fr_rgb = ST_RenovateNet_Fine(
        2048, n_class=num_classes, version='V0', use_p_map=True)
    self.fr_pose = ST_RenovateNet_Fine(
        512, n_class=num_classes, version='V0', use_p_map=True)


# ========== 修改2: forward方法（约第576行）==========
def forward(self, x: List[torch.Tensor]) -> Dict:
    x_rgb, x_pose = self.avg_pool(x[0]), self.avg_pool(x[1])
    x_rgb = x_rgb.view(x_rgb.size(0), -1)
    x_pose = x_pose.view(x_pose.size(0), -1)
    
    x_rgb = self.dropout_rgb(x_rgb)
    x_pose = self.dropout_pose(x_pose)
    
    cls_scores = dict()
    logits_rgb = self.fc_rgb(x_rgb)
    logits_pose = self.fc_pose(x_pose)
    
    if self.training:
        cls_scores['rgb'] = logits_rgb
        cls_scores['pose'] = logits_pose
        
        x_rgb1, x_pose1 = x[2], x[3]
        gt = x[4]
        
        x_rgb1 = x_rgb1.mean(dim=2)
        x_pose1 = x_pose1.mean(dim=2)
        
        if self.use_coarse:
            # 有粗分类的情况（MA-52）
            gt_coarse = x[5]
            logits_coarse_rgb = self.fc_rgb_coarse(x_rgb)
            logits_coarse_pose = self.fc_pose_coarse(x_pose)
            
            cls_scores['rgb_coarse'] = logits_coarse_rgb
            cls_scores['pose_coarse'] = logits_coarse_pose
            
            # 粗分类的FR loss
            coarse_fr_loss_rgb = self.fr_coarse_rgb(
                x_rgb1, gt_coarse.detach(), logits_coarse_rgb)
            coarse_fr_loss_pose = self.fr_coarse_pose(
                x_pose1, gt_coarse.detach(), logits_coarse_pose)
            
            # 细分类的FR loss（需要粗分类标签）
            fr_loss_rgb = self.fr_rgb(
                x_rgb1, gt_coarse.detach(), gt.detach(), 
                logits_coarse_rgb, logits_rgb)
            fr_loss_pose = self.fr_pose(
                x_pose1, gt_coarse.detach(), gt.detach(),
                logits_coarse_pose, logits_pose)
            
            # Tree loss
            hierarchy_loss_rgb = self.tree_loss_rgb(
                logits_coarse_rgb, logits_rgb, gt_coarse.detach(), gt.detach())
            hierarchy_loss_pose = self.tree_loss_pose(
                logits_coarse_pose, logits_pose, gt_coarse.detach(), gt.detach())
            
            cls_scores['fr_loss_rgb_coarse'] = coarse_fr_loss_rgb
            cls_scores['fr_loss_pose_coarse'] = coarse_fr_loss_pose
            cls_scores['hierarchy_loss_rgb'] = hierarchy_loss_rgb
            cls_scores['hierarchy_loss_pose'] = hierarchy_loss_pose
        else:
            # 无粗分类的情况（NTU-60）
            # 创建虚拟的粗分类标签（全部映射到类别0）
            gt_coarse = torch.zeros_like(gt)
            logits_coarse_rgb = torch.zeros(x_rgb.size(0), 1).to(x_rgb.device)
            logits_coarse_pose = torch.zeros(x_pose.size(0), 1).to(x_pose.device)
            
            # 简化的FR loss（只用细分类）
            fr_loss_rgb = self.fr_rgb.forward_simple(
                x_rgb1, gt.detach(), logits_rgb)  # 需要添加simple版本
            fr_loss_pose = self.fr_pose.forward_simple(
                x_pose1, gt.detach(), logits_pose)
        
        cls_scores['fr_loss_rgb'] = fr_loss_rgb
        cls_scores['fr_loss_pose'] = fr_loss_pose
    
    if not self.training:
        # 测试阶段的原型修正（无需粗分类）
        with torch.no_grad():
            rgb_proto = self.fr_rgb.spatio_cl_net.avg_f.permute(1, 0).cuda()
            pose_proto = self.fr_pose.spatio_cl_net.avg_f.permute(1, 0).cuda()
            
            logits_rgb_proto = self.fr_rgb.spatio_cl_net.cl_fc(x_rgb)
            logits_pose_proto = self.fr_pose.spatio_cl_net.cl_fc(x_pose)
            
            cos_sim_rgb = F.cosine_similarity(
                logits_rgb_proto.unsqueeze(1), rgb_proto.unsqueeze(0), dim=2)
            cos_sim_pose = F.cosine_similarity(
                logits_pose_proto.unsqueeze(1), pose_proto.unsqueeze(0), dim=2)
            
            cls_scores['rgb'] = logits_rgb + cos_sim_rgb * 5
            cls_scores['pose'] = logits_pose + cos_sim_pose * 1
    
    return cls_scores
```

### 第3步：修改RenovateNet支持无粗分类模式

在 `rgbpose_head.py` 的 `RenovateNet_Fine` 类中添加简化版forward：

```python
def forward_simple(self, feature, lbl, logit):
    """
    无粗分类的简化版本
    只使用细分类标签进行原型学习
    """
    # 特征投影
    feature = self.cl_fc(feature)
    feature = F.normalize(feature.permute(0, 2, 3, 1), dim=3)
    feature = feature.reshape(-1, self.h_channel)
    
    # 使用argmax作为虚拟的粗分类预测
    lbl_one_coarse = F.one_hot(torch.zeros_like(lbl), 1).to(torch.float32)
    pred_one_coarse = F.one_hot(torch.zeros_like(lbl), 1).to(torch.float32)
    
    # 细分类的one-hot
    lbl_one = F.one_hot(lbl.to(torch.int64), self.n_class).to(torch.float32)
    pred_one = F.one_hot(logit.argmax(1), self.n_class).to(torch.float32)
    
    # 计算TP/FN/FP（只基于细分类）
    mask, fn, fp = self.get_mask_fn_fp(
        lbl_one_coarse, lbl_one, 
        pred_one_coarse, pred_one,
        torch.zeros_like(logit), logit)
    
    # 其余计算保持不变
    f_mem, f_fn_1, f_fn_2, f_fn_3, f_fp = self.local_avg_tp_fn_fp(
        feature, mask, fn[0], fn[1], fn[2], fp)
    
    score_cl_fn, score_cl_fp = self.get_score(
        feature, f_mem, f_fn_1, f_fn_2, f_fn_3, f_fp,
        lbl_one, pred_one, lbl, use_p_map=self.use_p_map)
    
    return (self.loss(score_cl_fn, lbl) + self.loss(score_cl_fp, lbl)).mean()
```

### 第4步：修改MMRecognizer3D处理虚拟粗分类

在 `mmaction/models/recognizers/recognizer3d_mm.py` 中（约第56-65行）：

```python
def extract_feat(self, inputs: torch.Tensor, stage: str = 'neck',
                 data_samples: OptSampleList = None,
                 test_mode: bool = False, **kwargs):
    # ... existing code ...
    
    if self.with_neck:
        if stage == 'backbone':
            return x
        gts = [data.gt_labels.item for data in data_samples]
        
        # 检查是否有粗分类标签
        if hasattr(self.cls_head, 'use_coarse') and not self.cls_head.use_coarse:
            # NTU-60模式：使用虚拟粗分类
            gts_coarse = torch.zeros_like(torch.tensor(gts))
        else:
            # MA-52模式：从细分类映射到粗分类
            temp = torch.tensor(gts).to(torch.int32)
            gts_coarse = [fine2coarse(i) for i in temp]
        
        inputs['gt'] = gts
        inputs['gt_coarse'] = gts_coarse
        x = self.backbone(**inputs)
        # ... rest of code ...
```

---

## 🎯 方案C：自定义NTU-60的粗分类（最佳效果）

### 定义NTU-60的层次结构

根据NTU-60的动作语义，可以定义为8个粗类：

```python
# 在rgbpose_head.py开头添加
def action2body_ntu60(x):
    """NTU-60的层次映射"""
    if 0 <= x <= 11:
        return 0  # Daily Actions (drink, eat, brush teeth...)
    elif 12 <= x <= 23:
        return 1  # Interaction (handshake, hug, punch...)
    elif 24 <= x <= 35:
        return 2  # Health (sneeze, headache, chest pain...)
    elif 36 <= x <= 47:
        return 3  # Gesture (point, thumb up, ok sign...)
    elif 48 <= x <= 50:
        return 4  # Hand movements (typing, writing...)
    elif 51 <= x <= 55:
        return 5  # Body movements (walk, sit, stand...)
    elif 56 <= x <= 58:
        return 6  # Fall related
    else:
        return 7  # Other
```

然后修改配置：

```python
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,
    num_coarse_classes=8,  # NTU自定义8类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose', 'rgb_coarse', 'pose_coarse'],
    loss_weights=[1., 1., 0.5, 0.5],
    average_clips='prob',
    use_coarse=True,
    action2body_func='action2body_ntu60')  # 指定使用NTU的映射
```

---

## 📊 性能预期

### 基于现有结果推测

```
基础RGBPoseConv3D (NTU-60 XSub):
  • Top-1: 96.2-96.4% (README中报告)

加入PCAN后预期提升:
  • 原型学习: +1-2%
  • FN/FP校准: +0.5-1%
  • 预期Top-1: 97.5-98.0%
  
⚠️ 注意：
  • 没有TreeLoss（因为自定义粗分类不够精准）
  • 提升幅度可能不如MA-52（NTU-60类别较少，混淆度低）
```

---

## 🔄 完整的修改清单

### 必须修改的文件

| 文件 | 修改内容 | 难度 |
|------|---------|------|
| `rgbpose_conv3d.py` | in_channels: 28→25, num_classes: 52→60 | ⭐ 简单 |
| `rgbpose_head.py` | 添加use_coarse参数支持 | ⭐⭐ 中等 |
| `recognizer3d_mm.py` | 虚拟粗分类标签生成 | ⭐⭐ 中等 |
| `action2body函数` | 创建NTU版本（可选） | ⭐⭐⭐ 较难 |

### 数据准备

```bash
# 1. 准备NTU-60数据集
# 下载RGB视频和骨架标注
cd data
mkdir -p skeleton nturgbd_videos

# 2. 准备骨架数据（pkl格式）
# 参考：https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md

# 3. 骨架数据格式（25个关键点）
# NTU骨架点索引：0-24
# 需要转换为与MA-52相同的数据格式
```

---

## ⚠️ 主要挑战和解决方案

### 挑战1：骨架点数不同（28 vs 25）

**影响**：轻微（只需改配置）

```python
# MA-52: 28个点（OpenPose-28）
# NTU-60: 25个点（Kinect v2）

解决：
  pose_pathway=dict(
      in_channels=25,  # 修改这里
      # 其他参数保持不变
  )
```

### 挑战2：没有粗分类标签

**影响**：中等（需要修改代码逻辑）

**解决方案对比**：

| 方案 | 优点 | 缺点 | 预期效果 |
|------|------|------|---------|
| 去除PCAN | 简单 | 失去原型学习优势 | +0% |
| 虚拟粗分类 | 保留原型学习 | TreeLoss无效 | +1-1.5% |
| 自定义粗分类 | 保留所有PCAN特性 | 需要手动标注 | +1.5-2.5% |

### 挑战3：TreeLoss依赖固定的层次结构

**影响**：重大（硬编码了52类到7类的映射）

**解决方案**：

```python
# 方案1：去除TreeLoss（简单）
# 在forward中跳过hierarchy_loss的计算

# 方案2：改写TreeLoss以支持自定义映射（推荐）
class TreeLoss(nn.Module):
    def __init__(self, num_actions=52, num_bodies=7, 
                 action2body_fn=None):
        super().__init__()
        self.num_actions = num_actions
        self.num_bodies = num_bodies
        self.action2body_fn = action2body_fn or action2body
        self.stateSpace = self.generateStateSpace().cuda()
    
    def generateStateSpace(self):
        stat_list = np.eye(self.num_bodies + self.num_actions)
        for i in range(self.num_bodies, 
                      self.num_bodies + self.num_actions):
            action_id = i - self.num_bodies
            coarse_id = self.action2body_fn(action_id)
            stat_list[i][coarse_id] = 1
        return torch.tensor(stat_list)
```

---

## 🚀 快速开始（最小修改方案）

如果你想快速测试，用这个**最简配置**：

### 创建 `ntu60_simple.py`

```python
_base_ = './rgbpose_conv3d.py'

# 只修改关键参数
backbone_cfg = dict(
    type='RGBPoseConv3D',
    pose_pathway=dict(in_channels=25))  # 25个骨架点

head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # 60个类别
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],  # 只用RGB和Pose，去掉coarse
    loss_weights=[1., 1.],
    average_clips='prob')

model = dict(
    backbone=backbone_cfg,
    cls_head=head_cfg)

# 数据集配置
dataset_type = 'PoseDataset'
data_root = 'data/nturgbd_videos/'
ann_file_train = 'data/skeleton/ntu60_3d.pkl'
ann_file_val = 'data/skeleton/ntu60_3d.pkl'

# 左右骨架点（NTU-60）
left_kp = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

# ... pipeline配置保持类似 ...
```

**这个方案的限制**：
- ❌ 无法使用粗分类的FR loss
- ❌ 无法使用TreeLoss
- ✅ 保留细分类的原型学习
- ✅ 保留测试时的原型修正
- ✅ 保留FeatureInteraction

**预期效果**：约**+0.5-1%**的提升（相比基础RGBPoseConv3D）

---

## 📈 完整方案的工作量评估

| 任务 | 预计工作量 | 优先级 |
|------|----------|--------|
| 修改骨架点数 | 5分钟 | 🔴 必须 |
| 修改类别数 | 5分钟 | 🔴 必须 |
| 去除粗分类依赖 | 1-2小时 | 🟡 推荐 |
| 自定义NTU粗分类 | 4-8小时 | 🟢 可选 |
| 改写TreeLoss | 2-3小时 | 🟢 可选 |
| 重新训练init权重 | 8-12小时×2 | 🟡 推荐 |

### 总工作量

- **最小可行方案**：2-3小时（修改配置+简单代码调整）
- **完整PCAN方案**：2-3天（包括自定义分层+重新训练）

---

## 💡 推荐实施路线

### 阶段1：快速验证（1天）

```bash
# 1. 修改配置文件
# 2. 去除粗分类依赖（代码层面简化处理）
# 3. 直接测试基础性能

预期：能跑通，性能约+0.5-1%
```

### 阶段2：优化提升（2-3天）

```bash
# 1. 设计NTU-60的粗分类体系
# 2. 实现完整的PCAN功能
# 3. 重新训练所有权重

预期：最优性能，+1.5-2.5%
```

### 阶段3：对比实验（1天）

```bash
# 1. 对比去除PCAN vs 简化PCAN vs 完整PCAN
# 2. 消融实验
# 3. 发论文 :)
```

---

## 🎯 总结

### 可以在NTU-60上运行吗？

**✅ 可以！** 但需要修改。

### 整体变化大吗？

**⭐⭐⭐☆☆ 中等变化**

- **架构层面**：无需改变（双流+原型学习核心思想保留）
- **实现层面**：需要适度修改（去除或虚拟化粗分类）
- **性能预期**：仍然会有提升（+0.5-2.5%）

### 最快的方案

**使用简化PCAN**：
- 保留原型学习和测试修正
- 去除粗分类和TreeLoss
- 只改配置文件和少量代码
- 2-3小时完成

### 最优的方案

**完整适配PCAN**：
- 自定义NTU-60的粗分类
- 保留所有PCAN特性
- 重新训练权重
- 2-3天完成

需要我详细展开某个方案的具体代码吗？或者有其他疑问？

```plaintext
整体变化：⭐⭐⭐☆☆ (中等)

必须修改的部分：
  🔴 骨架点数：in_channels (28→25)
  🔴 类别数：num_classes (52→60)
  🔴 粗分类处理：移除或虚拟化
  🔴 TreeLoss：去掉或修改

可选修改的部分：
  🟡 原型数量：自动适配60类
  🟢 Backbone：无需修改
  🟢 双流架构：保持不变
```

```python
_base_ = '../../../_base_/default_runtime.py'

# ========== 修改1: Backbone配置 ==========
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        fusion_kernel=7,
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1),
        with_pool2=False),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        fusion_kernel=7,
        in_channels=25,  # ← 改为25 (NTU骨架点数)
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride_s=1,
        conv1_stride_t=1,
        pool1_stride_s=1,
        pool1_stride_t=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        with_pool2=False))

# ========== 修改2: Head配置 ==========
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← 改为60类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],
    loss_weights=[1., 1.],
    average_clips='prob')

data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    preprocessors=dict(
        imgs=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        heatmap_imgs=dict(type='ActionDataPreprocessor')))

model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    data_preprocessor=data_preprocessor)

# ========== 修改3: 数据配置 ==========
dataset_type = 'PoseDataset'
data_root = 'data/nturgbd_videos/'
ann_file = 'data/skeleton/ntu60_3d.pkl'  # 或 ntu60_2d.pkl

# NTU-60的骨架点定义（25个关节点）
left_kp = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

# ... 其他pipeline和dataloader配置类似 ...

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=3, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[10, 20],
        gamma=0.1)
]
```

```python
# ========== 修改1: __init__方法（约第520行）==========
def __init__(self,
             num_classes: int,
             in_channels: Tuple[int],
             loss_cls: Dict = dict(type='CrossEntropyLoss'),
             loss_components: List[str] = ['rgb', 'pose'],
             loss_weights: Union[float, Tuple[float]] = 1.,
             dropout: float = 0.5,
             init_std: float = 0.01,
             use_coarse: bool = True,  # ← 新增：是否使用粗分类
             num_coarse_classes: int = 7,  # ← 新增：粗分类数量
             **kwargs) -> None:
    
    super().__init__(num_classes, in_channels, loss_cls, **kwargs)
    
    # ... 原有代码 ...
    
    self.fc_rgb = nn.Linear(self.in_channels[0], num_classes)
    self.fc_pose = nn.Linear(self.in_channels[1], num_classes)
    
    # 根据use_coarse决定是否创建粗分类相关模块
    self.use_coarse = use_coarse
    if self.use_coarse:
        self.fc_rgb_coarse = nn.Linear(self.in_channels[0], num_coarse_classes)
        self.fc_pose_coarse = nn.Linear(self.in_channels[1], num_coarse_classes)
        
        self.fr_coarse_rgb = ST_RenovateNet(
            2048, 8, n_class=num_coarse_classes, h_channel=128)
        self.fr_coarse_pose = ST_RenovateNet(
            512, 32, n_class=num_coarse_classes, h_channel=128)
        
        self.tree_loss_rgb = TreeLoss()
        self.tree_loss_pose = TreeLoss()
    
    # 细粒度RenovateNet（永远需要）
    self.fr_rgb = ST_RenovateNet_Fine(
        2048, n_class=num_classes, version='V0', use_p_map=True)
    self.fr_pose = ST_RenovateNet_Fine(
        512, n_class=num_classes, version='V0', use_p_map=True)


# ========== 修改2: forward方法（约第576行）==========
def forward(self, x: List[torch.Tensor]) -> Dict:
    x_rgb, x_pose = self.avg_pool(x[0]), self.avg_pool(x[1])
    x_rgb = x_rgb.view(x_rgb.size(0), -1)
    x_pose = x_pose.view(x_pose.size(0), -1)
    
    x_rgb = self.dropout_rgb(x_rgb)
    x_pose = self.dropout_pose(x_pose)
    
    cls_scores = dict()
    logits_rgb = self.fc_rgb(x_rgb)
    logits_pose = self.fc_pose(x_pose)
    
    if self.training:
        cls_scores['rgb'] = logits_rgb
        cls_scores['pose'] = logits_pose
        
        x_rgb1, x_pose1 = x[2], x[3]
        gt = x[4]
        
        x_rgb1 = x_rgb1.mean(dim=2)
        x_pose1 = x_pose1.mean(dim=2)
        
        if self.use_coarse:
            # 有粗分类的情况（MA-52）
            gt_coarse = x[5]
            logits_coarse_rgb = self.fc_rgb_coarse(x_rgb)
            logits_coarse_pose = self.fc_pose_coarse(x_pose)
            
            cls_scores['rgb_coarse'] = logits_coarse_rgb
            cls_scores['pose_coarse'] = logits_coarse_pose
            
            # 粗分类的FR loss
            coarse_fr_loss_rgb = self.fr_coarse_rgb(
                x_rgb1, gt_coarse.detach(), logits_coarse_rgb)
            coarse_fr_loss_pose = self.fr_coarse_pose(
                x_pose1, gt_coarse.detach(), logits_coarse_pose)
            
            # 细分类的FR loss（需要粗分类标签）
            fr_loss_rgb = self.fr_rgb(
                x_rgb1, gt_coarse.detach(), gt.detach(), 
                logits_coarse_rgb, logits_rgb)
            fr_loss_pose = self.fr_pose(
                x_pose1, gt_coarse.detach(), gt.detach(),
                logits_coarse_pose, logits_pose)
            
            # Tree loss
            hierarchy_loss_rgb = self.tree_loss_rgb(
                logits_coarse_rgb, logits_rgb, gt_coarse.detach(), gt.detach())
            hierarchy_loss_pose = self.tree_loss_pose(
                logits_coarse_pose, logits_pose, gt_coarse.detach(), gt.detach())
            
            cls_scores['fr_loss_rgb_coarse'] = coarse_fr_loss_rgb
            cls_scores['fr_loss_pose_coarse'] = coarse_fr_loss_pose
            cls_scores['hierarchy_loss_rgb'] = hierarchy_loss_rgb
            cls_scores['hierarchy_loss_pose'] = hierarchy_loss_pose
        else:
            # 无粗分类的情况（NTU-60）
            # 创建虚拟的粗分类标签（全部映射到类别0）
            gt_coarse = torch.zeros_like(gt)
            logits_coarse_rgb = torch.zeros(x_rgb.size(0), 1).to(x_rgb.device)
            logits_coarse_pose = torch.zeros(x_pose.size(0), 1).to(x_pose.device)
            
            # 简化的FR loss（只用细分类）
            fr_loss_rgb = self.fr_rgb.forward_simple(
                x_rgb1, gt.detach(), logits_rgb)  # 需要添加simple版本
            fr_loss_pose = self.fr_pose.forward_simple(
                x_pose1, gt.detach(), logits_pose)
        
        cls_scores['fr_loss_rgb'] = fr_loss_rgb
        cls_scores['fr_loss_pose'] = fr_loss_pose
    
    if not self.training:
        # 测试阶段的原型修正（无需粗分类）
        with torch.no_grad():
            rgb_proto = self.fr_rgb.spatio_cl_net.avg_f.permute(1, 0).cuda()
            pose_proto = self.fr_pose.spatio_cl_net.avg_f.permute(1, 0).cuda()
            
            logits_rgb_proto = self.fr_rgb.spatio_cl_net.cl_fc(x_rgb)
            logits_pose_proto = self.fr_pose.spatio_cl_net.cl_fc(x_pose)
            
            cos_sim_rgb = F.cosine_similarity(
                logits_rgb_proto.unsqueeze(1), rgb_proto.unsqueeze(0), dim=2)
            cos_sim_pose = F.cosine_similarity(
                logits_pose_proto.unsqueeze(1), pose_proto.unsqueeze(0), dim=2)
            
            cls_scores['rgb'] = logits_rgb + cos_sim_rgb * 5
            cls_scores['pose'] = logits_pose + cos_sim_pose * 1
    
    return cls_scores
```

```python
def forward_simple(self, feature, lbl, logit):
    """
    无粗分类的简化版本
    只使用细分类标签进行原型学习
    """
    # 特征投影
    feature = self.cl_fc(feature)
    feature = F.normalize(feature.permute(0, 2, 3, 1), dim=3)
    feature = feature.reshape(-1, self.h_channel)
    
    # 使用argmax作为虚拟的粗分类预测
    lbl_one_coarse = F.one_hot(torch.zeros_like(lbl), 1).to(torch.float32)
    pred_one_coarse = F.one_hot(torch.zeros_like(lbl), 1).to(torch.float32)
    
    # 细分类的one-hot
    lbl_one = F.one_hot(lbl.to(torch.int64), self.n_class).to(torch.float32)
    pred_one = F.one_hot(logit.argmax(1), self.n_class).to(torch.float32)
    
    # 计算TP/FN/FP（只基于细分类）
    mask, fn, fp = self.get_mask_fn_fp(
        lbl_one_coarse, lbl_one, 
        pred_one_coarse, pred_one,
        torch.zeros_like(logit), logit)
    
    # 其余计算保持不变
    f_mem, f_fn_1, f_fn_2, f_fn_3, f_fp = self.local_avg_tp_fn_fp(
        feature, mask, fn[0], fn[1], fn[2], fp)
    
    score_cl_fn, score_cl_fp = self.get_score(
        feature, f_mem, f_fn_1, f_fn_2, f_fn_3, f_fp,
        lbl_one, pred_one, lbl, use_p_map=self.use_p_map)
    
    return (self.loss(score_cl_fn, lbl) + self.loss(score_cl_fp, lbl)).mean()
```

```python
def extract_feat(self, inputs: torch.Tensor, stage: str = 'neck',
                 data_samples: OptSampleList = None,
                 test_mode: bool = False, **kwargs):
    # ... existing code ...
    
    if self.with_neck:
        if stage == 'backbone':
            return x
        gts = [data.gt_labels.item for data in data_samples]
        
        # 检查是否有粗分类标签
        if hasattr(self.cls_head, 'use_coarse') and not self.cls_head.use_coarse:
            # NTU-60模式：使用虚拟粗分类
            gts_coarse = torch.zeros_like(torch.tensor(gts))
        else:
            # MA-52模式：从细分类映射到粗分类
            temp = torch.tensor(gts).to(torch.int32)
            gts_coarse = [fine2coarse(i) for i in temp]
        
        inputs['gt'] = gts
        inputs['gt_coarse'] = gts_coarse
        x = self.backbone(**inputs)
        # ... rest of code ...
```

```python
# 在rgbpose_head.py开头添加
def action2body_ntu60(x):
    """NTU-60的层次映射"""
    if 0 <= x <= 11:
        return 0  # Daily Actions (drink, eat, brush teeth...)
    elif 12 <= x <= 23:
        return 1  # Interaction (handshake, hug, punch...)
    elif 24 <= x <= 35:
        return 2  # Health (sneeze, headache, chest pain...)
    elif 36 <= x <= 47:
        return 3  # Gesture (point, thumb up, ok sign...)
    elif 48 <= x <= 50:
        return 4  # Hand movements (typing, writing...)
    elif 51 <= x <= 55:
        return 5  # Body movements (walk, sit, stand...)
    elif 56 <= x <= 58:
        return 6  # Fall related
    else:
        return 7  # Other
```

```python
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,
    num_coarse_classes=8,  # NTU自定义8类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose', 'rgb_coarse', 'pose_coarse'],
    loss_weights=[1., 1., 0.5, 0.5],
    average_clips='prob',
    use_coarse=True,
    action2body_func='action2body_ntu60')  # 指定使用NTU的映射
```

```plaintext
基础RGBPoseConv3D (NTU-60 XSub):
  • Top-1: 96.2-96.4% (README中报告)

加入PCAN后预期提升:
  • 原型学习: +1-2%
  • FN/FP校准: +0.5-1%
  • 预期Top-1: 97.5-98.0%
  
⚠️ 注意：
  • 没有TreeLoss（因为自定义粗分类不够精准）
  • 提升幅度可能不如MA-52（NTU-60类别较少，混淆度低）
```

```shellscript
# 1. 准备NTU-60数据集
# 下载RGB视频和骨架标注
cd data
mkdir -p skeleton nturgbd_videos

# 2. 准备骨架数据（pkl格式）
# 参考：https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md

# 3. 骨架数据格式（25个关键点）
# NTU骨架点索引：0-24
# 需要转换为与MA-52相同的数据格式
```

```python
# MA-52: 28个点（OpenPose-28）
# NTU-60: 25个点（Kinect v2）

解决：
  pose_pathway=dict(
      in_channels=25,  # 修改这里
      # 其他参数保持不变
  )
```

```python
# 方案1：去除TreeLoss（简单）
# 在forward中跳过hierarchy_loss的计算

# 方案2：改写TreeLoss以支持自定义映射（推荐）
class TreeLoss(nn.Module):
    def __init__(self, num_actions=52, num_bodies=7, 
                 action2body_fn=None):
        super().__init__()
        self.num_actions = num_actions
        self.num_bodies = num_bodies
        self.action2body_fn = action2body_fn or action2body
        self.stateSpace = self.generateStateSpace().cuda()
    
    def generateStateSpace(self):
        stat_list = np.eye(self.num_bodies + self.num_actions)
        for i in range(self.num_bodies, 
                      self.num_bodies + self.num_actions):
            action_id = i - self.num_bodies
            coarse_id = self.action2body_fn(action_id)
            stat_list[i][coarse_id] = 1
        return torch.tensor(stat_list)
```

```python
_base_ = './rgbpose_conv3d.py'

# 只修改关键参数
backbone_cfg = dict(
    type='RGBPoseConv3D',
    pose_pathway=dict(in_channels=25))  # 25个骨架点

head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # 60个类别
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],  # 只用RGB和Pose，去掉coarse
    loss_weights=[1., 1.],
    average_clips='prob')

model = dict(
    backbone=backbone_cfg,
    cls_head=head_cfg)

# 数据集配置
dataset_type = 'PoseDataset'
data_root = 'data/nturgbd_videos/'
ann_file_train = 'data/skeleton/ntu60_3d.pkl'
ann_file_val = 'data/skeleton/ntu60_3d.pkl'

# 左右骨架点（NTU-60）
left_kp = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

# ... pipeline配置保持类似 ...
```

```shellscript
# 1. 修改配置文件
# 2. 去除粗分类依赖（代码层面简化处理）
# 3. 直接测试基础性能

预期：能跑通，性能约+0.5-1%
```

```shellscript
# 1. 设计NTU-60的粗分类体系
# 2. 实现完整的PCAN功能
# 3. 重新训练所有权重

预期：最优性能，+1.5-2.5%
```

```shellscript
# 1. 对比去除PCAN vs 简化PCAN vs 完整PCAN
# 2. 消融实验
# 3. 发论文 :)
```

---

**User**

你帮我查询一下项目看看是否有处理ntud60数据集的脚本代码 我似乎在项目F:\lunwen\PCAN-main\mmaction2\configs\skeleton 这个文件夹下面看到了相关的代码，但我不知道这个代码能不能适用@configs 

---

**Cursor**

让我帮你详细查看项目中关于NTU-60数据集的配置和脚本。

## ✅ 好消息：项目已经支持NTU-60！

你的发现是对的！让我详细分析你找到的这些代码。

## 📁 发现的NTU-60相关代码

### 1. **基础配置文件**（在`configs/skeleton/posec3d`下）

```
✓ slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
✓ slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb.py
```

这些是**基础的PoseC3D**（只用骨架，不用RGB）在NTU-60上的配置。

### 2. **RGBPose配置**（在`configs/skeleton/posec3d/rgbpose_conv3d`下）

```
✓ rgb_only.py      - 只用RGB训练NTU-60
✓ pose_only.py     - 只用Pose训练NTU-60
✓ rgbpose_conv3d.py - 双流融合（但当前配置是MA-52）
```

### 3. **数据处理脚本**（在`tools/data/skeleton`下）

```
✓ compress_nturgbd.py      - 压缩NTU-60视频
✓ gen_ntu_rgbd_raw.py      - 转换骨架数据格式（3D，25个点）
✓ ntu_pose_extraction.py   - 提取2D姿态（17个点）
```

---

## 🔍 关键发现：配置文件的差异

### NTU-60配置 vs PCAN(MA-52)配置

| 特征 | NTU-60配置 | PCAN(MA-52)配置 | 能否直接用？ |
|------|-----------|----------------|------------|
| **骨架点数** | **17点**(2D) 或 **25点**(3D) | **28点**(OpenPose) | ❌ 需要修改 |
| **类别数** | **60类** | **52类** | ❌ 需要修改 |
| **粗分类** | ❌ 无 | ✓ 7类 | ❌ 冲突！ |
| **Head类型** | `I3DHead` | `RGBPoseHead`(带PCAN) | ❌ 不兼容 |
| **骨架格式** | Kinect v2 | OpenPose-28 | ❌ 不同 |

---

## 🎯 现有NTU-60配置能否用于PCAN？

### 答案：**不能直接使用！需要适配！**

### 原因分析

#### 1. **pose_only.py和rgb_only.py** 

```python
# configs/skeleton/posec3d/rgbpose_conv3d/pose_only.py
model = dict(
    type='Recognizer3D',  # ← 基础模型，不是MMRecognizer3D
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels=17,  # ← NTU-60的2D骨架
        num_classes=60),
    cls_head=dict(
        type='I3DHead',  # ← 基础Head，不是RGBPoseHead
        num_classes=60))
```

**问题**：
- ✅ 支持NTU-60（60类，17或25个点）
- ❌ **没有PCAN功能**（没有原型学习、没有RenovateNet）
- ❌ 不是双流架构（单独的RGB或Pose）

#### 2. **当前的rgbpose_conv3d.py**

```python
# configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py
backbone_cfg = dict(
    type='RGBPoseConv3D',
    pose_pathway=dict(
        in_channels=28,  # ← MA-52的OpenPose-28
        ...))

head_cfg = dict(
    type='RGBPoseHead',  # ← 包含PCAN功能
    num_classes=52,  # ← MA-52的52类
    ...)
```

**问题**：
- ✅ 有完整的PCAN功能
- ❌ 硬编码为MA-52（28点，52类）
- ❌ 依赖粗分类（7类body-level）

---

## 💡 适配方案总结

### 方案对比

| 方案 | 使用现有配置 | 需要修改 | PCAN功能 | 预期效果 |
|------|------------|---------|---------|---------|
| **A. 使用pose_only.py** | ✅ 直接用 | ❌ 无 | ❌ 无PCAN | 93-95% (基线) |
| **B. 修改rgbpose_conv3d.py** | ❌ 不能用 | ✅ 中等 | ⚠️ 简化PCAN | 96-97% |
| **C. 完全适配PCAN** | ❌ 不能用 | ✅ 较多 | ✅ 完整PCAN | 97-98% |

---

## 🚀 实用建议：分步实施

### 阶段1：验证基础流程（用现有配置）✅

**目的**：先确保NTU-60数据能跑通

```bash
# 使用现有的NTU-60配置（不带PCAN）
cd /home/zh/ChCode/codes01/mmaction2

# 测试Pose-only模型
python tools/test.py \
    configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    <checkpoint.pth>
```

**这个配置的特点**：
- ✓ 完全支持NTU-60（17个点，60类）
- ✓ 不需要修改代码
- ✓ 有预训练权重可下载
- ❌ 没有PCAN功能
- ❌ 只用Pose，不用RGB

### 阶段2：适配PCAN到NTU-60（推荐方案）⭐

创建新配置 `configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d_ntu60.py`：

```python
_base_ = './rgbpose_conv3d.py'

# ========== 修改1: 骨架点数 ==========
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        # RGB配置保持不变
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        fusion_kernel=7,
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1),
        with_pool2=False),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        fusion_kernel=7,
        in_channels=17,  # ← 改：NTU-60 2D骨架(17点)
        # 或 in_channels=25 for 3D骨架
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride_s=1,
        conv1_stride_t=1,
        pool1_stride_s=1,
        pool1_stride_t=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        with_pool2=False))

# ========== 修改2: 类别数 ==========
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← 改：NTU-60的60类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],  # ← 改：去掉coarse
    loss_weights=[1., 1.],
    average_clips='prob')

# ========== 修改3: 数据路径 ==========
dataset_type = 'PoseDataset'
data_root = 'data/nturgbd_videos/'
ann_file = 'data/skeleton/ntu60_2d.pkl'  # 使用2D骨架
ann_file_val = 'data/skeleton/ntu60_2d.pkl'
ann_file_test = 'data/skeleton/ntu60_2d.pkl'

# ========== 修改4: 骨架点定义（17点的左右定义）==========
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# 如果用25点（3D骨架），需要改为：
# left_kp = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
# right_kp = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

# Pipeline保持类似，但去掉skeletons定义（因为没有limb heatmap）
train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

# ... 其他dataloader配置类似MA-52 ...

# ========== 修改5: 训练配置 ==========
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=3, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# ========== 重要：去掉load_from或指向NTU的init权重 ==========
load_from = None  # 或指向NTU-60的初始化权重
```

### 阶段3：修改代码以去除粗分类依赖

由于PCAN的代码**硬编码了粗分类**（在`recognizer3d_mm.py`和`rgbpose_head.py`中），需要修改：

#### 修改1：`mmaction/models/recognizers/recognizer3d_mm.py`

```python
def fine2coarse_ntu60(x):
    """
    为NTU-60创建虚拟的粗分类
    方案1：全部映射到类别0（最简单）
    """
    return 0

# 或者根据NTU-60的语义定义8个粗类
def fine2coarse_ntu60_semantic(x):
    """
    基于NTU-60动作语义的粗分类
    """
    if 0 <= x <= 7:
        return 0  # Hand-arm movements
    elif 8 <= x <= 15:
        return 1  # Two-hand movements
    elif 16 <= x <= 25:
        return 2  # Interaction
    elif 26 <= x <= 31:
        return 3  # Body movements
    elif 32 <= x <= 39:
        return 4  # Pick/throw
    elif 40 <= x <= 49:
        return 5  # Health-related
    elif 50 <= x <= 56:
        return 6  # Common actions
    else:
        return 7  # Other

@MODELS.register_module()
class MMRecognizer3D(BaseRecognizer):
    def extract_feat(self, inputs, ...):
        # ... existing code ...
        
        gts = []
        for data in data_samples:
            gts.extend(data.gt_labels.item)
        gts = torch.stack(gts)
        temp = gts.cpu().numpy()
        
        # ← 修改这里：根据数据集选择映射函数
        if self.cls_head.num_classes == 60:
            # NTU-60模式
            gts_coarse = [fine2coarse_ntu60(i) for i in temp]
        else:
            # MA-52模式
            gts_coarse = [fine2coarse(i) for i in temp]
        
        gts_coarse = torch.from_numpy(np.array(gts_coarse)).cuda()
        # ... rest of code ...
```

#### 修改2：`mmaction/models/heads/rgbpose_head.py`

需要让`RenovateNet_Fine`支持不同的粗分类数量：

```python
class RenovateNet_Fine(nn.Module):
    def __init__(self, n_channel, n_class, alp=0.125, tmp=0.125, mom=0.9, 
                 h_channel=None, version='V0', pred_threshold=0.0, 
                 use_p_map=True, 
                 n_class_coarse=7):  # ← 新增参数
        super(RenovateNet_Fine, self).__init__()
        self.n_channel = n_channel
        self.h_channel = n_channel if h_channel is None else h_channel
        self.n_class = n_class
        self.n_class_coarse = n_class_coarse  # ← 使用传入的值
        # ... rest of code ...

# 在RGBPoseHead的__init__中：
class RGBPoseHead(BaseHead):
    def __init__(self, num_classes, in_channels, 
                 num_coarse_classes=7,  # ← 新增参数
                 ...):
        # ...
        self.fc_rgb_coarse = nn.Linear(self.in_channels[0], num_coarse_classes)
        self.fc_pose_coarse = nn.Linear(self.in_channels[1], num_coarse_classes)
        
        self.fr_coarse_rgb = ST_RenovateNet(
            2048, 8, n_class=num_coarse_classes, h_channel=128)
        self.fr_coarse_pose = ST_RenovateNet(
            512, 32, n_class=num_coarse_classes, h_channel=128)
        
        self.fr_rgb = ST_RenovateNet_Fine(
            2048, n_class=num_classes, 
            n_class_coarse=num_coarse_classes)  # ← 传递粗分类数
        # ... rest
```

#### 修改3：`TreeLoss`需要适配

```python
class TreeLoss(nn.Module):
    def __init__(self, num_actions=52, num_bodies=7, 
                 action2body_fn=None):  # ← 新增参数
        super(TreeLoss, self).__init__()
        self.num_actions = num_actions
        self.num_bodies = num_bodies
        self.action2body_fn = action2body_fn or action2body
        self.stateSpace = self.generateStateSpace().cuda()
        self.sig = nn.Sigmoid()
    
    def generateStateSpace(self):
        stat_list = np.eye(self.num_bodies + self.num_actions)
        for i in range(self.num_bodies, self.num_bodies + self.num_actions):
            action_id = i - self.num_bodies
            coarse_id = self.action2body_fn(action_id)
            stat_list[i][coarse_id] = 1
        return torch.tensor(stat_list)
```

---

## 📋 完整的适配checklist

### 必须修改的项

- [ ] **配置文件**：创建`rgbpose_conv3d_ntu60.py`
  - [ ] `pose_pathway.in_channels`: 28 → 17 (或25)
  - [ ] `num_classes`: 52 → 60
  - [ ] `left_kp/right_kp`: 更新为NTU的定义
  - [ ] `data_root`: 指向NTU-60路径
  - [ ] `ann_file`: 指向ntu60_2d.pkl或ntu60_3d.pkl
  - [ ] 去掉`skeletons`定义（MA-52特有）

- [ ] **recognizer3d_mm.py**（第10-24行）
  - [ ] 添加`fine2coarse_ntu60`函数
  - [ ] 修改`extract_feat`根据类别数选择映射

- [ ] **rgbpose_head.py**（第75行和第556-564行）
  - [ ] `RenovateNet_Fine.__init__`: 添加`n_class_coarse`参数
  - [ ] `RGBPoseHead.__init__`: 添加`num_coarse_classes`参数
  - [ ] 传递粗分类数到各个RenovateNet

- [ ] **rgbpose_head.py**（第35-64行）
  - [ ] `TreeLoss.__init__`: 支持动态的action2body映射
  - [ ] `generateStateSpace`: 使用传入的映射函数

### 可选优化项

- [ ] 定义NTU-60的语义粗分类（8类）
- [ ] 重新训练RGB-only和Pose-only的预训练权重
- [ ] 合并预训练权重为NTU版的init.pth

---

## 🎯 快速测试方案（最小改动）

如果你只想快速验证，可以用这个**hack方案**（不推荐生产使用）：

### 创建临时配置

```python
# configs/skeleton/posec3d/rgbpose_conv3d/quick_ntu60.py
_base_ = './rgbpose_conv3d.py'

# 核心修改
backbone_cfg = dict(
    pose_pathway=dict(in_channels=17))  # 只改骨架点数

head_cfg = dict(
    num_classes=60)  # 只改类别数

model = dict(
    backbone=backbone_cfg,
    cls_head=head_cfg)

# 数据路径
data_root = 'data/nturgbd_videos/'
ann_file = 'data/skeleton/ntu60_2d.pkl'
ann_file_val = 'data/skeleton/ntu60_2d.pkl'
ann_file_test = 'data/skeleton/ntu60_2d.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# 去掉MA-52特有的skeleton定义
train_pipeline = [
    dict(type='MMUniformSampleFrames', 
         clip_len=dict(RGB=8, Pose=32), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True,
         with_kp=True, with_limb=False, scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

# 其他pipeline类似...
```

**运行时修改代码**（临时hack，仅用于测试）：

```python
# 在recognizer3d_mm.py的fine2coarse函数中：
def fine2coarse(x):
    # Hack for NTU-60: 全部映射到0
    if x < 60:  # NTU-60
        return 0
    # 原MA-52逻辑
    elif x <= 4:
        return 0
    # ... rest
```

**这个方案的问题**：
- ⚠️ TreeLoss会报错（因为尺寸不匹配67→61）
- ⚠️ 粗分类失去意义（都是类别0）
- ⚠️ 但原型学习仍然有效（细分类层面）

---

## 📊 最终建议

### 根据你的需求选择：

#### 1. **只是想快速测试NTU-60**

→ 使用现有的`slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py`

**优点**：
- ✅ 开箱即用
- ✅ 有预训练权重

**缺点**：
- ❌ 没有PCAN功能
- ❌ 没有RGB信息

#### 2. **想验证PCAN在NTU-60上的效果**

→ 按照"阶段2"的方案，创建适配的配置+修改代码

**工作量**：2-3小时

**预期提升**：+0.5-1%（96.8% → 97.5-98%）

#### 3. **想发论文，做完整实验**

→ 完全适配PCAN（自定义NTU粗分类+完整修改）

**工作量**：2-3天

**预期提升**：+1-2%（96.8% → 98%+）

---

## 🔑 关键结论

### ✅ 现有配置能否用于PCAN？

**不能直接使用**，因为：
1. `pose_only.py`和`rgb_only.py`是基础模型，没有PCAN功能
2. 当前的`rgbpose_conv3d.py`硬编码为MA-52
3. PCAN依赖粗分类（NTU-60没有）

### ✅ 需要做哪些修改？

**核心修改（必须）**：
- 骨架点数：28→17/25
- 类别数：52→60
- 粗分类：创建虚拟映射

**代码修改（中等工作量）**：
- `recognizer3d_mm.py`：fine2coarse函数
- `rgbpose_head.py`：支持动态粗分类数
- 配置文件：创建NTU-60版本

### ✅ 整体变化大吗？

**⭐⭐⭐☆☆ 中等变化**

- **架构不变**：双流+原型学习的核心保留
- **配置需改**：骨架点、类别数、数据路径
- **代码需改**：去除MA-52特有的硬编码
- **预期工作量**：2-3小时（简化版）到2-3天（完整版）

---

需要我详细展开某个修改步骤的代码吗？或者帮你生成完整的配置文件？

```plaintext
✓ slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
✓ slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb.py
```

```plaintext
✓ rgb_only.py      - 只用RGB训练NTU-60
✓ pose_only.py     - 只用Pose训练NTU-60
✓ rgbpose_conv3d.py - 双流融合（但当前配置是MA-52）
```

```plaintext
✓ compress_nturgbd.py      - 压缩NTU-60视频
✓ gen_ntu_rgbd_raw.py      - 转换骨架数据格式（3D，25个点）
✓ ntu_pose_extraction.py   - 提取2D姿态（17个点）
```

```python
# configs/skeleton/posec3d/rgbpose_conv3d/pose_only.py
model = dict(
    type='Recognizer3D',  # ← 基础模型，不是MMRecognizer3D
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels=17,  # ← NTU-60的2D骨架
        num_classes=60),
    cls_head=dict(
        type='I3DHead',  # ← 基础Head，不是RGBPoseHead
        num_classes=60))
```

```python
# configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py
backbone_cfg = dict(
    type='RGBPoseConv3D',
    pose_pathway=dict(
        in_channels=28,  # ← MA-52的OpenPose-28
        ...))

head_cfg = dict(
    type='RGBPoseHead',  # ← 包含PCAN功能
    num_classes=52,  # ← MA-52的52类
    ...)
```

```shellscript
# 使用现有的NTU-60配置（不带PCAN）
cd /home/zh/ChCode/codes01/mmaction2

# 测试Pose-only模型
python tools/test.py \
    configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    <checkpoint.pth>
```

```python
_base_ = './rgbpose_conv3d.py'

# ========== 修改1: 骨架点数 ==========
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        # RGB配置保持不变
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        fusion_kernel=7,
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1),
        with_pool2=False),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        fusion_kernel=7,
        in_channels=17,  # ← 改：NTU-60 2D骨架(17点)
        # 或 in_channels=25 for 3D骨架
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride_s=1,
        conv1_stride_t=1,
        pool1_stride_s=1,
        pool1_stride_t=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        with_pool2=False))

# ========== 修改2: 类别数 ==========
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← 改：NTU-60的60类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],  # ← 改：去掉coarse
    loss_weights=[1., 1.],
    average_clips='prob')

# ========== 修改3: 数据路径 ==========
dataset_type = 'PoseDataset'
data_root = 'data/nturgbd_videos/'
ann_file = 'data/skeleton/ntu60_2d.pkl'  # 使用2D骨架
ann_file_val = 'data/skeleton/ntu60_2d.pkl'
ann_file_test = 'data/skeleton/ntu60_2d.pkl'

# ========== 修改4: 骨架点定义（17点的左右定义）==========
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# 如果用25点（3D骨架），需要改为：
# left_kp = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
# right_kp = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

# Pipeline保持类似，但去掉skeletons定义（因为没有limb heatmap）
train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

# ... 其他dataloader配置类似MA-52 ...

# ========== 修改5: 训练配置 ==========
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=3, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# ========== 重要：去掉load_from或指向NTU的init权重 ==========
load_from = None  # 或指向NTU-60的初始化权重
```

```python
def fine2coarse_ntu60(x):
    """
    为NTU-60创建虚拟的粗分类
    方案1：全部映射到类别0（最简单）
    """
    return 0

# 或者根据NTU-60的语义定义8个粗类
def fine2coarse_ntu60_semantic(x):
    """
    基于NTU-60动作语义的粗分类
    """
    if 0 <= x <= 7:
        return 0  # Hand-arm movements
    elif 8 <= x <= 15:
        return 1  # Two-hand movements
    elif 16 <= x <= 25:
        return 2  # Interaction
    elif 26 <= x <= 31:
        return 3  # Body movements
    elif 32 <= x <= 39:
        return 4  # Pick/throw
    elif 40 <= x <= 49:
        return 5  # Health-related
    elif 50 <= x <= 56:
        return 6  # Common actions
    else:
        return 7  # Other

@MODELS.register_module()
class MMRecognizer3D(BaseRecognizer):
    def extract_feat(self, inputs, ...):
        # ... existing code ...
        
        gts = []
        for data in data_samples:
            gts.extend(data.gt_labels.item)
        gts = torch.stack(gts)
        temp = gts.cpu().numpy()
        
        # ← 修改这里：根据数据集选择映射函数
        if self.cls_head.num_classes == 60:
            # NTU-60模式
            gts_coarse = [fine2coarse_ntu60(i) for i in temp]
        else:
            # MA-52模式
            gts_coarse = [fine2coarse(i) for i in temp]
        
        gts_coarse = torch.from_numpy(np.array(gts_coarse)).cuda()
        # ... rest of code ...
```

```python
class RenovateNet_Fine(nn.Module):
    def __init__(self, n_channel, n_class, alp=0.125, tmp=0.125, mom=0.9, 
                 h_channel=None, version='V0', pred_threshold=0.0, 
                 use_p_map=True, 
                 n_class_coarse=7):  # ← 新增参数
        super(RenovateNet_Fine, self).__init__()
        self.n_channel = n_channel
        self.h_channel = n_channel if h_channel is None else h_channel
        self.n_class = n_class
        self.n_class_coarse = n_class_coarse  # ← 使用传入的值
        # ... rest of code ...

# 在RGBPoseHead的__init__中：
class RGBPoseHead(BaseHead):
    def __init__(self, num_classes, in_channels, 
                 num_coarse_classes=7,  # ← 新增参数
                 ...):
        # ...
        self.fc_rgb_coarse = nn.Linear(self.in_channels[0], num_coarse_classes)
        self.fc_pose_coarse = nn.Linear(self.in_channels[1], num_coarse_classes)
        
        self.fr_coarse_rgb = ST_RenovateNet(
            2048, 8, n_class=num_coarse_classes, h_channel=128)
        self.fr_coarse_pose = ST_RenovateNet(
            512, 32, n_class=num_coarse_classes, h_channel=128)
        
        self.fr_rgb = ST_RenovateNet_Fine(
            2048, n_class=num_classes, 
            n_class_coarse=num_coarse_classes)  # ← 传递粗分类数
        # ... rest
```

```python
class TreeLoss(nn.Module):
    def __init__(self, num_actions=52, num_bodies=7, 
                 action2body_fn=None):  # ← 新增参数
        super(TreeLoss, self).__init__()
        self.num_actions = num_actions
        self.num_bodies = num_bodies
        self.action2body_fn = action2body_fn or action2body
        self.stateSpace = self.generateStateSpace().cuda()
        self.sig = nn.Sigmoid()
    
    def generateStateSpace(self):
        stat_list = np.eye(self.num_bodies + self.num_actions)
        for i in range(self.num_bodies, self.num_bodies + self.num_actions):
            action_id = i - self.num_bodies
            coarse_id = self.action2body_fn(action_id)
            stat_list[i][coarse_id] = 1
        return torch.tensor(stat_list)
```

```python
# configs/skeleton/posec3d/rgbpose_conv3d/quick_ntu60.py
_base_ = './rgbpose_conv3d.py'

# 核心修改
backbone_cfg = dict(
    pose_pathway=dict(in_channels=17))  # 只改骨架点数

head_cfg = dict(
    num_classes=60)  # 只改类别数

model = dict(
    backbone=backbone_cfg,
    cls_head=head_cfg)

# 数据路径
data_root = 'data/nturgbd_videos/'
ann_file = 'data/skeleton/ntu60_2d.pkl'
ann_file_val = 'data/skeleton/ntu60_2d.pkl'
ann_file_test = 'data/skeleton/ntu60_2d.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# 去掉MA-52特有的skeleton定义
train_pipeline = [
    dict(type='MMUniformSampleFrames', 
         clip_len=dict(RGB=8, Pose=32), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True,
         with_kp=True, with_limb=False, scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

# 其他pipeline类似...
```

```python
# 在recognizer3d_mm.py的fine2coarse函数中：
def fine2coarse(x):
    # Hack for NTU-60: 全部映射到0
    if x < 60:  # NTU-60
        return 0
    # 原MA-52逻辑
    elif x <= 4:
        return 0
    # ... rest
```

---

**User**

先用方案B吧 也就是适配PCAN到NTU-60（推荐方案）我现在已经有ntud60-sub的骨架序列以及rgb视频数据了，请你给出步骤 我一步一步来

---

**Cursor**

好的！让我帮你完整地适配PCAN到NTU-60数据集。首先让我检查一下你的数据格式。

完美！让我给你一个完整的分步指南，从数据准备到运行测试。

## 🚀 PCAN适配NTU-60完整指南

### 📋 准备工作检查

首先确认你的数据位置：

```bash
# 在服务器上运行
cd /home/zh/ChCode/codes01/mmaction2

# 检查你的NTU-60数据在哪里
# 通常应该在：
ls -lh <你的NTU60数据路径>/
```

请告诉我你的NTU-60数据的具体格式：
1. **骨架数据**：是`.skeleton`原始文件？还是已经是`.pkl`格式？
2. **RGB视频**：是`.avi`原始文件？还是已压缩的`.mp4`？
3. **数据路径**：具体在哪个目录？

---

## 📝 步骤1：创建数据目录结构

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 创建数据目录
mkdir -p data/skeleton
mkdir -p data/nturgbd_raw      # 原始视频（如果有）
mkdir -p data/nturgbd_videos   # 处理后的视频
```

---

## 📝 步骤2：准备骨架数据

### 方案A：如果你有`.skeleton`原始文件

```bash
# 将原始骨架文件放到一个目录
# 例如：data/ntu60_raw_skeletons/

# 运行转换脚本（生成25点的3D骨架）
python tools/data/skeleton/gen_ntu_rgbd_raw.py \
    --data-path <你的骨架文件路径> \
    --ignored-sample-path tools/data/skeleton/NTU_RGBD_samples_with_missing_skeletons.txt \
    --out-folder data/skeleton/ \
    --task ntu60

# 这会生成：data/skeleton/ntu60_3d.pkl（25个3D关键点）
```

### 方案B：如果你已有`.pkl`文件

```bash
# 直接复制到data/skeleton/
cp <你的pkl路径>/ntu60_*.pkl data/skeleton/

# 检查格式
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_3d.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f"Keys: {data.keys()}")
    print(f"Splits: {data['split'].keys()}")
    print(f"Samples: {len(data['annotations'])}")
    sample = data['annotations'][0]
    print(f"Keypoint shape: {sample['keypoint'].shape}")  # 应该是[M, T, V, C]
    # M: 人数, T: 帧数, V: 关键点数(17或25), C: 坐标维度(2或3)
EOF
```

### 方案C：如果只有2D骨架（17点）

```bash
# 下载MMAction2提供的预处理骨架
cd data/skeleton
wget https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl
# 或
wget https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_3d.pkl
```

---

## 📝 步骤3：准备RGB视频

### 如果有`.avi`原始视频

```bash
# 1. 放置原始视频
# 将NTU-60的RGB视频放到 data/nturgbd_raw/

# 2. 压缩视频（从1920x1080降到960x540，转mp4）
python tools/data/skeleton/compress_nturgbd.py

# 这会生成：data/nturgbd_videos/*.mp4
```

### 如果已有`.mp4`视频

```bash
# 直接复制到目标目录
cp -r <你的视频路径>/*.mp4 data/nturgbd_videos/
```

### 验证视频和骨架对应

```bash
# 检查视频数量和骨架数量是否匹配
python << 'EOF'
import pickle
import os

# 检查骨架
with open('data/skeleton/ntu60_3d.pkl', 'rb') as f:
    data = pickle.load(f)
    skeleton_names = [ann['frame_dir'] for ann in data['annotations']]
    print(f"骨架样本数: {len(skeleton_names)}")
    print(f"示例: {skeleton_names[0]}")  # 例如: S001C001P001R001A001

# 检查视频
videos = os.listdir('data/nturgbd_videos/')
print(f"视频数量: {len([v for v in videos if v.endswith('.mp4')])}")
print(f"示例: {videos[0]}")

# 检查是否匹配
sample_name = skeleton_names[0]
video_name = sample_name + '.mp4'
if video_name in videos:
    print(f"✓ 视频和骨架匹配！")
else:
    print(f"✗ 警告：{video_name} 不存在")
EOF
```

---

## 📝 步骤4：确定使用的骨架格式

根据你的骨架数据，选择配置：

```bash
# 检查骨架点数
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_3d.pkl', 'rb') as f:
    data = pickle.load(f)
    keypoint = data['annotations'][0]['keypoint']
    print(f"Keypoint shape: {keypoint.shape}")
    # [M, T, V, C]
    # V=17 → 使用2D骨架（COCO格式）
    # V=25 → 使用3D骨架（Kinect格式）
EOF
```

**记住这个数字**（V的值），后面配置会用到！

---

## 📝 步骤5：创建NTU-60配置文件

在 `configs/skeleton/posec3d/rgbpose_conv3d/` 目录下创建 `pcan_ntu60.py`：

```python
_base_ = '../../../_base_/default_runtime.py'

# ==========================================
# 模型配置
# ==========================================
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        fusion_kernel=7,
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1),
        with_pool2=False),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        fusion_kernel=7,
        in_channels=17,  # ← 如果是2D骨架用17，3D骨架用25
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride_s=1,
        conv1_stride_t=1,
        pool1_stride_s=1,
        pool1_stride_t=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        with_pool2=False))

head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← NTU-60的60类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],  # ← 只用rgb和pose，去掉coarse
    loss_weights=[1., 1.],
    average_clips='prob')

data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    preprocessors=dict(
        imgs=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        heatmap_imgs=dict(type='ActionDataPreprocessor')))

model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    data_preprocessor=data_preprocessor)

# ==========================================
# 数据配置
# ==========================================
dataset_type = 'PoseDataset'
data_root = './data/nturgbd_videos/'
ann_file = './data/skeleton/ntu60_3d.pkl'  # 或 ntu60_2d.pkl
ann_file_val = './data/skeleton/ntu60_3d.pkl'
ann_file_test = './data/skeleton/ntu60_3d.pkl'

# NTU-60的左右关键点（17点-2D骨架）
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# 如果是25点-3D骨架，使用：
# left_kp = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
# right_kp = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

val_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

test_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=10,  # 多clip测试
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        split='xsub_train',  # ← NTU-60的split
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        split='xsub_val',
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        split='xsub_val',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

# ==========================================
# 训练配置
# ==========================================
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=3, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[10, 20],
        gamma=0.1)
]

load_from = None  # 初始设置为None，稍后指定预训练权重

auto_scale_lr = dict(enable=False, base_batch_size=40)
```

**保存为**：`configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py`

---

## 📝 步骤6：修改代码以支持NTU-60（关键！）

### 修改1：`mmaction/models/recognizers/recognizer3d_mm.py`

在文件开头添加NTU-60的粗分类映射：

```python
# 第10行之后添加
def fine2coarse(x):
    # MA-52的映射（保持不变）
    if x<=4:
        return 0
    elif 5<=x<=10:
        return 1
    elif 11<=x<=23:
        return 2
    elif 24<=x<=31:
        return 3
    elif 32<=x<=37:
        return 4
    elif 38<=x<=47:
        return 5
    else:
        return 6

def fine2coarse_ntu60(x):
    """
    NTU-60的虚拟粗分类映射
    方案1：全部映射到类别0（最简单，但失去层次信息）
    """
    return 0

# 或者使用语义分组（更好）：
def fine2coarse_ntu60_semantic(x):
    """
    基于NTU-60动作语义的8类粗分类
    """
    if 0 <= x <= 9:
        return 0  # Drink, eat, brushing, reading (10个动作)
    elif 10 <= x <= 19:
        return 1  # Hand clapping, phone, camera (10个)
    elif 20 <= x <= 29:
        return 2  # Pickup, throw, sitting (10个)
    elif 30 <= x <= 39:
        return 3  # Standing, falling, kicking (10个)
    elif 40 <= x <= 44:
        return 4  # Punching, pushing, hugging (5个)
    elif 45 <= x <= 49:
        return 5  # Giving, handshaking (5个)
    elif 50 <= x <= 54:
        return 6  # Touch pocket, sneeze, staggering (5个)
    else:
        return 7  # Other (剩余)
```

然后修改`extract_feat`方法（第56-65行）：

```python
# 原代码：
gts = []
for data in data_samples:
    gts.extend(data.gt_labels.item)
gts = torch.stack(gts)
temp = gts.cpu().numpy()
gts_coarse = [fine2coarse(i) for i in temp]  # ← 这里
gts_coarse = torch.from_numpy(np.array(gts_coarse)).cuda()

# 修改为：
gts = []
for data in data_samples:
    gts.extend(data.gt_labels.item)
gts = torch.stack(gts)
temp = gts.cpu().numpy()

# 根据类别数判断数据集
max_label = temp.max()
if max_label >= 52:  # NTU-60 (标签0-59)
    gts_coarse = [fine2coarse_ntu60_semantic(i) for i in temp]
else:  # MA-52 (标签0-51)
    gts_coarse = [fine2coarse(i) for i in temp]

gts_coarse = torch.from_numpy(np.array(gts_coarse)).cuda()
```

### 修改2：`mmaction/models/heads/rgbpose_head.py`

修改`RenovateNet_Fine`支持不同的粗分类数（第68-89行）：

```python
class RenovateNet_Fine(nn.Module):
    def __init__(self, n_channel, n_class, alp=0.125, tmp=0.125, mom=0.9, 
                 h_channel=None, version='V0', pred_threshold=0.0, 
                 use_p_map=True,
                 n_class_coarse=7):  # ← 添加这个参数，默认7
        super(RenovateNet_Fine, self).__init__()
        self.n_channel = n_channel
        self.h_channel = n_channel if h_channel is None else h_channel
        self.n_class = n_class
        self.n_class_coarse = n_class_coarse  # ← 改：使用传入的值
        # ... 其余代码保持不变
```

修改`RGBPoseHead`的初始化（第548-567行）：

```python
# 在__init__参数中添加：
def __init__(self,
             num_classes: int,
             in_channels: Tuple[int],
             loss_cls: Dict = dict(type='CrossEntropyLoss'),
             loss_components: List[str] = ['rgb', 'pose'],
             loss_weights: Union[float, Tuple[float]] = 1.,
             dropout: float = 0.5,
             init_std: float = 0.01,
             num_coarse_classes: int = 7,  # ← 添加这个参数
             **kwargs) -> None:
    
    super().__init__(num_classes, in_channels, loss_cls, **kwargs)
    # ... existing code ...
    
    # 修改这些行：
    self.fc_rgb_coarse = nn.Linear(self.in_channels[0], num_coarse_classes)
    self.fc_pose_coarse = nn.Linear(self.in_channels[1], num_coarse_classes)
    
    self.fr_coarse_rgb = ST_RenovateNet(
        2048, 8, n_class=num_coarse_classes, h_channel=128, version='V0', use_p_map=True)
    self.fr_coarse_pose = ST_RenovateNet(
        512, 32, n_class=num_coarse_classes, h_channel=128, version='V0', use_p_map=True)
    
    self.fr_rgb = ST_RenovateNet_Fine(
        2048, n_class=num_classes, version='V0', use_p_map=True,
        n_class_coarse=num_coarse_classes)  # ← 传递粗分类数
    self.fr_pose = ST_RenovateNet_Fine(
        512, n_class=num_classes, version='V0', use_p_map=True,
        n_class_coarse=num_coarse_classes)  # ← 传递粗分类数
    
    self.tree_loss_rgb = TreeLoss()
    self.tree_loss_pose = TreeLoss()
```

### 修改3：`TreeLoss`适配（第35-64行）

```python
class TreeLoss(nn.Module):
    def __init__(self, num_actions=52, num_bodies=7):  # ← 添加参数
        super(TreeLoss, self).__init__()
        self.num_actions = num_actions
        self.num_bodies = num_bodies
        self.stateSpace = self.generateStateSpace().cuda()
        self.sig = nn.Sigmoid()

    def generateStateSpace(self):
        total_size = self.num_bodies + self.num_actions
        stat_list = np.eye(total_size)
        
        for i in range(self.num_bodies, total_size):
            temp = stat_list[i]
            index = np.where(temp > 0)[0]
            action_id = int(index) - self.num_bodies
            
            # 根据动作数判断使用哪个映射
            if self.num_actions == 60:
                coarse = fine2coarse_ntu60_semantic(action_id)
            else:
                coarse = action2body(action_id)
            
            stat_list[i][coarse] = 1
        stateSpace = torch.tensor(stat_list)
        return stateSpace
```

然后在`RGBPoseHead.__init__`中传递参数：

```python
self.tree_loss_rgb = TreeLoss(
    num_actions=num_classes, 
    num_bodies=num_coarse_classes)
self.tree_loss_pose = TreeLoss(
    num_actions=num_classes, 
    num_bodies=num_coarse_classes)
```

---

## 📝 步骤7：更新配置文件传递粗分类数

回到 `pcan_ntu60.py`，更新head配置：

```python
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,
    num_coarse_classes=8,  # ← 添加：NTU-60使用8个粗类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose', 'rgb_coarse', 'pose_coarse'],  # ← 改回来
    loss_weights=[1., 1., 0.5, 0.5],  # ← 添加粗分类权重
    average_clips='prob')
```

---

## 📝 步骤8：验证配置和数据

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 检查配置是否正确
python << 'EOF'
from mmengine.config import Config

cfg = Config.fromfile('configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py')
print("配置加载成功！")
print(f"类别数: {cfg.model.cls_head.num_classes}")
print(f"骨架点数: {cfg.model.backbone.pose_pathway.in_channels}")
print(f"数据路径: {cfg.data_root}")
print(f"标注文件: {cfg.ann_file}")
EOF

# 测试数据加载
python tools/analysis_tools/browse_dataset.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    --mode pipeline \
    --show-number 1
```

---

## 📝 步骤9：准备预训练权重（可选但推荐）

### 方案A：从头训练（不推荐）

```bash
# 直接训练，但效果可能不好
bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    4
```

### 方案B：使用NTU-60的预训练权重（推荐）

```bash
# 1. 下载NTU-60的RGB-only和Pose-only预训练权重
mkdir -p pretrained/ntu60
cd pretrained/ntu60

# RGB-only预训练
wget https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/rgbpose_conv3d/rgb_only_20230228-576b9f86.pth

# Pose-only预训练
wget https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/rgbpose_conv3d/pose_only_20230228-fa40054e.pth

# 2. 合并两个权重（参考MA-52的方法）
cd /home/zh/ChCode/codes01/mmaction2

python << 'EOF'
import torch

print("合并NTU-60的预训练权重...")

# 加载两个单独的权重
rgb_ckpt = torch.load('pretrained/ntu60/rgb_only_20230228-576b9f86.pth', 
                       map_location='cpu')
pose_ckpt = torch.load('pretrained/ntu60/pose_only_20230228-fa40054e.pth', 
                        map_location='cpu')

# 创建合并的权重
merged = {'state_dict': {}}

# 添加RGB路径的权重
for key, value in rgb_ckpt['state_dict'].items():
    if 'backbone' in key:
        new_key = key.replace('backbone', 'backbone.rgb_path')
        merged['state_dict'][new_key] = value

# 添加Pose路径的权重
for key, value in pose_ckpt['state_dict'].items():
    if 'backbone' in key:
        new_key = key.replace('backbone', 'backbone.pose_path')
        merged['state_dict'][new_key] = value

print(f"合并后的参数数量: {len(merged['state_dict'])}")

# 保存
torch.save(merged, 'pretrained/ntu60/rgbpose_ntu60_init.pth')
print("✓ 保存成功: pretrained/ntu60/rgbpose_ntu60_init.pth")
EOF
```

### 方案C：迁移MA-52的权重（次优）

```bash
# 使用MA-52训练好的backbone，只重新训练head
# 在配置文件中：
load_from = './pretrained/rgbpose_conv3d_init.pth'
# 但需要手动调整in_channels（28→17/25）
```

---

## 📝 步骤10：更新配置文件的load_from

修改 `pcan_ntu60.py` 的最后：

```python
# 如果有NTU-60的初始化权重
load_from = './pretrained/ntu60/rgbpose_ntu60_init.pth'

# 或者从头训练
# load_from = None
```

---

## 📝 步骤11：开始训练！

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 使用4张A100训练
bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    4

# 训练日志会保存在 work_dirs/pcan_ntu60/
```

---

## 📝 步骤12：测试和评估

```bash
# 训练完成后测试
bash tools/dist_test.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    work_dirs/pcan_ntu60/best_acc_top1_epoch_*.pth \
    4

# 或者使用单GPU测试
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    work_dirs/pcan_ntu60/best_acc_top1_epoch_*.pth
```

---

## 🔧 故障排查

### 问题1：`TreeLoss`尺寸不匹配

**报错**：`RuntimeError: size mismatch, m1: [8, 67], m2: [59, ...]`

**解决**：确保`TreeLoss`正确初始化：

```python
# 在RGBPoseHead.__init__中
self.tree_loss_rgb = TreeLoss(num_actions=60, num_bodies=8)
self.tree_loss_pose = TreeLoss(num_actions=60, num_bodies=8)
```

### 问题2：找不到视频文件

**报错**：`FileNotFoundError: S001C001P001R001A001.mp4`

**解决**：检查视频命名：

```bash
# NTU-60的标准命名（不带后缀）
ls data/nturgbd_videos/ | head -5
# 应该看到：S001C001P001R001A001.mp4

# 如果pkl中是不带后缀的
# 但视频文件是 .mp4
# 代码会自动添加 .mp4
```

### 问题3：骨架点数不匹配

**报错**：`RuntimeError: Expected 28 channels, got 17`

**解决**：确保配置文件的`in_channels`与实际骨架点数一致：

```bash
# 检查实际骨架点数
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_3d.pkl', 'rb') as f:
    data = pickle.load(f)
    V = data['annotations'][0]['keypoint'].shape[2]
    print(f"骨架点数: {V}")
    # V=17 → 配置用 in_channels=17
    # V=25 → 配置用 in_channels=25
EOF
```

---

## 📋 完整的文件清单

你需要创建/修改的文件：

```
mmaction2/
├── configs/skeleton/posec3d/rgbpose_conv3d/
│   └── pcan_ntu60.py  ← 新建配置文件
├── mmaction/models/recognizers/
│   └── recognizer3d_mm.py  ← 修改：添加fine2coarse_ntu60
├── mmaction/models/heads/
│   └── rgbpose_head.py  ← 修改：支持动态粗分类数
└── data/
    ├── skeleton/
    │   ├── ntu60_2d.pkl  ← 骨架数据（17点）
    │   └── ntu60_3d.pkl  ← 或3D骨架（25点）
    └── nturgbd_videos/
        └── *.mp4  ← RGB视频
```

---

## ✅ 快速检查清单

在开始训练前，确认：

- [ ] 骨架pkl文件存在：`data/skeleton/ntu60_*.pkl`
- [ ] RGB视频存在：`data/nturgbd_videos/*.mp4`
- [ ] 配置文件创建：`pcan_ntu60.py`
- [ ] 代码已修改：`recognizer3d_mm.py`
- [ ] 代码已修改：`rgbpose_head.py`（3处）
- [ ] 骨架点数匹配：配置中的`in_channels`=实际点数
- [ ] 类别数匹配：`num_classes=60`
- [ ] 粗分类数匹配：`num_coarse_classes=8`（或1）

---

## 💡 如果你现在就想开始

**最快的验证步骤**：

```bash
# 1. 告诉我你的数据格式
cd /home/zh/ChCode/codes01/mmaction2
ls data/skeleton/
ls data/nturgbd_videos/ | head -5

# 2. 检查骨架点数
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_3d.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f"骨架点数V: {data['annotations'][0]['keypoint'].shape[2]}")
EOF
```

**把这两个命令的输出告诉我**，我会根据你的实际数据格式，给出最精确的配置！

你准备好开始了吗？先确认你的数据格式，然后我们一步步来！🚀

```shellscript
# 在服务器上运行
cd /home/zh/ChCode/codes01/mmaction2

# 检查你的NTU-60数据在哪里
# 通常应该在：
ls -lh <你的NTU60数据路径>/
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 创建数据目录
mkdir -p data/skeleton
mkdir -p data/nturgbd_raw      # 原始视频（如果有）
mkdir -p data/nturgbd_videos   # 处理后的视频
```

```shellscript
# 将原始骨架文件放到一个目录
# 例如：data/ntu60_raw_skeletons/

# 运行转换脚本（生成25点的3D骨架）
python tools/data/skeleton/gen_ntu_rgbd_raw.py \
    --data-path <你的骨架文件路径> \
    --ignored-sample-path tools/data/skeleton/NTU_RGBD_samples_with_missing_skeletons.txt \
    --out-folder data/skeleton/ \
    --task ntu60

# 这会生成：data/skeleton/ntu60_3d.pkl（25个3D关键点）
```

```shellscript
# 直接复制到data/skeleton/
cp <你的pkl路径>/ntu60_*.pkl data/skeleton/

# 检查格式
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_3d.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f"Keys: {data.keys()}")
    print(f"Splits: {data['split'].keys()}")
    print(f"Samples: {len(data['annotations'])}")
    sample = data['annotations'][0]
    print(f"Keypoint shape: {sample['keypoint'].shape}")  # 应该是[M, T, V, C]
    # M: 人数, T: 帧数, V: 关键点数(17或25), C: 坐标维度(2或3)
EOF
```

```shellscript
# 下载MMAction2提供的预处理骨架
cd data/skeleton
wget https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl
# 或
wget https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_3d.pkl
```

```shellscript
# 1. 放置原始视频
# 将NTU-60的RGB视频放到 data/nturgbd_raw/

# 2. 压缩视频（从1920x1080降到960x540，转mp4）
python tools/data/skeleton/compress_nturgbd.py

# 这会生成：data/nturgbd_videos/*.mp4
```

```shellscript
# 直接复制到目标目录
cp -r <你的视频路径>/*.mp4 data/nturgbd_videos/
```

```shellscript
# 检查视频数量和骨架数量是否匹配
python << 'EOF'
import pickle
import os

# 检查骨架
with open('data/skeleton/ntu60_3d.pkl', 'rb') as f:
    data = pickle.load(f)
    skeleton_names = [ann['frame_dir'] for ann in data['annotations']]
    print(f"骨架样本数: {len(skeleton_names)}")
    print(f"示例: {skeleton_names[0]}")  # 例如: S001C001P001R001A001

# 检查视频
videos = os.listdir('data/nturgbd_videos/')
print(f"视频数量: {len([v for v in videos if v.endswith('.mp4')])}")
print(f"示例: {videos[0]}")

# 检查是否匹配
sample_name = skeleton_names[0]
video_name = sample_name + '.mp4'
if video_name in videos:
    print(f"✓ 视频和骨架匹配！")
else:
    print(f"✗ 警告：{video_name} 不存在")
EOF
```

```shellscript
# 检查骨架点数
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_3d.pkl', 'rb') as f:
    data = pickle.load(f)
    keypoint = data['annotations'][0]['keypoint']
    print(f"Keypoint shape: {keypoint.shape}")
    # [M, T, V, C]
    # V=17 → 使用2D骨架（COCO格式）
    # V=25 → 使用3D骨架（Kinect格式）
EOF
```

```python
_base_ = '../../../_base_/default_runtime.py'

# ==========================================
# 模型配置
# ==========================================
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        fusion_kernel=7,
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1),
        with_pool2=False),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        fusion_kernel=7,
        in_channels=17,  # ← 如果是2D骨架用17，3D骨架用25
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride_s=1,
        conv1_stride_t=1,
        pool1_stride_s=1,
        pool1_stride_t=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        with_pool2=False))

head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← NTU-60的60类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],  # ← 只用rgb和pose，去掉coarse
    loss_weights=[1., 1.],
    average_clips='prob')

data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    preprocessors=dict(
        imgs=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        heatmap_imgs=dict(type='ActionDataPreprocessor')))

model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    data_preprocessor=data_preprocessor)

# ==========================================
# 数据配置
# ==========================================
dataset_type = 'PoseDataset'
data_root = './data/nturgbd_videos/'
ann_file = './data/skeleton/ntu60_3d.pkl'  # 或 ntu60_2d.pkl
ann_file_val = './data/skeleton/ntu60_3d.pkl'
ann_file_test = './data/skeleton/ntu60_3d.pkl'

# NTU-60的左右关键点（17点-2D骨架）
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# 如果是25点-3D骨架，使用：
# left_kp = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
# right_kp = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

val_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

test_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=10,  # 多clip测试
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        split='xsub_train',  # ← NTU-60的split
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        split='xsub_val',
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        split='xsub_val',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

# ==========================================
# 训练配置
# ==========================================
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=3, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[10, 20],
        gamma=0.1)
]

load_from = None  # 初始设置为None，稍后指定预训练权重

auto_scale_lr = dict(enable=False, base_batch_size=40)
```

```python
# 第10行之后添加
def fine2coarse(x):
    # MA-52的映射（保持不变）
    if x<=4:
        return 0
    elif 5<=x<=10:
        return 1
    elif 11<=x<=23:
        return 2
    elif 24<=x<=31:
        return 3
    elif 32<=x<=37:
        return 4
    elif 38<=x<=47:
        return 5
    else:
        return 6

def fine2coarse_ntu60(x):
    """
    NTU-60的虚拟粗分类映射
    方案1：全部映射到类别0（最简单，但失去层次信息）
    """
    return 0

# 或者使用语义分组（更好）：
def fine2coarse_ntu60_semantic(x):
    """
    基于NTU-60动作语义的8类粗分类
    """
    if 0 <= x <= 9:
        return 0  # Drink, eat, brushing, reading (10个动作)
    elif 10 <= x <= 19:
        return 1  # Hand clapping, phone, camera (10个)
    elif 20 <= x <= 29:
        return 2  # Pickup, throw, sitting (10个)
    elif 30 <= x <= 39:
        return 3  # Standing, falling, kicking (10个)
    elif 40 <= x <= 44:
        return 4  # Punching, pushing, hugging (5个)
    elif 45 <= x <= 49:
        return 5  # Giving, handshaking (5个)
    elif 50 <= x <= 54:
        return 6  # Touch pocket, sneeze, staggering (5个)
    else:
        return 7  # Other (剩余)
```

```python
# 原代码：
gts = []
for data in data_samples:
    gts.extend(data.gt_labels.item)
gts = torch.stack(gts)
temp = gts.cpu().numpy()
gts_coarse = [fine2coarse(i) for i in temp]  # ← 这里
gts_coarse = torch.from_numpy(np.array(gts_coarse)).cuda()

# 修改为：
gts = []
for data in data_samples:
    gts.extend(data.gt_labels.item)
gts = torch.stack(gts)
temp = gts.cpu().numpy()

# 根据类别数判断数据集
max_label = temp.max()
if max_label >= 52:  # NTU-60 (标签0-59)
    gts_coarse = [fine2coarse_ntu60_semantic(i) for i in temp]
else:  # MA-52 (标签0-51)
    gts_coarse = [fine2coarse(i) for i in temp]

gts_coarse = torch.from_numpy(np.array(gts_coarse)).cuda()
```

```python
class RenovateNet_Fine(nn.Module):
    def __init__(self, n_channel, n_class, alp=0.125, tmp=0.125, mom=0.9, 
                 h_channel=None, version='V0', pred_threshold=0.0, 
                 use_p_map=True,
                 n_class_coarse=7):  # ← 添加这个参数，默认7
        super(RenovateNet_Fine, self).__init__()
        self.n_channel = n_channel
        self.h_channel = n_channel if h_channel is None else h_channel
        self.n_class = n_class
        self.n_class_coarse = n_class_coarse  # ← 改：使用传入的值
        # ... 其余代码保持不变
```

```python
# 在__init__参数中添加：
def __init__(self,
             num_classes: int,
             in_channels: Tuple[int],
             loss_cls: Dict = dict(type='CrossEntropyLoss'),
             loss_components: List[str] = ['rgb', 'pose'],
             loss_weights: Union[float, Tuple[float]] = 1.,
             dropout: float = 0.5,
             init_std: float = 0.01,
             num_coarse_classes: int = 7,  # ← 添加这个参数
             **kwargs) -> None:
    
    super().__init__(num_classes, in_channels, loss_cls, **kwargs)
    # ... existing code ...
    
    # 修改这些行：
    self.fc_rgb_coarse = nn.Linear(self.in_channels[0], num_coarse_classes)
    self.fc_pose_coarse = nn.Linear(self.in_channels[1], num_coarse_classes)
    
    self.fr_coarse_rgb = ST_RenovateNet(
        2048, 8, n_class=num_coarse_classes, h_channel=128, version='V0', use_p_map=True)
    self.fr_coarse_pose = ST_RenovateNet(
        512, 32, n_class=num_coarse_classes, h_channel=128, version='V0', use_p_map=True)
    
    self.fr_rgb = ST_RenovateNet_Fine(
        2048, n_class=num_classes, version='V0', use_p_map=True,
        n_class_coarse=num_coarse_classes)  # ← 传递粗分类数
    self.fr_pose = ST_RenovateNet_Fine(
        512, n_class=num_classes, version='V0', use_p_map=True,
        n_class_coarse=num_coarse_classes)  # ← 传递粗分类数
    
    self.tree_loss_rgb = TreeLoss()
    self.tree_loss_pose = TreeLoss()
```

```python
class TreeLoss(nn.Module):
    def __init__(self, num_actions=52, num_bodies=7):  # ← 添加参数
        super(TreeLoss, self).__init__()
        self.num_actions = num_actions
        self.num_bodies = num_bodies
        self.stateSpace = self.generateStateSpace().cuda()
        self.sig = nn.Sigmoid()

    def generateStateSpace(self):
        total_size = self.num_bodies + self.num_actions
        stat_list = np.eye(total_size)
        
        for i in range(self.num_bodies, total_size):
            temp = stat_list[i]
            index = np.where(temp > 0)[0]
            action_id = int(index) - self.num_bodies
            
            # 根据动作数判断使用哪个映射
            if self.num_actions == 60:
                coarse = fine2coarse_ntu60_semantic(action_id)
            else:
                coarse = action2body(action_id)
            
            stat_list[i][coarse] = 1
        stateSpace = torch.tensor(stat_list)
        return stateSpace
```

```python
self.tree_loss_rgb = TreeLoss(
    num_actions=num_classes, 
    num_bodies=num_coarse_classes)
self.tree_loss_pose = TreeLoss(
    num_actions=num_classes, 
    num_bodies=num_coarse_classes)
```

```python
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,
    num_coarse_classes=8,  # ← 添加：NTU-60使用8个粗类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose', 'rgb_coarse', 'pose_coarse'],  # ← 改回来
    loss_weights=[1., 1., 0.5, 0.5],  # ← 添加粗分类权重
    average_clips='prob')
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 检查配置是否正确
python << 'EOF'
from mmengine.config import Config

cfg = Config.fromfile('configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py')
print("配置加载成功！")
print(f"类别数: {cfg.model.cls_head.num_classes}")
print(f"骨架点数: {cfg.model.backbone.pose_pathway.in_channels}")
print(f"数据路径: {cfg.data_root}")
print(f"标注文件: {cfg.ann_file}")
EOF

# 测试数据加载
python tools/analysis_tools/browse_dataset.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    --mode pipeline \
    --show-number 1
```

```shellscript
# 直接训练，但效果可能不好
bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    4
```

```shellscript
# 1. 下载NTU-60的RGB-only和Pose-only预训练权重
mkdir -p pretrained/ntu60
cd pretrained/ntu60

# RGB-only预训练
wget https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/rgbpose_conv3d/rgb_only_20230228-576b9f86.pth

# Pose-only预训练
wget https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/rgbpose_conv3d/pose_only_20230228-fa40054e.pth

# 2. 合并两个权重（参考MA-52的方法）
cd /home/zh/ChCode/codes01/mmaction2

python << 'EOF'
import torch

print("合并NTU-60的预训练权重...")

# 加载两个单独的权重
rgb_ckpt = torch.load('pretrained/ntu60/rgb_only_20230228-576b9f86.pth', 
                       map_location='cpu')
pose_ckpt = torch.load('pretrained/ntu60/pose_only_20230228-fa40054e.pth', 
                        map_location='cpu')

# 创建合并的权重
merged = {'state_dict': {}}

# 添加RGB路径的权重
for key, value in rgb_ckpt['state_dict'].items():
    if 'backbone' in key:
        new_key = key.replace('backbone', 'backbone.rgb_path')
        merged['state_dict'][new_key] = value

# 添加Pose路径的权重
for key, value in pose_ckpt['state_dict'].items():
    if 'backbone' in key:
        new_key = key.replace('backbone', 'backbone.pose_path')
        merged['state_dict'][new_key] = value

print(f"合并后的参数数量: {len(merged['state_dict'])}")

# 保存
torch.save(merged, 'pretrained/ntu60/rgbpose_ntu60_init.pth')
print("✓ 保存成功: pretrained/ntu60/rgbpose_ntu60_init.pth")
EOF
```

```shellscript
# 使用MA-52训练好的backbone，只重新训练head
# 在配置文件中：
load_from = './pretrained/rgbpose_conv3d_init.pth'
# 但需要手动调整in_channels（28→17/25）
```

```python
# 如果有NTU-60的初始化权重
load_from = './pretrained/ntu60/rgbpose_ntu60_init.pth'

# 或者从头训练
# load_from = None
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 使用4张A100训练
bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    4

# 训练日志会保存在 work_dirs/pcan_ntu60/
```

```shellscript
# 训练完成后测试
bash tools/dist_test.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    work_dirs/pcan_ntu60/best_acc_top1_epoch_*.pth \
    4

# 或者使用单GPU测试
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    work_dirs/pcan_ntu60/best_acc_top1_epoch_*.pth
```

```python
# 在RGBPoseHead.__init__中
self.tree_loss_rgb = TreeLoss(num_actions=60, num_bodies=8)
self.tree_loss_pose = TreeLoss(num_actions=60, num_bodies=8)
```

```shellscript
# NTU-60的标准命名（不带后缀）
ls data/nturgbd_videos/ | head -5
# 应该看到：S001C001P001R001A001.mp4

# 如果pkl中是不带后缀的
# 但视频文件是 .mp4
# 代码会自动添加 .mp4
```

```shellscript
# 检查实际骨架点数
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_3d.pkl', 'rb') as f:
    data = pickle.load(f)
    V = data['annotations'][0]['keypoint'].shape[2]
    print(f"骨架点数: {V}")
    # V=17 → 配置用 in_channels=17
    # V=25 → 配置用 in_channels=25
EOF
```

```plaintext
mmaction2/
├── configs/skeleton/posec3d/rgbpose_conv3d/
│   └── pcan_ntu60.py  ← 新建配置文件
├── mmaction/models/recognizers/
│   └── recognizer3d_mm.py  ← 修改：添加fine2coarse_ntu60
├── mmaction/models/heads/
│   └── rgbpose_head.py  ← 修改：支持动态粗分类数
└── data/
    ├── skeleton/
    │   ├── ntu60_2d.pkl  ← 骨架数据（17点）
    │   └── ntu60_3d.pkl  ← 或3D骨架（25点）
    └── nturgbd_videos/
        └── *.mp4  ← RGB视频
```

```shellscript
# 1. 告诉我你的数据格式
cd /home/zh/ChCode/codes01/mmaction2
ls data/skeleton/
ls data/nturgbd_videos/ | head -5

# 2. 检查骨架点数
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_3d.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f"骨架点数V: {data['annotations'][0]['keypoint'].shape[2]}")
EOF
```

---

**User**

现在出现报错了：Traceback (most recent call last):

  File "tools/train.py", line 143, in <module>

    main()

  File "tools/train.py", line 139, in main

    runner.train()

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/runner.py", line 1728, in train

    self._train_loop = self.build_train_loop(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/runner.py", line 1520, in build_train_loop

    loop = LOOPS.build(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/registry/registry.py", line 570, in build

    return self.build_func(cfg, *args, **kwargs, registry=self)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/registry/build_functions.py", line 121, in build_from_cfg

    obj = obj_cls(**args)  # type: ignore

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py", line 46, in __init__

    super().__init__(runner, dataloader)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/base_loop.py", line 26, in __init__

    self.dataloader = runner.build_dataloader(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/runner.py", line 1370, in build_dataloader

    dataset = DATASETS.build(dataset_cfg)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/registry/registry.py", line 570, in build

    return self.build_func(cfg, *args, **kwargs, registry=self)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/registry/build_functions.py", line 121, in build_from_cfg

    obj = obj_cls(**args)  # type: ignore

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/datasets/pose_dataset.py", line 40, in __init__

    super().__init__(

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/datasets/base.py", line 48, in __init__

    super().__init__(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/dataset/base_dataset.py", line 247, in __init__

    self.full_init()

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/dataset/base_dataset.py", line 298, in full_init

    self.data_list = self.load_data_list()

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/datasets/pose_dataset.py", line 50, in load_data_list

    split, annos = data_list['split'], data_list['annotations']

TypeError: list indices must be integers or slices, not str

Traceback (most recent call last):

  File "tools/train.py", line 143, in <module>

    main()

  File "tools/train.py", line 139, in main

    runner.train()

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/runner.py", line 1728, in train

    self._train_loop = self.build_train_loop(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/runner.py", line 1520, in build_train_loop

    loop = LOOPS.build(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/registry/registry.py", line 570, in build

    return self.build_func(cfg, *args, **kwargs, registry=self)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/registry/build_functions.py", line 121, in build_from_cfg

    obj = obj_cls(**args)  # type: ignore

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py", line 46, in __init__

    super().__init__(runner, dataloader)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/base_loop.py", line 26, in __init__

    self.dataloader = runner.build_dataloader(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/runner.py", line 1370, in build_dataloader

    dataset = DATASETS.build(dataset_cfg)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/registry/registry.py", line 570, in build

    return self.build_func(cfg, *args, **kwargs, registry=self)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/registry/build_functions.py", line 121, in build_from_cfg

    obj = obj_cls(**args)  # type: ignore

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/datasets/pose_dataset.py", line 40, in __init__

    super().__init__(

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/datasets/base.py", line 48, in __init__

    super().__init__(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/dataset/base_dataset.py", line 247, in __init__

    self.full_init()

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/dataset/base_dataset.py", line 298, in full_init

    self.data_list = self.load_data_list()

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/datasets/pose_dataset.py", line 50, in load_data_list

    split, annos = data_list['split'], data_list['annotations']

TypeError: list indices must be integers or slices, not str

ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 3322297) of binary: /home/zh/anaconda3/envs/openmmlab/bin/python

Traceback (most recent call last):

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/runpy.py", line 194, in _run_module_as_main

    return _run_code(code, main_globals, None,

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/runpy.py", line 87, in _run_code

    exec(code, run_globals)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/torch/distributed/launch.py", line 196, in <module>

    main()

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/torch/distributed/launch.py", line 192, in main

    launch(args)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/torch/distributed/launch.py", line 177, in launch

    run(args)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/torch/distributed/run.py", line 785, in run

    elastic_launch(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 134, in __call__

    return launch_agent(self._config, self._entrypoint, list(args))

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent

    raise ChildFailedError(

torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 

============================================================

tools/train.py FAILED

------------------------------------------------------------

Failures:

[1]:

  time      : 2025-11-21_00:56:14

  host      : localhost.localdomain

  rank      : 1 (local_rank: 1)

  exitcode  : 1 (pid: 3322298)

  error_file: <N/A>

  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html

------------------------------------------------------------

Root Cause (first observed failure):

[0]:

  time      : 2025-11-21_00:56:14

  host      : localhost.localdomain

  rank      : 0 (local_rank: 0)

  exitcode  : 1 (pid: 3322297)

  error_file: <N/A>

  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html

============================================================   我pcan_ntu60.py这个文件的数据配置部分是这样的# ==========================================

# 数据配置

# ==========================================

dataset_type = 'PoseDataset'

data_root = '/home/zh/ChCode/codes01/mmaction2/data/nturgbd_videos/'

ann_file = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_train.pkl'  # 或 ntu60_2d.pkl

ann_file_val = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_val.pkl'

ann_file_test = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_val.pkl'     这个是这样配置么？ntu60_xsub_val.pkl和ntu60_xsub_val.pkl这两个文件是由ntu60_2d.pkl 拆分出来的，  此外骨骼点是17个，所有的修改我都直接修改在服务器上了，目前你看到的项目在我本地，本地没有修改，因此你看不到。然后现在报错是什么情况？  以及我运行这个检测数据的代码结果如下 (openmmlab) [zh@node01 mmaction2]$ python << 'EOF'

import pickle

import os



# 检查骨架

with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_2d.pkl', 'rb') as f:

    data = pickle.load(f)

    skeleton_names = [ann['frame_dir'] for ann in data['annotations']]

    print(f"骨架样本数: {len(skeleton_names)}")

    print(f"示例: {skeleton_names[0]}")  # 例如: S001C001P001R001A001



# 检查视频

videos = os.listdir('/home/zh/ChCode/codes01/mmaction2/data/nturgbd_videos/')

print(f"视频数量: {len([v for v in videos if v.endswith('.mp4')])}")

print(f"示例: {videos[0]}")



# 检查是否匹配

sample_name = skeleton_names[0]

video_name = sample_name + '.mp4'

if video_name in videos:

    print(f"✓ 视频和骨架匹配！")

else:

    print(f"✗ 警告：{video_name} 不存在")

EOF

骨架样本数: 56578

示例: S001C001P001R001A001

视频数量: 56578

示例: S007C003P028R001A021_rgb.mp4

✗ 警告：S001C001P001R001A001.mp4 不存在

---

**Cursor**

发现了两个问题！让我帮你逐一解决。

**找到问题了！有两个关键错误：**

## ❌ 问题1：pkl文件格式不对

错误信息：`TypeError: list indices must be integers or slices, not str`

**原因**：你的`ntu60_xsub_train.pkl`可能只是一个列表，而不是标准的字典格式。

### 检查pkl格式

```bash
python << 'EOF'
import pickle

# 检查train pkl
with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f"数据类型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"✓ 格式正确 - 字典")
        print(f"  Keys: {data.keys()}")
    elif isinstance(data, list):
        print(f"✗ 格式错误 - 列表")
        print(f"  需要转换为字典格式")
        print(f"  列表长度: {len(data)}")
EOF
```

### 解决方案：转换pkl格式

如果输出显示是**列表**，需要转换：

```bash
python << 'EOF'
import pickle

# 加载列表格式的数据
with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

print(f"训练集样本数: {len(train_data)}")
print(f"验证集样本数: {len(val_data)}")

# 转换为标准格式
standard_format = {
    'split': {
        'xsub_train': [item['frame_dir'] for item in train_data],
        'xsub_val': [item['frame_dir'] for item in val_data]
    },
    'annotations': train_data + val_data  # 合并所有标注
}

# 保存为标准格式
with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl', 'wb') as f:
    pickle.dump(standard_format, f)

print("✓ 转换完成！")
print(f"  总样本数: {len(standard_format['annotations'])}")
print(f"  Splits: {standard_format['split'].keys()}")
print(f"  训练集: {len(standard_format['split']['xsub_train'])}个")
print(f"  验证集: {len(standard_format['split']['xsub_val'])}个")
EOF
```

**然后修改配置文件**：

```python
# pcan_ntu60.py 中修改：
ann_file = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl'
ann_file_val = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl'
ann_file_test = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl'

# dataloader中需要指定split
train_dataloader = dict(
    # ... other config
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        split='xsub_train',  # ← 指定使用哪个split
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        split='xsub_val',  # ← 指定split
        pipeline=val_pipeline,
        test_mode=True))
```

---

## ❌ 问题2：视频命名不匹配

**骨架中**：`S001C001P001R001A001`  
**实际视频**：`S007C003P028R001A021_rgb.mp4`（有`_rgb`后缀）

### 解决方案A：重命名视频文件（推荐）

```bash
cd /home/zh/ChCode/codes01/mmaction2/data/nturgbd_videos

# 批量去除_rgb后缀
for file in *_rgb.mp4; do
    newname="${file/_rgb.mp4/.mp4}"
    mv "$file" "$newname"
done

# 验证
ls | head -5
# 应该看到：S001C001P001R001A001.mp4
```

### 解决方案B：修改pkl中的frame_dir

```bash
python << 'EOF'
import pickle

# 加载pkl
with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl', 'rb') as f:
    data = pickle.load(f)

# 给所有frame_dir添加_rgb后缀
for ann in data['annotations']:
    if not ann['frame_dir'].endswith('_rgb'):
        ann['frame_dir'] = ann['frame_dir'] + '_rgb'

# 保存
with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✓ 已添加_rgb后缀")
EOF
```

---

## 📋 完整的修复步骤

### 步骤1：检查并转换pkl格式

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 1. 检查格式
python << 'EOF'
import pickle

files = {
    'train': 'data/skeleton/ntu60_xsub_train.pkl',
    'val': 'data/skeleton/ntu60_xsub_val.pkl'
}

for name, path in files.items():
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"{name}: type={type(data)}, ", end='')
    if isinstance(data, list):
        print(f"len={len(data)}")
    elif isinstance(data, dict):
        print(f"keys={data.keys()}")
EOF
```

### 步骤2：如果是列表，运行转换脚本

```bash
python << 'EOF'
import pickle

print("转换pkl格式...")

# 加载分离的pkl
with open('data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
    
with open('data/skeleton/ntu60_xsub_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

# 确保是列表
if not isinstance(train_data, list):
    print("✗ train数据不是列表格式")
    exit(1)
if not isinstance(val_data, list):
    print("✗ val数据不是列表格式")
    exit(1)

print(f"训练集: {len(train_data)}个样本")
print(f"验证集: {len(val_data)}个样本")

# 检查是否需要添加_rgb后缀
sample = train_data[0]
print(f"示例frame_dir: {sample['frame_dir']}")

# 检查视频文件格式
import os
videos = os.listdir('data/nturgbd_videos/')
has_rgb_suffix = any('_rgb.mp4' in v for v in videos)
print(f"视频有_rgb后缀: {has_rgb_suffix}")

# 如果视频有_rgb后缀，给frame_dir也加上
if has_rgb_suffix:
    print("添加_rgb后缀到frame_dir...")
    for item in train_data + val_data:
        if not item['frame_dir'].endswith('_rgb'):
            item['frame_dir'] = item['frame_dir'] + '_rgb'

# 转换为标准格式
standard_format = {
    'split': {
        'xsub_train': [item['frame_dir'] for item in train_data],
        'xsub_val': [item['frame_dir'] for item in val_data]
    },
    'annotations': train_data + val_data
}

# 保存
output_path = 'data/skeleton/ntu60_xsub.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(standard_format, f)

print(f"\n✓ 转换完成！保存到: {output_path}")
print(f"  格式: dict with keys {list(standard_format.keys())}")
print(f"  Splits: {list(standard_format['split'].keys())}")
print(f"  总样本数: {len(standard_format['annotations'])}")
print(f"  训练集: {len(standard_format['split']['xsub_train'])}个")
print(f"  验证集: {len(standard_format['split']['xsub_val'])}个")

# 验证
sample = standard_format['annotations'][0]
print(f"\n示例数据:")
print(f"  frame_dir: {sample['frame_dir']}")
print(f"  label: {sample['label']}")
print(f"  keypoint shape: {sample['keypoint'].shape}")
EOF
```

### 步骤3：修正配置文件

修改 `pcan_ntu60.py` 的数据配置部分：

```python
# ==========================================
# 数据配置
# ==========================================
dataset_type = 'PoseDataset'
data_root = '/home/zh/ChCode/codes01/mmaction2/data/nturgbd_videos/'

# ← 改：使用合并后的pkl文件
ann_file = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
ann_file_val = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
ann_file_test = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'

# ... 其他配置 ...

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        split='xsub_train',  # ← 必须指定split
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        split='xsub_val',  # ← 必须指定split
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        split='xsub_val',  # ← 必须指定split
        pipeline=test_pipeline,
        test_mode=True))
```

### 步骤4：验证修复

```bash
# 运行转换脚本后，验证新pkl
python << 'EOF'
import pickle
import os

pkl_path = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("="*60)
print("PKL格式验证")
print("="*60)
print(f"✓ 类型: {type(data)}")
print(f"✓ Keys: {list(data.keys())}")
print(f"✓ Splits: {list(data['split'].keys())}")
print(f"✓ 总样本: {len(data['annotations'])}")

# 检查视频匹配
sample = data['annotations'][0]
frame_dir = sample['frame_dir']
print(f"\n示例frame_dir: {frame_dir}")

# 构建视频路径
video_path = os.path.join('data/nturgbd_videos', frame_dir + '.mp4')
if os.path.exists(video_path):
    print(f"✓ 视频存在: {video_path}")
else:
    print(f"✗ 视频不存在: {video_path}")
    # 尝试查找
    videos = os.listdir('data/nturgbd_videos')
    # 去掉_rgb.mp4看看能不能匹配
    base_name = frame_dir.replace('_rgb', '')
    if base_name + '.mp4' in videos:
        print(f"  提示: 找到 {base_name}.mp4")
        print(f"  建议: frame_dir去掉_rgb后缀")
    elif frame_dir + '_rgb.mp4' in videos:
        print(f"  提示: 找到 {frame_dir}_rgb.mp4")
        print(f"  建议: 视频去掉_rgb后缀")
EOF
```

### 步骤5：根据上面的输出选择修正方案

**方案A：如果frame_dir有`_rgb`但视频没有**

```bash
# 修改pkl，去掉_rgb
python << 'EOF'
import pickle

with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

# 去掉所有_rgb后缀
for ann in data['annotations']:
    ann['frame_dir'] = ann['frame_dir'].replace('_rgb', '')

# 更新split
data['split']['xsub_train'] = [s.replace('_rgb', '') for s in data['split']['xsub_train']]
data['split']['xsub_val'] = [s.replace('_rgb', '') for s in data['split']['xsub_val']]

# 保存
with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
    pickle.dump(data, f)
print("✓ 已去除_rgb后缀")
EOF
```

**方案B：如果视频有`_rgb`但frame_dir没有**

```bash
# 修改pkl，添加_rgb
python << 'EOF'
import pickle

with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

# 添加_rgb后缀
for ann in data['annotations']:
    if not ann['frame_dir'].endswith('_rgb'):
        ann['frame_dir'] = ann['frame_dir'] + '_rgb'

# 更新split
data['split']['xsub_train'] = [s if s.endswith('_rgb') else s+'_rgb' 
                                for s in data['split']['xsub_train']]
data['split']['xsub_val'] = [s if s.endswith('_rgb') else s+'_rgb' 
                              for s in data['split']['xsub_val']]

# 保存
with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
    pickle.dump(data, f)
print("✓ 已添加_rgb后缀")
EOF
```

---

## 📝 完整的修复流程（按顺序执行）

```bash
cd /home/zh/ChCode/codes01/mmaction2

# ===== 第1步：检查pkl格式 =====
echo "检查pkl格式..."
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"类型: {type(data)}")
if isinstance(data, dict):
    print("✓ 已经是字典格式")
elif isinstance(data, list):
    print("✗ 是列表格式，需要转换")
EOF

# ===== 第2步：转换格式（如果需要）=====
echo ""
echo "转换pkl格式..."
python << 'EOF'
import pickle
import os

# 加载
with open('data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('data/skeleton/ntu60_xsub_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

# 检查视频命名
videos = os.listdir('data/nturgbd_videos/')
has_rgb_suffix = videos[0].endswith('_rgb.mp4')

print(f"视频示例: {videos[0]}")
print(f"视频有_rgb后缀: {has_rgb_suffix}")
print(f"骨架示例: {train_data[0]['frame_dir']}")

# 统一命名：都加_rgb（因为视频有_rgb）
for item in train_data + val_data:
    if not item['frame_dir'].endswith('_rgb'):
        item['frame_dir'] = item['frame_dir'] + '_rgb'

# 标准格式
data = {
    'split': {
        'xsub_train': [item['frame_dir'] for item in train_data],
        'xsub_val': [item['frame_dir'] for item in val_data]
    },
    'annotations': train_data + val_data
}

# 保存
with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"✓ 保存到: data/skeleton/ntu60_xsub.pkl")
print(f"  训练集: {len(data['split']['xsub_train'])}")
print(f"  验证集: {len(data['split']['xsub_val'])}")
EOF

# ===== 第3步：验证视频匹配 =====
echo ""
echo "验证视频和骨架匹配..."
python << 'EOF'
import pickle
import os

with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

# 检查前10个样本
success = 0
for i in range(min(10, len(data['annotations']))):
    frame_dir = data['annotations'][i]['frame_dir']
    video_path = os.path.join('data/nturgbd_videos', frame_dir + '.mp4')
    if os.path.exists(video_path):
        success += 1
    else:
        print(f"✗ 缺失: {frame_dir}.mp4")

print(f"\n匹配率: {success}/10")
if success == 10:
    print("✓ 所有视频都匹配！")
else:
    print("⚠️ 有视频缺失，请检查命名")
EOF

# ===== 第4步：测试数据加载 =====
echo ""
echo "测试数据加载..."
python tools/analysis_tools/browse_dataset.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    --mode pipeline \
    --show-number 1 2>&1 | head -50

# 如果没报错，说明数据加载成功！
```

---

## 🎯 重新训练

修复后重新运行：

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 单GPU测试（先验证能跑通）
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    --work-dir work_dirs/pcan_ntu60_test

# 如果单GPU能跑，再用4GPU
bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    4
```

---

## 📊 问题总结

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `TypeError: list indices...` | pkl是列表不是字典 | 转换为标准格式 |
| 视频找不到 | frame_dir和视频名不匹配 | 统一命名（加或去_rgb） |
| split未指定 | dataloader缺少split参数 | 添加`split='xsub_train'` |

---

## ✅ 执行这个一键修复脚本

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 保存为 fix_ntu60_data.sh
cat > fix_ntu60_data.sh << 'SCRIPT'
#!/bin/bash

echo "=========================================="
echo "修复NTU-60数据格式"
echo "=========================================="

cd /home/zh/ChCode/codes01/mmaction2

# 转换pkl格式
python << 'EOF'
import pickle
import os

print("1. 加载pkl文件...")
with open('data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('data/skeleton/ntu60_xsub_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

print(f"   训练集: {len(train_data)}")
print(f"   验证集: {len(val_data)}")

print("\n2. 统一命名（添加_rgb）...")
for item in train_data + val_data:
    if not item['frame_dir'].endswith('_rgb'):
        item['frame_dir'] = item['frame_dir'] + '_rgb'

print("\n3. 转换为标准格式...")
data = {
    'split': {
        'xsub_train': [item['frame_dir'] for item in train_data],
        'xsub_val': [item['frame_dir'] for item in val_data]
    },
    'annotations': train_data + val_data
}

print("\n4. 保存...")
with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✓ 完成！")
print(f"  文件: data/skeleton/ntu60_xsub.pkl")
print(f"  总样本: {len(data['annotations'])}")

# 验证
sample = data['annotations'][0]
video_path = os.path.join('data/nturgbd_videos', sample['frame_dir'] + '.mp4')
if os.path.exists(video_path):
    print(f"✓ 视频匹配测试通过")
else:
    print(f"✗ 警告：{video_path} 不存在")
EOF

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
SCRIPT

# 运行
chmod +x fix_ntu60_data.sh
./fix_ntu60_data.sh
```

运行这个脚本后，**再重新训练**应该就可以了！

把脚本运行结果告诉我，如果还有问题我们继续调试！🚀

```shellscript
python << 'EOF'
import pickle

# 检查train pkl
with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f"数据类型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"✓ 格式正确 - 字典")
        print(f"  Keys: {data.keys()}")
    elif isinstance(data, list):
        print(f"✗ 格式错误 - 列表")
        print(f"  需要转换为字典格式")
        print(f"  列表长度: {len(data)}")
EOF
```

```shellscript
python << 'EOF'
import pickle

# 加载列表格式的数据
with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

print(f"训练集样本数: {len(train_data)}")
print(f"验证集样本数: {len(val_data)}")

# 转换为标准格式
standard_format = {
    'split': {
        'xsub_train': [item['frame_dir'] for item in train_data],
        'xsub_val': [item['frame_dir'] for item in val_data]
    },
    'annotations': train_data + val_data  # 合并所有标注
}

# 保存为标准格式
with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl', 'wb') as f:
    pickle.dump(standard_format, f)

print("✓ 转换完成！")
print(f"  总样本数: {len(standard_format['annotations'])}")
print(f"  Splits: {standard_format['split'].keys()}")
print(f"  训练集: {len(standard_format['split']['xsub_train'])}个")
print(f"  验证集: {len(standard_format['split']['xsub_val'])}个")
EOF
```

```python
# pcan_ntu60.py 中修改：
ann_file = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl'
ann_file_val = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl'
ann_file_test = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl'

# dataloader中需要指定split
train_dataloader = dict(
    # ... other config
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        split='xsub_train',  # ← 指定使用哪个split
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        split='xsub_val',  # ← 指定split
        pipeline=val_pipeline,
        test_mode=True))
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2/data/nturgbd_videos

# 批量去除_rgb后缀
for file in *_rgb.mp4; do
    newname="${file/_rgb.mp4/.mp4}"
    mv "$file" "$newname"
done

# 验证
ls | head -5
# 应该看到：S001C001P001R001A001.mp4
```

```shellscript
python << 'EOF'
import pickle

# 加载pkl
with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl', 'rb') as f:
    data = pickle.load(f)

# 给所有frame_dir添加_rgb后缀
for ann in data['annotations']:
    if not ann['frame_dir'].endswith('_rgb'):
        ann['frame_dir'] = ann['frame_dir'] + '_rgb'

# 保存
with open('/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub_merged.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✓ 已添加_rgb后缀")
EOF
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 1. 检查格式
python << 'EOF'
import pickle

files = {
    'train': 'data/skeleton/ntu60_xsub_train.pkl',
    'val': 'data/skeleton/ntu60_xsub_val.pkl'
}

for name, path in files.items():
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"{name}: type={type(data)}, ", end='')
    if isinstance(data, list):
        print(f"len={len(data)}")
    elif isinstance(data, dict):
        print(f"keys={data.keys()}")
EOF
```

```shellscript
python << 'EOF'
import pickle

print("转换pkl格式...")

# 加载分离的pkl
with open('data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
    
with open('data/skeleton/ntu60_xsub_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

# 确保是列表
if not isinstance(train_data, list):
    print("✗ train数据不是列表格式")
    exit(1)
if not isinstance(val_data, list):
    print("✗ val数据不是列表格式")
    exit(1)

print(f"训练集: {len(train_data)}个样本")
print(f"验证集: {len(val_data)}个样本")

# 检查是否需要添加_rgb后缀
sample = train_data[0]
print(f"示例frame_dir: {sample['frame_dir']}")

# 检查视频文件格式
import os
videos = os.listdir('data/nturgbd_videos/')
has_rgb_suffix = any('_rgb.mp4' in v for v in videos)
print(f"视频有_rgb后缀: {has_rgb_suffix}")

# 如果视频有_rgb后缀，给frame_dir也加上
if has_rgb_suffix:
    print("添加_rgb后缀到frame_dir...")
    for item in train_data + val_data:
        if not item['frame_dir'].endswith('_rgb'):
            item['frame_dir'] = item['frame_dir'] + '_rgb'

# 转换为标准格式
standard_format = {
    'split': {
        'xsub_train': [item['frame_dir'] for item in train_data],
        'xsub_val': [item['frame_dir'] for item in val_data]
    },
    'annotations': train_data + val_data
}

# 保存
output_path = 'data/skeleton/ntu60_xsub.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(standard_format, f)

print(f"\n✓ 转换完成！保存到: {output_path}")
print(f"  格式: dict with keys {list(standard_format.keys())}")
print(f"  Splits: {list(standard_format['split'].keys())}")
print(f"  总样本数: {len(standard_format['annotations'])}")
print(f"  训练集: {len(standard_format['split']['xsub_train'])}个")
print(f"  验证集: {len(standard_format['split']['xsub_val'])}个")

# 验证
sample = standard_format['annotations'][0]
print(f"\n示例数据:")
print(f"  frame_dir: {sample['frame_dir']}")
print(f"  label: {sample['label']}")
print(f"  keypoint shape: {sample['keypoint'].shape}")
EOF
```

```python
# ==========================================
# 数据配置
# ==========================================
dataset_type = 'PoseDataset'
data_root = '/home/zh/ChCode/codes01/mmaction2/data/nturgbd_videos/'

# ← 改：使用合并后的pkl文件
ann_file = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
ann_file_val = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
ann_file_test = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'

# ... 其他配置 ...

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        split='xsub_train',  # ← 必须指定split
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        split='xsub_val',  # ← 必须指定split
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        split='xsub_val',  # ← 必须指定split
        pipeline=test_pipeline,
        test_mode=True))
```

```shellscript
# 运行转换脚本后，验证新pkl
python << 'EOF'
import pickle
import os

pkl_path = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("="*60)
print("PKL格式验证")
print("="*60)
print(f"✓ 类型: {type(data)}")
print(f"✓ Keys: {list(data.keys())}")
print(f"✓ Splits: {list(data['split'].keys())}")
print(f"✓ 总样本: {len(data['annotations'])}")

# 检查视频匹配
sample = data['annotations'][0]
frame_dir = sample['frame_dir']
print(f"\n示例frame_dir: {frame_dir}")

# 构建视频路径
video_path = os.path.join('data/nturgbd_videos', frame_dir + '.mp4')
if os.path.exists(video_path):
    print(f"✓ 视频存在: {video_path}")
else:
    print(f"✗ 视频不存在: {video_path}")
    # 尝试查找
    videos = os.listdir('data/nturgbd_videos')
    # 去掉_rgb.mp4看看能不能匹配
    base_name = frame_dir.replace('_rgb', '')
    if base_name + '.mp4' in videos:
        print(f"  提示: 找到 {base_name}.mp4")
        print(f"  建议: frame_dir去掉_rgb后缀")
    elif frame_dir + '_rgb.mp4' in videos:
        print(f"  提示: 找到 {frame_dir}_rgb.mp4")
        print(f"  建议: 视频去掉_rgb后缀")
EOF
```

```shellscript
# 修改pkl，去掉_rgb
python << 'EOF'
import pickle

with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

# 去掉所有_rgb后缀
for ann in data['annotations']:
    ann['frame_dir'] = ann['frame_dir'].replace('_rgb', '')

# 更新split
data['split']['xsub_train'] = [s.replace('_rgb', '') for s in data['split']['xsub_train']]
data['split']['xsub_val'] = [s.replace('_rgb', '') for s in data['split']['xsub_val']]

# 保存
with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
    pickle.dump(data, f)
print("✓ 已去除_rgb后缀")
EOF
```

```shellscript
# 修改pkl，添加_rgb
python << 'EOF'
import pickle

with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

# 添加_rgb后缀
for ann in data['annotations']:
    if not ann['frame_dir'].endswith('_rgb'):
        ann['frame_dir'] = ann['frame_dir'] + '_rgb'

# 更新split
data['split']['xsub_train'] = [s if s.endswith('_rgb') else s+'_rgb' 
                                for s in data['split']['xsub_train']]
data['split']['xsub_val'] = [s if s.endswith('_rgb') else s+'_rgb' 
                              for s in data['split']['xsub_val']]

# 保存
with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
    pickle.dump(data, f)
print("✓ 已添加_rgb后缀")
EOF
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# ===== 第1步：检查pkl格式 =====
echo "检查pkl格式..."
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"类型: {type(data)}")
if isinstance(data, dict):
    print("✓ 已经是字典格式")
elif isinstance(data, list):
    print("✗ 是列表格式，需要转换")
EOF

# ===== 第2步：转换格式（如果需要）=====
echo ""
echo "转换pkl格式..."
python << 'EOF'
import pickle
import os

# 加载
with open('data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('data/skeleton/ntu60_xsub_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

# 检查视频命名
videos = os.listdir('data/nturgbd_videos/')
has_rgb_suffix = videos[0].endswith('_rgb.mp4')

print(f"视频示例: {videos[0]}")
print(f"视频有_rgb后缀: {has_rgb_suffix}")
print(f"骨架示例: {train_data[0]['frame_dir']}")

# 统一命名：都加_rgb（因为视频有_rgb）
for item in train_data + val_data:
    if not item['frame_dir'].endswith('_rgb'):
        item['frame_dir'] = item['frame_dir'] + '_rgb'

# 标准格式
data = {
    'split': {
        'xsub_train': [item['frame_dir'] for item in train_data],
        'xsub_val': [item['frame_dir'] for item in val_data]
    },
    'annotations': train_data + val_data
}

# 保存
with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"✓ 保存到: data/skeleton/ntu60_xsub.pkl")
print(f"  训练集: {len(data['split']['xsub_train'])}")
print(f"  验证集: {len(data['split']['xsub_val'])}")
EOF

# ===== 第3步：验证视频匹配 =====
echo ""
echo "验证视频和骨架匹配..."
python << 'EOF'
import pickle
import os

with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

# 检查前10个样本
success = 0
for i in range(min(10, len(data['annotations']))):
    frame_dir = data['annotations'][i]['frame_dir']
    video_path = os.path.join('data/nturgbd_videos', frame_dir + '.mp4')
    if os.path.exists(video_path):
        success += 1
    else:
        print(f"✗ 缺失: {frame_dir}.mp4")

print(f"\n匹配率: {success}/10")
if success == 10:
    print("✓ 所有视频都匹配！")
else:
    print("⚠️ 有视频缺失，请检查命名")
EOF

# ===== 第4步：测试数据加载 =====
echo ""
echo "测试数据加载..."
python tools/analysis_tools/browse_dataset.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    --mode pipeline \
    --show-number 1 2>&1 | head -50

# 如果没报错，说明数据加载成功！
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 单GPU测试（先验证能跑通）
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    --work-dir work_dirs/pcan_ntu60_test

# 如果单GPU能跑，再用4GPU
bash tools/dist_train.sh \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    4
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 保存为 fix_ntu60_data.sh
cat > fix_ntu60_data.sh << 'SCRIPT'
#!/bin/bash

echo "=========================================="
echo "修复NTU-60数据格式"
echo "=========================================="

cd /home/zh/ChCode/codes01/mmaction2

# 转换pkl格式
python << 'EOF'
import pickle
import os

print("1. 加载pkl文件...")
with open('data/skeleton/ntu60_xsub_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('data/skeleton/ntu60_xsub_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

print(f"   训练集: {len(train_data)}")
print(f"   验证集: {len(val_data)}")

print("\n2. 统一命名（添加_rgb）...")
for item in train_data + val_data:
    if not item['frame_dir'].endswith('_rgb'):
        item['frame_dir'] = item['frame_dir'] + '_rgb'

print("\n3. 转换为标准格式...")
data = {
    'split': {
        'xsub_train': [item['frame_dir'] for item in train_data],
        'xsub_val': [item['frame_dir'] for item in val_data]
    },
    'annotations': train_data + val_data
}

print("\n4. 保存...")
with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✓ 完成！")
print(f"  文件: data/skeleton/ntu60_xsub.pkl")
print(f"  总样本: {len(data['annotations'])}")

# 验证
sample = data['annotations'][0]
video_path = os.path.join('data/nturgbd_videos', sample['frame_dir'] + '.mp4')
if os.path.exists(video_path):
    print(f"✓ 视频匹配测试通过")
else:
    print(f"✗ 警告：{video_path} 不存在")
EOF

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
SCRIPT

# 运行
chmod +x fix_ntu60_data.sh
./fix_ntu60_data.sh
```

---

**User**

新的错误 Traceback (most recent call last):

  File "tools/train.py", line 143, in <module>

    main()

  File "tools/train.py", line 139, in main

    runner.train()

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/runner.py", line 1777, in train

    model = self.train_loop.run()  # type: ignore

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py", line 98, in run

    self.run_epoch()

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py", line 115, in run_epoch

    self.run_iter(idx, data_batch)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py", line 131, in run_iter

    outputs = self.runner.model.train_step(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/model/base_model/base_model.py", line 114, in train_step

    losses = self._run_forward(data, mode='loss')  # type: ignore

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/model/base_model/base_model.py", line 361, in _run_forward

    results = self(**data, mode=mode)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl

    return forward_call(*args, **kwargs)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/recognizers/base.py", line 262, in forward

    s=self.loss(inputs, data_samples, **kwargs)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/recognizers/base.py", line 176, in loss

    loss_cls = self.cls_head.loss(feats, data_samples, **loss_kwargs)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/heads/rgbpose_head.py", line 690, in loss

    return self.loss_by_feat(cls_scores, data_samples)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/heads/rgbpose_head.py", line 725, in loss_by_feat

    labels_body = labels.cpu().numpy()

RuntimeError: CUDA error: device-side assert triggered

CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.

For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.    以上这个错误是我执行CUDA_VISIBLE_DEVICES=1 python tools/train.py \

    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \

    --work-dir work_dirs/pcan_ntu60_test   因为目前卡0 有人用所以我换成了卡1  这是什么原因给报错啊

---

**Cursor**

**CUDA错误通常是标签越界！** 让我帮你诊断问题所在。

```python
labels_body = labels.cpu().numpy()
labels_body = np.array([action2body(i) for i in labels_body])
labels_body = torch.tensor(labels_body).cuda()
```

**找到问题了！** 这是典型的**标签越界**错误。让我帮你定位具体原因：

## 🔍 问题诊断

错误发生在第696-698行，调用`action2body`函数时：

```python
labels_body = labels.cpu().numpy()
labels_body = np.array([action2body(i) for i in labels_body])  # ← 这里！
```

**问题**：`action2body`函数是为MA-52设计的（处理0-51标签），但NTU-60的标签可能是：
- **1-60**（原始NTU标签）→ 需要减1变成0-59
- **0-59**（已转换） → 但`action2body`不认识这些值

## 🔧 快速诊断：检查标签范围

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 运行这个检查NTU-60标签的实际范围
CUDA_LAUNCH_BLOCKING=1 python << 'EOF'
import pickle
import numpy as np

# 加载NTU-60数据
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

# 检查所有标签
labels = [ann['label'] for ann in data['annotations']]
print(f"总样本数: {len(labels)}")
print(f"标签范围: [{min(labels)}, {max(labels)}]")
print(f"唯一标签数: {len(set(labels))}")

# 检查是否有越界
if min(labels) < 0:
    print(f"✗ 错误：有负数标签！")
if max(labels) > 59:
    print(f"✗ 错误：标签超过59！")
    print(f"  最大标签是: {max(labels)}")
    print(f"  这说明标签是1-60，需要减1")
elif max(labels) == 59:
    print(f"✓ 标签范围正确 (0-59)")
    
# 统计分布
from collections import Counter
label_dist = Counter(labels)
print(f"\n标签分布（前10个）:")
for label, count in sorted(label_dist.items())[:10]:
    print(f"  标签{label}: {count}个样本")
EOF
```

## 💡 解决方案（根据诊断结果）

### 情况A：如果标签是1-60（需要减1）

修改pkl文件：

```bash
python << 'EOF'
import pickle

print("修正NTU-60标签（1-60 → 0-59）...")

# 加载
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

# 检查并修正标签
labels_before = [ann['label'] for ann in data['annotations']]
print(f"修正前: 标签范围 [{min(labels_before)}, {max(labels_before)}]")

if min(labels_before) == 1:  # 标签从1开始
    print("✓ 检测到标签从1开始，执行减1操作...")
    for ann in data['annotations']:
        ann['label'] = ann['label'] - 1
    
    labels_after = [ann['label'] for ann in data['annotations']]
    print(f"修正后: 标签范围 [{min(labels_after)}, {max(labels_after)}]")
    
    # 保存
    with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("✓ 已保存修正后的pkl")
else:
    print("✓ 标签已经从0开始，无需修正")
EOF
```

### 情况B：如果标签是0-59（action2body函数问题）

修改 `mmaction/models/heads/rgbpose_head.py` 第17-31行：

```python
def action2body(x):
    """
    MA-52的action到body映射（0-51 → 0-6）
    """
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

def action2body_ntu60(x):
    """
    NTU-60的action到body映射（0-59 → 0-7）
    按每8个动作一组
    """
    if x < 0 or x > 59:
        print(f"警告：标签{x}越界！")
        return 0
    return min(x // 8, 7)  # 0-7, 8-15, ..., 56-59 → 0-7
```

然后修改第696-698行，使用NTU-60的映射：

```python
# 第696-698行修改为：
labels_body = labels.cpu().numpy()

# 根据最大标签判断使用哪个映射
max_label = labels_body.max()
if max_label >= 52:  # NTU-60
    labels_body = np.array([action2body_ntu60(i) for i in labels_body])
else:  # MA-52
    labels_body = np.array([action2body(i) for i in labels_body])

labels_body = torch.tensor(labels_body).cuda()
```

### 情况C：F.one_hot的num_classes错误

问题可能在第738行：

```python
labels = F.one_hot(labels, num_classes=self.num_classes)
```

如果`self.num_classes=60`但标签是1-60，会越界！

---

## 🚀 完整的修复脚本

```bash
cd /home/zh/ChCode/codes01/mmaction2

# ===== 第1步：诊断标签范围 =====
echo "=========================================="
echo "诊断标签范围"
echo "=========================================="

CUDA_LAUNCH_BLOCKING=1 python << 'EOF'
import pickle
import numpy as np

with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

labels = [ann['label'] for ann in data['annotations']]
print(f"标签范围: [{min(labels)}, {max(labels)}]")
print(f"唯一标签数: {len(set(labels))}")

if min(labels) == 0 and max(labels) == 59:
    print("✓ 标签格式正确 (0-59)")
    print("  问题在action2body函数")
elif min(labels) == 1 and max(labels) == 60:
    print("✗ 标签需要减1 (当前1-60，应该0-59)")
    print("  执行步骤2的修正脚本")
else:
    print(f"⚠️ 异常标签范围")
EOF

# ===== 第2步：修正标签（如果需要）=====
echo ""
echo "修正标签范围..."

python << 'EOF'
import pickle

with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

labels = [ann['label'] for ann in data['annotations']]
min_label, max_label = min(labels), max(labels)

if min_label == 1:
    print(f"修正标签: {min_label}-{max_label} → {min_label-1}-{max_label-1}")
    for ann in data['annotations']:
        ann['label'] = ann['label'] - 1
    
    with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("✓ 已保存")
else:
    print("✓ 标签无需修正")
EOF

# ===== 第3步：验证修正 =====
echo ""
echo "验证..."

python << 'EOF'
import pickle
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)
labels = [ann['label'] for ann in data['annotations']]
print(f"最终标签范围: [{min(labels)}, {max(labels)}]")
if min(labels) == 0 and max(labels) == 59:
    print("✓ 标签格式正确！")
else:
    print(f"✗ 仍有问题")
EOF

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
```

---

## 📝 修改代码文件

### 修改1：`mmaction/models/heads/rgbpose_head.py`

在第17行后添加NTU-60映射函数：

```python
def action2body(x):
    """MA-52: 0-51 → 0-6"""
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

def action2body_ntu60(x):
    """
    NTU-60: 0-59 → 0-7
    每8个动作一组
    """
    if x < 0 or x > 59:
        # 异常标签，返回0避免崩溃
        return 0
    return min(x // 8, 7)  # 0-7, 8-15, ..., 56-59 → 类别0-7
```

然后修改第696-698行：

```python
# 原代码（第696-698行）：
labels_body = labels.cpu().numpy()
labels_body = np.array([action2body(i) for i in labels_body])
labels_body = torch.tensor(labels_body).cuda()

# 修改为：
labels_body = labels.cpu().numpy()

# 根据标签范围判断数据集
max_label = int(labels_body.max())
if max_label >= 52:  # NTU-60 (0-59)
    labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
else:  # MA-52 (0-51)
    labels_body = np.array([action2body(int(i)) for i in labels_body])

labels_body = torch.tensor(labels_body, dtype=torch.long).cuda()
```

### 修改2：`mmaction/models/recognizers/recognizer3d_mm.py`

同样在第10行后添加NTU-60映射：

```python
def fine2coarse(x):
    """MA-52映射"""
    if x<=4:
        return 0
    elif 5<=x<=10:
        return 1
    elif 11<=x<=23:
        return 2
    elif 24<=x<=31:
        return 3
    elif 32<=x<=37:
        return 4
    elif 38<=x<=47:
        return 5
    else:
        return 6

def fine2coarse_ntu60(x):
    """NTU-60映射：0-59 → 0-7"""
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

然后修改第62行：

```python
# 原代码（第62行）：
gts_coarse=[fine2coarse(i) for i in temp]

# 修改为：
max_label = int(temp.max())
if max_label >= 52:  # NTU-60
    gts_coarse = [fine2coarse_ntu60(int(i)) for i in temp]
else:  # MA-52
    gts_coarse = [fine2coarse(int(i)) for i in temp]
```

---

## 🎯 完整的代码修改清单

由于你在ask模式，我把需要修改的代码都列出来，你手动修改：

### 文件1：`mmaction/models/heads/rgbpose_head.py`

**位置1**：第17行后添加

```python
def action2body_ntu60(x):
    """NTU-60: 0-59 → 0-7, 每8个动作一组"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

**位置2**：第696-698行修改

```python
# 原代码：
labels_body = labels.cpu().numpy()
labels_body = np.array([action2body(i) for i in labels_body])
labels_body = torch.tensor(labels_body).cuda()

# 改为：
labels_body = labels.cpu().numpy()
max_label = int(labels_body.max())
if max_label >= 52:
    labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
else:
    labels_body = np.array([action2body(int(i)) for i in labels_body])
labels_body = torch.tensor(labels_body, dtype=torch.long).cuda()
```

### 文件2：`mmaction/models/recognizers/recognizer3d_mm.py`

**位置1**：第24行后添加

```python
def fine2coarse_ntu60(x):
    """NTU-60: 0-59 → 0-7"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

**位置2**：第62行修改

```python
# 原代码：
gts_coarse=[fine2coarse(i) for i in temp]

# 改为：
max_label = int(temp.max())
if max_label >= 52:
    gts_coarse = [fine2coarse_ntu60(int(i)) for i in temp]
else:
    gts_coarse = [fine2coarse(int(i)) for i in temp]
```

---

## 🔄 修复后重新训练

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 先确保标签修正了
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)
labels = [ann['label'] for ann in data['annotations']]
print(f"标签范围: [{min(labels)}, {max(labels)}]")
assert min(labels) == 0 and max(labels) == 59, "标签范围不对！"
print("✓ 标签检查通过")
EOF

# 然后重新训练（使用CUDA_LAUNCH_BLOCKING=1便于调试）
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python tools/train.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    --work-dir work_dirs/pcan_ntu60_test
```

---

## 🎯 如果还报错，用这个超详细诊断

```bash
# 创建诊断脚本
cat > diagnose_labels.py << 'EOF'
import pickle
import numpy as np
import torch

print("="*60)
print("NTU-60标签诊断")
print("="*60)

# 1. 检查pkl
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

labels = np.array([ann['label'] for ann in data['annotations']])
print(f"\n1. 标签统计:")
print(f"   范围: [{labels.min()}, {labels.max()}]")
print(f"   唯一值: {len(np.unique(labels))}")
print(f"   数据类型: {labels.dtype}")

# 2. 测试action2body_ntu60
print(f"\n2. 测试粗分类映射:")
def action2body_ntu60(x):
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)

test_labels = [0, 10, 20, 30, 40, 50, 59]
for lbl in test_labels:
    coarse = action2body_ntu60(lbl)
    print(f"   {lbl} → {coarse}")

# 3. 测试所有标签的粗分类
coarse_labels = np.array([action2body_ntu60(int(i)) for i in labels])
print(f"\n3. 粗分类标签:")
print(f"   范围: [{coarse_labels.min()}, {coarse_labels.max()}]")
print(f"   唯一值: {len(np.unique(coarse_labels))}")

# 4. 测试one_hot
print(f"\n4. 测试one_hot:")
try:
    labels_tensor = torch.tensor(labels[:10], dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes=60)
    print(f"   ✓ one_hot成功: {one_hot.shape}")
except Exception as e:
    print(f"   ✗ one_hot失败: {e}")

# 5. 测试粗分类one_hot
print(f"\n5. 测试粗分类one_hot:")
try:
    coarse_tensor = torch.tensor(coarse_labels[:10], dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(coarse_tensor, num_classes=8)
    print(f"   ✓ one_hot成功: {one_hot.shape}")
except Exception as e:
    print(f"   ✗ one_hot失败: {e}")

print("\n" + "="*60)
print("诊断完成！")
print("="*60)
EOF

python diagnose_labels.py
```

---

## 📋 总结：需要做的修改

### ✅ 必须执行

1. **修正标签范围**（如果是1-60）
2. **添加`action2body_ntu60`函数**到两个文件
3. **修改两处调用**使用NTU-60的映射

### ✅ 配置文件确认

确保 `pcan_ntu60.py` 中：

```python
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← 确认是60
    num_coarse_classes=8,  # ← 添加这个！NTU-60用8个粗类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose', 'rgb_coarse', 'pose_coarse'],
    loss_weights=[1., 1., 0.5, 0.5],
    average_clips='prob')
```

---

## 🎯 快速测试checklist

修改完成后，按顺序检查：

```bash
# ✓ 1. 标签范围
python -c "import pickle; data=pickle.load(open('data/skeleton/ntu60_xsub.pkl','rb')); labels=[a['label'] for a in data['annotations']]; print(f'[{min(labels)}, {max(labels)}]')"
# 应该输出: [0, 59]

# ✓ 2. pkl格式
python -c "import pickle; data=pickle.load(open('data/skeleton/ntu60_xsub.pkl','rb')); print(type(data), list(data.keys()))"
# 应该输出: <class 'dict'> ['split', 'annotations']

# ✓ 3. split存在
python -c "import pickle; data=pickle.load(open('data/skeleton/ntu60_xsub.pkl','rb')); print(list(data['split'].keys()))"
# 应该输出: ['xsub_train', 'xsub_val']

# ✓ 4. 视频匹配
python -c "import pickle,os; data=pickle.load(open('data/skeleton/ntu60_xsub.pkl','rb')); f=data['annotations'][0]['frame_dir']; print('存在' if os.path.exists(f'data/nturgbd_videos/{f}.mp4') else '不存在')"
# 应该输出: 存在
```

全部通过后，再运行训练！

先运行这些检查，把结果告诉我，我们继续调试！🔧

```python
labels_body = labels.cpu().numpy()
labels_body = np.array([action2body(i) for i in labels_body])  # ← 这里！
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 运行这个检查NTU-60标签的实际范围
CUDA_LAUNCH_BLOCKING=1 python << 'EOF'
import pickle
import numpy as np

# 加载NTU-60数据
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

# 检查所有标签
labels = [ann['label'] for ann in data['annotations']]
print(f"总样本数: {len(labels)}")
print(f"标签范围: [{min(labels)}, {max(labels)}]")
print(f"唯一标签数: {len(set(labels))}")

# 检查是否有越界
if min(labels) < 0:
    print(f"✗ 错误：有负数标签！")
if max(labels) > 59:
    print(f"✗ 错误：标签超过59！")
    print(f"  最大标签是: {max(labels)}")
    print(f"  这说明标签是1-60，需要减1")
elif max(labels) == 59:
    print(f"✓ 标签范围正确 (0-59)")
    
# 统计分布
from collections import Counter
label_dist = Counter(labels)
print(f"\n标签分布（前10个）:")
for label, count in sorted(label_dist.items())[:10]:
    print(f"  标签{label}: {count}个样本")
EOF
```

```shellscript
python << 'EOF'
import pickle

print("修正NTU-60标签（1-60 → 0-59）...")

# 加载
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

# 检查并修正标签
labels_before = [ann['label'] for ann in data['annotations']]
print(f"修正前: 标签范围 [{min(labels_before)}, {max(labels_before)}]")

if min(labels_before) == 1:  # 标签从1开始
    print("✓ 检测到标签从1开始，执行减1操作...")
    for ann in data['annotations']:
        ann['label'] = ann['label'] - 1
    
    labels_after = [ann['label'] for ann in data['annotations']]
    print(f"修正后: 标签范围 [{min(labels_after)}, {max(labels_after)}]")
    
    # 保存
    with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("✓ 已保存修正后的pkl")
else:
    print("✓ 标签已经从0开始，无需修正")
EOF
```

```python
def action2body(x):
    """
    MA-52的action到body映射（0-51 → 0-6）
    """
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

def action2body_ntu60(x):
    """
    NTU-60的action到body映射（0-59 → 0-7）
    按每8个动作一组
    """
    if x < 0 or x > 59:
        print(f"警告：标签{x}越界！")
        return 0
    return min(x // 8, 7)  # 0-7, 8-15, ..., 56-59 → 0-7
```

```python
# 第696-698行修改为：
labels_body = labels.cpu().numpy()

# 根据最大标签判断使用哪个映射
max_label = labels_body.max()
if max_label >= 52:  # NTU-60
    labels_body = np.array([action2body_ntu60(i) for i in labels_body])
else:  # MA-52
    labels_body = np.array([action2body(i) for i in labels_body])

labels_body = torch.tensor(labels_body).cuda()
```

```python
labels = F.one_hot(labels, num_classes=self.num_classes)
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# ===== 第1步：诊断标签范围 =====
echo "=========================================="
echo "诊断标签范围"
echo "=========================================="

CUDA_LAUNCH_BLOCKING=1 python << 'EOF'
import pickle
import numpy as np

with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

labels = [ann['label'] for ann in data['annotations']]
print(f"标签范围: [{min(labels)}, {max(labels)}]")
print(f"唯一标签数: {len(set(labels))}")

if min(labels) == 0 and max(labels) == 59:
    print("✓ 标签格式正确 (0-59)")
    print("  问题在action2body函数")
elif min(labels) == 1 and max(labels) == 60:
    print("✗ 标签需要减1 (当前1-60，应该0-59)")
    print("  执行步骤2的修正脚本")
else:
    print(f"⚠️ 异常标签范围")
EOF

# ===== 第2步：修正标签（如果需要）=====
echo ""
echo "修正标签范围..."

python << 'EOF'
import pickle

with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

labels = [ann['label'] for ann in data['annotations']]
min_label, max_label = min(labels), max(labels)

if min_label == 1:
    print(f"修正标签: {min_label}-{max_label} → {min_label-1}-{max_label-1}")
    for ann in data['annotations']:
        ann['label'] = ann['label'] - 1
    
    with open('data/skeleton/ntu60_xsub.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("✓ 已保存")
else:
    print("✓ 标签无需修正")
EOF

# ===== 第3步：验证修正 =====
echo ""
echo "验证..."

python << 'EOF'
import pickle
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)
labels = [ann['label'] for ann in data['annotations']]
print(f"最终标签范围: [{min(labels)}, {max(labels)}]")
if min(labels) == 0 and max(labels) == 59:
    print("✓ 标签格式正确！")
else:
    print(f"✗ 仍有问题")
EOF

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
```

```python
def action2body(x):
    """MA-52: 0-51 → 0-6"""
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

def action2body_ntu60(x):
    """
    NTU-60: 0-59 → 0-7
    每8个动作一组
    """
    if x < 0 or x > 59:
        # 异常标签，返回0避免崩溃
        return 0
    return min(x // 8, 7)  # 0-7, 8-15, ..., 56-59 → 类别0-7
```

```python
# 原代码（第696-698行）：
labels_body = labels.cpu().numpy()
labels_body = np.array([action2body(i) for i in labels_body])
labels_body = torch.tensor(labels_body).cuda()

# 修改为：
labels_body = labels.cpu().numpy()

# 根据标签范围判断数据集
max_label = int(labels_body.max())
if max_label >= 52:  # NTU-60 (0-59)
    labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
else:  # MA-52 (0-51)
    labels_body = np.array([action2body(int(i)) for i in labels_body])

labels_body = torch.tensor(labels_body, dtype=torch.long).cuda()
```

```python
def fine2coarse(x):
    """MA-52映射"""
    if x<=4:
        return 0
    elif 5<=x<=10:
        return 1
    elif 11<=x<=23:
        return 2
    elif 24<=x<=31:
        return 3
    elif 32<=x<=37:
        return 4
    elif 38<=x<=47:
        return 5
    else:
        return 6

def fine2coarse_ntu60(x):
    """NTU-60映射：0-59 → 0-7"""
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

```python
# 原代码（第62行）：
gts_coarse=[fine2coarse(i) for i in temp]

# 修改为：
max_label = int(temp.max())
if max_label >= 52:  # NTU-60
    gts_coarse = [fine2coarse_ntu60(int(i)) for i in temp]
else:  # MA-52
    gts_coarse = [fine2coarse(int(i)) for i in temp]
```

```python
def action2body_ntu60(x):
    """NTU-60: 0-59 → 0-7, 每8个动作一组"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

```python
# 原代码：
labels_body = labels.cpu().numpy()
labels_body = np.array([action2body(i) for i in labels_body])
labels_body = torch.tensor(labels_body).cuda()

# 改为：
labels_body = labels.cpu().numpy()
max_label = int(labels_body.max())
if max_label >= 52:
    labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
else:
    labels_body = np.array([action2body(int(i)) for i in labels_body])
labels_body = torch.tensor(labels_body, dtype=torch.long).cuda()
```

```python
def fine2coarse_ntu60(x):
    """NTU-60: 0-59 → 0-7"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

```python
# 原代码：
gts_coarse=[fine2coarse(i) for i in temp]

# 改为：
max_label = int(temp.max())
if max_label >= 52:
    gts_coarse = [fine2coarse_ntu60(int(i)) for i in temp]
else:
    gts_coarse = [fine2coarse(int(i)) for i in temp]
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 先确保标签修正了
python << 'EOF'
import pickle
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)
labels = [ann['label'] for ann in data['annotations']]
print(f"标签范围: [{min(labels)}, {max(labels)}]")
assert min(labels) == 0 and max(labels) == 59, "标签范围不对！"
print("✓ 标签检查通过")
EOF

# 然后重新训练（使用CUDA_LAUNCH_BLOCKING=1便于调试）
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python tools/train.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    --work-dir work_dirs/pcan_ntu60_test
```

```shellscript
# 创建诊断脚本
cat > diagnose_labels.py << 'EOF'
import pickle
import numpy as np
import torch

print("="*60)
print("NTU-60标签诊断")
print("="*60)

# 1. 检查pkl
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

labels = np.array([ann['label'] for ann in data['annotations']])
print(f"\n1. 标签统计:")
print(f"   范围: [{labels.min()}, {labels.max()}]")
print(f"   唯一值: {len(np.unique(labels))}")
print(f"   数据类型: {labels.dtype}")

# 2. 测试action2body_ntu60
print(f"\n2. 测试粗分类映射:")
def action2body_ntu60(x):
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)

test_labels = [0, 10, 20, 30, 40, 50, 59]
for lbl in test_labels:
    coarse = action2body_ntu60(lbl)
    print(f"   {lbl} → {coarse}")

# 3. 测试所有标签的粗分类
coarse_labels = np.array([action2body_ntu60(int(i)) for i in labels])
print(f"\n3. 粗分类标签:")
print(f"   范围: [{coarse_labels.min()}, {coarse_labels.max()}]")
print(f"   唯一值: {len(np.unique(coarse_labels))}")

# 4. 测试one_hot
print(f"\n4. 测试one_hot:")
try:
    labels_tensor = torch.tensor(labels[:10], dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes=60)
    print(f"   ✓ one_hot成功: {one_hot.shape}")
except Exception as e:
    print(f"   ✗ one_hot失败: {e}")

# 5. 测试粗分类one_hot
print(f"\n5. 测试粗分类one_hot:")
try:
    coarse_tensor = torch.tensor(coarse_labels[:10], dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(coarse_tensor, num_classes=8)
    print(f"   ✓ one_hot成功: {one_hot.shape}")
except Exception as e:
    print(f"   ✗ one_hot失败: {e}")

print("\n" + "="*60)
print("诊断完成！")
print("="*60)
EOF

python diagnose_labels.py
```

```python
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← 确认是60
    num_coarse_classes=8,  # ← 添加这个！NTU-60用8个粗类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose', 'rgb_coarse', 'pose_coarse'],
    loss_weights=[1., 1., 0.5, 0.5],
    average_clips='prob')
```

```shellscript
# ✓ 1. 标签范围
python -c "import pickle; data=pickle.load(open('data/skeleton/ntu60_xsub.pkl','rb')); labels=[a['label'] for a in data['annotations']]; print(f'[{min(labels)}, {max(labels)}]')"
# 应该输出: [0, 59]

# ✓ 2. pkl格式
python -c "import pickle; data=pickle.load(open('data/skeleton/ntu60_xsub.pkl','rb')); print(type(data), list(data.keys()))"
# 应该输出: <class 'dict'> ['split', 'annotations']

# ✓ 3. split存在
python -c "import pickle; data=pickle.load(open('data/skeleton/ntu60_xsub.pkl','rb')); print(list(data['split'].keys()))"
# 应该输出: ['xsub_train', 'xsub_val']

# ✓ 4. 视频匹配
python -c "import pickle,os; data=pickle.load(open('data/skeleton/ntu60_xsub.pkl','rb')); f=data['annotations'][0]['frame_dir']; print('存在' if os.path.exists(f'data/nturgbd_videos/{f}.mp4') else '不存在')"
# 应该输出: 存在
```

---

**User**

再次报错 Traceback (most recent call last):

  File "tools/train.py", line 143, in <module>

    main()

  File "tools/train.py", line 139, in main

    runner.train()

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/runner.py", line 1777, in train

    model = self.train_loop.run()  # type: ignore

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py", line 98, in run

    self.run_epoch()

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py", line 115, in run_epoch

    self.run_iter(idx, data_batch)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py", line 131, in run_iter

    outputs = self.runner.model.train_step(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/model/base_model/base_model.py", line 114, in train_step

    losses = self._run_forward(data, mode='loss')  # type: ignore

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/model/base_model/base_model.py", line 361, in _run_forward

    results = self(**data, mode=mode)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl

    return forward_call(*args, **kwargs)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/recognizers/base.py", line 262, in forward

    s=self.loss(inputs, data_samples, **kwargs)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/recognizers/base.py", line 176, in loss

    loss_cls = self.cls_head.loss(feats, data_samples, **loss_kwargs)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/heads/rgbpose_head.py", line 700, in loss

    return self.loss_by_feat(cls_scores, data_samples)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/heads/rgbpose_head.py", line 737, in loss_by_feat

    max_label = labels_body.max()

UnboundLocalError: local variable 'labels_body' referenced before assignment

---

**Cursor**

**好的！这是变量作用域问题。** 让我查看完整的`loss_by_feat`方法：

```python
max_label = int(labels_body.max())
if max_label >= 52:
    labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
```

**看到问题了！** 你把代码放错位置了。`labels_body`在`loss_by_feat`方法中定义（第696行），但你在`loss_by_scores`方法中使用了它（第737行）。

让我给你完整的正确修改：

## 📝 正确的修改位置

### `mmaction/models/heads/rgbpose_head.py` 完整修改

**第1处：在文件开头添加函数（第32行之后）**

```python
def action2body(x):
    """MA-52映射"""
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

def action2body_ntu60(x):
    """NTU-60映射：0-59 → 0-7"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

**第2处：修改loss_by_feat方法（第676-714行）**

找到整个`loss_by_feat`方法，完整替换为：

```python
def loss_by_feat(self, cls_scores: Dict[str, torch.Tensor],
                 data_samples: SampleList) -> Dict:
    """Calculate the loss based on the features extracted by the head."""
    labels = torch.stack([x.gt_labels.item for x in data_samples])
    labels = labels.squeeze()

    if labels.shape == torch.Size([]):
        labels = labels.unsqueeze(0)
    elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
            and cls_scores.size()[0] == 1:
        labels = labels.unsqueeze(0)

    losses = dict()
    
    # ========== 在循环前判断数据集类型 ==========
    max_label = int(labels.max().item())
    is_ntu60 = (max_label >= 52)  # 标签范围判断
    
    for loss_name, weight in zip(self.loss_components, self.loss_weights):
        cls_score1 = cls_scores[loss_name]
        loss_cls = self.loss_by_scores(cls_score1, labels)
        loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name}_loss_cls'] *= weight
        losses.update(loss_cls)

        # ========== 修改：使用正确的映射函数 ==========
        labels_body = labels.cpu().numpy()
        if is_ntu60:  # NTU-60
            labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
        else:  # MA-52
            labels_body = np.array([action2body(int(i)) for i in labels_body])
        labels_body = torch.tensor(labels_body, dtype=torch.long).cuda()

        cls_score2 = cls_scores[loss_name+'_coarse']
        loss_name = loss_name+'_coarse'
        loss_cls = self.loss_by_scores(cls_score2, labels_body)
        loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name}_loss_cls'] *= weight
        losses.update(loss_cls)

    if self.training:
        losses['rgb_fr_coarse_loss'] = cls_scores['fr_loss_rgb_coarse'] / 5
        losses['pose_fr_coarse_loss'] = cls_scores['fr_loss_pose_coarse'] / 5
        losses['rgb_fr_loss'] = cls_scores['fr_loss_rgb']/5
        losses['pose_fr_loss'] = cls_scores['fr_loss_pose']/5
        losses['hierarchy_rgb_loss'] = cls_scores['hierarchy_loss_rgb']
        losses['hierarchy_pose_loss'] = cls_scores['hierarchy_loss_pose']
    return losses
```

**注意**：`loss_by_scores`方法（第716-748行）**不需要修改**！

---

## 📝 `mmaction/models/recognizers/recognizer3d_mm.py` 修改

**第1处：添加函数（第24行之后）**

```python
def fine2coarse(x):
    """MA-52映射"""
    if x<=4:
        return 0
    elif 5<=x<=10:
        return 1
    elif 11<=x<=23:
        return 2
    elif 24<=x<=31:
        return 3
    elif 32<=x<=37:
        return 4
    elif 38<=x<=47:
        return 5
    else:
        return 6

def fine2coarse_ntu60(x):
    """NTU-60映射：0-59 → 0-7"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

**第2处：修改extract_feat方法（第56-65行）**

```python
# 原代码：
gts=[]
for data in data_samples:
    gts.extend(data.gt_labels.item)
gts=torch.stack(gts)
temp=gts.cpu().numpy()
gts_coarse=[fine2coarse(i) for i in temp]
gts_coarse=torch.from_numpy(np.array(gts_coarse)).cuda()

# 修改为：
gts=[]
for data in data_samples:
    gts.extend(data.gt_labels.item)
gts=torch.stack(gts)
temp=gts.cpu().numpy()

# 判断数据集类型
max_label = int(temp.max())
if max_label >= 52:  # NTU-60
    gts_coarse = [fine2coarse_ntu60(int(i)) for i in temp]
else:  # MA-52
    gts_coarse = [fine2coarse(int(i)) for i in temp]

gts_coarse=torch.from_numpy(np.array(gts_coarse)).cuda()
```

---

## 🔑 关键提醒：作用域问题

```python
# ❌ 错误的修改（你现在的情况）
def loss_by_feat(...):
    # ... 
    for loss_name in ...:
        labels_body = ...  # 在循环内定义
    
    # 在循环外使用 labels_body ← 错误！
    max_label = labels_body.max()  # UnboundLocalError

# ✓ 正确的修改
def loss_by_feat(...):
    labels = ...
    max_label = labels.max()  # ← 在循环前判断
    is_ntu60 = (max_label >= 52)
    
    for loss_name in ...:
        labels_body = ...
        if is_ntu60:  # ← 在循环内使用判断结果
            labels_body = action2body_ntu60(...)
        else:
            labels_body = action2body(...)
```

---

## 📋 完整的代码文件内容（直接复制）

### 文件1：`mmaction/models/heads/rgbpose_head.py`

在**第17-31行**，完整替换为：

```python
def action2body(x):
    """MA-52: action (0-51) → body (0-6)"""
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

def action2body_ntu60(x):
    """NTU-60: action (0-59) → body (0-7), 每8个动作一组"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

在**第663-714行**，完整替换`loss_by_feat`方法：

```python
def loss_by_feat(self, cls_scores: Dict[str, torch.Tensor],
                 data_samples: SampleList) -> Dict:
    """Calculate the loss based on the features extracted by the head.

    Args:
        cls_scores (dict[str, torch.Tensor]): The dict of
            classification scores,
        data_samples (list[:obj:`ActionDataSample`]): The batch
            data samples.

    Returns:
        dict: A dictionary of loss components.
    """
    labels = torch.stack([x.gt_labels.item for x in data_samples])
    labels = labels.squeeze()

    if labels.shape == torch.Size([]):
        labels = labels.unsqueeze(0)
    elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
            and cls_scores.size()[0] == 1:
        # Fix a bug when training with soft labels and batch size is 1.
        # When using soft labels, `labels` and `cls_score` share the same
        # shape.
        labels = labels.unsqueeze(0)

    # ========== 判断数据集（在循环前）==========
    max_label = int(labels.max().item())
    is_ntu60 = (max_label >= 52)  # NTU-60: 0-59, MA-52: 0-51
    
    losses = dict()
    for loss_name, weight in zip(self.loss_components, self.loss_weights):
        cls_score1 = cls_scores[loss_name]
        loss_cls = self.loss_by_scores(cls_score1, labels)
        loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name}_loss_cls'] *= weight
        losses.update(loss_cls)

        # ========== 计算粗分类标签 ==========
        labels_body = labels.cpu().numpy()
        if is_ntu60:
            labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
        else:
            labels_body = np.array([action2body(int(i)) for i in labels_body])
        labels_body = torch.tensor(labels_body, dtype=torch.long).cuda()

        cls_score2 = cls_scores[loss_name+'_coarse']
        loss_name = loss_name+'_coarse'
        loss_cls = self.loss_by_scores(cls_score2, labels_body)
        loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name}_loss_cls'] *= weight
        losses.update(loss_cls)

    if self.training:
        losses['rgb_fr_coarse_loss'] = cls_scores['fr_loss_rgb_coarse'] / 5
        losses['pose_fr_coarse_loss'] = cls_scores['fr_loss_pose_coarse'] / 5
        losses['rgb_fr_loss'] = cls_scores['fr_loss_rgb']/5
        losses['pose_fr_loss'] = cls_scores['fr_loss_pose']/5
        losses['hierarchy_rgb_loss'] = cls_scores['hierarchy_loss_rgb']
        losses['hierarchy_pose_loss'] = cls_scores['hierarchy_loss_pose']
    return losses
```

**第716-748行的`loss_by_scores`方法保持不变！**

---

### 文件2：`mmaction/models/recognizers/recognizer3d_mm.py`

在**第10-24行**，完整替换为：

```python
def fine2coarse(x):
    """MA-52: 0-51 → 0-6"""
    if x<=4:
        return 0
    elif 5<=x<=10:
        return 1
    elif 11<=x<=23:
        return 2
    elif 24<=x<=31:
        return 3
    elif 32<=x<=37:
        return 4
    elif 38<=x<=47:
        return 5
    else:
        return 6

def fine2coarse_ntu60(x):
    """NTU-60: 0-59 → 0-7"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

在**第56-65行**，修改为：

```python
#get gt_label
gts=[]
for data in data_samples:
    gts.extend(data.gt_labels.item)
gts=torch.stack(gts)
temp=gts.cpu().numpy()

# 判断数据集
max_label = int(temp.max())
if max_label >= 52:  # NTU-60
    gts_coarse = [fine2coarse_ntu60(int(i)) for i in temp]
else:  # MA-52
    gts_coarse = [fine2coarse(int(i)) for i in temp]

gts_coarse=torch.from_numpy(np.array(gts_coarse)).cuda()
```

---

## 🎯 检查你的修改是否正确

运行这个验证脚本：

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 检查rgbpose_head.py的修改
python << 'EOF'
import sys
sys.path.insert(0, '/home/zh/ChCode/codes01/mmaction2')

# 尝试导入函数
try:
    from mmaction.models.heads.rgbpose_head import action2body, action2body_ntu60
    print("✓ action2body函数导入成功")
    print("✓ action2body_ntu60函数导入成功")
    
    # 测试映射
    print("\nMA-52测试:")
    for x in [0, 5, 11, 24, 32, 38, 50]:
        print(f"  {x} → {action2body(x)}")
    
    print("\nNTU-60测试:")
    for x in [0, 8, 16, 24, 32, 40, 48, 56, 59]:
        print(f"  {x} → {action2body_ntu60(x)}")
        
except ImportError as e:
    print(f"✗ 导入失败: {e}")
except Exception as e:
    print(f"✗ 函数定义有问题: {e}")
EOF

# 检查recognizer3d_mm.py的修改
python << 'EOF'
import sys
sys.path.insert(0, '/home/zh/ChCode/codes01/mmaction2')

try:
    from mmaction.models.recognizers.recognizer3d_mm import fine2coarse, fine2coarse_ntu60
    print("\n✓ fine2coarse函数导入成功")
    print("✓ fine2coarse_ntu60函数导入成功")
    
    # 测试
    print("\nNTU-60测试:")
    for x in [0, 10, 20, 30, 40, 50, 59]:
        print(f"  {x} → {fine2coarse_ntu60(x)}")
        
except ImportError as e:
    print(f"✗ 导入失败: {e}")
except Exception as e:
    print(f"✗ 函数定义有问题: {e}")
EOF
```

---

## 🔍 如果还有问题，完整检查

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 创建完整的测试脚本
cat > test_ntu60_setup.py << 'EOF'
import torch
import pickle
import numpy as np
import sys
sys.path.insert(0, '/home/zh/ChCode/codes01/mmaction2')

print("="*60)
print("NTU-60完整检查")
print("="*60)

# 1. 检查pkl数据
print("\n1. 检查pkl数据...")
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

labels = [ann['label'] for ann in data['annotations']]
print(f"   标签范围: [{min(labels)}, {max(labels)}]")
print(f"   样本数: {len(labels)}")

if min(labels) != 0 or max(labels) != 59:
    print(f"   ✗ 错误：标签应该是0-59！")
    exit(1)
else:
    print(f"   ✓ 标签范围正确")

# 2. 检查函数导入
print("\n2. 检查函数导入...")
try:
    from mmaction.models.heads.rgbpose_head import action2body_ntu60
    from mmaction.models.recognizers.recognizer3d_mm import fine2coarse_ntu60
    print("   ✓ 所有函数导入成功")
except ImportError as e:
    print(f"   ✗ 导入失败: {e}")
    exit(1)

# 3. 测试映射函数
print("\n3. 测试映射函数...")
test_labels = np.array([0, 10, 20, 30, 40, 50, 59])
mapped = [action2body_ntu60(int(x)) for x in test_labels]
print(f"   Action: {test_labels.tolist()}")
print(f"   Body:   {mapped}")

if max(mapped) <= 7 and min(mapped) >= 0:
    print("   ✓ 映射结果在0-7范围内")
else:
    print(f"   ✗ 映射越界！")
    exit(1)

# 4. 模拟一个batch的处理
print("\n4. 模拟batch处理...")
batch_labels = torch.tensor([0, 15, 30, 45, 59], dtype=torch.long)
print(f"   模拟标签: {batch_labels.tolist()}")

max_label = int(batch_labels.max().item())
print(f"   最大标签: {max_label}")

is_ntu60 = (max_label >= 52)
print(f"   判定为NTU-60: {is_ntu60}")

labels_body = batch_labels.cpu().numpy()
if is_ntu60:
    labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
else:
    labels_body = np.array([action2body(int(i)) for i in labels_body])

print(f"   粗分类结果: {labels_body.tolist()}")
print(f"   粗分类范围: [{labels_body.min()}, {labels_body.max()}]")

# 5. 测试one_hot
print("\n5. 测试one_hot编码...")
try:
    labels_tensor = torch.tensor(labels_body, dtype=torch.long).cuda()
    one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes=8)
    print(f"   ✓ one_hot成功: {one_hot.shape}")
except Exception as e:
    print(f"   ✗ one_hot失败: {e}")
    exit(1)

print("\n" + "="*60)
print("✓ 所有检查通过！可以开始训练")
print("="*60)
EOF

# 运行测试
python test_ntu60_setup.py
```

---

## 🚀 验证通过后重新训练

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 清理之前的错误日志
rm -rf work_dirs/pcan_ntu60_test

# 重新训练（保持CUDA_LAUNCH_BLOCKING=1便于调试）
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python tools/train.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    --work-dir work_dirs/pcan_ntu60_test

# 如果看到正常的训练日志（loss下降），就成功了！
```

---

## 📊 预期的正常输出

修复后，你应该看到：

```
11/21 01:XX:XX - mmengine - INFO - Epoch(train) [1][10/XXXX]
    lr: 7.5000e-03
    eta: X:XX:XX
    time: X.XXXX
    data_time: X.XXXX
    memory: XXXX
    rgb_loss_cls: X.XXXX
    pose_loss_cls: X.XXXX
    rgb_coarse_loss_cls: X.XXXX  ← 粗分类损失
    pose_coarse_loss_cls: X.XXXX
    rgb_fr_loss: X.XXXX  ← 原型学习损失
    pose_fr_loss: X.XXXX
    hierarchy_rgb_loss: X.XXXX  ← 层次约束损失
    hierarchy_pose_loss: X.XXXX
```

---

## ⚠️ 重要提醒

1. **不要修改`loss_by_scores`方法**（第716-748行）
2. **只修改`loss_by_feat`方法**（第663-714行）
3. **在循环前判断数据集类型**（添加`is_ntu60`变量）
4. **在循环内使用`is_ntu60`**（不要在循环外访问`labels_body`）

---

## 🔧 如果还报错

把以下信息发给我：

```bash
# 1. 标签范围
python -c "import pickle; d=pickle.load(open('data/skeleton/ntu60_xsub.pkl','rb')); l=[a['label'] for a in d['annotations']]; print(f'[{min(l)}, {max(l)}]')"

# 2. 函数测试结果
python test_ntu60_setup.py 2>&1 | tail -20

# 3. 训练错误的完整堆栈
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python tools/train.py configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py --work-dir work_dirs/test 2>&1 | tail -50
```

先运行`test_ntu60_setup.py`检查，如果全部通过，就可以正常训练了！🚀

```python
def action2body(x):
    """MA-52映射"""
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

def action2body_ntu60(x):
    """NTU-60映射：0-59 → 0-7"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

```python
def loss_by_feat(self, cls_scores: Dict[str, torch.Tensor],
                 data_samples: SampleList) -> Dict:
    """Calculate the loss based on the features extracted by the head."""
    labels = torch.stack([x.gt_labels.item for x in data_samples])
    labels = labels.squeeze()

    if labels.shape == torch.Size([]):
        labels = labels.unsqueeze(0)
    elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
            and cls_scores.size()[0] == 1:
        labels = labels.unsqueeze(0)

    losses = dict()
    
    # ========== 在循环前判断数据集类型 ==========
    max_label = int(labels.max().item())
    is_ntu60 = (max_label >= 52)  # 标签范围判断
    
    for loss_name, weight in zip(self.loss_components, self.loss_weights):
        cls_score1 = cls_scores[loss_name]
        loss_cls = self.loss_by_scores(cls_score1, labels)
        loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name}_loss_cls'] *= weight
        losses.update(loss_cls)

        # ========== 修改：使用正确的映射函数 ==========
        labels_body = labels.cpu().numpy()
        if is_ntu60:  # NTU-60
            labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
        else:  # MA-52
            labels_body = np.array([action2body(int(i)) for i in labels_body])
        labels_body = torch.tensor(labels_body, dtype=torch.long).cuda()

        cls_score2 = cls_scores[loss_name+'_coarse']
        loss_name = loss_name+'_coarse'
        loss_cls = self.loss_by_scores(cls_score2, labels_body)
        loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name}_loss_cls'] *= weight
        losses.update(loss_cls)

    if self.training:
        losses['rgb_fr_coarse_loss'] = cls_scores['fr_loss_rgb_coarse'] / 5
        losses['pose_fr_coarse_loss'] = cls_scores['fr_loss_pose_coarse'] / 5
        losses['rgb_fr_loss'] = cls_scores['fr_loss_rgb']/5
        losses['pose_fr_loss'] = cls_scores['fr_loss_pose']/5
        losses['hierarchy_rgb_loss'] = cls_scores['hierarchy_loss_rgb']
        losses['hierarchy_pose_loss'] = cls_scores['hierarchy_loss_pose']
    return losses
```

```python
def fine2coarse(x):
    """MA-52映射"""
    if x<=4:
        return 0
    elif 5<=x<=10:
        return 1
    elif 11<=x<=23:
        return 2
    elif 24<=x<=31:
        return 3
    elif 32<=x<=37:
        return 4
    elif 38<=x<=47:
        return 5
    else:
        return 6

def fine2coarse_ntu60(x):
    """NTU-60映射：0-59 → 0-7"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

```python
# 原代码：
gts=[]
for data in data_samples:
    gts.extend(data.gt_labels.item)
gts=torch.stack(gts)
temp=gts.cpu().numpy()
gts_coarse=[fine2coarse(i) for i in temp]
gts_coarse=torch.from_numpy(np.array(gts_coarse)).cuda()

# 修改为：
gts=[]
for data in data_samples:
    gts.extend(data.gt_labels.item)
gts=torch.stack(gts)
temp=gts.cpu().numpy()

# 判断数据集类型
max_label = int(temp.max())
if max_label >= 52:  # NTU-60
    gts_coarse = [fine2coarse_ntu60(int(i)) for i in temp]
else:  # MA-52
    gts_coarse = [fine2coarse(int(i)) for i in temp]

gts_coarse=torch.from_numpy(np.array(gts_coarse)).cuda()
```

```python
# ❌ 错误的修改（你现在的情况）
def loss_by_feat(...):
    # ... 
    for loss_name in ...:
        labels_body = ...  # 在循环内定义
    
    # 在循环外使用 labels_body ← 错误！
    max_label = labels_body.max()  # UnboundLocalError

# ✓ 正确的修改
def loss_by_feat(...):
    labels = ...
    max_label = labels.max()  # ← 在循环前判断
    is_ntu60 = (max_label >= 52)
    
    for loss_name in ...:
        labels_body = ...
        if is_ntu60:  # ← 在循环内使用判断结果
            labels_body = action2body_ntu60(...)
        else:
            labels_body = action2body(...)
```

```python
def action2body(x):
    """MA-52: action (0-51) → body (0-6)"""
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

def action2body_ntu60(x):
    """NTU-60: action (0-59) → body (0-7), 每8个动作一组"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

```python
def loss_by_feat(self, cls_scores: Dict[str, torch.Tensor],
                 data_samples: SampleList) -> Dict:
    """Calculate the loss based on the features extracted by the head.

    Args:
        cls_scores (dict[str, torch.Tensor]): The dict of
            classification scores,
        data_samples (list[:obj:`ActionDataSample`]): The batch
            data samples.

    Returns:
        dict: A dictionary of loss components.
    """
    labels = torch.stack([x.gt_labels.item for x in data_samples])
    labels = labels.squeeze()

    if labels.shape == torch.Size([]):
        labels = labels.unsqueeze(0)
    elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
            and cls_scores.size()[0] == 1:
        # Fix a bug when training with soft labels and batch size is 1.
        # When using soft labels, `labels` and `cls_score` share the same
        # shape.
        labels = labels.unsqueeze(0)

    # ========== 判断数据集（在循环前）==========
    max_label = int(labels.max().item())
    is_ntu60 = (max_label >= 52)  # NTU-60: 0-59, MA-52: 0-51
    
    losses = dict()
    for loss_name, weight in zip(self.loss_components, self.loss_weights):
        cls_score1 = cls_scores[loss_name]
        loss_cls = self.loss_by_scores(cls_score1, labels)
        loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name}_loss_cls'] *= weight
        losses.update(loss_cls)

        # ========== 计算粗分类标签 ==========
        labels_body = labels.cpu().numpy()
        if is_ntu60:
            labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
        else:
            labels_body = np.array([action2body(int(i)) for i in labels_body])
        labels_body = torch.tensor(labels_body, dtype=torch.long).cuda()

        cls_score2 = cls_scores[loss_name+'_coarse']
        loss_name = loss_name+'_coarse'
        loss_cls = self.loss_by_scores(cls_score2, labels_body)
        loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name}_loss_cls'] *= weight
        losses.update(loss_cls)

    if self.training:
        losses['rgb_fr_coarse_loss'] = cls_scores['fr_loss_rgb_coarse'] / 5
        losses['pose_fr_coarse_loss'] = cls_scores['fr_loss_pose_coarse'] / 5
        losses['rgb_fr_loss'] = cls_scores['fr_loss_rgb']/5
        losses['pose_fr_loss'] = cls_scores['fr_loss_pose']/5
        losses['hierarchy_rgb_loss'] = cls_scores['hierarchy_loss_rgb']
        losses['hierarchy_pose_loss'] = cls_scores['hierarchy_loss_pose']
    return losses
```

```python
def fine2coarse(x):
    """MA-52: 0-51 → 0-6"""
    if x<=4:
        return 0
    elif 5<=x<=10:
        return 1
    elif 11<=x<=23:
        return 2
    elif 24<=x<=31:
        return 3
    elif 32<=x<=37:
        return 4
    elif 38<=x<=47:
        return 5
    else:
        return 6

def fine2coarse_ntu60(x):
    """NTU-60: 0-59 → 0-7"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

```python
#get gt_label
gts=[]
for data in data_samples:
    gts.extend(data.gt_labels.item)
gts=torch.stack(gts)
temp=gts.cpu().numpy()

# 判断数据集
max_label = int(temp.max())
if max_label >= 52:  # NTU-60
    gts_coarse = [fine2coarse_ntu60(int(i)) for i in temp]
else:  # MA-52
    gts_coarse = [fine2coarse(int(i)) for i in temp]

gts_coarse=torch.from_numpy(np.array(gts_coarse)).cuda()
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 检查rgbpose_head.py的修改
python << 'EOF'
import sys
sys.path.insert(0, '/home/zh/ChCode/codes01/mmaction2')

# 尝试导入函数
try:
    from mmaction.models.heads.rgbpose_head import action2body, action2body_ntu60
    print("✓ action2body函数导入成功")
    print("✓ action2body_ntu60函数导入成功")
    
    # 测试映射
    print("\nMA-52测试:")
    for x in [0, 5, 11, 24, 32, 38, 50]:
        print(f"  {x} → {action2body(x)}")
    
    print("\nNTU-60测试:")
    for x in [0, 8, 16, 24, 32, 40, 48, 56, 59]:
        print(f"  {x} → {action2body_ntu60(x)}")
        
except ImportError as e:
    print(f"✗ 导入失败: {e}")
except Exception as e:
    print(f"✗ 函数定义有问题: {e}")
EOF

# 检查recognizer3d_mm.py的修改
python << 'EOF'
import sys
sys.path.insert(0, '/home/zh/ChCode/codes01/mmaction2')

try:
    from mmaction.models.recognizers.recognizer3d_mm import fine2coarse, fine2coarse_ntu60
    print("\n✓ fine2coarse函数导入成功")
    print("✓ fine2coarse_ntu60函数导入成功")
    
    # 测试
    print("\nNTU-60测试:")
    for x in [0, 10, 20, 30, 40, 50, 59]:
        print(f"  {x} → {fine2coarse_ntu60(x)}")
        
except ImportError as e:
    print(f"✗ 导入失败: {e}")
except Exception as e:
    print(f"✗ 函数定义有问题: {e}")
EOF
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 创建完整的测试脚本
cat > test_ntu60_setup.py << 'EOF'
import torch
import pickle
import numpy as np
import sys
sys.path.insert(0, '/home/zh/ChCode/codes01/mmaction2')

print("="*60)
print("NTU-60完整检查")
print("="*60)

# 1. 检查pkl数据
print("\n1. 检查pkl数据...")
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

labels = [ann['label'] for ann in data['annotations']]
print(f"   标签范围: [{min(labels)}, {max(labels)}]")
print(f"   样本数: {len(labels)}")

if min(labels) != 0 or max(labels) != 59:
    print(f"   ✗ 错误：标签应该是0-59！")
    exit(1)
else:
    print(f"   ✓ 标签范围正确")

# 2. 检查函数导入
print("\n2. 检查函数导入...")
try:
    from mmaction.models.heads.rgbpose_head import action2body_ntu60
    from mmaction.models.recognizers.recognizer3d_mm import fine2coarse_ntu60
    print("   ✓ 所有函数导入成功")
except ImportError as e:
    print(f"   ✗ 导入失败: {e}")
    exit(1)

# 3. 测试映射函数
print("\n3. 测试映射函数...")
test_labels = np.array([0, 10, 20, 30, 40, 50, 59])
mapped = [action2body_ntu60(int(x)) for x in test_labels]
print(f"   Action: {test_labels.tolist()}")
print(f"   Body:   {mapped}")

if max(mapped) <= 7 and min(mapped) >= 0:
    print("   ✓ 映射结果在0-7范围内")
else:
    print(f"   ✗ 映射越界！")
    exit(1)

# 4. 模拟一个batch的处理
print("\n4. 模拟batch处理...")
batch_labels = torch.tensor([0, 15, 30, 45, 59], dtype=torch.long)
print(f"   模拟标签: {batch_labels.tolist()}")

max_label = int(batch_labels.max().item())
print(f"   最大标签: {max_label}")

is_ntu60 = (max_label >= 52)
print(f"   判定为NTU-60: {is_ntu60}")

labels_body = batch_labels.cpu().numpy()
if is_ntu60:
    labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
else:
    labels_body = np.array([action2body(int(i)) for i in labels_body])

print(f"   粗分类结果: {labels_body.tolist()}")
print(f"   粗分类范围: [{labels_body.min()}, {labels_body.max()}]")

# 5. 测试one_hot
print("\n5. 测试one_hot编码...")
try:
    labels_tensor = torch.tensor(labels_body, dtype=torch.long).cuda()
    one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes=8)
    print(f"   ✓ one_hot成功: {one_hot.shape}")
except Exception as e:
    print(f"   ✗ one_hot失败: {e}")
    exit(1)

print("\n" + "="*60)
print("✓ 所有检查通过！可以开始训练")
print("="*60)
EOF

# 运行测试
python test_ntu60_setup.py
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 清理之前的错误日志
rm -rf work_dirs/pcan_ntu60_test

# 重新训练（保持CUDA_LAUNCH_BLOCKING=1便于调试）
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python tools/train.py \
    configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py \
    --work-dir work_dirs/pcan_ntu60_test

# 如果看到正常的训练日志（loss下降），就成功了！
```

```plaintext
11/21 01:XX:XX - mmengine - INFO - Epoch(train) [1][10/XXXX]
    lr: 7.5000e-03
    eta: X:XX:XX
    time: X.XXXX
    data_time: X.XXXX
    memory: XXXX
    rgb_loss_cls: X.XXXX
    pose_loss_cls: X.XXXX
    rgb_coarse_loss_cls: X.XXXX  ← 粗分类损失
    pose_coarse_loss_cls: X.XXXX
    rgb_fr_loss: X.XXXX  ← 原型学习损失
    pose_fr_loss: X.XXXX
    hierarchy_rgb_loss: X.XXXX  ← 层次约束损失
    hierarchy_pose_loss: X.XXXX
```

```shellscript
# 1. 标签范围
python -c "import pickle; d=pickle.load(open('data/skeleton/ntu60_xsub.pkl','rb')); l=[a['label'] for a in d['annotations']]; print(f'[{min(l)}, {max(l)}]')"

# 2. 函数测试结果
python test_ntu60_setup.py 2>&1 | tail -20

# 3. 训练错误的完整堆栈
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python tools/train.py configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py --work-dir work_dirs/test 2>&1 | tail -50
```

---

**User**

还是报错，你所说的那几个文件都按照你的要求改了，只不过服务器上的改过的文件，你看不到而已，现在运行还是报错 信息如下：(openmmlab) [zh@node01 mmaction2]$ python -c "import pickle; d=pickle.load(open('data/skeleton/ntu60_xsub.pkl','rb')); l=[a['label'] for a in d['annotations']]; print(f'[{min(l)}, {max(l)}]')"

[0, 59]   (openmmlab) [zh@node01 mmaction2]$ python test_ntu60_setup.py 2>&1 | tail -20

   ✓ 所有函数导入成功

3. 测试映射函数...

   Action: [0, 10, 20, 30, 40, 50, 59]

   Body:   [0, 1, 2, 3, 5, 6, 7]

   ✓ 映射结果在0-7范围内

4. 模拟batch处理...

   模拟标签: [0, 15, 30, 45, 59]

   最大标签: 59

   判定为NTU-60: True

   粗分类结果: [0, 1, 3, 5, 7]

   粗分类范围: [0, 7]

5. 测试one_hot编码...

   ✓ one_hot成功: torch.Size([5, 8])

============================================================

✓ 所有检查通过！可以开始训练

============================================================

(openmmlab) [zh@node01 mmaction2]$ CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python tools/train.py configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py --work-dir work_dirs/test 2>&1 | tail -50

11/21 01:54:12 - mmengine - WARNING - "HardDiskBackend" is the alias of "LocalBackend" and the former will be deprecated in future.

11/21 01:54:12 - mmengine - INFO - Checkpoints will be saved to /home/zh/ChCode/codes01/mmaction2/work_dirs/test.

/opt/conda/conda-bld/pytorch_1682343998658/work/aten/src/ATen/native/cuda/Loss.cu:240: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [0,0,0] Assertion `t >= 0 && t < n_classes` failed.

/opt/conda/conda-bld/pytorch_1682343998658/work/aten/src/ATen/native/cuda/Loss.cu:240: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [2,0,0] Assertion `t >= 0 && t < n_classes` failed.

/opt/conda/conda-bld/pytorch_1682343998658/work/aten/src/ATen/native/cuda/Loss.cu:240: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [3,0,0] Assertion `t >= 0 && t < n_classes` failed.

/opt/conda/conda-bld/pytorch_1682343998658/work/aten/src/ATen/native/cuda/Loss.cu:240: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [4,0,0] Assertion `t >= 0 && t < n_classes` failed.

/opt/conda/conda-bld/pytorch_1682343998658/work/aten/src/ATen/native/cuda/Loss.cu:240: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [5,0,0] Assertion `t >= 0 && t < n_classes` failed.

/opt/conda/conda-bld/pytorch_1682343998658/work/aten/src/ATen/native/cuda/Loss.cu:240: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [6,0,0] Assertion `t >= 0 && t < n_classes` failed.

/opt/conda/conda-bld/pytorch_1682343998658/work/aten/src/ATen/native/cuda/Loss.cu:240: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [7,0,0] Assertion `t >= 0 && t < n_classes` failed.

/opt/conda/conda-bld/pytorch_1682343998658/work/aten/src/ATen/native/cuda/Loss.cu:240: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [9,0,0] Assertion `t >= 0 && t < n_classes` failed.

Traceback (most recent call last):

  File "tools/train.py", line 143, in <module>

    main()

  File "tools/train.py", line 139, in main

    runner.train()

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/runner.py", line 1777, in train

    model = self.train_loop.run()  # type: ignore

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py", line 98, in run

    self.run_epoch()

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py", line 115, in run_epoch

    self.run_iter(idx, data_batch)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py", line 131, in run_iter

    outputs = self.runner.model.train_step(

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/model/base_model/base_model.py", line 114, in train_step

    losses = self._run_forward(data, mode='loss')  # type: ignore

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmengine/model/base_model/base_model.py", line 361, in _run_forward

    results = self(**data, mode=mode)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl

    return forward_call(*args, **kwargs)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/recognizers/base.py", line 262, in forward

    s=self.loss(inputs, data_samples, **kwargs)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/recognizers/base.py", line 176, in loss

    loss_cls = self.cls_head.loss(feats, data_samples, **loss_kwargs)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/heads/rgbpose_head.py", line 700, in loss

    return self.loss_by_feat(cls_scores, data_samples)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/heads/rgbpose_head.py", line 735, in loss_by_feat

    loss_cls = self.loss_by_scores(cls_score1, labels)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/heads/rgbpose_head.py", line 790, in loss_by_scores

    loss_cls = self.loss_cls(cls_scores, labels)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl

    return forward_call(*args, **kwargs)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/losses/base.py", line 39, in forward

    ret = self._forward(*args, **kwargs)

  File "/home/zh/ChCode/codes01/mmaction2/mmaction/models/losses/cross_entropy_loss.py", line 86, in _forward

    loss_cls = F.cross_entropy(cls_score, label, **kwargs)

  File "/home/zh/anaconda3/envs/openmmlab/lib/python3.8/site-packages/torch/nn/functional.py", line 3029, in cross_entropy

    return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)

RuntimeError: CUDA error: device-side assert triggered

Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.   

---

**Cursor**

```plaintext
Assertion `t >= 0 && t < n_classes` failed.
```

**错误信息很明确了！** `Assertion t >= 0 && t < n_classes failed` 说明标签超出了分类器的类别数。

问题不在标签（标签是0-59正确的），而在**分类器的输出维度不匹配**！

## 🔍 诊断：分类器维度

很可能是：
- 标签：0-59 ✓
- 分类器输出：只有52维（从MA-52继承） ✗

运行这个诊断：

```bash
cd /home/zh/ChCode/codes01/mmaction2

python << 'EOF'
import torch
from mmengine.config import Config
from mmaction.registry import MODELS

print("检查模型维度...")

# 加载配置
cfg = Config.fromfile('configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py')

print(f"配置中的num_classes: {cfg.model.cls_head.num_classes}")
print(f"配置中的num_coarse_classes: {cfg.model.cls_head.get('num_coarse_classes', '未设置')}")

# 构建模型
import sys
sys.path.insert(0, '.')
import mmaction

try:
    model = MODELS.build(cfg.model)
    
    # 检查分类器维度
    print(f"\n实际模型维度:")
    print(f"  fc_rgb输出: {model.cls_head.fc_rgb.out_features}")
    print(f"  fc_pose输出: {model.cls_head.fc_pose.out_features}")
    print(f"  fc_rgb_coarse输出: {model.cls_head.fc_rgb_coarse.out_features}")
    print(f"  fc_pose_coarse输出: {model.cls_head.fc_pose_coarse.out_features}")
    
    # 检查原型维度
    print(f"\n原型维度:")
    print(f"  fr_rgb.avg_f: {model.cls_head.fr_rgb.spatio_cl_net.avg_f.shape}")
    print(f"  fr_pose.avg_f: {model.cls_head.fr_pose.spatio_cl_net.avg_f.shape}")
    
    if model.cls_head.fc_rgb.out_features != 60:
        print(f"\n✗ 错误：fc_rgb应该是60维，但实际是{model.cls_head.fc_rgb.out_features}维！")
        print("  配置文件的num_classes没有生效")
    else:
        print(f"\n✓ 分类器维度正确")
        
except Exception as e:
    print(f"✗ 构建模型失败: {e}")
    import traceback
    traceback.print_exc()
EOF
```

## 💡 问题根源

我猜测问题是**配置文件中缺少`num_coarse_classes`参数**！

### 检查你的配置文件

确认 `pcan_ntu60.py` 中的head_cfg是这样的：

```python
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← 必须有
    num_coarse_classes=8,  # ← 必须添加这个！
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose', 'rgb_coarse', 'pose_coarse'],
    loss_weights=[1., 1., 0.5, 0.5],
    average_clips='prob')
```

**如果缺少`num_coarse_classes=8`**，模型会使用默认值7，导致维度不匹配！

---

## 🔧 修复方案：确保RGBPoseHead接受num_coarse_classes

### 检查`mmaction/models/heads/rgbpose_head.py`的__init__

在第520行左右，`__init__`方法应该是这样：

```python
def __init__(self,
             num_classes: int,
             in_channels: Tuple[int],
             loss_cls: Dict = dict(type='CrossEntropyLoss'),
             loss_components: List[str] = ['rgb', 'pose'],
             loss_weights: Union[float, Tuple[float]] = 1.,
             dropout: float = 0.5,
             init_std: float = 0.01,
             num_coarse_classes: int = 7,  # ← 必须添加这个参数！
             **kwargs) -> None:
    
    super().__init__(num_classes, in_channels, loss_cls, **kwargs)
    
    # ... 中间代码 ...
    
    # 使用num_coarse_classes而不是硬编码的7
    self.fc_rgb_coarse = nn.Linear(self.in_channels[0], num_coarse_classes)
    self.fc_pose_coarse = nn.Linear(self.in_channels[1], num_coarse_classes)
    
    self.fr_coarse_rgb = ST_RenovateNet(
        2048, 8, n_class=num_coarse_classes, h_channel=128, ...)
    self.fr_coarse_pose = ST_RenovateNet(
        512, 32, n_class=num_coarse_classes, h_channel=128, ...)
    
    self.fr_rgb = ST_RenovateNet_Fine(
        2048, n_class=num_classes, ..., 
        n_class_coarse=num_coarse_classes)  # ← 传递给RenovateNet_Fine
    self.fr_pose = ST_RenovateNet_Fine(
        512, n_class=num_classes, ...,
        n_class_coarse=num_coarse_classes)
    
    self.tree_loss_rgb = TreeLoss(num_actions=num_classes, num_bodies=num_coarse_classes)
    self.tree_loss_pose = TreeLoss(num_actions=num_classes, num_bodies=num_coarse_classes)
```

---

## 📝 完整的代码修改清单（第3版 - 最终版）

由于我看不到你服务器上的修改，让我给出**完整的**需要修改的部分：

### 文件1：`mmaction/models/heads/rgbpose_head.py`

#### 位置1：第17-32行（添加NTU-60映射）

```python
def action2body(x):
    """MA-52: 0-51 → 0-6"""
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

def action2body_ntu60(x):
    """NTU-60: 0-59 → 0-7"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

#### 位置2：第35-64行（修改TreeLoss）

```python
class TreeLoss(nn.Module):
    def __init__(self, num_actions=52, num_bodies=7):
        super(TreeLoss, self).__init__()
        self.num_actions = num_actions
        self.num_bodies = num_bodies
        self.stateSpace = self.generateStateSpace().cuda()
        self.sig = nn.Sigmoid()

    def forward(self, pred_body, pred_action, labels_body, labels_action):
        pred_body = self.sig(pred_body)
        pred_action = self.sig(pred_action)
        pred_fusion = torch.cat((pred_body, pred_action), dim=1)
        labels_action = labels_action + self.num_bodies  # ← 改：使用动态值
        index = torch.mm(self.stateSpace.to(torch.float32), pred_fusion.T)
        joint = torch.exp(index)
        z = torch.sum(joint, dim=0)
        loss = torch.zeros(pred_fusion.shape[0], dtype=torch.float64).cuda()
        for i in range(len(labels_action)):
            marginal = torch.sum(torch.index_select(
                joint[:, i], 0, torch.where(self.stateSpace[:, labels_action[i]] > 0)[0]))
            loss[i] = -torch.log(marginal / z[i])
        return torch.mean(loss)

    def generateStateSpace(self):
        total_size = self.num_bodies + self.num_actions
        stat_list = np.eye(total_size)
        
        for i in range(self.num_bodies, total_size):
            temp = stat_list[i]
            index = np.where(temp > 0)[0]
            action_id = int(index) - self.num_bodies
            
            # 根据动作数判断数据集
            if self.num_actions == 60:
                coarse = action2body_ntu60(action_id)
            else:
                coarse = action2body(action_id)
            
            stat_list[i][coarse] = 1
        
        stateSpace = torch.tensor(stat_list)
        return stateSpace
```

#### 位置3：第68-89行（修改RenovateNet_Fine）

```python
class RenovateNet_Fine(nn.Module):
    def __init__(self, n_channel, n_class, alp=0.125, tmp=0.125, mom=0.9, 
                 h_channel=None, version='V0', pred_threshold=0.0, use_p_map=True,
                 n_class_coarse=7):  # ← 添加这个参数
        super(RenovateNet_Fine, self).__init__()
        self.n_channel = n_channel
        self.h_channel = n_channel if h_channel is None else h_channel
        self.n_class = n_class
        self.n_class_coarse = n_class_coarse  # ← 使用参数值

        self.alp = alp
        self.tmp = tmp
        self.mom = mom

        self.avg_f = nn.Parameter(torch.randn(
            h_channel, n_class), requires_grad=False)
        self.cl_fc = nn.Linear(self.n_channel, self.h_channel)

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.version = version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map
```

#### 位置4：第510-575行（修改RGBPoseHead.__init__）

```python
def __init__(self,
             num_classes: int,
             in_channels: Tuple[int],
             loss_cls: Dict = dict(type='CrossEntropyLoss'),
             loss_components: List[str] = ['rgb', 'pose'],
             loss_weights: Union[float, Tuple[float]] = 1.,
             dropout: float = 0.5,
             init_std: float = 0.01,
             num_coarse_classes: int = 7,  # ← 添加这个参数！
             **kwargs) -> None:
    
    super().__init__(num_classes, in_channels, loss_cls, **kwargs)
    
    # 处理dropout
    if isinstance(dropout, float):
        self.dropout = {'rgb': dropout, 'pose': dropout}
    else:
        self.dropout = dropout

    # 处理loss_weights
    if not isinstance(loss_weights, (tuple, list)):
        loss_weights = (loss_weights, ) * len(loss_components)
    assert len(loss_weights) == len(loss_components)
    self.loss_components = loss_components
    self.loss_weights = loss_weights

    self.dropout_rgb = nn.Dropout(p=self.dropout['rgb'])
    self.dropout_pose = nn.Dropout(p=self.dropout['pose'])
    self.init_std = init_std

    # ========== 关键：使用num_coarse_classes ==========
    self.fc_rgb = nn.Linear(self.in_channels[0], num_classes)
    self.fc_pose = nn.Linear(self.in_channels[1], num_classes)
    self.fc_rgb_coarse = nn.Linear(self.in_channels[0], num_coarse_classes)
    self.fc_pose_coarse = nn.Linear(self.in_channels[1], num_coarse_classes)
    
    self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))

    self.fr_coarse_rgb = ST_RenovateNet(
        2048, 8, n_class=num_coarse_classes, h_channel=128, version='V0', use_p_map=True)
    self.fr_coarse_pose = ST_RenovateNet(
        512, 32, n_class=num_coarse_classes, h_channel=128, version='V0', use_p_map=True)

    self.fr_rgb = ST_RenovateNet_Fine(
        2048, n_class=num_classes, version='V0', use_p_map=True,
        n_class_coarse=num_coarse_classes)
    self.fr_pose = ST_RenovateNet_Fine(
        512, n_class=num_classes, version='V0', use_p_map=True,
        n_class_coarse=num_coarse_classes)

    self.tree_loss_rgb = TreeLoss(num_actions=num_classes, num_bodies=num_coarse_classes)
    self.tree_loss_pose = TreeLoss(num_actions=num_classes, num_bodies=num_coarse_classes)
```

#### 位置5：第663-714行（完整的loss_by_feat）

```python
def loss_by_feat(self, cls_scores: Dict[str, torch.Tensor],
                 data_samples: SampleList) -> Dict:
    """Calculate the loss based on the features extracted by the head."""
    labels = torch.stack([x.gt_labels.item for x in data_samples])
    labels = labels.squeeze()

    if labels.shape == torch.Size([]):
        labels = labels.unsqueeze(0)
    elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
            and cls_scores.size()[0] == 1:
        labels = labels.unsqueeze(0)

    # 判断数据集类型（在循环前）
    max_label = int(labels.max().item())
    is_ntu60 = (max_label >= 52)

    losses = dict()
    for loss_name, weight in zip(self.loss_components, self.loss_weights):
        cls_score1 = cls_scores[loss_name]
        loss_cls = self.loss_by_scores(cls_score1, labels)
        loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name}_loss_cls'] *= weight
        losses.update(loss_cls)

        # 计算粗分类标签
        labels_body = labels.cpu().numpy()
        if is_ntu60:
            labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
        else:
            labels_body = np.array([action2body(int(i)) for i in labels_body])
        labels_body = torch.tensor(labels_body, dtype=torch.long).cuda()

        cls_score2 = cls_scores[loss_name+'_coarse']
        loss_name_coarse = loss_name+'_coarse'
        loss_cls = self.loss_by_scores(cls_score2, labels_body)
        loss_cls = {loss_name_coarse + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name_coarse}_loss_cls'] *= weight
        losses.update(loss_cls)

    if self.training:
        losses['rgb_fr_coarse_loss'] = cls_scores['fr_loss_rgb_coarse'] / 5
        losses['pose_fr_coarse_loss'] = cls_scores['fr_loss_pose_coarse'] / 5
        losses['rgb_fr_loss'] = cls_scores['fr_loss_rgb']/5
        losses['pose_fr_loss'] = cls_scores['fr_loss_pose']/5
        losses['hierarchy_rgb_loss'] = cls_scores['hierarchy_loss_rgb']
        losses['hierarchy_pose_loss'] = cls_scores['hierarchy_loss_pose']
    return losses
```

---

## 📝 配置文件最终版本

完整的 `configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py`：

```python
_base_ = '../../../_base_/default_runtime.py'

# ==========================================
# 模型配置
# ==========================================
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        fusion_kernel=7,
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1),
        with_pool2=False),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        fusion_kernel=7,
        in_channels=17,  # NTU-60 2D骨架
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride_s=1,
        conv1_stride_t=1,
        pool1_stride_s=1,
        pool1_stride_t=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        with_pool2=False))

head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← NTU-60的60类
    num_coarse_classes=8,  # ← NTU-60的8个粗类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose', 'rgb_coarse', 'pose_coarse'],
    loss_weights=[1., 1., 0.5, 0.5],
    average_clips='prob')

data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    preprocessors=dict(
        imgs=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        heatmap_imgs=dict(type='ActionDataPreprocessor')))

model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    data_preprocessor=data_preprocessor)

# ==========================================
# 数据配置
# ==========================================
dataset_type = 'PoseDataset'
data_root = '/home/zh/ChCode/codes01/mmaction2/data/nturgbd_videos/'
ann_file = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
ann_file_val = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
ann_file_test = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, 
         with_kp=True, with_limb=False, scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32), 
         num_clips=1, test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True,
         with_kp=True, with_limb=False, scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

test_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32),
         num_clips=10, test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True,
         with_kp=True, with_limb=False, scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        split='xsub_train',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        split='xsub_val',
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        split='xsub_val',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_begin=3, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=30, by_epoch=True,
         milestones=[10, 20], gamma=0.1)
]

load_from = None

auto_scale_lr = dict(enable=False, base_batch_size=40)
```

---

## 🎯 最终验证（运行这个）

```bash
cd /home/zh/ChCode/codes01/mmaction2

# 完整验证脚本
python << 'EOF'
import torch
from mmengine.config import Config
from mmaction.registry import MODELS
import sys
sys.path.insert(0, '.')
import mmaction

print("="*60)
print("最终验证")
print("="*60)

# 1. 加载配置
cfg = Config.fromfile('configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py')
print(f"\n1. 配置文件:")
print(f"   num_classes: {cfg.model.cls_head.num_classes}")
print(f"   num_coarse_classes: {cfg.model.cls_head.get('num_coarse_classes', '未设置!!!')}")

if cfg.model.cls_head.get('num_coarse_classes') != 8:
    print(f"   ✗ 错误：配置文件中缺少num_coarse_classes=8")
    print(f"   请在head_cfg中添加： num_coarse_classes=8,")
    exit(1)

# 2. 构建模型
print(f"\n2. 构建模型...")
cfg.load_from = None  # 不加载checkpoint
model = MODELS.build(cfg.model)

print(f"   ✓ 模型构建成功")
print(f"\n3. 检查分类器维度:")
print(f"   fc_rgb: {model.cls_head.fc_rgb.weight.shape[0]}类")
print(f"   fc_pose: {model.cls_head.fc_pose.weight.shape[0]}类")
print(f"   fc_rgb_coarse: {model.cls_head.fc_rgb_coarse.weight.shape[0]}类")
print(f"   fc_pose_coarse: {model.cls_head.fc_pose_coarse.weight.shape[0]}类")

if model.cls_head.fc_rgb.weight.shape[0] != 60:
    print(f"   ✗ 错误：fc_rgb应该是60维！")
    exit(1)
if model.cls_head.fc_rgb_coarse.weight.shape[0] != 8:
    print(f"   ✗ 错误：fc_rgb_coarse应该是8维！")
    exit(1)

print(f"\n4. 检查原型维度:")
print(f"   fr_rgb.avg_f: {model.cls_head.fr_rgb.spatio_cl_net.avg_f.shape}")
print(f"   fr_coarse_rgb.avg_f: {model.cls_head.fr_coarse_rgb.spatio_cl_net.avg_f.shape}")

if model.cls_head.fr_rgb.spatio_cl_net.avg_f.shape[1] != 60:
    print(f"   ✗ 错误：细分类原型应该是60类！")
    exit(1)
if model.cls_head.fr_coarse_rgb.spatio_cl_net.avg_f.shape[1] != 8:
    print(f"   ✗ 错误：粗分类原型应该是8类！")
    exit(1)

print(f"\n5. 模拟前向传播...")
# 模拟数据
batch_size = 2
x_rgb = torch.randn(batch_size, 2048, 8, 1, 1).cuda()
x_pose = torch.randn(batch_size, 512, 32, 1, 1).cuda()
x_rgb_mid = torch.randn(batch_size, 2048, 7, 7).cuda()
x_pose_mid = torch.randn(batch_size, 512, 7, 7).cuda()
gt = torch.tensor([10, 45], dtype=torch.long).cuda()
gt_coarse = torch.tensor([1, 5], dtype=torch.long).cuda()

x = [x_rgb, x_pose, x_rgb_mid, x_pose_mid, gt, gt_coarse]

model = model.cuda()
model.train()

try:
    with torch.no_grad():
        outputs = model.cls_head(x)
    print(f"   ✓ 前向传播成功")
    print(f"   输出keys: {list(outputs.keys())}")
except Exception as e:
    print(f"   ✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("✓ 所有验证通过！可以开始训练！")
print("="*60)
EOF
```

**如果这个脚本通过了，就可以正常训练。如果失败，把完整输出发给我。**

运行结果是什么？特别关注：
1. `num_coarse_classes`是否为8
2. `fc_rgb`维度是否为60
3. `fc_rgb_coarse`维度是否为8

把验证脚本的输出告诉我！🔍

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

python << 'EOF'
import torch
from mmengine.config import Config
from mmaction.registry import MODELS

print("检查模型维度...")

# 加载配置
cfg = Config.fromfile('configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py')

print(f"配置中的num_classes: {cfg.model.cls_head.num_classes}")
print(f"配置中的num_coarse_classes: {cfg.model.cls_head.get('num_coarse_classes', '未设置')}")

# 构建模型
import sys
sys.path.insert(0, '.')
import mmaction

try:
    model = MODELS.build(cfg.model)
    
    # 检查分类器维度
    print(f"\n实际模型维度:")
    print(f"  fc_rgb输出: {model.cls_head.fc_rgb.out_features}")
    print(f"  fc_pose输出: {model.cls_head.fc_pose.out_features}")
    print(f"  fc_rgb_coarse输出: {model.cls_head.fc_rgb_coarse.out_features}")
    print(f"  fc_pose_coarse输出: {model.cls_head.fc_pose_coarse.out_features}")
    
    # 检查原型维度
    print(f"\n原型维度:")
    print(f"  fr_rgb.avg_f: {model.cls_head.fr_rgb.spatio_cl_net.avg_f.shape}")
    print(f"  fr_pose.avg_f: {model.cls_head.fr_pose.spatio_cl_net.avg_f.shape}")
    
    if model.cls_head.fc_rgb.out_features != 60:
        print(f"\n✗ 错误：fc_rgb应该是60维，但实际是{model.cls_head.fc_rgb.out_features}维！")
        print("  配置文件的num_classes没有生效")
    else:
        print(f"\n✓ 分类器维度正确")
        
except Exception as e:
    print(f"✗ 构建模型失败: {e}")
    import traceback
    traceback.print_exc()
EOF
```

```python
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← 必须有
    num_coarse_classes=8,  # ← 必须添加这个！
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose', 'rgb_coarse', 'pose_coarse'],
    loss_weights=[1., 1., 0.5, 0.5],
    average_clips='prob')
```

```python
def __init__(self,
             num_classes: int,
             in_channels: Tuple[int],
             loss_cls: Dict = dict(type='CrossEntropyLoss'),
             loss_components: List[str] = ['rgb', 'pose'],
             loss_weights: Union[float, Tuple[float]] = 1.,
             dropout: float = 0.5,
             init_std: float = 0.01,
             num_coarse_classes: int = 7,  # ← 必须添加这个参数！
             **kwargs) -> None:
    
    super().__init__(num_classes, in_channels, loss_cls, **kwargs)
    
    # ... 中间代码 ...
    
    # 使用num_coarse_classes而不是硬编码的7
    self.fc_rgb_coarse = nn.Linear(self.in_channels[0], num_coarse_classes)
    self.fc_pose_coarse = nn.Linear(self.in_channels[1], num_coarse_classes)
    
    self.fr_coarse_rgb = ST_RenovateNet(
        2048, 8, n_class=num_coarse_classes, h_channel=128, ...)
    self.fr_coarse_pose = ST_RenovateNet(
        512, 32, n_class=num_coarse_classes, h_channel=128, ...)
    
    self.fr_rgb = ST_RenovateNet_Fine(
        2048, n_class=num_classes, ..., 
        n_class_coarse=num_coarse_classes)  # ← 传递给RenovateNet_Fine
    self.fr_pose = ST_RenovateNet_Fine(
        512, n_class=num_classes, ...,
        n_class_coarse=num_coarse_classes)
    
    self.tree_loss_rgb = TreeLoss(num_actions=num_classes, num_bodies=num_coarse_classes)
    self.tree_loss_pose = TreeLoss(num_actions=num_classes, num_bodies=num_coarse_classes)
```

```python
def action2body(x):
    """MA-52: 0-51 → 0-6"""
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

def action2body_ntu60(x):
    """NTU-60: 0-59 → 0-7"""
    x = int(x)
    if x < 0 or x > 59:
        return 0
    return min(x // 8, 7)
```

```python
class TreeLoss(nn.Module):
    def __init__(self, num_actions=52, num_bodies=7):
        super(TreeLoss, self).__init__()
        self.num_actions = num_actions
        self.num_bodies = num_bodies
        self.stateSpace = self.generateStateSpace().cuda()
        self.sig = nn.Sigmoid()

    def forward(self, pred_body, pred_action, labels_body, labels_action):
        pred_body = self.sig(pred_body)
        pred_action = self.sig(pred_action)
        pred_fusion = torch.cat((pred_body, pred_action), dim=1)
        labels_action = labels_action + self.num_bodies  # ← 改：使用动态值
        index = torch.mm(self.stateSpace.to(torch.float32), pred_fusion.T)
        joint = torch.exp(index)
        z = torch.sum(joint, dim=0)
        loss = torch.zeros(pred_fusion.shape[0], dtype=torch.float64).cuda()
        for i in range(len(labels_action)):
            marginal = torch.sum(torch.index_select(
                joint[:, i], 0, torch.where(self.stateSpace[:, labels_action[i]] > 0)[0]))
            loss[i] = -torch.log(marginal / z[i])
        return torch.mean(loss)

    def generateStateSpace(self):
        total_size = self.num_bodies + self.num_actions
        stat_list = np.eye(total_size)
        
        for i in range(self.num_bodies, total_size):
            temp = stat_list[i]
            index = np.where(temp > 0)[0]
            action_id = int(index) - self.num_bodies
            
            # 根据动作数判断数据集
            if self.num_actions == 60:
                coarse = action2body_ntu60(action_id)
            else:
                coarse = action2body(action_id)
            
            stat_list[i][coarse] = 1
        
        stateSpace = torch.tensor(stat_list)
        return stateSpace
```

```python
class RenovateNet_Fine(nn.Module):
    def __init__(self, n_channel, n_class, alp=0.125, tmp=0.125, mom=0.9, 
                 h_channel=None, version='V0', pred_threshold=0.0, use_p_map=True,
                 n_class_coarse=7):  # ← 添加这个参数
        super(RenovateNet_Fine, self).__init__()
        self.n_channel = n_channel
        self.h_channel = n_channel if h_channel is None else h_channel
        self.n_class = n_class
        self.n_class_coarse = n_class_coarse  # ← 使用参数值

        self.alp = alp
        self.tmp = tmp
        self.mom = mom

        self.avg_f = nn.Parameter(torch.randn(
            h_channel, n_class), requires_grad=False)
        self.cl_fc = nn.Linear(self.n_channel, self.h_channel)

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.version = version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map
```

```python
def __init__(self,
             num_classes: int,
             in_channels: Tuple[int],
             loss_cls: Dict = dict(type='CrossEntropyLoss'),
             loss_components: List[str] = ['rgb', 'pose'],
             loss_weights: Union[float, Tuple[float]] = 1.,
             dropout: float = 0.5,
             init_std: float = 0.01,
             num_coarse_classes: int = 7,  # ← 添加这个参数！
             **kwargs) -> None:
    
    super().__init__(num_classes, in_channels, loss_cls, **kwargs)
    
    # 处理dropout
    if isinstance(dropout, float):
        self.dropout = {'rgb': dropout, 'pose': dropout}
    else:
        self.dropout = dropout

    # 处理loss_weights
    if not isinstance(loss_weights, (tuple, list)):
        loss_weights = (loss_weights, ) * len(loss_components)
    assert len(loss_weights) == len(loss_components)
    self.loss_components = loss_components
    self.loss_weights = loss_weights

    self.dropout_rgb = nn.Dropout(p=self.dropout['rgb'])
    self.dropout_pose = nn.Dropout(p=self.dropout['pose'])
    self.init_std = init_std

    # ========== 关键：使用num_coarse_classes ==========
    self.fc_rgb = nn.Linear(self.in_channels[0], num_classes)
    self.fc_pose = nn.Linear(self.in_channels[1], num_classes)
    self.fc_rgb_coarse = nn.Linear(self.in_channels[0], num_coarse_classes)
    self.fc_pose_coarse = nn.Linear(self.in_channels[1], num_coarse_classes)
    
    self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))

    self.fr_coarse_rgb = ST_RenovateNet(
        2048, 8, n_class=num_coarse_classes, h_channel=128, version='V0', use_p_map=True)
    self.fr_coarse_pose = ST_RenovateNet(
        512, 32, n_class=num_coarse_classes, h_channel=128, version='V0', use_p_map=True)

    self.fr_rgb = ST_RenovateNet_Fine(
        2048, n_class=num_classes, version='V0', use_p_map=True,
        n_class_coarse=num_coarse_classes)
    self.fr_pose = ST_RenovateNet_Fine(
        512, n_class=num_classes, version='V0', use_p_map=True,
        n_class_coarse=num_coarse_classes)

    self.tree_loss_rgb = TreeLoss(num_actions=num_classes, num_bodies=num_coarse_classes)
    self.tree_loss_pose = TreeLoss(num_actions=num_classes, num_bodies=num_coarse_classes)
```

```python
def loss_by_feat(self, cls_scores: Dict[str, torch.Tensor],
                 data_samples: SampleList) -> Dict:
    """Calculate the loss based on the features extracted by the head."""
    labels = torch.stack([x.gt_labels.item for x in data_samples])
    labels = labels.squeeze()

    if labels.shape == torch.Size([]):
        labels = labels.unsqueeze(0)
    elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
            and cls_scores.size()[0] == 1:
        labels = labels.unsqueeze(0)

    # 判断数据集类型（在循环前）
    max_label = int(labels.max().item())
    is_ntu60 = (max_label >= 52)

    losses = dict()
    for loss_name, weight in zip(self.loss_components, self.loss_weights):
        cls_score1 = cls_scores[loss_name]
        loss_cls = self.loss_by_scores(cls_score1, labels)
        loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name}_loss_cls'] *= weight
        losses.update(loss_cls)

        # 计算粗分类标签
        labels_body = labels.cpu().numpy()
        if is_ntu60:
            labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
        else:
            labels_body = np.array([action2body(int(i)) for i in labels_body])
        labels_body = torch.tensor(labels_body, dtype=torch.long).cuda()

        cls_score2 = cls_scores[loss_name+'_coarse']
        loss_name_coarse = loss_name+'_coarse'
        loss_cls = self.loss_by_scores(cls_score2, labels_body)
        loss_cls = {loss_name_coarse + '_' + k: v for k, v in loss_cls.items()}
        loss_cls[f'{loss_name_coarse}_loss_cls'] *= weight
        losses.update(loss_cls)

    if self.training:
        losses['rgb_fr_coarse_loss'] = cls_scores['fr_loss_rgb_coarse'] / 5
        losses['pose_fr_coarse_loss'] = cls_scores['fr_loss_pose_coarse'] / 5
        losses['rgb_fr_loss'] = cls_scores['fr_loss_rgb']/5
        losses['pose_fr_loss'] = cls_scores['fr_loss_pose']/5
        losses['hierarchy_rgb_loss'] = cls_scores['hierarchy_loss_rgb']
        losses['hierarchy_pose_loss'] = cls_scores['hierarchy_loss_pose']
    return losses
```

```python
_base_ = '../../../_base_/default_runtime.py'

# ==========================================
# 模型配置
# ==========================================
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        fusion_kernel=7,
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1),
        with_pool2=False),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        fusion_kernel=7,
        in_channels=17,  # NTU-60 2D骨架
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride_s=1,
        conv1_stride_t=1,
        pool1_stride_s=1,
        pool1_stride_t=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        with_pool2=False))

head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,  # ← NTU-60的60类
    num_coarse_classes=8,  # ← NTU-60的8个粗类
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose', 'rgb_coarse', 'pose_coarse'],
    loss_weights=[1., 1., 0.5, 0.5],
    average_clips='prob')

data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    preprocessors=dict(
        imgs=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        heatmap_imgs=dict(type='ActionDataPreprocessor')))

model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    data_preprocessor=data_preprocessor)

# ==========================================
# 数据配置
# ==========================================
dataset_type = 'PoseDataset'
data_root = '/home/zh/ChCode/codes01/mmaction2/data/nturgbd_videos/'
ann_file = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
ann_file_val = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
ann_file_test = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, 
         with_kp=True, with_limb=False, scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32), 
         num_clips=1, test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True,
         with_kp=True, with_limb=False, scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

test_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32),
         num_clips=10, test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True,
         with_kp=True, with_limb=False, scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        split='xsub_train',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        split='xsub_val',
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        split='xsub_val',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_begin=3, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=30, by_epoch=True,
         milestones=[10, 20], gamma=0.1)
]

load_from = None

auto_scale_lr = dict(enable=False, base_batch_size=40)
```

```shellscript
cd /home/zh/ChCode/codes01/mmaction2

# 完整验证脚本
python << 'EOF'
import torch
from mmengine.config import Config
from mmaction.registry import MODELS
import sys
sys.path.insert(0, '.')
import mmaction

print("="*60)
print("最终验证")
print("="*60)

# 1. 加载配置
cfg = Config.fromfile('configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py')
print(f"\n1. 配置文件:")
print(f"   num_classes: {cfg.model.cls_head.num_classes}")
print(f"   num_coarse_classes: {cfg.model.cls_head.get('num_coarse_classes', '未设置!!!')}")

if cfg.model.cls_head.get('num_coarse_classes') != 8:
    print(f"   ✗ 错误：配置文件中缺少num_coarse_classes=8")
    print(f"   请在head_cfg中添加： num_coarse_classes=8,")
    exit(1)

# 2. 构建模型
print(f"\n2. 构建模型...")
cfg.load_from = None  # 不加载checkpoint
model = MODELS.build(cfg.model)

print(f"   ✓ 模型构建成功")
print(f"\n3. 检查分类器维度:")
print(f"   fc_rgb: {model.cls_head.fc_rgb.weight.shape[0]}类")
print(f"   fc_pose: {model.cls_head.fc_pose.weight.shape[0]}类")
print(f"   fc_rgb_coarse: {model.cls_head.fc_rgb_coarse.weight.shape[0]}类")
print(f"   fc_pose_coarse: {model.cls_head.fc_pose_coarse.weight.shape[0]}类")

if model.cls_head.fc_rgb.weight.shape[0] != 60:
    print(f"   ✗ 错误：fc_rgb应该是60维！")
    exit(1)
if model.cls_head.fc_rgb_coarse.weight.shape[0] != 8:
    print(f"   ✗ 错误：fc_rgb_coarse应该是8维！")
    exit(1)

print(f"\n4. 检查原型维度:")
print(f"   fr_rgb.avg_f: {model.cls_head.fr_rgb.spatio_cl_net.avg_f.shape}")
print(f"   fr_coarse_rgb.avg_f: {model.cls_head.fr_coarse_rgb.spatio_cl_net.avg_f.shape}")

if model.cls_head.fr_rgb.spatio_cl_net.avg_f.shape[1] != 60:
    print(f"   ✗ 错误：细分类原型应该是60类！")
    exit(1)
if model.cls_head.fr_coarse_rgb.spatio_cl_net.avg_f.shape[1] != 8:
    print(f"   ✗ 错误：粗分类原型应该是8类！")
    exit(1)

print(f"\n5. 模拟前向传播...")
# 模拟数据
batch_size = 2
x_rgb = torch.randn(batch_size, 2048, 8, 1, 1).cuda()
x_pose = torch.randn(batch_size, 512, 32, 1, 1).cuda()
x_rgb_mid = torch.randn(batch_size, 2048, 7, 7).cuda()
x_pose_mid = torch.randn(batch_size, 512, 7, 7).cuda()
gt = torch.tensor([10, 45], dtype=torch.long).cuda()
gt_coarse = torch.tensor([1, 5], dtype=torch.long).cuda()

x = [x_rgb, x_pose, x_rgb_mid, x_pose_mid, gt, gt_coarse]

model = model.cuda()
model.train()

try:
    with torch.no_grad():
        outputs = model.cls_head(x)
    print(f"   ✓ 前向传播成功")
    print(f"   输出keys: {list(outputs.keys())}")
except Exception as e:
    print(f"   ✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("✓ 所有验证通过！可以开始训练！")
print("="*60)
EOF
```

