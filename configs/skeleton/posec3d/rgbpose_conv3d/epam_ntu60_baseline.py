_base_ = '../../../_base_/default_runtime.py'

# ==========================================
# 自定义导入：注册EPAM模块
# ==========================================
custom_imports = dict(
    imports=[
        'mmaction.models.backbones.epam_backbone',
        'mmaction.models.recognizers.epam_recognizer'
    ],
    allow_failed_imports=False)

# ==========================================
# EPAM Backbone + NTU60数据集配置
# 使用EPAM-Net论文中的backbone替换原有的RGBPoseConv3D
# 保留原有的RGBPoseHead（包含原型学习等高级功能）
# ==========================================

# Backbone配置：使用EPAM Backbone
backbone_cfg = dict(
    type='EPAMBackbone',
    num_classes=60,
    rgb_pretrained=None,  # 如有预训练权重，填入路径
    pose_pretrained=None,  # 如有预训练权重，填入路径
    attention_type='CBAM_spatial_efficient_temporal',
    freeze_rgb=False,
    freeze_pose=False,
    return_both_streams=True
)

# 分类头配置：RGBPoseHead（保留原型学习功能）
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,
    num_coarse_classes=8,
    in_channels=[432, 216],  # EPAM输出: RGB=432, Pose=216 (不同于原始的[2048, 512])
    loss_components=['rgb', 'pose'],
    loss_weights=[1.0, 1.2, 0.5, 0.9],  # [rgb_fine, pose_fine, rgb_coarse, pose_coarse]
    average_clips='prob'
)

# 数据预处理器
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    preprocessors=dict(
        imgs=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        heatmap_imgs=dict(type='ActionDataPreprocessor')))

# 模型：使用EPAM Recognizer
model = dict(
    type='EPAMRecognizer',
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

# ==========================================
# 训练Pipeline
# ==========================================
train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=16, Pose=48),  # EPAM: RGB=16帧, Pose=48帧
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),  # EPAM RGB输入: 224x224
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),  # 生成56x56的热图 (224 * 0.25 = 56)
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

val_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=16, Pose=48),
        num_clips=1,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
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

# ==========================================
# 测试Pipeline（TTA：5 clips）
# ==========================================
test_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=16, Pose=48),
        num_clips=5,  # 使用5个clips进行测试时增强
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
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

# ==========================================
# DataLoader配置
# ==========================================
train_dataloader = dict(
    batch_size=8,  # EPAM相对轻量，可以用稍大的batch size
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

# ==========================================
# 训练配置
# ==========================================
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=50,  # 初始训练50个epochs
    val_begin=1, 
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ==========================================
# 优化器配置
# ==========================================
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', 
        lr=0.01,  # 初始学习率
        momentum=0.9, 
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# ==========================================
# 学习率调度策略
# ==========================================
param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[20, 40],  # 在第20和40个epoch降低学习率
        gamma=0.1,
        begin=0,
        end=50)
]

# 如果从其他模型继续训练，设置此项
# load_from = 'work_dirs/epam_ntu60_baseline/best_checkpoint.pth'
load_from = None
resume = False

auto_scale_lr = dict(enable=False, base_batch_size=32)

# ==========================================
# Hooks配置
# ==========================================
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5,
        save_best='acc/RGBPose_1:1_top1',
        rule='greater',
        max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

# ==========================================
# 运行时配置
# ==========================================
# 工作目录
work_dir = './work_dirs/epam_ntu60_baseline'

# 日志配置
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

# 可视化后端 (可选)
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

# ==========================================
# 说明
# ==========================================
# 主要变更：
# 1. Backbone: RGBPoseConv3D -> EPAMBackbone
#    - RGB特征: 2048 -> 432 通道
#    - Pose特征: 512 -> 216 通道
#    - 新增: Pose引导的注意力机制
#    - 使用X3D + Temporal Shift Module
#
# 2. Recognizer: MMRecognizer3D -> EPAMRecognizer
#    - 适配EPAM Backbone的输入输出
#    - 保持与原有训练流程兼容
#
# 3. Head: RGBPoseHead (保持不变)
#    - 保留原型学习功能
#    - 保留层次化损失
#    - 仅调整in_channels以匹配新backbone
#
# 4. 数据Pipeline: 调整clip_len
#    - RGB: 8帧 -> 16帧 (EPAM要求)
#    - Pose: 32帧 -> 48帧 (EPAM要求，密集采样)
#    - RGB尺寸: 224x224 (EPAM标准)
#    - Pose热图: 56x56 (scaling=0.25)

