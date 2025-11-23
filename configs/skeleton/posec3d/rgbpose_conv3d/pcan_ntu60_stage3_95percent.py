_base_ = '../../../_base_/default_runtime.py'

# ==========================================
# 阶段3: 冲刺95% (激进优化)
# 从阶段2的85-90%基础上，极限冲刺
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
        in_channels=17,
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
    num_classes=60,
    num_coarse_classes=8,
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],
    loss_weights=[1.0, 1.5, 0.6, 1.2],  # ← 阶段3：恢复激进权重
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

# ==========================================
# 训练Pipeline（激进数据增强 - 冲刺95%）
# ==========================================
train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.45, 1.0)),  # ← 阶段3：0.50→0.45
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3, hue=0.12),  # ← 阶段3：增强
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

# ==========================================
# 测试Pipeline（TTA：10 clips - 极限优化）
# ==========================================
test_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8, Pose=32),
        num_clips=10,  # ← 阶段3：5→10，最强TTA
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
    batch_size=16,
    num_workers=16,
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
# 训练配置（阶段3：50 epochs冲刺95%）
# ==========================================
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=50,  # ← 阶段3：继续50个epoch
    val_begin=1, 
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置（激进优化）
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', 
        lr=0.01,              # ← 阶段3：提高到0.01
        momentum=0.9, 
        weight_decay=0.0003), # ← 阶段3：降低正则化，让模型学得更充分
    clip_grad=dict(max_norm=30, norm_type=2))  # ← 阶段3：30

# 学习率调度策略（短warmup + Cosine退火）
param_scheduler = [
    dict(
        type='LinearLR',      # ← 阶段3：加入短warmup
        start_factor=0.5,
        by_epoch=True,
        begin=0,
        end=3),
    dict(
        type='CosineAnnealingLR',
        T_max=47,
        eta_min=1e-6,
        by_epoch=True,
        begin=3,
        end=50)
]

# ← 从阶段2最佳checkpoint继续（85-90%）
# 需要手动更新为阶段2的最佳checkpoint路径
load_from = 'work_dirs/pcan_ntu60_stage2/best_acc_RGBPose_1:1_top1_epoch_XX.pth'  # ← 需要更新XX
resume = False

auto_scale_lr = dict(enable=False, base_batch_size=40)

# ==========================================
# Hooks配置
# ==========================================
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5,
        save_best='acc/RGBPose_1:1_top1',
        rule='greater',
        max_keep_ckpts=15),      # ← 阶段3：保留更多checkpoint
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='acc/RGBPose_1:1_top1',
        patience=20,  # ← 阶段3：更宽容（20 epoch）
        min_delta=0.0002))  # ← 阶段3：更精细的阈值


