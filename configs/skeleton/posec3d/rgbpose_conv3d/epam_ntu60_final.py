_base_ = '../../../_base_/default_runtime.py'

# ==========================================
# EPAM NTU60 最终优化版本
# 
# 综合两次失败教训的最终方案：
# 1. lr=0.002（折中：0.001太慢，0.003太抖）
# 2. 添加WarmUp（前5 epochs逐渐升温，稳定初期）
# 3. T_max=50（快速衰减）
# 4. 恢复原始loss_weights（不降低特征修复权重）
# 5. weight_decay=0.0002（中等正则化）
# ==========================================

custom_imports = dict(
    imports=[
        'mmaction.models.backbones.epam_backbone',
        'mmaction.models.recognizers.epam_recognizer'
    ],
    allow_failed_imports=False)

randomness = dict(seed=100, deterministic=False)

# Backbone配置
backbone_cfg = dict(
    type='EPAMBackbone',
    num_classes=60,
    rgb_pretrained=None,
    pose_pretrained=None,
    attention_type='CBAM_spatial_efficient_temporal',
    freeze_rgb=False,
    freeze_pose=False,
    return_both_streams=True
)

# Head配置 - 恢复原始loss_weights
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,
    num_coarse_classes=8,
    in_channels=[432, 216],
    loss_components=['rgb', 'pose'],
    loss_weights=[1.0, 1.2, 0.5, 0.9],  # 恢复原始权重，不削弱特征修复
    dropout=0.5,  # 恢复到0.5
    average_clips='prob'
)

# 数据预处理
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    preprocessors=dict(
        imgs=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        heatmap_imgs=dict(type='ActionDataPreprocessor')))

# Model
model = dict(
    type='EPAMRecognizer',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    data_preprocessor=data_preprocessor)

# 数据集路径
ann_file = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
ann_file_val = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
ann_file_test = '/home/zh/ChCode/codes01/mmaction2/data/skeleton/ntu60_xsub.pkl'
data_root = '/home/zh/ChCode/codes01/mmaction2/data/nturgbd_videos/'
data_root_val = '/home/zh/ChCode/codes01/mmaction2/data/nturgbd_videos/'

dataset_type = 'PoseDataset'

left_kp = (1, 3, 5, 7, 9, 11, 13, 15)
right_kp = (2, 4, 6, 8, 10, 12, 14, 16)

train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=16, Pose=48), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', 
         sigma=0.7,
         use_score=True, 
         with_kp=True,
         with_limb=False,
         scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=16, Pose=48), num_clips=1, test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='GeneratePoseTarget',
         sigma=0.7,
         use_score=True,
         with_kp=True,
         with_limb=False,
         scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

test_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=16, Pose=48), num_clips=5, test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='GeneratePoseTarget',
         sigma=0.7,
         use_score=True,
         with_kp=True,
         with_limb=False,
         scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='xsub_train',
            data_prefix=dict(video=data_root))))

val_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        split='xsub_val',
        data_prefix=dict(video=data_root_val),
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        split='xsub_val',
        data_prefix=dict(video=data_root_val),
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# 训练策略
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,  # 缩短到80，因为使用了WarmUp
    val_begin=1,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置：关键折中
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.002,            # 折中：0.001和0.003之间
        momentum=0.9,
        weight_decay=0.0002  # 中等正则化
    ),
    clip_grad=dict(max_norm=40, norm_type=2)
)

# 学习率调度器：添加WarmUp + 快速衰减
param_scheduler = [
    # WarmUp阶段：前5个epochs从0.0002逐渐升到0.002
    dict(
        type='LinearLR',
        start_factor=0.1,    # 从lr*0.1 = 0.0002开始
        by_epoch=True,
        begin=0,
        end=5                # 前5个epochs预热
    ),
    # CosineAnnealing阶段：5-50 epochs快速衰减
    dict(
        type='CosineAnnealingLR',
        T_max=45,            # 50-5=45 epochs衰减
        eta_min=1e-6,
        by_epoch=True,
        begin=5,
        end=50
    )
]

# Hooks配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='acc/RGBPose_1:1_top1',
        rule='greater',
        max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=50, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    runtime_info=dict(type='RuntimeInfoHook'))

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# 日志配置
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)
log_level = 'INFO'
load_from = None
resume = False

# 工作目录
work_dir = './work_dirs/epam_ntu60_final'

# 自动学习率缩放
auto_scale_lr = dict(enable=False, base_batch_size=16)

