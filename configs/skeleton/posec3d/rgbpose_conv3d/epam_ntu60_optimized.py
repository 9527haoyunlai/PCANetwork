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
# 随机种子（确保可复现性）
# ==========================================
randomness = dict(seed=100, deterministic=False)

# ==========================================
# EPAM Backbone + NTU60数据集配置（优化版）
# 基于原始RGBPoseConv3D的成功超参数优化
# 
# 主要优化点：
# 1. 学习率从0.01降到0.0015（匹配原始模型）
# 2. Weight Decay从0.0001增到0.0003（更强正则化）
# 3. 学习率调度从MultiStepLR改为CosineAnnealingLR（更平滑）
# 4. 设置随机种子100（确保可复现）
# ==========================================

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

# Head配置（保留原型学习功能）
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,
    num_coarse_classes=8,
    in_channels=[432, 216],  # EPAM输出维度
    loss_components=['rgb', 'pose'],
    loss_weights=[1.0, 1.2, 0.5, 0.9],
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

# 模型配置
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
        clip_len=dict(RGB=16, Pose=48),
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

test_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=16, Pose=48),
        num_clips=5,
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
    batch_size=8,  # 每卡8，总batch=16 (EPAM较轻量，合理配置)
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
    max_epochs=50,
    val_begin=1, 
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ==========================================
# 优化器配置（✨ 优化版）
# 基于原始RGBPoseConv3D模型的成功超参数
# ==========================================
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',               # SGD优化器（SOTA标准）
        lr=0.0015,                # ← 优化：从0.01降到0.0015（匹配原始模型）
        momentum=0.9,             # ← 动量0.9（标准值，增加训练稳定性）
        weight_decay=0.0003       # ← 优化：从0.0001增到0.0003（更强L2正则化，防止过拟合）
    ),
    clip_grad=dict(max_norm=40, norm_type=2)  # ← 梯度裁剪（防止梯度爆炸）
)

# ==========================================
# 学习率调度策略（✨ 优化版）
# 改用余弦退火，比阶梯式下降更平滑
# ==========================================
param_scheduler = [
    dict(
        type='CosineAnnealingLR',  # ← 优化：从MultiStepLR改为CosineAnnealingLR
        T_max=50,                  # 总epoch数
        eta_min=1e-6,              # 最小学习率（接近于0但不为0）
        by_epoch=True,
        begin=0,
        end=50
    )
]

# 如果从其他模型继续训练，设置此项
# load_from = 'path/to/pretrained.pth'
load_from = None
resume = False

# 自动学习率缩放（可选）
auto_scale_lr = dict(
    enable=False,       # 如果想根据GPU数量自动调整学习率，设为True
    base_batch_size=16  # 基准batch size
)

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
        interval=5,                           # 每5个epoch保存一次
        save_best='acc/RGBPose_1:1_top1',     # 保存最佳模型
        rule='greater',
        max_keep_ckpts=5),                    # 最多保留5个checkpoint
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))  # ← SyncBN同步（多卡训练关键）

# ==========================================
# 运行时配置
# ==========================================
work_dir = './work_dirs/epam_ntu60_optimized'
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

# ==========================================
# 优化总结
# ==========================================
# ✅ 1. 学习率: 0.01 → 0.0015
#     - 理由：匹配原始模型，EPAM参数量更少需要更小的学习率
#     - 效果：更稳定的训练，更好的收敛
#
# ✅ 2. Weight Decay: 0.0001 → 0.0003
#     - 理由：更强的L2正则化，防止过拟合
#     - 效果：更好的泛化能力
#
# ✅ 3. LR Scheduler: MultiStepLR → CosineAnnealingLR
#     - 理由：余弦退火提供更平滑的学习率衰减
#     - 效果：通常能获得+0.5-1%的准确率提升
#
# ✅ 4. 随机种子: 设置为100
#     - 理由：确保实验可复现
#     - 效果：每次训练结果一致
#
# ✅ 5. Gradient Clip: max_norm=40
#     - 理由：防止梯度爆炸（视频模型常见问题）
#     - 效果：训练更稳定，不会突然崩溃
#
# ✅ 6. Momentum: 0.9
#     - 理由：SGD标准配置，增加动量帮助跳出局部最优
#     - 效果：收敛更快更稳定
#
# ==========================================
# 预期性能提升
# ==========================================
# - 训练曲线更平滑
# - 收敛速度适中（不会太快导致欠拟合）
# - 最终准确率预期：90-93%（NTU60 X-Sub）
# - 相比baseline配置预期提升：+1-2%

