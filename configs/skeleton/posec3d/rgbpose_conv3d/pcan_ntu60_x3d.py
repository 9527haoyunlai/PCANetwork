_base_ = '../../../_base_/default_runtime.py'

# ==========================================
# X3D TemporalShift Backbone 配置
# 目标：突破87%瓶颈，冲刺90-93%
# ==========================================

# 导入自定义backbone
import sys
sys.path.insert(0, '/home/zh/ChCode/codes01/mmaction2/emap_backbone')

# RGB Backbone: X3D TemporalShift (Standalone版本，纯PyTorch)
rgb_backbone = dict(
    gamma_w=1,              # 宽度因子
    gamma_b=2.25,           # 瓶颈因子
    gamma_d=2.2,            # 深度因子
    se_style='half',        # SE模块（特征增强）
    se_ratio=1/16,
    use_swish=True,         # Swish激活（优于ReLU）
    in_channels=3,
    num_stages=4,
    spatial_strides=(2, 2, 2, 2),
    frozen_stages=-1,
    fold_div=8,            # TemporalShift的fold division
    with_cp=False          # 不使用checkpoint
)

# Pose Backbone: X3D TemporalShift Pose (Standalone版本，纯PyTorch)
pose_backbone = dict(
    gamma_d=1,              # Pose使用更小模型
    in_channels=17,         # 17个骨骼关节点
    base_channels=24,
    num_stages=3,
    stage_blocks=(5, 11, 7),  # 每stage的block数
    spatial_strides=(2, 2, 2),
    conv1_stride=1,
    se_ratio=1/16,
    use_swish=True,
    frozen_stages=-1,
    fold_div=4             # TemporalShift的fold division
)

# Head配置 - 关键：匹配X3D输出通道
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,
    num_coarse_classes=8,
    in_channels=[432, 216],  # ← X3D输出通道 (RGB=432, Pose=216)
    loss_components=['rgb', 'pose'],
    loss_weights=[1.0, 1.5, 0.6, 1.2],  # 保持阶段1成功权重
    average_clips='prob'
)

# 数据预处理 - 使用自定义的RGBPoseDataPreprocessor
data_preprocessor = dict(
    type='RGBPoseDataPreprocessor',
    preprocessors=dict(
        imgs=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCTHW'),
        heatmap_imgs=dict(type='ActionDataPreprocessor')))

# 模型定义
model = dict(
    type='RGBPoseX3DRecognizer',  # 使用自定义双backbone Recognizer
    rgb_backbone=rgb_backbone,
    pose_backbone=pose_backbone,
    cls_head=head_cfg,
    data_preprocessor=data_preprocessor
)

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

# 训练数据pipeline - X3D双模态
train_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=16, Pose=48),  # RGB用16帧，Pose用48帧
        num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

# 验证数据pipeline
val_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=16, Pose=48),
        num_clips=1,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

# 测试数据pipeline
test_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=16, Pose=48),
        num_clips=1,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
]

# 训练dataloader - 优化版
train_dataloader = dict(
    batch_size=16,  # 增加到16（从12）
    num_workers=16,  # 增加到16（从8）
    persistent_workers=True,
    prefetch_factor=2,  # 添加预取
    pin_memory=True,    # 添加pin memory加速
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        # ❌ 删除 RepeatDataset 包装
        type=dataset_type,  # 直接使用PoseDataset
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline,
        split='xsub_train'
    )
)

# 验证dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),  # 添加视频路径前缀
        pipeline=val_pipeline,
        split='xsub_val',
        test_mode=True))

# 测试dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),  # 添加视频路径前缀
        pipeline=test_pipeline,
        split='xsub_val',
        test_mode=True))

# 评估配置
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# ==========================================
# 训练配置（X3D优化策略）
# ==========================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,  # X3D收敛快，80 epochs足够
    val_begin=5,
    val_interval=2)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器（X3D适配）
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.012,           # X3D适合稍高学习率
        momentum=0.9,
        weight_decay=0.0002),  # 轻量模型，较小weight_decay
    clip_grad=dict(max_norm=40, norm_type=2))

# 学习率调度（Warmup + CosineAnnealing）
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5),  # 5 epoch warmup
    dict(
        type='CosineAnnealingLR',
        T_max=75,
        eta_min=1e-6,
        by_epoch=True,
        begin=5,
        end=80)
]

# ==========================================
# Hooks配置
# ==========================================
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='acc/RGBPose_1:1_top1',
        max_keep_ckpts=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='acc/RGBPose_1:1_top1',
        patience=15,
        min_delta=0.0005))

# ==========================================
# 运行时配置
# ==========================================
# Resume / Load
load_from = None  # 从零开始训练X3D
resume = False

# 工作目录
work_dir = './work_dirs/pcan_ntu60_x3d'

# 日志配置
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
log_level = 'INFO'

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# 可视化
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

