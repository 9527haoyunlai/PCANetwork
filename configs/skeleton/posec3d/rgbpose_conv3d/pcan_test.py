ann_file = './data/ma52/MA-52_openpose_28kp/MA52_train.pkl'
ann_file_test = './data/ma52/MA-52_openpose_28kp/MA52_test.pkl'
ann_file_val = './data/ma52/MA-52_openpose_28kp/MA52_val.pkl'
auto_scale_lr = dict(base_batch_size=10, enable=False)
backbone_cfg = dict(
    channel_ratio=4,
    pose_pathway=dict(
        base_channels=32,
        conv1_kernel=(
            1,
            7,
            7,
        ),
        conv1_stride_s=1,
        conv1_stride_t=1,
        dilations=(
            1,
            1,
            1,
        ),
        fusion_kernel=7,
        in_channels=28,
        inflate=(
            0,
            1,
            1,
        ),
        lateral=True,
        lateral_activate=(
            0,
            1,
            1,
        ),
        lateral_infl=16,
        lateral_inv=True,
        num_stages=3,
        out_indices=(2, ),
        pool1_stride_s=1,
        pool1_stride_t=1,
        spatial_strides=(
            2,
            2,
            2,
        ),
        stage_blocks=(
            4,
            6,
            3,
        ),
        temporal_strides=(
            1,
            1,
            1,
        ),
        with_pool2=False),
    rgb_pathway=dict(
        base_channels=64,
        conv1_kernel=(
            1,
            7,
            7,
        ),
        fusion_kernel=7,
        inflate=(
            0,
            0,
            1,
            1,
        ),
        lateral=True,
        lateral_activate=[
            0,
            0,
            1,
            1,
        ],
        lateral_infl=1,
        num_stages=4,
        with_pool2=False),
    speed_ratio=4,
    type='RGBPoseConv3D')
data_preprocessor = dict(
    preprocessors=dict(
        heatmap_imgs=dict(type='ActionDataPreprocessor'),
        imgs=dict(
            format_shape='NCTHW',
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            std=[
                58.395,
                57.12,
                57.375,
            ],
            type='ActionDataPreprocessor')),
    type='MultiModalDataPreprocessor')
data_root = './data/ma52/raw_videos/'
dataset_type = 'PoseDataset'
default_hooks = dict(
    checkpoint=dict(interval=10, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=200, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
head_cfg = dict(
    average_clips='prob',
    in_channels=[
        2048,
        512,
    ],
    loss_components=[
        'rgb',
        'pose',
        'rgb_coarse',
        'pose_coarse',
    ],
    loss_weights=[
        1.0,
        1.0,
        0.5,
        0.5,
    ],
    num_classes=52,
    type='RGBPoseHead')
left_kp = [
    2,
    3,
    4,
    8,
    9,
    10,
    14,
    16,
]
left_limb = (
    0,
    1,
    6,
    7,
    9,
    11,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)
load_from = './pretrained/rgbpose_conv3d_init.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        channel_ratio=4,
        pose_pathway=dict(
            base_channels=32,
            conv1_kernel=(
                1,
                7,
                7,
            ),
            conv1_stride_s=1,
            conv1_stride_t=1,
            dilations=(
                1,
                1,
                1,
            ),
            fusion_kernel=7,
            in_channels=28,
            inflate=(
                0,
                1,
                1,
            ),
            lateral=True,
            lateral_activate=(
                0,
                1,
                1,
            ),
            lateral_infl=16,
            lateral_inv=True,
            num_stages=3,
            out_indices=(2, ),
            pool1_stride_s=1,
            pool1_stride_t=1,
            spatial_strides=(
                2,
                2,
                2,
            ),
            stage_blocks=(
                4,
                6,
                3,
            ),
            temporal_strides=(
                1,
                1,
                1,
            ),
            with_pool2=False),
        rgb_pathway=dict(
            base_channels=64,
            conv1_kernel=(
                1,
                7,
                7,
            ),
            fusion_kernel=7,
            inflate=(
                0,
                0,
                1,
                1,
            ),
            lateral=True,
            lateral_activate=[
                0,
                0,
                1,
                1,
            ],
            lateral_infl=1,
            num_stages=4,
            with_pool2=False),
        speed_ratio=4,
        type='RGBPoseConv3D'),
    cls_head=dict(
        average_clips='prob',
        in_channels=[
            2048,
            512,
        ],
        loss_components=[
            'rgb',
            'pose',
            'rgb_coarse',
            'pose_coarse',
        ],
        loss_weights=[
            1.0,
            1.0,
            0.5,
            0.5,
        ],
        num_classes=52,
        type='RGBPoseHead'),
    data_preprocessor=dict(
        preprocessors=dict(
            heatmap_imgs=dict(type='ActionDataPreprocessor'),
            imgs=dict(
                format_shape='NCTHW',
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                type='ActionDataPreprocessor')),
        type='MultiModalDataPreprocessor'),
    type='MMRecognizer3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.0075, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=30,
        gamma=0.1,
        milestones=[
            10,
            20,
        ],
        type='MultiStepLR'),
]
resume = False
right_kp = [
    5,
    6,
    7,
    11,
    12,
    13,
    15,
    17,
]
right_limb = (
    2,
    3,
    4,
    5,
    8,
    10,
    13,
    15,
    22,
    23,
    24,
    25,
    26,
)
skeletons = (
    (
        4,
        3,
    ),
    (
        3,
        2,
    ),
    (
        7,
        6,
    ),
    (
        6,
        5,
    ),
    (
        13,
        12,
    ),
    (
        12,
        11,
    ),
    (
        10,
        9,
    ),
    (
        9,
        8,
    ),
    (
        11,
        5,
    ),
    (
        8,
        2,
    ),
    (
        5,
        1,
    ),
    (
        2,
        1,
    ),
    (
        0,
        1,
    ),
    (
        15,
        0,
    ),
    (
        14,
        0,
    ),
    (
        17,
        15,
    ),
    (
        16,
        14,
    ),
    (
        23,
        4,
    ),
    (
        24,
        4,
    ),
    (
        25,
        4,
    ),
    (
        26,
        4,
    ),
    (
        27,
        4,
    ),
    (
        18,
        7,
    ),
    (
        19,
        7,
    ),
    (
        20,
        7,
    ),
    (
        21,
        7,
    ),
    (
        22,
        7,
    ),
)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='./data/ma52/MA-52_openpose_28kp/MA52_test.pkl',
        data_prefix=dict(video='./data/ma52/raw_videos/'),
        pipeline=[
            dict(
                clip_len=dict(Pose=32, RGB=8),
                num_clips=10,
                test_mode=True,
                type='MMUniformSampleFrames'),
            dict(type='MMDecode'),
            dict(allow_imgpad=True, hw_ratio=1.0, type='MMCompact'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(
                scaling=0.25,
                sigma=0.7,
                type='GeneratePoseTarget',
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                collect_keys=(
                    'imgs',
                    'heatmap_imgs',
                ),
                type='PackActionInputs'),
        ],
        test_mode=True,
        type='PoseDataset'),
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='AccMetric'),
]
test_pipeline = [
    dict(
        clip_len=dict(Pose=32, RGB=8),
        num_clips=10,
        test_mode=True,
        type='MMUniformSampleFrames'),
    dict(type='MMDecode'),
    dict(allow_imgpad=True, hw_ratio=1.0, type='MMCompact'),
    dict(keep_ratio=False, scale=(
        256,
        256,
    ), type='Resize'),
    dict(
        scaling=0.25,
        sigma=0.7,
        type='GeneratePoseTarget',
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(collect_keys=(
        'imgs',
        'heatmap_imgs',
    ), type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=30, type='EpochBasedTrainLoop', val_begin=3, val_interval=1)
train_dataloader = dict(
    batch_size=10,
    dataset=dict(
        ann_file='./data/ma52/MA-52_openpose_28kp/MA52_train.pkl',
        data_prefix=dict(video='./data/ma52/raw_videos/'),
        pipeline=[
            dict(
                clip_len=dict(Pose=32, RGB=8),
                num_clips=1,
                type='MMUniformSampleFrames'),
            dict(type='MMDecode'),
            dict(allow_imgpad=True, hw_ratio=1.0, type='MMCompact'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(area_range=(
                0.56,
                1.0,
            ), type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(
                flip_ratio=0.5,
                left_kp=[
                    2,
                    3,
                    4,
                    8,
                    9,
                    10,
                    14,
                    16,
                ],
                right_kp=[
                    5,
                    6,
                    7,
                    11,
                    12,
                    13,
                    15,
                    17,
                ],
                type='Flip'),
            dict(
                scaling=0.25,
                sigma=0.7,
                type='GeneratePoseTarget',
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                collect_keys=(
                    'imgs',
                    'heatmap_imgs',
                ),
                type='PackActionInputs'),
        ],
        type='PoseDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        clip_len=dict(Pose=32, RGB=8),
        num_clips=1,
        type='MMUniformSampleFrames'),
    dict(type='MMDecode'),
    dict(allow_imgpad=True, hw_ratio=1.0, type='MMCompact'),
    dict(keep_ratio=False, scale=(
        256,
        256,
    ), type='Resize'),
    dict(area_range=(
        0.56,
        1.0,
    ), type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(
        flip_ratio=0.5,
        left_kp=[
            2,
            3,
            4,
            8,
            9,
            10,
            14,
            16,
        ],
        right_kp=[
            5,
            6,
            7,
            11,
            12,
            13,
            15,
            17,
        ],
        type='Flip'),
    dict(
        scaling=0.25,
        sigma=0.7,
        type='GeneratePoseTarget',
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(collect_keys=(
        'imgs',
        'heatmap_imgs',
    ), type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='./data/ma52/MA-52_openpose_28kp/MA52_val.pkl',
        data_prefix=dict(video='./data/ma52/raw_videos/'),
        pipeline=[
            dict(
                clip_len=dict(Pose=32, RGB=8),
                num_clips=1,
                test_mode=True,
                type='MMUniformSampleFrames'),
            dict(type='MMDecode'),
            dict(allow_imgpad=True, hw_ratio=1.0, type='MMCompact'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(
                scaling=0.25,
                sigma=0.7,
                type='GeneratePoseTarget',
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                collect_keys=(
                    'imgs',
                    'heatmap_imgs',
                ),
                type='PackActionInputs'),
        ],
        test_mode=True,
        type='PoseDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AccMetric'),
]
val_pipeline = [
    dict(
        clip_len=dict(Pose=32, RGB=8),
        num_clips=1,
        test_mode=True,
        type='MMUniformSampleFrames'),
    dict(type='MMDecode'),
    dict(allow_imgpad=True, hw_ratio=1.0, type='MMCompact'),
    dict(keep_ratio=False, scale=(
        256,
        256,
    ), type='Resize'),
    dict(
        scaling=0.25,
        sigma=0.7,
        type='GeneratePoseTarget',
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(collect_keys=(
        'imgs',
        'heatmap_imgs',
    ), type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
