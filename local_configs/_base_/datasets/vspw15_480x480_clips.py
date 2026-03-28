# dataset settings
dataset_type = 'VSPWCrossDomainDataset_clips'
data_root = '/data1/wangcl/dataset/open_video_DGSS/VSPW_val'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (480, 480)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(853, 480), ratio_range=(0.5, 2.0), process_clips=True),
    dict(type='RandomCrop_clips', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip_clips', prob=0.5),
    dict(type='PhotoMetricDistortion_clips'),
    dict(type='Normalize_clips', **img_norm_cfg),
    dict(type='Pad_clips', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle_clips'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(853, 480),
        flip=False,
        transforms=[
            dict(type='AlignedResize_clips', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip_clips'),
            dict(type='Normalize_clips', **img_norm_cfg),
            dict(type='ImageToTensor_clips', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='ColorImage',
        ann_dir='Label_classes15',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split='val',
        pipeline=test_pipeline,
        dilation=[-9, -6, -3],
        flip_video=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='ColorImage',
        ann_dir='Label_classes15',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split='val',
        pipeline=test_pipeline,
        dilation=[-9, -6, -3],
        flip_video=False,
        mamba_mode=False))
