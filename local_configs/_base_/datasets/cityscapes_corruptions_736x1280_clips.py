# dataset settings for Cityscapes sequence corruptions
dataset_type = 'CityscapesCorruptionsDataset_clips'
data_root = '/home/wangcl/data/open_video_DGSS/cityscapes_sequence'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (480, 480)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 480),
        flip=False,
        transforms=[
            dict(type='AlignedResize_clips', keep_ratio=False, size_divisor=32),
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
        img_dir='leftImg8bit_sequence_Corruptions',
        ann_dir='gtFine',
        img_suffix='_leftImg8bit.png',
        seg_map_suffix='_gtFine_label14TrainIds.png',
        split=None,
        pipeline=test_pipeline,
        dilation=[-9, -6, -3],
        flip_video=False,
        group_by_camera=False,
        reduce_zero_label=False,
        ignore_index=255,
        corruption=None),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit_sequence_Corruptions',
        ann_dir='gtFine',
        img_suffix='_leftImg8bit.png',
        seg_map_suffix='_gtFine_label14TrainIds.png',
        split=None,
        pipeline=test_pipeline,
        dilation=[-9, -6, -3],
        flip_video=False,
        group_by_camera=False,
        reduce_zero_label=False,
        ignore_index=255,
        corruption=None))
