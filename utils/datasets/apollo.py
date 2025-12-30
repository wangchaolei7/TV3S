import os
import os.path as osp
import random
import re

from mmcv.utils import print_log
from mmseg.datasets import DATASETS
from mmseg.utils import get_root_logger

from .custom import CustomDataset_cityscape_clips


@DATASETS.register_module()
class ApolloScapeDataset_clips(CustomDataset_cityscape_clips):
    """ApolloScape 15Label dataset with clip sampling from sequence folders."""

    CLASSES = (
        'background',
        'class_0',
        'class_1',
        'class_2',
        'class_3',
        'class_4',
        'class_5',
        'class_6',
        'class_7',
        'class_8',
        'class_9',
        'class_10',
        'class_11',
        'class_12',
        'class_13',
    )

    PALETTE = [
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
    ]

    def __init__(
        self,
        pipeline,
        img_dir,
        img_suffix='.jpg',
        ann_dir=None,
        seg_map_suffix='_bin.png',
        split=None,
        data_root=None,
        test_mode=False,
        ignore_index=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        dilation=(-9, -6, -3),
        flip_video=True,
        group_by_camera=True,
        istraining=None,
        gene_prototype=False,
        mamba_mode=False,
    ):
        if istraining is None:
            istraining = split == 'train' and not test_mode

        self._group_by_camera = group_by_camera
        self._camera_pattern = re.compile(r'_Camera_(\d+)')
        self._flip_video = flip_video
        self.split_name = split
        split_for_super = None

        super().__init__(
            pipeline=pipeline,
            img_dir=img_dir,
            img_suffix=img_suffix,
            ann_dir=ann_dir,
            seg_map_suffix=seg_map_suffix,
            split=split_for_super,
            data_root=data_root,
            test_mode=test_mode,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            classes=classes,
            palette=palette,
            dilation=list(dilation),
            istraining=istraining,
            gene_prototype=gene_prototype,
            mamba_mode=mamba_mode,
        )

        self.flip_video = self._flip_video
        self._build_sequence_index()
        self._filter_img_infos()

    def _split_relpath(self, rel_path):
        rel_path = osp.normpath(rel_path)
        seq_dir = osp.dirname(rel_path)
        if seq_dir == '.':
            seq_dir = ''
        frame = osp.basename(rel_path)
        return seq_dir, frame

    def _extract_camera_tag(self, rel_path):
        if not self._group_by_camera:
            return None
        match = self._camera_pattern.search(rel_path)
        if not match:
            return None
        return f'Camera_{match.group(1)}'

    def _sequence_key_from_relpath(self, rel_path):
        seq_dir, frame = self._split_relpath(rel_path)
        camera = self._extract_camera_tag(rel_path)
        if camera:
            seq_key = osp.join(seq_dir, camera) if seq_dir else camera
        else:
            seq_key = seq_dir
        return seq_dir, seq_key, frame

    def _split_sequence_key(self, seq_key):
        if not self._group_by_camera:
            return seq_key, None
        parts = seq_key.split(osp.sep)
        if parts and parts[-1].startswith('Camera_'):
            seq_dir = osp.join(*parts[:-1]) if len(parts) > 1 else ''
            return seq_dir, parts[-1]
        return seq_key, None

    def _build_sequence_index(self):
        sequences = {}
        for info in self.img_infos:
            seq_dir, seq_key, _ = self._sequence_key_from_relpath(info['filename'])
            sequences[seq_key] = seq_dir

        self.seq_frames = {}
        self.seq_indices = {}
        missing_dirs = []

        for seq_key, seq_dir in sorted(sequences.items()):
            seq_path = osp.join(self.img_dir, seq_dir) if seq_dir else self.img_dir
            if not osp.isdir(seq_path):
                missing_dirs.append(seq_path)
                continue
            frames = [
                f for f in os.listdir(seq_path)
                if f.lower().endswith(self.img_suffix.lower())
            ]
            _, camera = self._split_sequence_key(seq_key)
            if camera:
                frames = [f for f in frames if camera in f]
            frames.sort()
            if not frames:
                continue
            self.seq_frames[seq_key] = frames
            self.seq_indices[seq_key] = {f: i for i, f in enumerate(frames)}

        if missing_dirs:
            print_log(
                f'Found {len(missing_dirs)} image folders missing in ApolloScape',
                logger=get_root_logger(),
            )

    def _filter_img_infos(self):
        valid_infos = []
        missing = 0
        for info in self.img_infos:
            seq_dir, seq_key, frame = self._sequence_key_from_relpath(info['filename'])
            if seq_key not in self.seq_indices:
                missing += 1
                continue
            if frame not in self.seq_indices[seq_key]:
                missing += 1
                continue
            info['sequence_dir'] = seq_dir
            info['sequence_key'] = seq_key
            info['frame_name'] = frame
            valid_infos.append(info)
        if missing:
            print_log(
                f'Filtered {missing} annotations without matching images',
                logger=get_root_logger(),
            )
        self.img_infos = valid_infos

    def _resolve_frame(self, seq_key, anchor_frame, offset):
        frames = self.seq_frames.get(seq_key)
        if not frames:
            return anchor_frame
        anchor_idx = self.seq_indices[seq_key].get(anchor_frame, 0)
        target_idx = anchor_idx + offset
        if target_idx < 0:
            target_idx = 0
        elif target_idx >= len(frames):
            target_idx = len(frames) - 1
        return frames[target_idx]

    def _build_img_anns(self, img_info, ann_info, dilation_used):
        seq_dir = img_info.get('sequence_dir')
        seq_key = img_info.get('sequence_key')
        frame = img_info.get('frame_name')
        if seq_key is None or frame is None or seq_dir is None:
            seq_dir, seq_key, frame = self._sequence_key_from_relpath(img_info['filename'])

        img_anns = []
        dilation_used = list(dilation_used or [])
        for offset in dilation_used:
            frame_name = self._resolve_frame(seq_key, frame, offset)
            rel_path = osp.join(seq_dir, frame_name) if seq_dir else frame_name
            img_info_one = dict(filename=rel_path, ann=dict(seg_map=ann_info['seg_map']))
            img_anns.append([img_info_one, img_info_one['ann']])

        img_anns.append([img_info, ann_info])
        return img_anns

    def _pack_results(self, img_anns, include_gt):
        img_info_clips, ann_info_clips, seg_fields_clips = [], [], []
        img_prefix_clips, seg_prefix_clips, filename_clips = [], [], []
        ori_filename_clips, img_clips = [], []
        img_shape_clips, ori_shape_clips, pad_shape_clips = [], [], []
        scale_factor_clips, img_norm_cfg_clips = [], []
        gt_semantic_seg_clips = []

        for img_info_one, ann_info_one in img_anns:
            results = dict(img_info=img_info_one, ann_info=ann_info_one)
            self.pre_pipeline(results)
            self.pipeline_load(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results['seg_fields'])
            img_prefix_clips.append(results['img_prefix'])
            seg_prefix_clips.append(results['seg_prefix'])
            filename_clips.append(results['filename'])
            ori_filename_clips.append(results['ori_filename'])
            img_clips.append(results['img'])
            img_shape_clips.append(results['img_shape'])
            ori_shape_clips.append(results['ori_shape'])
            pad_shape_clips.append(results['pad_shape'])
            scale_factor_clips.append(results['scale_factor'])
            img_norm_cfg_clips.append(results['img_norm_cfg'])
            if include_gt:
                gt_semantic_seg_clips.append(results['gt_semantic_seg'])

        results_new = dict(
            img_info=img_info_clips[-1],
            ann_info=ann_info_clips[-1],
            seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],
            seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],
            ori_filename=ori_filename_clips[-1],
            img=img_clips,
            img_shape=img_shape_clips[-1],
            ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],
            scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1],
        )
        if include_gt:
            results_new['gt_semantic_seg'] = gt_semantic_seg_clips
        return self.pipeline_process(results_new)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        dilation_used = self.dilation
        if self.flip_video and random.random() < 0.5:
            dilation_used = [-offset for offset in dilation_used]

        img_anns = self._build_img_anns(img_info, ann_info, dilation_used)
        return self._pack_results(img_anns, include_gt=True)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        dilation_used = []
        if not self.mamba_mode:
            dilation_used = self.dilation

        img_anns = self._build_img_anns(img_info, ann_info, dilation_used)
        return self._pack_results(img_anns, include_gt=False)
