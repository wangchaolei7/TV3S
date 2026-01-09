#!/usr/bin/env python
import argparse
import os
import re
import sys
import tempfile
from functools import partial

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from PIL import Image
from torch.utils.data import DataLoader, Sampler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SSP_ROOT = os.environ.get("SSP_ROOT", "/home/wangcl/SSP")
if os.path.isdir(SSP_ROOT) and SSP_ROOT not in sys.path:
    sys.path.insert(0, SSP_ROOT)

try:
    from vis_utils.visualization import (
        color_predictions,
        inverse_normalize,
        pred_to_mask,
    )
except Exception as exc:
    raise ImportError(
        "Failed to import SSP visualization helpers. Ensure SSP_ROOT is "
        "set and /home/wangcl/SSP/vis_utils is available."
    ) from exc

EVAL_CROP_SIZE = (480, 480)

MVC_STRIDE = 4
MVC_N_LIST = (8, 16)
CITYS_SIM_THRESH = 20.0

CLASSES_15 = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "sky",
    "person",
    "rider",
    "car",
    "Truck_Bus",
    "motorcycle",
)

PALETTE_15 = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 60, 100],
    [0, 0, 230],
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="TV3S eval with OVDG-style metrics (mIoU/mAcc/aAcc + mVC)."
    )
    parser.add_argument("config", help="Config file path")
    parser.add_argument("checkpoint", help="Checkpoint file path")
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Directory to save metrics and optional predictions",
    )
    parser.add_argument(
        "--split", default="val", help="Dataset split to evaluate (val/test)"
    )
    parser.add_argument(
        "--write-res",
        dest="write_res",
        action="store_true",
        help="Write prediction masks (and colored masks) to disk",
    )
    parser.add_argument(
        "--no-write-res",
        dest="write_res",
        action="store_false",
        help="Do not write prediction masks (default)",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute metrics only (implies --no-write-res)",
    )
    parser.add_argument(
        "--metrics-on",
        action="store_true",
        help="No-op (metrics are always computed); kept for compatibility.",
    )
    parser.set_defaults(write_res=False)
    return parser.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return True
    return dist.is_initialized()


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process():
    return get_rank() == 0


def all_reduce_tensor(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _update_confusion(confusion, pred, label, n_classes, ignore_index):
    pred = torch.as_tensor(pred, dtype=torch.int64).view(-1)
    label = torch.as_tensor(label, dtype=torch.int64).view(-1)
    if ignore_index is not None:
        mask = label != ignore_index
        pred = pred[mask]
        label = label[mask]
    if pred.numel() == 0:
        return confusion
    idx = label * n_classes + pred
    confusion += torch.bincount(idx, minlength=n_classes * n_classes).reshape(
        n_classes, n_classes
    )
    return confusion


def _iou_from_confusion(confusion):
    intersection = torch.diag(confusion).float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
    iou = intersection / (union + 1e-6)
    return iou, union


def _summarize_confusion(confusion):
    conf = confusion.astype(np.float64)
    gt_sum = conf.sum(axis=1)
    pred_sum = conf.sum(axis=0)
    intersect = np.diag(conf)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = intersect / gt_sum
        iou = intersect / (gt_sum + pred_sum - intersect)
    total = conf.sum()
    a_acc = float(intersect.sum() / total) if total > 0 else float("nan")
    valid_mask = ~(np.isnan(acc) | (iou == 0))
    if valid_mask.any():
        m_acc = float(np.nanmean(acc[valid_mask]))
        m_iou = float(np.nanmean(iou[valid_mask]))
    else:
        m_acc = float("nan")
        m_iou = float("nan")
    return a_acc, m_acc, m_iou, acc, iou


def _infer_sequence_key(rel_path):
    rel_path = os.path.normpath(rel_path)
    seq_dir = os.path.dirname(rel_path)
    frame = os.path.basename(rel_path)
    match = re.search(r"_Camera_(\d+)", frame)
    if match:
        camera = f"Camera_{match.group(1)}"
        return os.path.join(seq_dir, camera) if seq_dir else camera
    return seq_dir if seq_dir else ""


def _is_cityscapes_sequence(path):
    if not path:
        return False
    return (
        "origin_leftImg8bit_sequence" in path
        or "leftImg8bit_sequence_Corruptions" in path
    )


def _build_sequence_indices(dataset):
    if not hasattr(dataset, "img_infos"):
        return None
    seq_to_items = {}
    for idx, info in enumerate(dataset.img_infos):
        rel = info.get("filename")
        seq_key = info.get("sequence_key") or _infer_sequence_key(rel)
        seq_to_items.setdefault(seq_key, []).append((rel, idx))
    seq_to_indices = {}
    for seq_key, items in seq_to_items.items():
        items.sort(key=lambda x: x[0])
        seq_to_indices[seq_key] = [idx for _, idx in items]
    return seq_to_indices


class SequenceDistributedSampler(Sampler):
    def __init__(self, dataset, world_size=1, rank=0):
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self._indices = self._build_indices()

    def _build_indices(self):
        seq_to_indices = _build_sequence_indices(self.dataset)
        if not seq_to_indices:
            return list(range(len(self.dataset)))
        seq_keys = sorted(seq_to_indices.keys())
        rank_seq_keys = seq_keys[self.rank :: self.world_size]
        indices = []
        for key in rank_seq_keys:
            indices.extend(seq_to_indices[key])
        return indices

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)


def _resize_hwc(img, target_hw):
    return mmcv.imresize(
        img, (target_hw[1], target_hw[0]), interpolation="bilinear"
    )


def _get_anchor_image(data, img_meta):
    img = data.get("img")
    if hasattr(img, "data"):
        img = img.data
    if isinstance(img, (list, tuple)):
        img = img[0]
    if not torch.is_tensor(img):
        return None
    if img.dim() == 5:
        img = img[0]
    if img.dim() == 4:
        img = img[-1]
    elif img.dim() != 3:
        return None
    img = img.detach().cpu().numpy()
    norm_cfg = img_meta.get("img_norm_cfg", {}) if isinstance(img_meta, dict) else {}
    mean = norm_cfg.get("mean")
    std = norm_cfg.get("std")
    to_rgb = norm_cfg.get("to_rgb", True)
    if mean is not None and std is not None:
        mean = np.asarray(mean, dtype=np.float32) / 255.0
        std = np.asarray(std, dtype=np.float32) / 255.0
        img = inverse_normalize(img, mean=mean, std=std)
    else:
        img = inverse_normalize(img)
    if not to_rgb:
        img = img[..., ::-1]
    return img


def _load_raw_frame(img_path, to_rgb, target_hw=None):
    img = mmcv.imread(img_path, flag="color")
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    if to_rgb:
        img = img[..., ::-1]
    if target_hw is not None:
        img = _resize_hwc(img, target_hw)
    img = img.astype(np.float32)
    return img.transpose(2, 0, 1)


class MVCTracker:
    def __init__(self, ignore_index, stride=4, mvc_n=(8, 16), citys_sim_thresh=20.0):
        self.ignore_index = int(ignore_index)
        self.stride = max(1, int(stride))
        self.mvc_n_list = sorted({max(1, int(x)) for x in mvc_n})
        self.citys_sim_thresh = float(citys_sim_thresh)
        self.seq_cache = {}
        self.last_seq_key = None
        self.mvc_sum = {n: 0.0 for n in self.mvc_n_list}
        self.mvc_cnt = {n: 0 for n in self.mvc_n_list}

    def _downsample(self, arr):
        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if self.stride > 1:
            return arr[:: self.stride, :: self.stride]
        return arr

    def _load_small_gray_image(self, img_path, h_small, w_small):
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            return None
        img = img.resize((w_small, h_small), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def _compute_vc_dense(self, preds, gts, n):
        length = len(preds)
        if length < n:
            return None
        vals = []
        for start in range(0, length - n + 1):
            pred_win = preds[start : start + n]
            gt_win = gts[start : start + n]

            gt0 = gt_win[0]
            gt_equal = np.ones(gt0.shape, dtype=bool)
            gt_valid = np.ones(gt0.shape, dtype=bool)
            for gt in gt_win:
                gt_valid &= gt != self.ignore_index
            for gt in gt_win[1:]:
                gt_equal &= gt == gt0
            gt_common = gt_equal & gt_valid
            denom = int(gt_common.sum())
            if denom == 0:
                continue

            pred0 = pred_win[0]
            pred_equal = np.ones(pred0.shape, dtype=bool)
            for pred in pred_win[1:]:
                pred_equal &= pred == pred0

            num = int((gt_common & pred_equal & (pred0 == gt0)).sum())
            vals.append(num / denom)
        if not vals:
            return None
        return float(np.mean(vals))

    def _compute_vc_sparse_by_valid_windows(self, preds, gts, n):
        length = len(preds)
        if length < n:
            return None
        vals = []
        for start in range(0, length - n + 1):
            gt_win = gts[start : start + n]
            if any(x is None for x in gt_win):
                continue
            vc = self._compute_vc_dense(
                preds[start : start + n],
                [x for x in gt_win if x is not None],
                n,
            )
            if vc is not None:
                vals.append(vc)
        if not vals:
            return None
        return float(np.mean(vals))

    def _compute_vc_citys_sparse(self, preds, imgs, ref_gt, ref_img, n):
        length = len(preds)
        if length < n:
            return None
        if ref_gt is None or ref_img is None:
            return None
        if len(imgs) != length:
            return None
        if any(img is None for img in imgs):
            return None

        vals = []
        ref_gt_valid = ref_gt != self.ignore_index
        for start in range(0, length - n + 1):
            pred_win = preds[start : start + n]
            img_win = imgs[start : start + n]

            static_all = np.ones(ref_img.shape, dtype=bool)
            for img_t in img_win:
                diff = np.abs(img_t.astype(np.int16) - ref_img.astype(np.int16))
                static_all &= diff <= self.citys_sim_thresh

            m_common = static_all & ref_gt_valid
            denom = int(m_common.sum())
            if denom == 0:
                continue

            pred0 = pred_win[0]
            pred_equal = np.ones(pred0.shape, dtype=bool)
            for pred in pred_win[1:]:
                pred_equal &= pred == pred0

            num = int((m_common & pred_equal & (pred0 == ref_gt)).sum())
            vals.append(num / denom)
        if not vals:
            return None
        return float(np.mean(vals))

    def update(self, seq_key, pred, gt, img_path, use_citys_sparse):
        if seq_key is None:
            seq_key = ""
        if self.last_seq_key is not None and self.last_seq_key != seq_key:
            self._flush_seq(self.last_seq_key)
        self.last_seq_key = seq_key

        pred_small = self._downsample(pred)
        if pred_small is None:
            return
        if pred_small.dtype != np.uint8:
            pred_small = pred_small.astype(np.uint8, copy=False)
        gt_small = self._downsample(gt) if gt is not None else None
        if gt_small is not None and gt_small.dtype != np.uint8:
            gt_small = gt_small.astype(np.uint8, copy=False)

        cache = self.seq_cache.get(seq_key)
        if cache is None:
            cache = {
                "preds": [],
                "gts": [],
                "imgs": [],
                "ref_gt": None,
                "ref_img": None,
                "use_citys_sparse": bool(use_citys_sparse),
            }
            self.seq_cache[seq_key] = cache
        else:
            cache["use_citys_sparse"] = cache["use_citys_sparse"] or bool(use_citys_sparse)

        cache["preds"].append(pred_small)
        cache["gts"].append(gt_small)

        if use_citys_sparse:
            img_small = None
            if img_path:
                img_small = self._load_small_gray_image(
                    img_path, pred_small.shape[0], pred_small.shape[1]
                )
            cache["imgs"].append(img_small)
            if gt_small is not None and img_small is not None:
                cache["ref_gt"] = gt_small
                cache["ref_img"] = img_small

    def _flush_seq(self, seq_key):
        cache = self.seq_cache.pop(seq_key, None)
        if not cache:
            return

        preds = cache.get("preds", [])
        gts = cache.get("gts", [])
        imgs = cache.get("imgs", [])
        ref_gt = cache.get("ref_gt")
        ref_img = cache.get("ref_img")
        use_citys_sparse = cache.get("use_citys_sparse", False)

        if not preds:
            return

        dense_gt = len(gts) == len(preds) and all(x is not None for x in gts)

        for n in self.mvc_n_list:
            vc_val = None
            if dense_gt:
                vc_val = self._compute_vc_dense(
                    preds, [x for x in gts if x is not None], n
                )
            elif use_citys_sparse:
                vc_val = self._compute_vc_citys_sparse(preds, imgs, ref_gt, ref_img, n)
            else:
                vc_val = self._compute_vc_sparse_by_valid_windows(preds, gts, n)
            if vc_val is not None and not np.isnan(vc_val):
                self.mvc_sum[n] += float(vc_val)
                self.mvc_cnt[n] += 1

    def finalize(self):
        for seq_key in list(self.seq_cache.keys()):
            self._flush_seq(seq_key)
        self.seq_cache.clear()
        self.last_seq_key = None


def _prepare_tmp_dir(save_dir):
    tmp_dir = os.path.join(save_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ.setdefault("TMPDIR", tmp_dir)
    tempfile.tempdir = tmp_dir


def _resolve_colors(dataset, n_classes, ignore_index):
    colors = getattr(dataset, "colors", None) or getattr(dataset, "COLORS", None)
    if n_classes == 15:
        if ignore_index and ignore_index > 0:
            return {0: (0, 0, 0), **{idx + 1: tuple(color) for idx, color in enumerate(PALETTE_15)}}
        return {idx: tuple(color) for idx, color in enumerate(PALETTE_15)}
    if colors:
        return colors
    palette = getattr(dataset, "PALETTE", None)
    if palette:
        if ignore_index and ignore_index > 0:
            return {0: (0, 0, 0), **{idx + 1: tuple(color) for idx, color in enumerate(palette[:n_classes])}}
        return {idx: tuple(color) for idx, color in enumerate(palette[:n_classes])}
    return None


def _prepare_output_dirs(save_dir, split, write_res):
    if not write_res:
        return None
    dirs = {
        "mask": os.path.join(save_dir, split),
        "colored": os.path.join(save_dir, f"{split}_colored"),
        "blended": os.path.join(save_dir, f"{split}_blended"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def _save_predictions_ssp(pred, rel_path, out_dirs, colors, ignore_index, frame_img=None):
    if out_dirs is None:
        return
    rel_out = os.path.splitext(rel_path)[0] + ".png"
    mask = pred_to_mask(pred.copy(), ignore_index)
    mask_path = os.path.join(out_dirs["mask"], rel_out)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(mask_path)

    if colors is None:
        return
    color_arr = color_predictions(pred.copy(), colors=colors, ignore_index=ignore_index)
    color_path = os.path.join(out_dirs["colored"], rel_out)
    os.makedirs(os.path.dirname(color_path), exist_ok=True)
    Image.fromarray(color_arr.astype(np.uint8)).save(color_path)

    if frame_img is None:
        return
    if frame_img.shape[:2] != pred.shape[:2]:
        frame_img = mmcv.imresize(
            frame_img, (pred.shape[1], pred.shape[0]), interpolation="bilinear"
        )
    _, blended = color_predictions(
        pred.copy(), colors=colors, ignore_index=ignore_index, blend_img=frame_img
    )
    blended_path = os.path.join(out_dirs["blended"], rel_out)
    os.makedirs(os.path.dirname(blended_path), exist_ok=True)
    blended.save(blended_path)


def _override_test_pipeline_scale(pipeline, crop_size):
    img_scale = (crop_size[1], crop_size[0])
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        step_type = step.get("type")
        if step_type == "MultiScaleFlipAug":
            step["img_scale"] = img_scale
            transforms = step.get("transforms")
            if transforms:
                _override_test_pipeline_scale(transforms, crop_size)
        elif step_type in (
            "AlignedResize_clips",
            "AlignedResize",
            "Resize",
            "Resize_clips",
        ):
            step["img_scale"] = img_scale
            step["keep_ratio"] = False


def main():
    args = parse_args()
    if args.metrics_only:
        args.write_res = False

    _prepare_tmp_dir(args.save_dir)

    # Register custom models/datasets.
    import utils  # noqa: F401

    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    distributed = init_distributed()
    rank = get_rank()
    world_size = get_world_size()

    data_cfg = cfg.data.get(args.split, cfg.data.test)
    if data_cfg.get("type") == "RepeatDataset":
        data_cfg = data_cfg["dataset"]
    data_cfg = mmcv.ConfigDict(data_cfg)
    data_cfg.test_mode = True
    if hasattr(data_cfg, "pipeline"):
        _override_test_pipeline_scale(data_cfg.pipeline, EVAL_CROP_SIZE)
    if hasattr(cfg.data, "test"):
        cfg.data.test.test_mode = True
    dataset = build_dataset(data_cfg)

    sampler = SequenceDistributedSampler(dataset, world_size=world_size, rank=rank) if distributed else None
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.data.get("workers_per_gpu", 2),
        collate_fn=partial(collate, samples_per_gpu=1),
        pin_memory=True,
        drop_last=False,
    )

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if cfg.model.get("decode_head", None) and cfg.model.decode_head.get("decoder_params", None):
        cfg.model.decode_head.decoder_params.test_mode = True
        cfg.model.decode_head.decoder_params.val_mode = 0

    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    classes = None
    palette = None
    if "meta" in checkpoint:
        classes = checkpoint["meta"].get("CLASSES", getattr(dataset, "CLASSES", None))
        palette = checkpoint["meta"].get("PALETTE", getattr(dataset, "PALETTE", None))
    else:
        classes = getattr(dataset, "CLASSES", None)
        palette = getattr(dataset, "PALETTE", None)

    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
    else:
        model = MMDataParallel(model, device_ids=[0])

    base_model = model.module if hasattr(model, "module") else model

    n_classes = len(classes) if classes is not None else cfg.model.decode_head.num_classes
    if classes is None and n_classes == 15:
        classes = CLASSES_15
    if palette is None and n_classes == 15:
        palette = PALETTE_15
    base_model.CLASSES = classes
    base_model.PALETTE = palette
    ignore_index = getattr(dataset, "ignore_index", 255)
    colors = _resolve_colors(dataset, n_classes, ignore_index)
    out_dirs = _prepare_output_dirs(args.save_dir, args.split, args.write_res)
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    mvc_tracker = MVCTracker(
        ignore_index=ignore_index,
        stride=MVC_STRIDE,
        mvc_n=MVC_N_LIST,
        citys_sim_thresh=CITYS_SIM_THRESH,
    )
    reduce_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ann_map = {}
    seg_suffix = getattr(dataset, "seg_map_suffix", None)
    img_suffix = getattr(dataset, "img_suffix", None)
    if dataset.ann_dir and seg_suffix and img_suffix:
        for ann_rel in mmcv.scandir(dataset.ann_dir, seg_suffix, recursive=True):
            img_rel = ann_rel[: -len(seg_suffix)] + img_suffix
            ann_map[img_rel] = ann_rel
    else:
        for info in getattr(dataset, "img_infos", []):
            rel = info.get("filename")
            seg = info.get("ann", {}).get("seg_map")
            if rel is not None and seg is not None:
                ann_map[rel] = seg

    total = len(sampler) if sampler is not None else len(dataset)
    prog_bar = mmcv.ProgressBar(total) if is_main_process() else None
    model.eval()
    for data in data_loader:
        with torch.no_grad():
            for d in range(len(data["img_metas"])):
                data["img_metas"][d] = data["img_metas"][d].data[0]
            result = model(return_loss=False, rescale=True, **data)

        pred = result[0] if isinstance(result, list) else result
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        img_meta = data["img_metas"][0]
        if isinstance(img_meta, list):
            img_meta = img_meta[0]
        rel_path = img_meta.get("ori_filename", "")
        if isinstance(rel_path, (list, tuple)):
            rel_path = rel_path[0]
        filename = img_meta.get("filename")
        if isinstance(filename, (list, tuple)):
            filename = filename[0]
        if not filename and rel_path:
            filename = os.path.join(dataset.img_dir, rel_path)

        pred = pred.astype(np.int64, copy=False)
        gt = None
        gt_rel = ann_map.get(rel_path)
        if gt_rel is not None:
            gt_path = os.path.join(dataset.ann_dir, gt_rel)
            gt = mmcv.imread(gt_path, flag="unchanged")
            if gt.ndim == 3:
                gt = gt[:, :, 0]
            if gt.shape != pred.shape:
                gt = mmcv.imresize(gt, pred.shape[::-1], interpolation="nearest")
            confusion = _update_confusion(confusion, pred, gt, n_classes, ignore_index)

        seq_key = img_meta.get("sequence_key") or _infer_sequence_key(rel_path)
        use_citys_sparse = _is_cityscapes_sequence(filename or rel_path)
        mvc_tracker.update(seq_key, pred, gt, filename, use_citys_sparse)

        if args.write_res:
            frame_vis = _get_anchor_image(data, img_meta)
            if frame_vis is None and filename:
                try:
                    frame = _load_raw_frame(
                        filename,
                        img_meta.get("img_norm_cfg", {}).get("to_rgb", False),
                        target_hw=EVAL_CROP_SIZE,
                    )
                except FileNotFoundError:
                    frame = None
                if frame is not None:
                    frame_vis = np.clip(
                        frame.transpose(1, 2, 0), 0, 255
                    ).astype(np.uint8)
                    if not img_meta.get("img_norm_cfg", {}).get("to_rgb", True):
                        frame_vis = frame_vis[..., ::-1]
            _save_predictions_ssp(
                pred, rel_path, out_dirs, colors, ignore_index, frame_img=frame_vis
            )

        if prog_bar is not None:
            prog_bar.update()

    mvc_tracker.finalize()
    mvc_n_list = list(mvc_tracker.mvc_n_list)
    mvc_sum = np.array([mvc_tracker.mvc_sum[n] for n in mvc_n_list], dtype=np.float64)
    mvc_cnt = np.array([mvc_tracker.mvc_cnt[n] for n in mvc_n_list], dtype=np.float64)

    if distributed:
        confusion = all_reduce_tensor(confusion.to(reduce_device)).cpu()
        mvc_sum_tensor = all_reduce_tensor(torch.tensor(mvc_sum, device=reduce_device))
        mvc_cnt_tensor = all_reduce_tensor(torch.tensor(mvc_cnt, device=reduce_device))
        mvc_sum = mvc_sum_tensor.cpu().numpy()
        mvc_cnt = mvc_cnt_tensor.cpu().numpy()

    a_acc, m_acc, m_iou, per_class_acc, per_class_iou = _summarize_confusion(
        confusion.cpu().numpy()
    )
    mvc_vals = np.divide(mvc_sum, np.maximum(mvc_cnt, 1.0))
    mvc_vals = np.where(mvc_cnt > 0, mvc_vals, np.nan)

    a_acc_pct = a_acc * 100.0 if not np.isnan(a_acc) else float("nan")
    m_acc_pct = m_acc * 100.0 if not np.isnan(m_acc) else float("nan")
    m_iou_pct = m_iou * 100.0 if not np.isnan(m_iou) else float("nan")
    per_class_iou_pct = per_class_iou * 100.0
    per_class_acc_pct = per_class_acc * 100.0
    mvc_pct = mvc_vals * 100.0

    if is_main_process():
        if base_model.CLASSES:
            class_names = list(base_model.CLASSES)
        elif n_classes == 15:
            class_names = list(CLASSES_15)
        else:
            class_names = [f"class_{idx}" for idx in range(n_classes)]
        os.makedirs(args.save_dir, exist_ok=True)
        metrics_path = os.path.join(args.save_dir, f"log_metrics_{args.split}.txt")
        with open(metrics_path, "a") as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"mIoU = {m_iou_pct:.2f}\n")
            f.write(f"mAcc = {m_acc_pct:.2f}\n")
            f.write(f"aAcc = {a_acc_pct:.2f}\n")
            for n, mvc_val, mvc_count in zip(mvc_n_list, mvc_pct, mvc_cnt):
                if np.isnan(mvc_val):
                    f.write(f"mVC{n} = nan (videos={int(mvc_count)})\n")
                else:
                    f.write(f"mVC{n} = {mvc_val:.2f} (videos={int(mvc_count)})\n")
            f.write("per class IoU (%):\n")
            for idx, name in enumerate(class_names):
                f.write(f"  {name}: {per_class_iou_pct[idx]:.5f}\n")
            f.write("per class Acc (%):\n")
            for idx, name in enumerate(class_names):
                f.write(f"  {name}: {per_class_acc_pct[idx]:.5f}\n")
            f.write("\n")
        mvc_parts = [
            f"mVC{n} = {val:.2f}" if not np.isnan(val) else f"mVC{n} = nan"
            for n, val in zip(mvc_n_list, mvc_pct)
        ]
        print(
            " | ".join(
                [
                    f"mIoU = {m_iou_pct:.2f}",
                    f"mAcc = {m_acc_pct:.2f}",
                    f"aAcc = {a_acc_pct:.2f}",
                ]
                + mvc_parts
            )
        )
        print(f"Metrics saved to {metrics_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
