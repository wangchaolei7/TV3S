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
    from RAFT_core.raft import RAFT
except Exception as exc:
    raise ImportError(
        "Failed to import RAFT from SSP. Set SSP_ROOT or ensure "
        "/home/wangcl/SSP is available."
    ) from exc
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

SSP_COLORS_15 = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (244, 35, 232),
    3: (70, 70, 70),
    4: (102, 102, 156),
    5: (190, 153, 153),
    6: (153, 153, 153),
    7: (250, 170, 30),
    8: (220, 220, 0),
    9: (107, 142, 35),
    10: (152, 251, 152),
    11: (70, 130, 180),
    12: (220, 20, 60),
    13: (255, 0, 0),
    14: (0, 0, 142),
    15: (0, 60, 100),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="TV3S eval with SSP-style metrics (mIoU + TC + per-class mIoU)."
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
        help="Do not write prediction masks",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute metrics only (implies --no-write-res)",
    )
    parser.add_argument(
        "--metrics-on",
        action="store_true",
        help="Alias flag for enabling metrics (default behavior).",
    )
    parser.set_defaults(write_res=True)
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


def _mean_iou(pred, label, n_classes, ignore_index):
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    confusion = _update_confusion(confusion, pred, label, n_classes, ignore_index)
    iou, union = _iou_from_confusion(confusion)
    valid = union > 0
    return iou[valid].mean() if valid.any() else torch.tensor(0.0)


def flowwarp(x, flo):
    b, c, h, w = x.size()
    xx = torch.arange(0, w).view(1, -1).repeat(h, 1)
    yy = torch.arange(0, h).view(-1, 1).repeat(1, w)
    xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
    yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = grid + flo
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(w - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(h - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(
        x, vgrid, mode="nearest", align_corners=False
    )
    return output


def temporal_consistency_pair(
    frame, next_frame, pred, next_pred, model_raft, n_classes, device, ignore_index
):
    frame = torch.from_numpy(frame).unsqueeze(0).to(device)
    next_frame = torch.from_numpy(next_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        model_raft.eval()
        _, flow = model_raft(frame, next_frame, iters=20, test_mode=True)
    flow = flow.detach().cpu()
    pred = torch.from_numpy(pred)
    next_pred = torch.from_numpy(next_pred)
    next_pred = next_pred.unsqueeze(0).unsqueeze(0).float()
    warp_pred = flowwarp(next_pred, flow).int().squeeze(1)
    pred = pred.unsqueeze(0)
    return _mean_iou(warp_pred, pred, n_classes, ignore_index)


def get_flow_model(raft_weights):
    model_raft = RAFT()
    state = torch.load(raft_weights, map_location="cpu", weights_only=True)
    new_state = {}
    for k, v in state.items():
        name = k[7:] if k.startswith("module.") else k
        new_state[name] = v
    model_raft.load_state_dict(new_state)
    return model_raft


def _infer_sequence_key(rel_path):
    rel_path = os.path.normpath(rel_path)
    seq_dir = os.path.dirname(rel_path)
    frame = os.path.basename(rel_path)
    match = re.search(r"_Camera_(\d+)", frame)
    if match:
        camera = f"Camera_{match.group(1)}"
        return os.path.join(seq_dir, camera) if seq_dir else camera
    return seq_dir if seq_dir else ""


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


def _prepare_tmp_dir(save_dir):
    tmp_dir = os.path.join(save_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ.setdefault("TMPDIR", tmp_dir)
    tempfile.tempdir = tmp_dir


def _resolve_colors(dataset, n_classes, ignore_index):
    colors = getattr(dataset, "colors", None) or getattr(dataset, "COLORS", None)
    if colors:
        return colors
    if n_classes == 15:
        if ignore_index and ignore_index > 0:
            return SSP_COLORS_15
        return {idx: SSP_COLORS_15[idx + 1] for idx in range(n_classes)}
    palette = getattr(dataset, "PALETTE", None)
    if palette:
        colors = {0: (0, 0, 0)}
        for idx, color in enumerate(palette[:n_classes]):
            if ignore_index and ignore_index > 0:
                colors[idx + 1] = tuple(color)
            else:
                colors[idx] = tuple(color)
        return colors
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


def _is_consecutive(dataset, seq_key, prev_frame, curr_frame):
    if hasattr(dataset, "seq_indices") and seq_key in dataset.seq_indices:
        idx_map = dataset.seq_indices[seq_key]
        if prev_frame in idx_map and curr_frame in idx_map:
            return idx_map[curr_frame] - idx_map[prev_frame] == 1
    return True


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
    base_model.CLASSES = classes
    base_model.PALETTE = palette

    n_classes = len(classes) if classes is not None else cfg.model.decode_head.num_classes
    ignore_index = getattr(dataset, "ignore_index", 255)
    colors = _resolve_colors(dataset, n_classes, ignore_index)
    out_dirs = _prepare_output_dirs(args.save_dir, args.split, args.write_res)
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    tc_sum = torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")
    tc_count = torch.tensor(0.0, device=tc_sum.device)

    raft_weights = os.path.join(SSP_ROOT, "RAFT_core", "raft-things.pth-no-zip")
    model_raft = get_flow_model(raft_weights).to(tc_sum.device)

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

    prev_by_seq = {}
    total = len(sampler) if sampler is not None else len(dataset)
    prog_bar = mmcv.ProgressBar(total) if is_main_process() else None
    model.eval()
    for data in data_loader:
        with torch.no_grad():
            for d in range(len(data["img_metas"])):
                data["img_metas"][d] = data["img_metas"][d].data[0]
            result = model(return_loss=False, rescale=True, **data)

        pred = result[0] if isinstance(result, list) else result
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

        gt_rel = ann_map.get(rel_path)
        if gt_rel is not None:
            gt_path = os.path.join(dataset.ann_dir, gt_rel)
            gt = mmcv.imread(gt_path, flag="unchanged")
            if gt.ndim == 3:
                gt = gt[:, :, 0]
            if gt.shape != pred.shape:
                gt = mmcv.imresize(gt, pred.shape[::-1], interpolation="nearest")
            pred = pred.astype(np.int64, copy=False)
            confusion = _update_confusion(confusion, pred, gt, n_classes, ignore_index)

        seq_key = _infer_sequence_key(rel_path)
        frame_name = os.path.basename(rel_path)
        try:
            frame = _load_raw_frame(
                filename,
                img_meta.get("img_norm_cfg", {}).get("to_rgb", False),
                target_hw=EVAL_CROP_SIZE,
            )
        except FileNotFoundError:
            frame = None

        if args.write_res:
            frame_vis = _get_anchor_image(data, img_meta)
            if frame_vis is None and frame is not None:
                frame_vis = np.clip(frame.transpose(1, 2, 0), 0, 255).astype(np.uint8)
                if not img_meta.get("img_norm_cfg", {}).get("to_rgb", True):
                    frame_vis = frame_vis[..., ::-1]
            _save_predictions_ssp(
                pred, rel_path, out_dirs, colors, ignore_index, frame_img=frame_vis
            )

        if frame is not None:
            pred_tc = pred
            if pred_tc.shape != EVAL_CROP_SIZE:
                pred_tc = mmcv.imresize(
                    pred_tc.astype(np.uint8),
                    (EVAL_CROP_SIZE[1], EVAL_CROP_SIZE[0]),
                    interpolation="nearest",
                ).astype(np.int64)
            if seq_key in prev_by_seq:
                prev_frame, prev_pred, prev_name = prev_by_seq[seq_key]
                if _is_consecutive(dataset, seq_key, prev_name, frame_name):
                    tc_val = temporal_consistency_pair(
                        prev_frame,
                        frame,
                        prev_pred,
                        pred_tc,
                        model_raft,
                        n_classes,
                        tc_sum.device,
                        ignore_index,
                    )
                    tc_sum += tc_val.to(tc_sum.device)
                    tc_count += 1
            prev_by_seq[seq_key] = (frame, pred_tc, frame_name)

        if prog_bar is not None:
            prog_bar.update()

    if distributed:
        confusion = all_reduce_tensor(confusion.to(tc_sum.device)).cpu()
        tc_sum = all_reduce_tensor(tc_sum)
        tc_count = all_reduce_tensor(tc_count)

    iou, union = _iou_from_confusion(confusion)
    valid = union > 0
    global_miou = iou[valid].mean().item() if valid.any() else 0.0
    per_class_miou = iou.tolist()
    tc_avg = (tc_sum / tc_count.clamp(min=1)).item()

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        metrics_path = os.path.join(args.save_dir, f"log_metrics_{args.split}.txt")
        with open(metrics_path, "a") as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"mIoU = {global_miou:.4f}\n")
            f.write(f"Temporal Consistency = {tc_avg:.4f}\n")
            f.write("per class mIoU:\n")
            if base_model.CLASSES:
                for idx, name in enumerate(base_model.CLASSES):
                    f.write(f"  {name}: {per_class_miou[idx]:.5f}\n")
            else:
                for idx, val in enumerate(per_class_miou):
                    f.write(f"  class_{idx}: {val:.5f}\n")
            f.write("\n")
        print(f"mIoU = {global_miou:.4f} | TC = {tc_avg:.4f}")
        print(f"Metrics saved to {metrics_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
