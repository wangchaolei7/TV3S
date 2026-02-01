#!/usr/bin/env python
import os
import sys

import mmcv
import numpy as np
import torch
from PIL import Image
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SSP_ROOT = os.environ.get("SSP_ROOT", "/home/wangcl/SSP")
if os.path.isdir(SSP_ROOT) and SSP_ROOT not in sys.path:
    sys.path.insert(0, SSP_ROOT)

# Register custom datasets/pipelines.
import utils  # noqa: F401

from vis_utils.visualization import color_predictions, pred_to_mask

# Editable parameters.
CONFIG_PATH = "local_configs/tv3s/B5/tv3s_realshift_w20_s10.b5.480x480.kitti36015.160k.py"
CHECKPOINT_PATH = "/data1/wangcl/project/TV3S/kitti360/iter_11000.pth"
OUTPUT_DIR = "/data1/wangcl/project/TV3S/kitti360/single_img_inference"
IMAGE_PATHS = [
    "/home/wangcl/data/open_video_DGSS/ApolloScape/train/ColorImage/Record008/171206_030550389_Camera_5.jpg",
]

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


def _prepare_output_dirs(save_dir):
    dirs = {
        "mask": os.path.join(save_dir, "pred"),
        "colored": os.path.join(save_dir, "pred_colored"),
        "blended": os.path.join(save_dir, "pred_blended"),
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


def _build_data_from_image(img_path, pipeline_load, pipeline_process):
    results = dict(img_info={"filename": img_path}, ann_info=None)
    results["img_prefix"] = None
    results["seg_prefix"] = None
    results["seg_fields"] = []
    results = pipeline_load(results)

    results_new = dict(
        img_info=results.get("img_info", {"filename": img_path}),
        ann_info=results.get("ann_info"),
        seg_fields=results.get("seg_fields", []),
        img_prefix=results.get("img_prefix"),
        seg_prefix=results.get("seg_prefix"),
        filename=results.get("filename", img_path),
        ori_filename=results.get("ori_filename", os.path.basename(img_path)),
        img=[results["img"]],
        img_shape=results.get("img_shape"),
        ori_shape=results.get("ori_shape"),
        pad_shape=results.get("pad_shape"),
        scale_factor=results.get("scale_factor"),
        img_norm_cfg=results.get("img_norm_cfg"),
    )

    data = pipeline_process(results_new)
    data = collate([data], samples_per_gpu=1)
    if "img_metas" in data:
        for d in range(len(data["img_metas"])):
            data["img_metas"][d] = data["img_metas"][d].data[0]
    return data


def main():
    cfg = mmcv.Config.fromfile(CONFIG_PATH)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    data_cfg = cfg.data.get("test", cfg.data.get("val"))
    if data_cfg.get("type") == "RepeatDataset":
        data_cfg = data_cfg["dataset"]
    data_cfg = mmcv.ConfigDict(data_cfg)
    data_cfg.test_mode = True

    pipeline = data_cfg.pipeline
    pipeline_load = Compose(pipeline[:1])
    pipeline_process = Compose(pipeline[1:])

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if cfg.model.get("decode_head", None) and cfg.model.decode_head.get("decoder_params", None):
        cfg.model.decode_head.decoder_params.test_mode = True
        cfg.model.decode_head.decoder_params.val_mode = 0

    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, CHECKPOINT_PATH, map_location="cpu")
    classes = None
    palette = None
    if "meta" in checkpoint:
        classes = checkpoint["meta"].get("CLASSES")
        palette = checkpoint["meta"].get("PALETTE")

    n_classes = len(classes) if classes is not None else cfg.model.decode_head.num_classes
    if classes is None and n_classes == 15:
        classes = CLASSES_15

    class _DatasetView:
        def __init__(self, palette):
            self.PALETTE = palette

    ignore_index = data_cfg.get("ignore_index", 255)
    colors = _resolve_colors(_DatasetView(palette), n_classes, ignore_index)

    if torch.cuda.is_available():
        model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    out_dirs = _prepare_output_dirs(OUTPUT_DIR)

    for img_path in IMAGE_PATHS:
        if not os.path.isfile(img_path):
            print(f"[skip] missing image: {img_path}")
            continue

        data = _build_data_from_image(img_path, pipeline_load, pipeline_process)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        pred = result[0] if isinstance(result, list) else result
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()

        img_meta = data.get("img_metas")
        if isinstance(img_meta, list):
            img_meta = img_meta[0]
        if isinstance(img_meta, list):
            img_meta = img_meta[0]
        img_norm = img_meta.get("img_norm_cfg", {}) if isinstance(img_meta, dict) else {}
        to_rgb = img_norm.get("to_rgb", True)
        frame_img = mmcv.imread(img_path, flag="color")
        if to_rgb:
            frame_img = frame_img[..., ::-1]

        rel_path = os.path.basename(img_path)
        _save_predictions_ssp(pred, rel_path, out_dirs, colors, ignore_index, frame_img=frame_img)
        print(f"[saved] {rel_path}")


if __name__ == "__main__":
    main()
