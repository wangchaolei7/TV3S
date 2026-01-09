#!/usr/bin/env bash
set -euo pipefail

SCENES=(fog frost snow spatter)
CKPT="/data1/wangcl/project/TV3S/camvid/iter_62000.pth"
SAVE_ROOT="/data1/wangcl/project/TV3S/camvid/cityscapes_corruptions"

for scene in "${SCENES[@]}"; do
  save_dir="${SAVE_ROOT}/${scene}"
  mkdir -p "${save_dir}"
  torchrun --nproc_per_node=6 tools/eval_ssp_video.py \
    "local_configs/tv3s/B5/cityscapes_corruptions/tv3s_b5_city_${scene}.py" \
    "${CKPT}" \
    --save-dir "${save_dir}" \
    --split val \
    --write-res
done
