#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-6}"
CKPT="${CKPT:-/data1/wangcl/project/TV3S/kitti360/iter_11000.pth}"
OUT_ROOT="${OUT_ROOT:-/data1/wangcl/project/TV3S/kitti360}"

run_eval() {
  local name="$1"
  local cfg="$2"
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  echo "[eval] ${name}"
  torchrun --nproc_per_node="${NPROC}" tools/eval_ssp_video.py \
    "${cfg}" "${CKPT}" \
    --save-dir "${out_dir}" \
    --split val \
    --metrics-only
}

# run_eval "apollo" "local_configs/tv3s/B5/tv3s_realshift_w20_s10.b5.480x480.apollo15.160k.py"
# run_eval "camvid" "local_configs/tv3s/B5/tv3s_realshift_w20_s10.b5.480x480.camvid15.160k.py"
run_eval "kitti360" "local_configs/tv3s/B5/tv3s_realshift_w20_s10.b5.480x480.kitti36015.160k.py"
# run_eval "cityscapes_origin" "local_configs/tv3s/B5/cityscapes_corruptions/tv3s_b5_city_origin.py"
# run_eval "cityscapes_fog" "local_configs/tv3s/B5/cityscapes_corruptions/tv3s_b5_city_fog.py"
# run_eval "cityscapes_frost" "local_configs/tv3s/B5/cityscapes_corruptions/tv3s_b5_city_frost.py"
# run_eval "cityscapes_snow" "local_configs/tv3s/B5/cityscapes_corruptions/tv3s_b5_city_snow.py"
# run_eval "cityscapes_spatter" "local_configs/tv3s/B5/cityscapes_corruptions/tv3s_b5_city_spatter.py"
