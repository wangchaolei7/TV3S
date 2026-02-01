#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-6}"
CKPT="${CKPT:-/data1/wangcl/project/TV3S/kitti360/iter_11000.pth}"
OUT_ROOT="${OUT_ROOT:-/data1/wangcl/project/TV3S/kitti360}"
METRICS_ONLY="${METRICS_ONLY:-0}"

CITY_ROOT_ORIGIN="${CITY_ROOT_ORIGIN:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence}"
CITY_ROOT_CORR="${CITY_ROOT_CORR:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions}"
CITY_ROOT_LABELS="${CITY_ROOT_LABELS:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence/gtFine}"

ORIGIN_SEQ_PATH="${ORIGIN_SEQ_PATH:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence/origin_leftImg8bit_sequence/munster/seq173}"
SEQ_CITY="$(basename "$(dirname "${ORIGIN_SEQ_PATH}")")"
SEQ_NAME="$(basename "${ORIGIN_SEQ_PATH}")"
CITY_SEQ="${SEQ_CITY}/${SEQ_NAME}"

run_eval() {
  local name="$1"
  local cfg="$2"
  shift 2
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  echo "[eval] ${name}"
  local common_args=()
  if [[ "${METRICS_ONLY}" == "1" ]]; then
    common_args+=(--metrics-only)
  else
    common_args+=(--write-res)
  fi
  torchrun --nproc_per_node="${NPROC}" tools/eval_ssp_video.py \
    "${cfg}" "${CKPT}" \
    --save-dir "${out_dir}" \
    --split val \
    "${common_args[@]}" \
    "$@"
}

run_eval "cityscapes_origin_${SEQ_CITY}_${SEQ_NAME}" \
  "local_configs/tv3s/B5/cityscapes_corruptions/tv3s_b5_city_origin.py" \
  --corruption origin_leftImg8bit_sequence \
  --city-root-images "${CITY_ROOT_ORIGIN}" \
  --city-root-labels "${CITY_ROOT_LABELS}" \
  --city-seq "${CITY_SEQ}"

for corruption in fog frost snow spatter; do
  run_eval "cityscapes_${corruption}_${SEQ_CITY}_${SEQ_NAME}" \
    "local_configs/tv3s/B5/cityscapes_corruptions/tv3s_b5_city_${corruption}.py" \
    --corruption "${corruption}" \
    --city-root-images "${CITY_ROOT_CORR}" \
    --city-root-labels "${CITY_ROOT_LABELS}" \
    --city-seq "${CITY_SEQ}"
done
