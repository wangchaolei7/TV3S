#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-2}"
CKPT="${CKPT:-/data1/wangcl/project/TV3S/apollo/iter_14000.pth}"
OUT_ROOT="${OUT_ROOT:-/data1/wangcl/project/TV3S/apollo}"
METRICS_ONLY="${METRICS_ONLY:-0}"

CITY_ROOT_ORIGIN="${CITY_ROOT_ORIGIN:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence}"
CITY_ROOT_CORR="${CITY_ROOT_CORR:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions}"
CITY_ROOT_LABELS="${CITY_ROOT_LABELS:-/home/wangcl/data/open_video_DGSS/cityscapes_sequence/gtFine}"

CAMVID_SEQ_PATH="${CAMVID_SEQ_PATH:-/home/wangcl/data/open_video_DGSS/CamVid/val/images/Seq05VD}"
KITTI360_SEQ_PATHS=(
  "/home/wangcl/data/open_video_DGSS/kitti360_sequence/val/data_2d_raw/2013_05_28_drive_0007_sync/image_00"
  "/home/wangcl/data/open_video_DGSS/kitti360_sequence/val/data_2d_raw/2013_05_28_drive_0009_sync/image_00"
)
CITY_SEQ_ORIGIN_PATHS=(
  "/data1/wangcl/dataset/open_video_DGSS/cityscapes_sequence/origin_leftImg8bit_sequence/munster/seq2"
)
CITY_SEQ_CORR_PATHS=(
  "/data1/wangcl/dataset/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/fog/lindau/seq42"
  "/data1/wangcl/dataset/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/frost/munster/seq35"
  "/data1/wangcl/dataset/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/snow/munster/seq107"
  "/data1/wangcl/dataset/open_video_DGSS/cityscapes_sequence/leftImg8bit_sequence_Corruptions/spatter/munster/seq88"
)

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

# camvid_seq="$(basename "${CAMVID_SEQ_PATH}")"
# run_eval "camvid_${camvid_seq}" \
#   "local_configs/tv3s/B5/tv3s_realshift_w20_s10.b5.480x480.camvid15.160k.py" \
#   --city-seq "${CAMVID_SEQ_PATH}"

# for seq_path in "${KITTI360_SEQ_PATHS[@]}"; do
#   drive="$(basename "$(dirname "${seq_path}")")"
#   run_eval "kitti360_${drive}" \
#     "local_configs/tv3s/B5/tv3s_realshift_w20_s10.b5.480x480.kitti36015.160k.py" \
#     --city-seq "${seq_path}"
# done

for seq_path in "${CITY_SEQ_ORIGIN_PATHS[@]}"; do
  seq_city="$(basename "$(dirname "${seq_path}")")"
  seq_name="$(basename "${seq_path}")"
  city_seq="${seq_city}/${seq_name}"
  run_eval "cityscapes_origin_${seq_city}_${seq_name}" \
    "local_configs/tv3s/B5/cityscapes_corruptions/tv3s_b5_city_origin.py" \
    --corruption origin_leftImg8bit_sequence \
    --city-root-images "${CITY_ROOT_ORIGIN}" \
    --city-root-labels "${CITY_ROOT_LABELS}" \
    --city-seq "${city_seq}"
done

for seq_path in "${CITY_SEQ_CORR_PATHS[@]}"; do
  corruption="$(basename "$(dirname "$(dirname "${seq_path}")")")"
  seq_city="$(basename "$(dirname "${seq_path}")")"
  seq_name="$(basename "${seq_path}")"
  city_seq="${seq_city}/${seq_name}"
  run_eval "cityscapes_${corruption}_${seq_city}_${seq_name}" \
    "local_configs/tv3s/B5/cityscapes_corruptions/tv3s_b5_city_${corruption}.py" \
    --corruption "${corruption}" \
    --city-root-images "${CITY_ROOT_CORR}" \
    --city-root-labels "${CITY_ROOT_LABELS}" \
    --city-seq "${city_seq}"
done
