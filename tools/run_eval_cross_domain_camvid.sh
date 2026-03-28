#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-6}"
MASTER_PORT="${MASTER_PORT:-29500}"
CKPT="${CKPT:-/data1/wangcl/project/TV3S/camvid/iter_62000.pth}"
OUT_ROOT="${OUT_ROOT:-/data1/wangcl/project/TV3S/camvid}"
METRICS_ONLY="${METRICS_ONLY:-1}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-cross_domain}"
TARGETS="${TARGETS:-uavid,vspw}"

if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_BIN="$(command -v torchrun)"
else
  TORCHRUN_BIN="/usr/local/anaconda3/envs/TV3S/bin/torchrun"
fi

run_eval() {
  local name="$1"
  local cfg="$2"
  shift 2
  local out_dir="${OUT_ROOT}/${OUTPUT_PREFIX}/${name}"
  mkdir -p "${out_dir}"
  echo "[eval] ${name}"
  local common_args=()
  if [[ "${METRICS_ONLY}" == "1" ]]; then
    common_args+=(--metrics-only)
  else
    common_args+=(--write-res)
  fi
  "${TORCHRUN_BIN}" --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" tools/eval_ssp_video.py \
    "${cfg}" "${CKPT}" \
    --save-dir "${out_dir}" \
    --split val \
    "${common_args[@]}" \
    "$@"
}

IFS=',' read -r -a target_list <<< "${TARGETS}"
for target in "${target_list[@]}"; do
  case "${target}" in
    uavid)
      run_eval "uavid_val" \
        "local_configs/tv3s/B5/tv3s_realshift_w20_s10.b5.480x480.uavid15.160k.py" \
        "$@"
      ;;
    vspw)
      run_eval "vspw_val" \
        "local_configs/tv3s/B5/tv3s_realshift_w20_s10.b5.480x480.vspw15.160k.py" \
        "$@"
      ;;
    "")
      ;;
    *)
      echo "Unknown target: ${target}" >&2
      exit 1
      ;;
  esac
done
