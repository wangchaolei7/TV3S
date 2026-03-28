#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_eval_cross_domain_camvid.sh" "$@"
bash "${SCRIPT_DIR}/run_eval_cross_domain_apollo.sh" "$@"
bash "${SCRIPT_DIR}/run_eval_cross_domain_kitti360.sh" "$@"
