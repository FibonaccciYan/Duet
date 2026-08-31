#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_type="${MODEL_TYPE:-llada}"
if [[ "${model_type}" == "sdar" ]]; then
  default_output_root="${script_dir}/../../sdar_exp/humaneval_matrix/sequential"
else
  default_output_root="${script_dir}/../../llada_exp/humaneval_matrix"
fi
exec env \
  BENCHMARK=humaneval \
  MODEL_TYPE="${model_type}" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-${default_output_root}}" \
  bash "${script_dir}/run_gsm8k_matrix.sh" "$@"
