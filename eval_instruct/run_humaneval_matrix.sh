#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env \
  BENCHMARK=humaneval \
  OUTPUT_ROOT="${OUTPUT_ROOT:-${script_dir}/../../llada_exp/humaneval_matrix}" \
  bash "${script_dir}/run_gsm8k_matrix.sh" "$@"
