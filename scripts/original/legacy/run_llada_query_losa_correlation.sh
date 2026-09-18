#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python}"
output_dir="${OUTPUT_DIR:-experiments/query_losa_correlation/llada_layer1_topk5}"

"${python_bin}" \
  scripts/original/legacy/llada_query_losa_correlation.py \
  --selection_layer 1 --selection_interval 1 --query_ratio 0.7 \
  --losa_active_topk 5 --block_length 32 --steps 32 --gen_length 32 \
  --output_dir "${output_dir}" "$@"
