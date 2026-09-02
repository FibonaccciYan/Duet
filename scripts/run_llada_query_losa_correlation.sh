#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/ysy/anaconda3/envs/llada/bin/python}"
output_dir="${OUTPUT_DIR:-experiments/query_losa_correlation/llada_layer1_topk5}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" "${python_bin}" \
  scripts/analyze_llada_query_losa_correlation.py \
  --selection_layer 1 --selection_interval 1 --query_ratio 0.7 \
  --losa_active_topk 5 --block_length 32 --steps 32 --gen_length 32 \
  --output_dir "${output_dir}" "$@"
