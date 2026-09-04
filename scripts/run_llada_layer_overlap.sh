#!/usr/bin/env bash
set -euo pipefail

output_root="${OUTPUT_DIR:-experiments/layer_overlap_results/llada_block32}"
python_bin="${PYTHON_BIN:-/home/ysy/anaconda3/envs/llada/bin/python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" "${python_bin}" scripts/analyze_llada_layer_overlap.py \
  --block_length 32 --steps 32 --gen_length 32 \
  --query_ratio 0.5 --layer_candidate_ratio 0.5 0.6 0.7 0.8 0.9 1.0 \
  --plot_overlap_stats --output_dir "${output_root}"
