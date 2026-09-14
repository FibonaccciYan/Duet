#!/usr/bin/env bash
set -euo pipefail

output_root="${OUTPUT_DIR:-experiments/layer_overlap_results/sdar_block32}"
python_bin="${PYTHON_BIN:-/home/ysy/anaconda3/envs/sparse/bin/python}"

for strategy in \
  low_confidence_dynamic \
  low_confidence_static \
  sequential \
  entropy_bounded
do
  echo "Running ${strategy}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" "${python_bin}" scripts/analyze_sdar_layer_overlap.py \
    --block_length 32 \
    --steps 32 \
    --gen_length 32 \
    --strategy "${strategy}" \
    --layer_candidate_ratio 0.25 0.5 0.75 1.0 \
    --plot_overlap_stats \
    --output_dir "${output_root}/${strategy}"
done
