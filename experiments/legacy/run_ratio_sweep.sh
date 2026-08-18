#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ysy/anaconda3/envs/llada/bin/python}"

RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

echo "Running LLaDA layer recall sweep with ratios: ${RATIOS[*]}"

EXTRA_ARGS=()
if [[ "${LOCAL_DEMO_SPARSE_ATTN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--local_demo_sparse_attn --demo_sparse_ratio "${DEMO_SPARSE_RATIO:-0.5}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/trace_generation.py" \
  --gen_length "${GEN_LENGTH:-128}" \
  --block_length "${BLOCK_LENGTH:-32}" \
  --steps "${STEPS:-32}" \
  --enable_layer_candidate_ratios \
  --layer_candidate_ratio "${RATIOS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@" \
  --no_plot_attentions
