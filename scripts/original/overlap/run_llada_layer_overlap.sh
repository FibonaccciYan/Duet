#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

output_root="${OUTPUT_DIR:-experiments/layer_overlap_results/llada_block32}"
python_bin="${PYTHON_BIN:-python}"
model_path="${MODEL_PATH:-${LLADA_MODEL_PATH:-}}"

if [[ -z "${model_path}" ]]; then
  echo "Set MODEL_PATH or LLADA_MODEL_PATH" >&2
  exit 2
fi

"${python_bin}" scripts/original/overlap/llada_layer_overlap.py \
  --model_path "${model_path}" \
  --block_length 32 --steps 32 --gen_length 32 \
  --query_ratio 0.5 --layer_candidate_ratio 0.5 0.6 0.7 0.8 0.9 1.0 \
  --plot_overlap_stats --output_dir "${output_root}"
