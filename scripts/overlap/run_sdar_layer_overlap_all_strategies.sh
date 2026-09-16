#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

output_root="${OUTPUT_DIR:-experiments/layer_overlap_results/sdar_block32}"
python_bin="${PYTHON_BIN:-python}"
model_path="${MODEL_PATH:-${SDAR_MODEL_PATH:-}}"

if [[ -z "${model_path}" ]]; then
  echo "Set MODEL_PATH or SDAR_MODEL_PATH" >&2
  exit 2
fi

for strategy in \
  low_confidence_dynamic \
  low_confidence_static \
  sequential \
  entropy_bounded
do
  echo "Running ${strategy}"
  "${python_bin}" scripts/overlap/sdar_layer_overlap.py \
    --model_path "${model_path}" \
    --block_length 32 \
    --steps 32 \
    --gen_length 32 \
    --strategy "${strategy}" \
    --layer_candidate_ratio 0.25 0.5 0.75 1.0 \
    --plot_overlap_stats \
    --output_dir "${output_root}/${strategy}"
done
