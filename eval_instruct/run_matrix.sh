#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
model_type="${MODEL_TYPE:-llada}"
output_root="${OUTPUT_ROOT:-${repo_root}/../${model_type}_exp/method_matrix}"
read -r -a methods <<<"${METHODS:-dense focus losa sparse}"
read -r -a benchmarks <<<"${BENCHMARKS:-gsm8k humaneval math mmlu}"

mkdir -p "${output_root}/logs"

for method in "${methods[@]}"; do
  for benchmark in "${benchmarks[@]}"; do
    job_root="${output_root}/${method}"
    done_file="${job_root}/${benchmark}/.done"
    log_path="${output_root}/logs/${method}_${benchmark}.log"
    if [[ "${SKIP_COMPLETED:-true}" == "true" && -f "${done_file}" ]]; then
      echo "skip completed ${model_type} ${method}/${benchmark}"
      continue
    fi
    echo "${model_type} ${method}/${benchmark} -> ${job_root}/${benchmark}"
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
      continue
    fi
    METHOD="${method}" MODEL_TYPE="${model_type}" BENCHMARK="${benchmark}" \
      OUTPUT_ROOT="${job_root}" bash "${script_dir}/eval.sh" \
      >"${log_path}" 2>&1
    touch "${done_file}"
  done
done
