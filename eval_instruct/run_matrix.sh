#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
model_type="${MODEL_TYPE:-llada}"
output_root="${OUTPUT_ROOT:-${repo_root}/../${model_type}_exp/method_matrix}"
read -r -a methods <<<"${METHODS:-dense focus losa sparse}"
read -r -a benchmarks <<<"${BENCHMARKS:-gsm8k humaneval math mmlu longbench}"

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
    if [[ "${benchmark}" == "longbench" ]]; then
      if [[ "${model_type}" == "sdar" ]]; then
        python_bin="${PYTHON:-/home/ysy/anaconda3/envs/dream/bin/python}"
      else
        python_bin="${PYTHON:-/home/ysy/anaconda3/envs/llada/bin/python}"
      fi
      args=(
        "${repo_root}/scripts/run_longbench_quality.py"
        --family "${model_type}"
        --method "${method}"
        --output_dir "${job_root}/longbench"
        --block_length "${BLOCK_LENGTH:-32}"
        --steps "${STEPS:-32}"
        --focus_alpha "${FOCUS_ALPHA:-1.5}"
        --losa_token_budget "${PAPER_LOSA_TOKEN_BUDGET:-256}"
      )
      [[ -z "${MODEL:-}" ]] || args+=(--model_path "${MODEL}")
      [[ -z "${LONGBENCH_DATA:-}" ]] || args+=(--data_dir "${LONGBENCH_DATA}")
      [[ -z "${LIMIT:-}" ]] || args+=(--limit "${LIMIT}")
      [[ -z "${DTYPE:-}" ]] || args+=(--dtype "${DTYPE}")
      [[ -z "${THRESHOLD:-}" ]] || args+=(--threshold "${THRESHOLD}")
      [[ -z "${QUERY_DENSE_THRESHOLD:-}" ]] || args+=(--query_dense_threshold "${QUERY_DENSE_THRESHOLD}")
      [[ -z "${EDITING_THRESHOLD:-}" ]] || args+=(--editing_threshold "${EDITING_THRESHOLD}")
      [[ -z "${REMASKING_STRATEGY:-}" ]] || args+=(--remasking_strategy "${REMASKING_STRATEGY}")
      case "${MOE_EXPERT_PATCH:-}" in
        1|true|yes|y) args+=(--moe_expert_patch) ;;
        0|false|no|n) args+=(--no-moe_expert_patch) ;;
      esac
      PYTHONPATH="${repo_root}:${PYTHONPATH:-}" "${python_bin}" "${args[@]}" \
        >"${log_path}" 2>&1
    else
      METHOD="${method}" MODEL_TYPE="${model_type}" BENCHMARK="${benchmark}" \
        OUTPUT_ROOT="${job_root}" bash "${script_dir}/eval.sh" \
        >"${log_path}" 2>&1
    fi
    touch "${done_file}"
  done
done
