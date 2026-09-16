#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
eval_script="${script_dir}/eval.sh"
benchmark="${BENCHMARK:-gsm8k}"
model_type="${MODEL_TYPE:-llada}"
output_root="${OUTPUT_ROOT:-${repo_root}/../${model_type}_exp/${benchmark}_matrix/sequential}"
base_port="${BASE_PROCESS_PORT:-12350}"
case "${model_type}" in
  llada)
    python_bin="${PYTHON:-/home/ysy/anaconda3/envs/sparse/bin/python}"
    block_length="${BLOCK_LENGTH:-32}"
    steps="${STEPS:-32}"
    ;;
  sdar)
    python_bin="${PYTHON:-/home/ysy/anaconda3/envs/sparse/bin/python}"
    block_length="${BLOCK_LENGTH:-32}"
    steps="${STEPS:-32}"
    ;;
  *)
    echo "Unsupported MODEL_TYPE: ${model_type} (expected llada or sdar)" >&2
    exit 2
    ;;
esac

mkdir -p "${output_root}/logs"

run_lane() {
  local gpu="$1"
  local losa_name="$2"
  local losa_enabled="$3"
  local losa_topk="$4"
  local port="$5"
  local combo method query_sparse prefix_sparse combo_root log_path

  for combo in dense prefix query prefix_query; do
    case "${combo}" in
      dense)
        method=dense
        query_sparse=false
        prefix_sparse=false
        ;;
      prefix)
        method=sparse
        query_sparse=false
        prefix_sparse=true
        ;;
      query)
        method=sparse
        query_sparse=true
        prefix_sparse=false
        ;;
      prefix_query)
        method=sparse
        query_sparse=true
        prefix_sparse=true
        ;;
    esac
    if [[ "${losa_enabled}" == "true" ]]; then
      method=sparse
    fi

    combo_root="${output_root}/${losa_name}/${combo}"
    log_path="${output_root}/logs/${losa_name}_${combo}.log"
    if [[ "${SKIP_COMPLETED:-true}" == "true" ]] &&
      [[ -n "$(find "${combo_root}" -type f -name 'results_*.json' -print -quit 2>/dev/null)" ]]; then
      echo "[GPU ${gpu}] skip completed ${benchmark} ${losa_name}/${combo}"
      continue
    fi
    echo "[GPU ${gpu}] ${benchmark} ${losa_name}/${combo} -> ${combo_root}"
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
      continue
    fi

    if ! CUDA_VISIBLE_DEVICES="${gpu}" \
      MAIN_PROCESS_PORT="${port}" \
      MODEL_TYPE="${model_type}" \
      PYTHON="${python_bin}" \
      BENCHMARK="${benchmark}" \
      BLOCK_LENGTH="${block_length}" \
      STEPS="${steps}" \
      METHOD="${method}" \
      SPARSE_DLM=true \
      QUERY_SPARSE="${query_sparse}" \
      PREFIX_SPARSE="${prefix_sparse}" \
      LOSA="${losa_enabled}" \
      LOSA_ACTIVE_TOPK="${losa_topk}" \
      OUTPUT_ROOT="${combo_root}" \
        bash "${eval_script}" >"${log_path}" 2>&1; then
      echo "[GPU ${gpu}] ${losa_name}/${combo} failed; see ${log_path}" >&2
      return 1
    fi
  done
}

if [[ "${model_type}" == "sdar" ]]; then
  run_lane 3 losa_off false 5 "${base_port}" &
  pid_3=$!
  run_lane 4 losa_topk5 true 5 "$((base_port + 1))" &
  pid_4=$!
  run_lane 5 losa_topk32 true 32 "$((base_port + 2))" &
  pid_5=$!
  pids=("${pid_3}" "${pid_4}" "${pid_5}")
else
  run_lane 3 losa_off false 5 "${base_port}" &
  pid_3=$!
  run_lane 4 losa_topk5 true 5 "$((base_port + 1))" &
  pid_4=$!
  run_lane 5 losa_topk32 true 32 "$((base_port + 2))" &
  pid_5=$!
  pids=("${pid_3}" "${pid_4}" "${pid_5}")
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
