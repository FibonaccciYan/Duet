#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${script_dir}"

python_bin="${PYTHON:-/home/ysy/anaconda3/envs/llada/bin/python}"
model="${MODEL:-/data0/ysy/models/LLaDA2.1-mini}"
port="${MAIN_PROCESS_PORT:-12335}"
benchmark="${BENCHMARK:-mmlu}"
sparse_dlm="${SPARSE_DLM:-true}"
block_length="${BLOCK_LENGTH:-32}"
steps="${STEPS:-32}"
sparse_ratio="${SPARSE_DLM_RATIO:-0.5}"

case "${sparse_dlm,,}" in
  1|true|yes|y)
    sparse_dlm=true
    ;;
  0|false|no|n)
    sparse_dlm=false
    ;;
  *)
    echo "Unsupported SPARSE_DLM value: ${sparse_dlm}" >&2
    exit 2
    ;;
esac

if [[ "${sparse_dlm}" == "true" ]]; then
  default_output="output_reproduce/block_cache_sparse_dlm_block${block_length}_ratio${sparse_ratio}"
else
  default_output="output_reproduce/default"
fi
output_root="${OUTPUT_ROOT:-${default_output}}"

model_args="pretrained=${model},trust_remote_code=true,dtype=${DTYPE:-bfloat16},attn_implementation=${ATTN_IMPLEMENTATION:-sdpa},sparse_dlm=${sparse_dlm},sparse_dlm_ratio=${sparse_ratio},sparse_dlm_top_k=${SPARSE_DLM_TOP_K:-64},sparse_dlm_selection_interval=${SPARSE_DLM_SELECTION_INTERVAL:-4},sparse_dlm_dense_fallback_mask_count=${SPARSE_DLM_DENSE_FALLBACK_MASK_COUNT:-4},block_length=${block_length},steps=${steps},temperature=${TEMPERATURE:-0.0},threshold=${THRESHOLD:-0.5},editing_threshold=${EDITING_THRESHOLD:-0.0},num_to_transfer=${NUM_TO_TRANSFER:-1},mask_id=${MASK_ID:-156895},eos_id=${EOS_ID:-156892}"

minerva_tasks="minerva_math_algebra,minerva_math_counting_and_prob,minerva_math_geometry,minerva_math_intermediate_algebra,minerva_math_num_theory,minerva_math_prealgebra,minerva_math_precalc"

run_eval() {
  local name="$1"
  local tasks="$2"
  local default_gen_length="$3"
  local default_fewshot="$4"
  shift 4
  local gen_length="${GEN_LENGTH:-${default_gen_length}}"
  local num_fewshot="${NUM_FEWSHOT:-${default_fewshot}}"
  local args=(
    -m accelerate.commands.launch
    --main_process_port "${port}"
    -m lm_eval
    --model llada
    --model_args "${model_args},gen_length=${gen_length}"
    --tasks "${tasks}"
    --device "${DEVICE:-cuda}"
    --batch_size 1
    --num_fewshot "${num_fewshot}"
    --output_path "${output_root}/${name}"
    --log_samples
    --confirm_run_unsafe_code
    --apply_chat_template
  )

  if [[ -n "${LIMIT:-}" ]]; then
    args+=(--limit "${LIMIT}")
  fi

  if [[ "${name}" == "humaneval" ]]; then
    HF_ALLOW_CODE_EVAL=1 PYTHONPATH="${repo_root}:${script_dir}:${PYTHONPATH:-}" \
      "${python_bin}" "${args[@]}" "$@"
  else
    PYTHONPATH="${repo_root}:${script_dir}:${PYTHONPATH:-}" \
      "${python_bin}" "${args[@]}" "$@"
  fi
}

case "${benchmark}" in
  mmlu)
    run_eval mmlu mmlu_generative 128 4 "$@"
    ;;
  gsm8k)
    run_eval gsm8k gsm8k_cot 256 0 "$@"
    ;;
  math)
    run_eval math "${minerva_tasks}" 512 0 "$@"
    ;;
  humaneval)
    run_eval humaneval humaneval_instruct 768 0 "$@"
    ;;
  all)
    run_eval mmlu mmlu_generative 128 4 "$@"
    run_eval gsm8k gsm8k_cot 256 0 "$@"
    run_eval math "${minerva_tasks}" 512 0 "$@"
    run_eval humaneval humaneval_instruct 768 0 "$@"
    ;;
  *)
    echo "Unsupported BENCHMARK: ${benchmark} (expected mmlu, gsm8k, math, humaneval, or all)" >&2
    exit 2
    ;;
esac
