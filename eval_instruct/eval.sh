#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${script_dir}"

model_type="${MODEL_TYPE:-llada}"
port="${MAIN_PROCESS_PORT:-12335}"
benchmark="${BENCHMARK:-gsm8k}"
sparse_dlm="${SPARSE_DLM:-true}"
method="${METHOD:-}"
case "${model_type}" in
  llada)
    default_model=/data0/ysy/models/LLaDA2.1-mini
    default_python=/home/ysy/anaconda3/envs/llada/bin/python
    block_length="${BLOCK_LENGTH:-32}"
    steps="${STEPS:-32}"
    mask_id="${MASK_ID:-156895}"
    eos_id="${EOS_ID:-156892}"
    default_ratio=0.7
    default_selection_interval=4
    default_query_dense_threshold=0
    default_refresh_step=2
    default_selection_layer=1
    default_prefix_min_length=0
    default_prefix_chunk_size=1024
    ;;
  sdar)
    default_model=/data0/ysy/models/SDAR-8B-Chat-b32
    default_python=/home/ysy/anaconda3/envs/dream/bin/python
    block_length="${BLOCK_LENGTH:-32}"
    steps="${STEPS:-32}"
    mask_id="${MASK_ID:-151669}"
    default_ratio=0.5
    default_selection_interval=1
    default_query_dense_threshold=0
    default_refresh_step=-1
    default_selection_layer=5
    default_prefix_min_length=0
    default_prefix_chunk_size=1024
    ;;
  *)
    echo "Unsupported MODEL_TYPE: ${model_type} (expected llada or sdar)" >&2
    exit 2
    ;;
esac

if [[ -z "${method}" ]]; then
  case "${sparse_dlm,,}" in
    1|true|yes|y) method=sparse ;;
    0|false|no|n) method=dense ;;
    *)
      echo "Unsupported SPARSE_DLM value: ${sparse_dlm}" >&2
      exit 2
      ;;
  esac
fi
case "${method}" in
  sparse|dense|focus|losa) ;;
  *)
    echo "Unsupported METHOD: ${method} (expected sparse, dense, focus, or losa)" >&2
    exit 2
    ;;
esac

if [[ "${method}" == "sparse" ]]; then
  default_llada_threshold=0.5
  default_sdar_threshold=1.0
  default_editing_threshold=0.0
  default_sdar_remasking=sequential
else
  default_llada_threshold=0.95
  default_sdar_threshold=0.85
  default_editing_threshold=0.9
  if [[ "${method}" == "focus" ]]; then
    default_sdar_remasking=low_confidence_dynamic
  else
    default_sdar_remasking=sequential
  fi
fi
model="${MODEL:-${default_model}}"
python_bin="${PYTHON:-${default_python}}"
query_sparse="${QUERY_SPARSE:-true}"
if [[ "${model_type}" == "sdar" ]]; then
  prefix_sparse="${PREFIX_SPARSE:-false}"
else
  prefix_sparse="${PREFIX_SPARSE:-true}"
fi
prefix_budget="${PREFIX_TOKEN_BUDGET:-256}"
losa="${LOSA:-false}"
losa_active_topk="${LOSA_ACTIVE_TOPK:-5}"
losa_score_mode="${LOSA_SCORE_MODE:-query}"
losa_key_samples="${LOSA_KEY_SAMPLES:-32}"
output_path="${OUTPUT_PATH:-default}"

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

default_output_root="${repo_root}/../${model_type}_exp/${output_path}"
output_root="${OUTPUT_ROOT:-${default_output_root}}"

if [[ "${model_type}" == "sdar" ]]; then
  model_args="pretrained=${model},trust_remote_code=true,dtype=${DTYPE:-float16},attn_implementation=${ATTN_IMPLEMENTATION:-sdpa},method=${method},moe_expert_patch=${MOE_EXPERT_PATCH:-false},block_length=${block_length},steps=${steps},temperature=${TEMPERATURE:-0.0},threshold=${THRESHOLD:-${default_sdar_threshold}},remasking_strategy=${REMASKING_STRATEGY:-${default_sdar_remasking}},eb_threshold=${EB_THRESHOLD:-0.35},mask_id=${mask_id}"
  if [[ -n "${EOS_ID:-}" ]]; then
    model_args+=",eos_id=${EOS_ID}"
  fi
else
  model_args="pretrained=${model},trust_remote_code=true,dtype=${DTYPE:-bfloat16},attn_implementation=${ATTN_IMPLEMENTATION:-sdpa},method=${method},moe_expert_patch=${MOE_EXPERT_PATCH:-true},block_length=${block_length},steps=${steps},temperature=${TEMPERATURE:-0.0},threshold=${THRESHOLD:-${default_llada_threshold}},editing_threshold=${EDITING_THRESHOLD:-${default_editing_threshold}},num_to_transfer=${NUM_TO_TRANSFER:-1},mask_id=${mask_id},eos_id=${eos_id}"
fi

case "${method}" in
  sparse)
    model_args+=",sparse_dlm_ratio=${SPARSE_DLM_RATIO:-${default_ratio}},sparse_dlm_top_k=${SPARSE_DLM_TOP_K:-64},sparse_dlm_selection_interval=${SPARSE_DLM_SELECTION_INTERVAL:-${default_selection_interval}},query_dense_threshold=${QUERY_DENSE_THRESHOLD:-${default_query_dense_threshold}},sparse_dlm_refresh_step=${SPARSE_DLM_REFRESH_STEP:-${default_refresh_step}},sparse_dlm_selection_layer=${SPARSE_DLM_SELECTION_LAYER:-${default_selection_layer}},sparse_dlm_deep_only_transfer=${SPARSE_DLM_DEEP_ONLY_TRANSFER:-false},query_sparse=${query_sparse},prefix_sparse=${prefix_sparse},prefix_min_prefix_length=${PREFIX_MIN_PREFIX_LENGTH:-${default_prefix_min_length}},prefix_token_budget=${prefix_budget},prefix_chunk_size=${PREFIX_CHUNK_SIZE:-${default_prefix_chunk_size}},prefix_share_layer_pairs=${PREFIX_SHARE_LAYER_PAIRS:-false},prefix_selector=${PREFIX_SELECTOR:-raw_l1},losa=${losa},losa_active_topk=${losa_active_topk},losa_score_mode=${losa_score_mode},losa_key_samples=${losa_key_samples}"
    if [[ "${model_type}" == "llada" ]]; then
      model_args+=",query_min_prefix_length=${QUERY_MIN_PREFIX_LENGTH:-0},query_losa_union=${QUERY_LOSA_UNION:-false}"
    fi
    ;;
  focus)
    model_args+=",focus_alpha=${FOCUS_ALPHA:-1.5}"
    ;;
  losa)
    model_args+=",paper_losa_page_size=${PAPER_LOSA_PAGE_SIZE:-16},paper_losa_token_budget=${PAPER_LOSA_TOKEN_BUDGET:-256},paper_losa_active_topk=${PAPER_LOSA_ACTIVE_TOPK:-5},paper_losa_gqa_mode=${PAPER_LOSA_GQA_MODE:-per_query_head},paper_losa_backend=${PAPER_LOSA_BACKEND:-auto}"
    ;;
esac
minerva_tasks="minerva_math_algebra,minerva_math_counting_and_prob,minerva_math_geometry,minerva_math_intermediate_algebra,minerva_math_num_theory,minerva_math_prealgebra,minerva_math_precalc"
gsm8k_task=gsm8k
gsm8k_gen_length=256
if [[ "${model_type}" == "sdar" ]]; then
  gsm8k_task=gsm8k_sdar
  gsm8k_gen_length=4096
fi

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
    --model "${model_type}"
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
    local samples_path
    local reeval_output
    HF_ALLOW_CODE_EVAL=1 PYTHONPATH="${repo_root}:${script_dir}:${PYTHONPATH:-}" \
      "${python_bin}" "${args[@]}" "$@"
    samples_path="$(find "${output_root}/humaneval" -type f -name 'samples_humaneval_instruct_*.jsonl' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -z "${samples_path}" ]]; then
      echo "HumanEval completed but no samples JSONL was found under ${output_root}/humaneval" >&2
      return 1
    fi
    reeval_output="$(dirname "${samples_path}")/reeval_indent_normalized.json"
    echo "Re-evaluating HumanEval samples: ${samples_path}"
    HF_ALLOW_CODE_EVAL=1 PYTHONPATH="${repo_root}:${script_dir}:${PYTHONPATH:-}" \
      "${python_bin}" "${script_dir}/reevaluate_humaneval.py" \
      "${samples_path}" --label "${output_path}/${name}" --output "${reeval_output}"
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
    run_eval gsm8k "${gsm8k_task}" "${gsm8k_gen_length}" 0 "$@"
    ;;
  math)
    run_eval math "${minerva_tasks}" 512 0 "$@"
    ;;
  humaneval)
    run_eval humaneval humaneval_instruct 768 0 "$@"
    ;;
  all)
    run_eval mmlu mmlu_generative 128 4 "$@"
    run_eval gsm8k "${gsm8k_task}" "${gsm8k_gen_length}" 0 "$@"
    run_eval math "${minerva_tasks}" 512 0 "$@"
    run_eval humaneval humaneval_instruct 768 0 "$@"
    ;;
  *)
    echo "Unsupported BENCHMARK: ${benchmark} (expected mmlu, gsm8k, math, humaneval, or all)" >&2
    exit 2
    ;;
esac
