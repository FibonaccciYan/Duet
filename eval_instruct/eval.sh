#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${script_dir}"

model_type="${MODEL_TYPE:-llada}"
port="${MAIN_PROCESS_PORT:-12335}"
benchmark="${BENCHMARK:-gsm8k}"
sparse_dlm="${SPARSE_DLM:-true}"
method="${METHOD:-${RUNTIME_MODE:-}}"

case "${model_type}" in
  llada)
    block_length="${BLOCK_LENGTH:-32}"
    steps="${STEPS:-32}"
    mask_id="${MASK_ID:-156895}"
    eos_id="${EOS_ID:-156892}"
    default_ratio=0.7
    default_selection_interval=1
    default_query_dense_threshold=0
    default_refresh_step=0
    default_selection_layer=1
    default_prefix_min_length=0
    default_prefix_chunk_size=1024
    ;;
  sdar)
    block_length="${BLOCK_LENGTH:-32}"
    steps="${STEPS:-16}"
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

default_sdar_threshold=0.95
default_sdar_remasking=sequential

default_llada_threshold=0.7
default_editing_threshold=0.5
resolved_sdar_remasking="${REMASKING_STRATEGY:-${default_sdar_remasking}}"
case "${model_type}" in
  llada) model_env="${LLADA_MODEL_PATH:-}" ;;
  sdar) model_env="${SDAR_MODEL_PATH:-}" ;;
esac
model="${MODEL:-${model_env}}"
python_bin="${PYTHON:-$(command -v python 2>/dev/null || true)}"
if [[ -z "${model}" ]]; then
  if [[ "${model_type}" == "llada" ]]; then
    echo "LLADA_MODEL_PATH is required (or pass MODEL explicitly)" >&2
  else
    echo "SDAR_MODEL_PATH is required (or pass MODEL explicitly)" >&2
  fi
  exit 2
fi
if [[ -z "${python_bin}" ]]; then
  echo "Python was not found; set PYTHON to the current environment interpreter" >&2
  exit 2
fi

query_sparse="${QUERY_SPARSE:-true}"
prefix_sparse="${PREFIX_SPARSE:-true}"
if [[ "${model_type}" == "sdar" ]]; then
  default_deep_only_transfer=true
else
  default_deep_only_transfer=false
fi
prefix_budget="${PREFIX_TOKEN_BUDGET:-256}"
losa="${LOSA:-false}"
losa_active_topk="${LOSA_ACTIVE_TOPK:-5}"
losa_score_mode="${LOSA_SCORE_MODE:-query}"
losa_key_samples="${LOSA_KEY_SAMPLES:-32}"
output_path="${OUTPUT_PATH:-default}"

case "${sparse_dlm,,}" in
  1|true|yes|y) sparse_dlm=true ;;
  0|false|no|n) sparse_dlm=false ;;
  *)
    echo "Unsupported SPARSE_DLM value: ${sparse_dlm}" >&2
    exit 2
    ;;
esac

default_output_root="${repo_root}/../${model_type}_exp/${output_path}"
output_root="${OUTPUT_ROOT:-${default_output_root}}"

if [[ "${model_type}" == "sdar" ]]; then
  model_args="pretrained=${model},trust_remote_code=true,dtype=${DTYPE:-float16},attn_implementation=${ATTN_IMPLEMENTATION:-sdpa},method=${method},implementation=${IMPLEMENTATION:-optimized},moe_expert_patch=${MOE_EXPERT_PATCH:-false},block_length=${block_length},steps=${steps},temperature=${TEMPERATURE:-0.0},threshold=${THRESHOLD:-${default_sdar_threshold}},remasking_strategy=${resolved_sdar_remasking},eb_threshold=${EB_THRESHOLD:-0.35},mask_id=${mask_id}"
  if [[ -n "${EOS_ID:-}" ]]; then
    model_args+=",eos_id=${EOS_ID}"
  fi
else
  model_args="pretrained=${model},trust_remote_code=true,dtype=${DTYPE:-bfloat16},attn_implementation=${ATTN_IMPLEMENTATION:-sdpa},method=${method},implementation=${IMPLEMENTATION:-optimized},moe_expert_patch=${MOE_EXPERT_PATCH:-true},block_length=${block_length},steps=${steps},temperature=${TEMPERATURE:-0.0},threshold=${THRESHOLD:-${default_llada_threshold}},editing_threshold=${EDITING_THRESHOLD:-${default_editing_threshold}},num_to_transfer=${NUM_TO_TRANSFER:-1},mask_id=${mask_id},eos_id=${eos_id}"
fi

if [[ -n "${MAX_PROMPT_LEN:-}" ]]; then
  model_args+=",max_prompt_len=${MAX_PROMPT_LEN}"
fi

case "${method}" in
  sparse)
    case "${PREFIX_STRICT_BUDGET:-true}" in
      1|true|True|TRUE|yes|y) ;;
      *)
        echo "Paper evaluation requires PREFIX_STRICT_BUDGET=true" >&2
        exit 2
        ;;
    esac
    model_args+=",prefix_strict_budget=true"
    model_args+=",sparse_dlm_ratio=${SPARSE_DLM_RATIO:-${default_ratio}}"
    model_args+=",sparse_dlm_top_k=${SPARSE_DLM_TOP_K:-64}"
    model_args+=",sparse_dlm_selection_interval=${SPARSE_DLM_SELECTION_INTERVAL:-${default_selection_interval}}"
    model_args+=",query_dense_threshold=${QUERY_DENSE_THRESHOLD:-${default_query_dense_threshold}}"
    model_args+=",sparse_dlm_refresh_step=${SPARSE_DLM_REFRESH_STEP:-${default_refresh_step}}"
    model_args+=",sparse_dlm_selection_layer=${SPARSE_DLM_SELECTION_LAYER:-${default_selection_layer}}"
    model_args+=",sparse_dlm_deep_only_transfer=${SPARSE_DLM_DEEP_ONLY_TRANSFER:-${default_deep_only_transfer}}"
    model_args+=",query_sparse=${query_sparse},prefix_sparse=${prefix_sparse}"
    model_args+=",prefix_min_prefix_length=${PREFIX_MIN_PREFIX_LENGTH:-${default_prefix_min_length}}"
    model_args+=",prefix_token_budget=${prefix_budget}"
    model_args+=",prefix_chunk_size=${PREFIX_CHUNK_SIZE:-${default_prefix_chunk_size}}"
    model_args+=",prefix_share_layer_pairs=${PREFIX_SHARE_LAYER_PAIRS:-false}"
    model_args+=",prefix_selector=${PREFIX_SELECTOR:-qk_tc}"
    model_args+=",prefix_dense_before_query_selection=${PREFIX_DENSE_BEFORE_QUERY_SELECTION:-true}"
    model_args+=",prefix_rescreen_full_kv=${PREFIX_RESCREEN_FULL_KV:-true}"
    model_args+=",losa=${losa},losa_active_topk=${losa_active_topk}"
    model_args+=",losa_score_mode=${losa_score_mode},losa_key_samples=${losa_key_samples}"
    if [[ "${model_type}" == "llada" ]]; then
      model_args+=",query_min_prefix_length=${QUERY_MIN_PREFIX_LENGTH:-0}"
      model_args+=",query_losa_union=${QUERY_LOSA_UNION:-false}"
    fi
    ;;
  focus)
    model_args+=",focus_alpha=${FOCUS_ALPHA:-1.5}"
    ;;
  losa)
    model_args+=",paper_losa_page_size=${PAPER_LOSA_PAGE_SIZE:-16}"
    model_args+=",paper_losa_token_budget=${PAPER_LOSA_TOKEN_BUDGET:-256}"
    model_args+=",paper_losa_active_topk=${PAPER_LOSA_ACTIVE_TOPK:-5}"
    model_args+=",paper_losa_gqa_mode=${PAPER_LOSA_GQA_MODE:-per_query_head}"
    model_args+=",paper_losa_backend=${PAPER_LOSA_BACKEND:-auto}"
    model_args+=",paper_losa_kv_stats=${PAPER_LOSA_KV_STATS:-true}"
    model_args+=",paper_losa_kv_stats_chunk_size=${PAPER_LOSA_KV_STATS_CHUNK_SIZE:-1024}"
    model_args+=",paper_losa_kv_stats_output_dir=${PAPER_LOSA_KV_STATS_OUTPUT_DIR:-${output_root}/kv_stats}"
    model_args+=",paper_losa_kv_stats_include_heads=${PAPER_LOSA_KV_STATS_INCLUDE_HEADS:-false}"
    model_args+=",paper_losa_kv_stats_compact=${PAPER_LOSA_KV_STATS_COMPACT:-false}"
    ;;
esac

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
  if [[ "${model_type}" == "llada" ]]; then default_gen_length=16384; fi
  local gen_length="${GEN_LENGTH:-${default_gen_length}}"
  local num_fewshot="${NUM_FEWSHOT:-${default_fewshot}}"
  local args=(
    -m accelerate.commands.launch
    --main_process_port "${port}"
    -m "${EVAL_ENTRY:-scripts.common.eval_entry}"
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
  if [[ -n "${EVAL_SEED:-}" ]]; then
    args+=(--seed "${EVAL_SEED}")
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
    run_eval mmlu mmlu_generative 128 5 "$@"
    ;;
  gsm8k)
    run_eval gsm8k "${gsm8k_task}" "${gsm8k_gen_length}" 0 "$@"
    ;;
  math500)
    run_eval math500 math_500 2048 0 "$@"
    ;;
  humaneval)
    run_eval humaneval humaneval_instruct 768 0 "$@"
    ;;
  all)
    run_eval mmlu mmlu_generative 128 5 "$@"
    run_eval gsm8k "${gsm8k_task}" "${gsm8k_gen_length}" 0 "$@"
    run_eval math500 math_500 2048 0 "$@"
    run_eval humaneval humaneval_instruct 768 0 "$@"
    ;;
  *)
    echo "Unsupported BENCHMARK: ${benchmark} (expected mmlu, gsm8k, math500, humaneval, or all)" >&2
    exit 2
    ;;
esac
