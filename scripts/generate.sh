#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"

model_type="${MODEL_TYPE:-llada}"
case "${model_type}" in
  llada)
    default_model_path=/data0/ysy/models/LLaDA2.1-mini
    default_python=/home/ysy/anaconda3/envs/llada/bin/python
    default_ratio=0.7
    default_selection_interval=4
    default_query_dense_threshold=20
    default_query_min_prefix_length=24576
    default_selection_layer=1
    default_refresh_step=2
    default_prefix_chunk_size=1024
    ;;
  sdar)
    default_model_path=/data0/ysy/models/SDAR-8B-Chat-b32
    default_python=/home/ysy/anaconda3/envs/dream/bin/python
    default_refresh_step=-1
    default_moe_expert_patch=false
    default_ratio=0.5
    default_selection_interval=1
    default_query_dense_threshold=0
    default_query_min_prefix_length=0
    default_prefix_min_prefix_length=24576
    default_selection_layer=5
    default_prefix_chunk_size=1024
    ;;
  *)
    echo "Unsupported MODEL_TYPE: ${model_type} (expected llada or sdar)" >&2
    exit 2
    ;;
esac
if [[ "${model_type}" == "llada" ]]; then
  default_refresh_step=2
  default_moe_expert_patch=true
  default_prefix_min_prefix_length=4096
fi
model_path="${MODEL_PATH:-${default_model_path}}"
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="${PYTHON}"
else
  python_bin="${default_python}"
fi
pattern="${PATTERN:-patch}"

args=(
  scripts/generate.py
  --model "${model_type}"
  --pattern "${pattern}"
  --model_path "${model_path}"
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --gen_length "${GEN_LENGTH:-512}"
  --temperature "${TEMPERATURE:-0.0}"
  --editing_threshold "${EDITING_THRESHOLD:-0.0}"
  --num_to_transfer "${NUM_TO_TRANSFER:-1}"
  --sparse_dlm_ratio "${SPARSE_DLM_RATIO:-${default_ratio}}"
  --sparse_dlm_top_k "${SPARSE_DLM_TOP_K:-64}"
  --sparse_dlm_selection_interval "${SPARSE_DLM_SELECTION_INTERVAL:-${default_selection_interval}}"
  --query_dense_threshold "${QUERY_DENSE_THRESHOLD:-${default_query_dense_threshold}}"
  --query_min_prefix_length "${QUERY_MIN_PREFIX_LENGTH:-${default_query_min_prefix_length}}"
  --block_length "${BLOCK_LENGTH:-32}"
  --steps "${STEPS:-32}"
  --sparse_dlm_refresh_step "${SPARSE_DLM_REFRESH_STEP:-${default_refresh_step}}"
  --sparse_dlm_selection_layer "${SPARSE_DLM_SELECTION_LAYER:-${default_selection_layer}}"
  --sparse_dlm_deep_only_transfer "${SPARSE_DLM_DEEP_ONLY_TRANSFER:-false}"
  --query_sparse "${QUERY_SPARSE:-true}"
  --query_losa_union "${QUERY_LOSA_UNION:-false}"
  --prefix_min_prefix_length "${PREFIX_MIN_PREFIX_LENGTH:-${default_prefix_min_prefix_length}}"
  --prefix_token_budget "${PREFIX_TOKEN_BUDGET:-256}"
  --prefix_chunk_size "${PREFIX_CHUNK_SIZE:-${default_prefix_chunk_size}}"
  --losa "${LOSA:-false}"
  --losa_active_topk "${LOSA_ACTIVE_TOPK:-5}"
  --losa_score_mode "${LOSA_SCORE_MODE:-query}"
  --losa_key_samples "${LOSA_KEY_SAMPLES:-32}"
  --moe_expert_patch "${MOE_EXPERT_PATCH:-${default_moe_expert_patch}}"
  --prompt "${PROMPT:-Write a short story about history.}"
)

if [[ "${model_type}" == "sdar" ]]; then
  args+=(
    --remasking_strategy "${REMASKING_STRATEGY:-sequential}"
    --eb_threshold "${EB_THRESHOLD:-0.35}"
  )
fi

if [[ -n "${PREFIX_SPARSE:-}" ]]; then
  args+=(--prefix_sparse "${PREFIX_SPARSE}")
fi

if [[ -n "${THRESHOLD:-}" ]]; then
  args+=(--threshold "${THRESHOLD}")
fi

if [[ -n "${SPARSE_DLM_SELECTION_INTERVAL:-}" ]]; then
  args+=(--sparse_dlm_selection_interval "${SPARSE_DLM_SELECTION_INTERVAL}")
fi

if [[ -n "${QUERY_DENSE_THRESHOLD:-}" ]]; then
  args+=(--query_dense_threshold "${QUERY_DENSE_THRESHOLD}")
fi

if [[ -n "${MASK_ID:-}" ]]; then
  args+=(--mask_id "${MASK_ID}")
fi

if [[ -n "${EOS_ID:-}" ]]; then
  args+=(--eos_id "${EOS_ID}")
fi

if [[ -n "${TOP_P:-}" ]]; then
  args+=(--top_p "${TOP_P}")
fi

if [[ -n "${TOP_K:-}" ]]; then
  args+=(--top_k "${TOP_K}")
fi

"${python_bin}" "${args[@]}" "$@"
