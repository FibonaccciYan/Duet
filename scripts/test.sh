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
    ;;
  sdar)
    default_model_path=/data0/ysy/models/SDAR-8B-Chat
    default_python=/home/ysy/anaconda3/envs/dream/bin/python
    ;;
  *)
    echo "Unsupported MODEL_TYPE: ${model_type} (expected llada or sdar)" >&2
    exit 2
    ;;
esac
model_path="${MODEL_PATH:-${default_model_path}}"
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="${PYTHON}"
else
  python_bin="${default_python}"
fi
pattern="${PATTERN:-patch}"

args=(
  scripts/test.py
  --model "${model_type}"
  --pattern "${pattern}"
  --model_path "${model_path}"
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --gen_length "${GEN_LENGTH:-512}"
  --temperature "${TEMPERATURE:-0.0}"
  --remasking_strategy "${REMASKING_STRATEGY:-low_confidence_dynamic}"
  --eb_threshold "${EB_THRESHOLD:-0.35}"
  --editing_threshold "${EDITING_THRESHOLD:-0.0}"
  --num_to_transfer "${NUM_TO_TRANSFER:-1}"
  --sparse_dlm_ratio "${SPARSE_DLM_RATIO:-0.5}"
  --sparse_dlm_top_k "${SPARSE_DLM_TOP_K:-64}"
  --sparse_dlm_refresh_step "${SPARSE_DLM_REFRESH_STEP:-2}"
  --query_sparse "${QUERY_SPARSE:-true}"
  --prefix_token_budget "${PREFIX_TOKEN_BUDGET:-256}"
  --prefix_chunk_size "${PREFIX_CHUNK_SIZE:-256}"
  --losa "${LOSA:-false}"
  --losa_active_topk "${LOSA_ACTIVE_TOPK:-5}"
  --moe_expert_patch "${MOE_EXPERT_PATCH:-true}"
  --prompt "${PROMPT:-Write a short story about history.}"
)

if [[ -n "${PREFIX_SPARSE:-}" ]]; then
  args+=(--prefix_sparse "${PREFIX_SPARSE}")
fi

if [[ -n "${BLOCK_LENGTH:-}" ]]; then
  args+=(--block_length "${BLOCK_LENGTH}")
fi

if [[ -n "${STEPS:-}" ]]; then
  args+=(--steps "${STEPS}")
fi

if [[ -n "${THRESHOLD:-}" ]]; then
  args+=(--threshold "${THRESHOLD}")
fi

if [[ -n "${SPARSE_DLM_SELECTION_INTERVAL:-}" ]]; then
  args+=(--sparse_dlm_selection_interval "${SPARSE_DLM_SELECTION_INTERVAL}")
fi

if [[ -n "${SPARSE_DLM_DENSE_FALLBACK_MASK_COUNT:-}" ]]; then
  args+=(--sparse_dlm_dense_fallback_mask_count "${SPARSE_DLM_DENSE_FALLBACK_MASK_COUNT}")
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
