#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"

python_bin="${PYTHON:-/home/ysy/anaconda3/envs/llada/bin/python}"
pattern="${PATTERN:-block_cache_sparse_dlm}"

args=(
  scripts/test.py
  --pattern "${pattern}"
  --model_path "${MODEL_PATH:-/data0/ysy/models/LLaDA2.1-mini}"
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --gen_length "${GEN_LENGTH:-512}"
  --block_length "${BLOCK_LENGTH:-32}"
  --steps "${STEPS:-32}"
  --temperature "${TEMPERATURE:-0.0}"
  --threshold "${THRESHOLD:-0.5}"
  --editing_threshold "${EDITING_THRESHOLD:-0.0}"
  --num_to_transfer "${NUM_TO_TRANSFER:-1}"
  --sparse_dlm_ratio "${SPARSE_DLM_RATIO:-0.5}"
  --sparse_dlm_top_k "${SPARSE_DLM_TOP_K:-64}"
  --sparse_dlm_selection_interval "${SPARSE_DLM_SELECTION_INTERVAL:-4}"
  --sparse_dlm_dense_fallback_mask_count "${SPARSE_DLM_DENSE_FALLBACK_MASK_COUNT:-4}"
  --query_sparse "${QUERY_SPARSE:-true}"
  --prefix_sparse "${PREFIX_SPARSE:-true}"
  --prefix_token_budget "${PREFIX_TOKEN_BUDGET:-256}"
  --prefix_chunk_size "${PREFIX_CHUNK_SIZE:-256}"
  --prompt "${PROMPT:-Write a short story about history.}"
)

if [[ -n "${TOP_P:-}" ]]; then
  args+=(--top_p "${TOP_P}")
fi

if [[ -n "${TOP_K:-}" ]]; then
  args+=(--top_k "${TOP_K}")
fi

"${python_bin}" "${args[@]}" "$@"
