#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ysy/anaconda3/envs/llada/bin/python}"
MODE="${MODE:-sparse}"

EXTRA_ARGS=()
case "${MODE}" in
  vanilla)
    EXTRA_ARGS+=(--use_model_generate)
    ;;
  trace)
    ;;
  sparse)
    EXTRA_ARGS+=(
      --local_demo_sparse_attn
      --demo_sparse_ratio "${DEMO_SPARSE_RATIO:-0.5}"
      --demo_sparse_mode "${DEMO_SPARSE_MODE:-kv}"
      --demo_sparse_dense_fallback_mask_count "${DEMO_SPARSE_DENSE_FALLBACK_MASK_COUNT:-4}"
    )
    ;;
  block_cache_sparse)
    EXTRA_ARGS+=(
      --block_cache_sparse_dlm
      --demo_sparse_ratio "${DEMO_SPARSE_RATIO:-0.5}"
      --sparse_dlm_top_k "${SPARSE_DLM_TOP_K:-64}"
      --sparse_dlm_selection_interval "${SPARSE_DLM_SELECTION_INTERVAL:-4}"
      --demo_sparse_dense_fallback_mask_count "${DEMO_SPARSE_DENSE_FALLBACK_MASK_COUNT:-4}"
    )
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use vanilla, trace, sparse, or block_cache_sparse." >&2
    exit 1
    ;;
esac

"${PYTHON_BIN}" "${SCRIPT_DIR}/trace_generation.py" \
  "${EXTRA_ARGS[@]}" \
  --no_plot_attentions \
  "$@"
