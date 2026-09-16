#!/usr/bin/env bash
# Card B chain: SDAR (mmlu, humaneval) -> LLaDA 2.1 (mmlu, gsm8k)
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${script_dir}/../quality/instruct_dense_quality.sh"

echo "=== chain B: SDAR 8B ==="
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_sdar_dense_b32_thr085 \
  BENCHMARKS="mmlu humaneval" "$DRIVER" sdar retest_sdar 3 12339 \
  || { echo "sdar chain FAILED"; exit 1; }

echo "=== chain B: LLaDA 2.1 mmlu+gsm8k ==="
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_llada21_dense_thr07 \
  MODEL=/data0/ysy/models/LLaDA2.1-mini THRESHOLD=0.7 EDITING_THRESHOLD=0.5 \
  BENCHMARKS="mmlu gsm8k" "$DRIVER" llada retest_llada21 3 12339 \
  || { echo "llada21 chain FAILED"; exit 1; }

echo "CHAIN B DONE"
