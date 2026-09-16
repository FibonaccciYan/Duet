#!/usr/bin/env bash
# Card A chain: LLaDA 2.0 (mmlu, gsm8k, humaneval) -> LLaDA 2.1 (humaneval)
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${script_dir}/../quality/instruct_dense_quality.sh"

echo "=== chain A: LLaDA 2.0 ==="
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_llada20_dense_thr095 \
  MODEL=/data0/gs/models/LLaDA2.0-mini THRESHOLD=0.95 EDITING_THRESHOLD=1.0 \
  BENCHMARKS="mmlu gsm8k humaneval" "$DRIVER" llada retest_llada20 0 12338 \
  || { echo "llada20 chain FAILED"; exit 1; }

echo "=== chain A: LLaDA 2.1 humaneval ==="
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_llada21_dense_thr07 \
  MODEL=/data0/ysy/models/LLaDA2.1-mini THRESHOLD=0.7 EDITING_THRESHOLD=0.5 \
  BENCHMARKS="humaneval" "$DRIVER" llada retest_llada21 0 12338 \
  || { echo "llada21 chain FAILED"; exit 1; }

echo "CHAIN A DONE"
