#!/usr/bin/env bash
# Card A chain: LLaDA 2.0 (mmlu, gsm8k, humaneval) -> LLaDA 2.1 (humaneval)
set -u
DRIVER=/data0/gs/SparseDLM_LLaDA_SDAR/scripts/driver_eval_dense.sh
export HF_DATASETS_CACHE=/data0/gs/.hf_cache_llada20_dense_thr07

echo "=== chain A: LLaDA 2.0 ==="
env MODEL=/data0/gs/models/LLaDA2.0-mini THRESHOLD=0.7 EDITING_THRESHOLD=0.5 \
  BENCHMARKS="mmlu gsm8k humaneval" "$DRIVER" llada retest_llada20 0 12338 \
  || { echo "llada20 chain FAILED"; exit 1; }

echo "=== chain A: LLaDA 2.1 humaneval ==="
env MODEL=/data0/ysy/models/LLaDA2.1-mini THRESHOLD=0.7 EDITING_THRESHOLD=0.5 \
  BENCHMARKS="humaneval" "$DRIVER" llada retest_llada21 0 12338 \
  || { echo "llada21 chain FAILED"; exit 1; }

echo "CHAIN A DONE"
