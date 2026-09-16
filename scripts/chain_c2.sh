#!/usr/bin/env bash
# Card 2 (GPU 1): LLaDA 2.0 MMLU + HumanEval (~9.5h)
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${script_dir}/driver_eval_dense.sh"
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_llada20_dense_thr095 \
  MODEL=/data0/gs/models/LLaDA2.0-mini THRESHOLD=0.95 EDITING_THRESHOLD=1.0 \
  BENCHMARKS="mmlu humaneval" "$DRIVER" llada retest_llada20 1 12339 \
  || { echo "card2 llada20 FAILED"; exit 1; }
echo "CARD 2 DONE"
