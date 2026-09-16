#!/usr/bin/env bash
# Card 4 (GPU 3): LLaDA 2.1 MMLU + HumanEval + SDAR HumanEval (~9.5h)
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${script_dir}/driver_eval_dense.sh"
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_llada21_dense_thr07 \
  MODEL=/data0/ysy/models/LLaDA2.1-mini THRESHOLD=0.7 EDITING_THRESHOLD=0.5 \
  BENCHMARKS="mmlu humaneval" "$DRIVER" llada retest_llada21 3 12341 \
  || { echo "card4 llada21 FAILED"; exit 1; }
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_sdar_dense_b32_thr085 \
  BENCHMARKS="humaneval" "$DRIVER" sdar retest_sdar 3 12341 \
  || { echo "card4 sdar humaneval FAILED"; exit 1; }
echo "CARD 4 DONE"
