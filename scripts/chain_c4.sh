#!/usr/bin/env bash
# Card 4 (GPU 3): LLaDA 2.1 MMLU + HumanEval + SDAR HumanEval (~9.5h)
set -u
DRIVER=/data0/gs/SparseDLM_LLaDA_SDAR/scripts/driver_eval_dense.sh
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_llada20_dense_thr095 \
  MODEL=/data0/ysy/models/LLaDA2.1-mini THRESHOLD=0.7 EDITING_THRESHOLD=0.5 \
  BENCHMARKS="mmlu humaneval" "$DRIVER" llada retest_llada21 3 12341 \
  || { echo "card4 llada21 FAILED"; exit 1; }
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_sdar_dense_b32_thr095 \
  BENCHMARKS="humaneval" "$DRIVER" sdar retest_sdar 3 12341 \
  || { echo "card4 sdar humaneval FAILED"; exit 1; }
echo "CARD 4 DONE"
