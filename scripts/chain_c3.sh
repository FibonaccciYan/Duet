#!/usr/bin/env bash
# Card 3 (GPU 2): LLaDA 2.0 GSM8K + LLaDA 2.1 GSM8K (~9.5h)
set -u
DRIVER=/data0/gs/SparseDLM_LLaDA_SDAR/scripts/driver_eval_dense.sh
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_llada20_dense_thr07 \
  MODEL=/data0/gs/models/LLaDA2.0-mini THRESHOLD=0.7 EDITING_THRESHOLD=0.5 \
  BENCHMARKS="gsm8k" "$DRIVER" llada retest_llada20 2 12340 \
  || { echo "card3 llada20 gsm8k FAILED"; exit 1; }
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_llada21_dense_thr07 \
  MODEL=/data0/ysy/models/LLaDA2.1-mini THRESHOLD=0.7 EDITING_THRESHOLD=0.5 \
  BENCHMARKS="gsm8k" "$DRIVER" llada retest_llada21 2 12340 \
  || { echo "card3 llada21 gsm8k FAILED"; exit 1; }
echo "CARD 3 DONE"
