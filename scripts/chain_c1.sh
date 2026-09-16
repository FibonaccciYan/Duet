#!/usr/bin/env bash
# Card 1 (GPU 0): SDAR MMLU (~8h)
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${script_dir}/driver_eval_dense.sh"
env HF_DATASETS_CACHE=/data0/gs/.hf_cache_sdar_dense_b32_thr085 \
  BENCHMARKS="mmlu" "$DRIVER" sdar retest_sdar 0 12338 \
  || { echo "card1 SDAR mmlu FAILED"; exit 1; }
echo "CARD 1 DONE"
