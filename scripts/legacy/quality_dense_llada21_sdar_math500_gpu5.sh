#!/usr/bin/env bash
# Card B (GPU 5): LLaDA 2.1 MATH-500 full -> SDAR MATH-500 full
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}/eval_instruct" || exit 1
export CUDA_VISIBLE_DEVICES=5
export RUNTIME_MODE=dense
export BENCHMARK=math500
export HF_DATASETS_CACHE=/data0/gs/.hf_cache_math500
export http_proxy=http://114.212.80.7:21087
export https_proxy=http://114.212.80.7:21087
export NUM_FEWSHOT=0

env MODEL_TYPE=llada MODEL=/data0/ysy/models/LLaDA2.1-mini THRESHOLD=0.7 EDITING_THRESHOLD=0.5 \
  GEN_LENGTH=16384 OUTPUT_ROOT=/data0/gs/eval_results/retest_math500_llada21 \
  MAIN_PROCESS_PORT=12339 bash ./eval.sh || { echo "LLaDA21 MATH500 FULL FAILED"; exit 1; }

env MODEL_TYPE=sdar MODEL=/data0/ysy/models/SDAR-8B-Chat-b32 THRESHOLD=0.85 \
  REMASKING_STRATEGY=low_confidence_static BLOCK_LENGTH=32 STEPS=32 GEN_LENGTH=4096 \
  OUTPUT_ROOT=/data0/gs/eval_results/retest_math500_sdar \
  MAIN_PROCESS_PORT=12340 bash ./eval.sh || { echo "SDAR MATH500 FULL FAILED"; exit 1; }

echo "CARD B DONE"
