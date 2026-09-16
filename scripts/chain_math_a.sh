#!/usr/bin/env bash
# Card A (GPU 4): LLaDA 2.0 MATH-500 full
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}/eval_instruct" || exit 1
export CUDA_VISIBLE_DEVICES=4
export RUNTIME_MODE=dense
export BENCHMARK=math500
export HF_DATASETS_CACHE=/data0/gs/.hf_cache_math500
export http_proxy=http://114.212.80.7:21087
export https_proxy=http://114.212.80.7:21087
export NUM_FEWSHOT=0
env MODEL_TYPE=llada MODEL=/data0/gs/models/LLaDA2.0-mini THRESHOLD=0.95 EDITING_THRESHOLD=1.0 \
  GEN_LENGTH=16384 OUTPUT_ROOT=/data0/gs/eval_results/retest_math500_llada20 \
  MAIN_PROCESS_PORT=12338 bash ./eval.sh || { echo "LLaDA20 MATH500 FULL FAILED"; exit 1; }
echo "CARD A DONE"
